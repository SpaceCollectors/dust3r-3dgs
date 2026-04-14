"""
Surface-Constrained Gaussian Splatting

Initialize splats on mesh surface, train with surface anchor loss
to prevent floating. Yields progress for live viewport rendering.
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
import math
import time

import importlib.util, sys, os as _os

def _import_local(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_dir = _os.path.dirname(_os.path.abspath(__file__))
_train = _import_local("_local_train", _os.path.join(_dir, "train.py"))
_rast = _import_local("_local_rast", _os.path.join(_dir, "rasterizer.py"))

render_gaussians = _rast.render_gaussians
render_gaussians_gsplat = _train.render_gaussians_gsplat
load_colmap_dataset = _train.load_colmap_dataset
C0 = _train.C0
rgb_to_sh = _train.rgb_to_sh
knn_distances = _train.knn_distances
create_optimizers = _train.create_optimizers
depth_loss_fn = _train.depth_loss_fn
anisotropy_loss_fn = _train.anisotropy_loss_fn
scale_loss_fn = _train.scale_loss_fn
render_depth_from_pts3d = _train.render_depth_from_pts3d
_ssim = _train._ssim
save_ply = _train.save_ply


def sample_mesh_surface(verts, faces, colors, n_samples=50000):
    """Sample points uniformly on mesh triangles.
    Returns (points, normals, colors) as float32 numpy arrays."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    # Face areas for weighted sampling
    cross = np.cross(v1 - v0, v2 - v0)
    areas = np.linalg.norm(cross, axis=-1) * 0.5
    normals = cross / (np.linalg.norm(cross, axis=-1, keepdims=True) + 1e-8)
    probs = areas / (areas.sum() + 1e-8)

    # Sample faces proportional to area
    face_idx = np.random.choice(len(faces), size=n_samples, p=probs)

    # Random barycentric coordinates
    r1 = np.random.rand(n_samples, 1).astype(np.float32)
    r2 = np.random.rand(n_samples, 1).astype(np.float32)
    sqrt_r1 = np.sqrt(r1)
    bary = np.concatenate([1 - sqrt_r1, sqrt_r1 * (1 - r2), sqrt_r1 * r2], axis=1)

    # Interpolate positions
    fv0 = v0[face_idx]; fv1 = v1[face_idx]; fv2 = v2[face_idx]
    points = bary[:, 0:1] * fv0 + bary[:, 1:2] * fv1 + bary[:, 2:3] * fv2
    sample_normals = normals[face_idx]

    # Interpolate vertex colors
    c0 = colors[faces[face_idx, 0]].astype(np.float32)
    c1 = colors[faces[face_idx, 1]].astype(np.float32)
    c2 = colors[faces[face_idx, 2]].astype(np.float32)
    sample_colors = (bary[:, 0:1] * c0 + bary[:, 1:2] * c1 + bary[:, 2:3] * c2)

    return points.astype(np.float32), sample_normals.astype(np.float32), sample_colors.astype(np.uint8)


def init_surface_splats(points, normals, colors, device='cuda'):
    """Initialize splats from surface samples with normal-aligned orientations."""
    N = len(points)
    print(f"    Initializing {N:,d} splats...")

    means = torch.from_numpy(points).float().to(device)

    # Scale from KNN distances — start as flat discs that OVERLAP neighbors
    avg_dist = knn_distances(points, k=3)
    scales_xy = np.log(np.clip(avg_dist * 1.0, 1e-7, None))   # full neighbor distance (overlap)
    scales_z = np.log(np.clip(avg_dist * 0.2, 1e-7, None))    # 5x thinner along normal
    scales = np.stack([scales_xy, scales_xy, scales_z], axis=-1)
    scales = torch.from_numpy(scales).float().to(device)

    # Vectorized quaternion from Z-axis to normal (Rodrigues)
    z = np.array([0, 0, 1], dtype=np.float32)
    n = normals.copy()
    n /= (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8)

    # cross(z, n) and dot(z, n)
    axis = np.cross(z, n)  # (N, 3)
    axis_len = np.linalg.norm(axis, axis=-1)  # (N,)
    cos_angle = n[:, 2]  # dot([0,0,1], n) = n_z

    # Half-angle quaternion: q = [cos(a/2), sin(a/2)*axis]
    # Using: angle = arccos(cos_angle), half = angle/2
    # But faster: cos(a/2) = sqrt((1+cos_a)/2), sin(a/2) = sqrt((1-cos_a)/2)
    half_cos = np.sqrt(np.clip((1 + cos_angle) / 2, 0, 1))  # (N,)
    half_sin = np.sqrt(np.clip((1 - cos_angle) / 2, 0, 1))  # (N,)

    # Normalize axis (where valid)
    valid = axis_len > 1e-6
    axis[valid] /= axis_len[valid, None]
    axis[~valid] = [1, 0, 0]  # arbitrary for aligned/anti-aligned

    quats_np = np.zeros((N, 4), dtype=np.float32)
    quats_np[:, 0] = half_cos          # w
    quats_np[:, 1] = half_sin * axis[:, 0]  # x
    quats_np[:, 2] = half_sin * axis[:, 1]  # y
    quats_np[:, 3] = half_sin * axis[:, 2]  # z

    # Fix anti-aligned case (cos_angle ~ -1): 180 deg around X
    anti = cos_angle < -0.999
    quats_np[anti] = [0, 1, 0, 0]

    quats = torch.from_numpy(quats_np).float().to(device)

    # Opacity — start visible so splats aren't pruned before they learn
    opacities = torch.full((N,), 0.5, device=device)  # sigmoid(0.5) ~ 0.62

    # SH colors
    rgb = torch.from_numpy(colors / 255.0).float().to(device)
    sh0 = rgb_to_sh(rgb).unsqueeze(1)
    sh_degree = 3
    num_sh = (sh_degree + 1) ** 2
    shN = torch.zeros(N, num_sh - 1, 3, device=device)

    splats = torch.nn.ParameterDict({
        'means': torch.nn.Parameter(means),
        'scales': torch.nn.Parameter(scales),
        'quats': torch.nn.Parameter(quats),
        'opacities': torch.nn.Parameter(opacities),
        'sh0': torch.nn.Parameter(sh0),
        'shN': torch.nn.Parameter(shN),
    })
    return splats


def _quat_to_rotmat(q):
    """Convert quaternions (N, 4) [w,x,y,z] to rotation matrices (N, 3, 3)."""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([
        1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y),
            2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x),
            2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)


def flatness_loss(scales):
    """Penalize splats that aren't flat (disc-like).
    Encourages the smallest scale axis to be much smaller than the other two."""
    exp_s = torch.exp(scales)  # (N, 3)
    sorted_s, _ = exp_s.sort(dim=-1)  # ascending: thin, mid, thick
    thin = sorted_s[:, 0]
    mid = sorted_s[:, 1]
    # Ratio of thinnest to middle axis — want this close to 0
    ratio = thin / (mid + 1e-8)
    return ratio.mean()


def normal_align_loss(quats, scales, normals_t):
    """Penalize splats whose thin axis doesn't align with the surface normal.
    The thin axis (smallest scale) should be parallel to the surface normal."""
    R = _quat_to_rotmat(quats)  # (N, 3, 3) — columns are local axes

    # Find which axis is thinnest per splat
    exp_s = torch.exp(scales)
    thin_idx = exp_s.argmin(dim=-1)  # (N,) index 0,1,2

    # Extract the thin axis from rotation matrix
    # R[:, :, i] is the i-th local axis
    thin_axis = R[torch.arange(len(R), device=R.device), :, thin_idx]  # (N, 3)

    # Alignment: |dot(thin_axis, normal)| should be 1
    dots = (thin_axis * normals_t).sum(dim=-1).abs()  # (N,)
    return (1.0 - dots).mean()


def surface_anchor_loss(means, anchor_tree, anchor_pts_t, _cache={}):
    """L2 distance from each splat to nearest mesh point.
    Caches nearest-neighbor indices and only recomputes every 50 steps."""
    N = means.shape[0]
    cache_key = id(anchor_tree)
    # Recompute NN indices when splat count changes (densification) or every 50 calls
    step_count = _cache.get('step', 0)
    if (_cache.get('key') != cache_key or
        _cache.get('N') != N or
        step_count % 50 == 0):
        with torch.no_grad():
            pts_np = means.detach().cpu().numpy()
            _, idx = anchor_tree.query(pts_np)
            _cache['idx'] = torch.from_numpy(idx.astype(np.int64)).to(means.device)
            _cache['key'] = cache_key
            _cache['N'] = N
    _cache['step'] = step_count + 1
    nearest = anchor_pts_t[_cache['idx']]
    return F.mse_loss(means, nearest)


def smooth_splat_field(splats, strength=0.1, k=6):
    """Laplacian smoothing: blend each splat's properties with its neighbors.
    strength=0 is no smoothing, strength=1 fully replaces with neighbor average."""
    if strength <= 0:
        return
    from scipy.spatial import cKDTree

    with torch.no_grad():
        pts = splats['means'].data.cpu().numpy()
        tree = cKDTree(pts)
        _, nn_idx = tree.query(pts, k=k + 1)  # includes self
        nn_idx = nn_idx[:, 1:]  # exclude self -> (N, k)
        nn_idx_t = torch.from_numpy(nn_idx.astype(np.int64)).to(splats['means'].device)

        # Smooth scales: neighbors should have similar size
        scales = splats['scales'].data  # (N, 3)
        nn_scales = scales[nn_idx_t]  # (N, k, 3)
        mean_scales = nn_scales.mean(dim=1)  # (N, 3)
        splats['scales'].data.lerp_(mean_scales, strength)

        # Smooth quaternions: neighbors should have similar orientation
        # Linear average + renormalize (valid for small deviations)
        quats = splats['quats'].data  # (N, 4)
        nn_quats = quats[nn_idx_t]  # (N, k, 4)
        # Flip quaternions to same hemisphere as center (q and -q are same rotation)
        dots = (nn_quats * quats.unsqueeze(1)).sum(dim=-1, keepdim=True)  # (N, k, 1)
        nn_quats = torch.where(dots < 0, -nn_quats, nn_quats)
        mean_quats = nn_quats.mean(dim=1)  # (N, 4)
        mean_quats = F.normalize(mean_quats, dim=-1)
        splats['quats'].data.lerp_(mean_quats, strength)
        splats['quats'].data.copy_(F.normalize(splats['quats'].data, dim=-1))

        # Smooth SH DC color: neighbors should have similar base color
        sh0 = splats['sh0'].data  # (N, 1, 3)
        nn_sh0 = sh0[nn_idx_t]  # (N, k, 1, 3)
        mean_sh0 = nn_sh0.mean(dim=1)  # (N, 1, 3)
        splats['sh0'].data.lerp_(mean_sh0, strength * 0.5)  # less aggressive on color


def _compute_freq_gap(rendered, gt):
    """Compute where the rendered image is missing high-frequency detail.
    Returns a per-pixel score: high = needs more splats."""
    # Laplacian as frequency proxy (3x3 kernel)
    lap_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                              device=rendered.device, dtype=torch.float32)
    lap_kernel = lap_kernel.view(1, 1, 3, 3).expand(3, 1, 3, 3)

    # (H, W, 3) -> (1, 3, H, W)
    gt_p = gt.permute(2, 0, 1).unsqueeze(0)
    rd_p = rendered.detach().permute(2, 0, 1).unsqueeze(0)

    detail_gt = F.conv2d(gt_p, lap_kernel, padding=1, groups=3).abs().mean(dim=1, keepdim=True)
    detail_rd = F.conv2d(rd_p, lap_kernel, padding=1, groups=3).abs().mean(dim=1, keepdim=True)

    # Where GT has detail but render doesn't
    freq_gap = F.relu(detail_gt - detail_rd).squeeze(0).squeeze(0)  # (H, W)
    return freq_gap


def _compute_image_gradients(gt):
    """Compute image gradient direction at each pixel. Returns (H, W, 2) gradient vectors."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           device=gt.device, dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           device=gt.device, dtype=torch.float32).view(1, 1, 3, 3)
    gray = gt.mean(dim=-1, keepdim=True).permute(2, 0, 1).unsqueeze(0)  # (1,1,H,W)
    gx = F.conv2d(gray, sobel_x, padding=1).squeeze()  # (H, W)
    gy = F.conv2d(gray, sobel_y, padding=1).squeeze()  # (H, W)
    return torch.stack([gx, gy], dim=-1)  # (H, W, 2)


def _spawn_splats_at_gaps(freq_gap, alpha, w2c, K, H, W, cloud_pts, cloud_normals,
                          cloud_colors, cloud_tree, n_spawn=200, device='cuda'):
    """Spawn new splats where frequency gap or alpha holes are worst.
    Returns new splat parameters (means, normals, colors) in world space."""
    # Combined score: frequency gap + alpha holes
    alpha_gap = F.relu(0.5 - alpha.squeeze(-1))  # (H, W) — high where alpha < 0.5
    score = freq_gap + alpha_gap * 2.0  # weight alpha holes more

    # Sample top-scoring pixels (stochastic — weighted by score)
    score_flat = score.reshape(-1)
    if score_flat.sum() < 1e-8:
        return None
    probs = score_flat / score_flat.sum()
    n_spawn = min(n_spawn, (score_flat > 0).sum().item(), len(cloud_pts))
    if n_spawn == 0:
        return None

    pixel_idx = torch.multinomial(probs, n_spawn, replacement=False)
    py = (pixel_idx // W).float()
    px = (pixel_idx % W).float()

    # Unproject pixels to rays
    fx, fy = K[0, 0].item(), K[1, 1].item()
    cx, cy_k = K[0, 2].item(), K[1, 2].item()
    dirs_cam = torch.stack([
        (px - cx) / fx,
        (py - cy_k) / fy,
        torch.ones(n_spawn, device=device)
    ], dim=-1)  # (n_spawn, 3)

    # Camera position in world space
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    cam_pos = -R.T @ t  # (3,)

    # Ray directions in world space
    dirs_world = (R.T @ dirs_cam.T).T  # (n_spawn, 3)
    dirs_world = dirs_world / (dirs_world.norm(dim=-1, keepdim=True) + 1e-8)

    # Find nearest cloud point to each ray
    # Approximate: project cloud into this camera, find nearest cloud point to each pixel
    cloud_pts_t = torch.from_numpy(cloud_pts).float().to(device)
    cloud_cam = cloud_pts_t @ R.T + t[None, :]
    valid = cloud_cam[:, 2] > 0.1
    cloud_u = cloud_cam[:, 0] / cloud_cam[:, 2] * fx + cx
    cloud_v = cloud_cam[:, 1] / cloud_cam[:, 2] * fy + cy_k

    # For each spawn pixel, find nearest projected cloud point
    spawn_pts = []
    spawn_normals = []
    spawn_colors = []
    cloud_uv = torch.stack([cloud_u, cloud_v], dim=-1)  # (M, 2)
    spawn_uv = torch.stack([px, py], dim=-1)  # (n_spawn, 2)

    # Batch nearest neighbor in 2D projection space
    # Use chunked distance to avoid OOM
    valid_mask = valid.cpu().numpy()
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        return None
    cloud_uv_valid = cloud_uv[valid].cpu().numpy()

    from scipy.spatial import cKDTree
    tree_2d = cKDTree(cloud_uv_valid)
    spawn_uv_np = spawn_uv.cpu().numpy()
    _, nn_idx = tree_2d.query(spawn_uv_np)
    orig_idx = valid_idx[nn_idx]

    new_means = cloud_pts[orig_idx]
    new_normals = cloud_normals[orig_idx] if cloud_normals is not None else np.zeros_like(new_means)
    new_colors = cloud_colors[orig_idx] if cloud_colors is not None else np.full_like(new_means, 128, dtype=np.uint8)

    return new_means, new_normals, new_colors


def train_surface_splats(
    mesh_data=None, point_cloud=None, colmap_dir=None,
    device='cuda', iterations=3000, max_resolution=1024,
    n_samples=50000,
    anchor_weight_start=0.1, anchor_weight_end=0.001,
    depth_lambda=0.5, aniso_lambda=0.0,
    flatness_lambda=0.01, normal_lambda=0.05,
    opacity_decay=0.001, prune_threshold=0.002,
    max_scale=2.0,
    target_splats=0,
    strategy_name='simple',
    multi_view=False,
    multi_view_count=2,  # how many views per step when multi_view=True
    smooth_strength=0.0,
    coverage_lambda=0.5,
    stop_flag=None,
):
    """Generator: train surface-constrained splats, yield progress dicts.

    Accepts either mesh_data=(verts, faces, colors) or point_cloud=(points, colors).
    If mesh is provided, samples surface points. If point cloud, uses directly.
    """
    from scipy.spatial import cKDTree

    if mesh_data is not None:
        verts, faces, colors_mesh = mesh_data
        print(f"Surface splat training: {len(verts):,d} verts, {len(faces):,d} faces")
        print(f"  Sampling {n_samples:,d} points on mesh surface...")
        points, normals, scolors = sample_mesh_surface(verts, faces, colors_mesh, n_samples)
        anchor_pts = verts.astype(np.float32)
    elif point_cloud is not None:
        cloud_pts, cloud_cols = point_cloud
        print(f"Splat training from point cloud: {len(cloud_pts):,d} points")
        # Subsample if too many
        if len(cloud_pts) > n_samples:
            idx = np.random.choice(len(cloud_pts), n_samples, replace=False)
            points = cloud_pts[idx].astype(np.float32)
            scolors = cloud_cols[idx].astype(np.uint8)
        else:
            points = cloud_pts.astype(np.float32)
            scolors = cloud_cols.astype(np.uint8)
        # Estimate normals from KNN
        tree_tmp = cKDTree(points)
        normals = np.zeros_like(points)
        _, nn_idx = tree_tmp.query(points, k=8)
        for i in range(len(points)):
            neighbors = points[nn_idx[i, 1:]]
            centered = neighbors - points[i]
            if len(centered) >= 3:
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                normals[i] = vh[-1]  # smallest singular vector = normal
        normals /= (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8)
        anchor_pts = points
    else:
        raise ValueError("Must provide mesh_data or point_cloud")

    # Build anchor KD-tree
    anchor_tree = cKDTree(anchor_pts)
    anchor_pts_t = torch.from_numpy(anchor_pts).to(device)

    # Load camera data
    print(f"  Loading COLMAP dataset from {colmap_dir}")
    images, _, _ = load_colmap_dataset(colmap_dir, max_resolution=max_resolution)
    print(f"  {len(images)} training views")

    if len(images) == 0:
        yield {'step': 0, 'total': iterations, 'loss': 0, 'n_splats': 0, 'done': True}
        return

    # Scene scale
    cam_centers = np.array([img['c2w'][:3, 3] for img in images])
    if len(cam_centers) > 1:
        from scipy.spatial.distance import pdist
        scene_scale = float(np.median(pdist(cam_centers)))
    else:
        scene_scale = 1.0

    # Depth priors
    print("  Computing depth priors...")
    depth_maps = []
    all_pts3d = np.concatenate([anchor_pts, points], axis=0)
    for img in images:
        dm = render_depth_from_pts3d(all_pts3d, img['w2c'], img['K'], img['H'], img['W'])
        depth_maps.append(torch.from_numpy(dm).float().to(device))

    # Initialize splats
    print(f"  Initializing {len(points):,d} splats...")
    splats = init_surface_splats(points, normals, scolors, device)
    optimizers = create_optimizers(splats, scene_scale)

    # Compute max allowed scale from initial splat density
    init_knn = knn_distances(points, k=3)
    max_scale_abs = float(np.median(init_knn) * 3.0)
    print(f"  Max splat scale: {max_scale_abs:.4f} (3x median KNN={np.median(init_knn):.4f})")

    # Store per-splat surface normals (not a parameter — fixed targets)
    n_norm = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8)
    splat_normals = torch.from_numpy(n_norm).float().to(device)
    means_scheduler = optim.lr_scheduler.ExponentialLR(
        optimizers['means'], gamma=0.01 ** (1.0 / iterations))

    # ── Cloud reservoir for adaptive spawning ──
    # Keep the full point cloud available for spawning new splats
    cloud_pts_full = points.copy()  # (M, 3) all cloud points
    cloud_normals_full = n_norm.copy()  # (M, 3) normalized normals
    cloud_colors_full = scolors.copy()  # (M, 3) uint8 colors

    # ── Strategy setup ──
    gsplat_strategy = None
    strategy_state = {}
    use_absgrad = False

    if strategy_name == 'mcmc':
        from gsplat.strategy import MCMCStrategy
        cap = target_splats if target_splats > 0 else 500000
        gsplat_strategy = MCMCStrategy(
            cap_max=cap, noise_lr=5e4,
            refine_start_iter=200, refine_stop_iter=int(iterations * 0.85),
            refine_every=100, min_opacity=max(prune_threshold, 0.005),
        )
        strategy_state = gsplat_strategy.initialize_state()
        strategy_state['splat_normals'] = splat_normals  # auto-synced by ops
        gsplat_strategy.check_sanity(splats, optimizers)
        print(f"  Strategy: MCMC (cap={cap:,d})")

    elif strategy_name == 'absgrad':
        from gsplat.strategy import DefaultStrategy
        gsplat_strategy = DefaultStrategy(
            absgrad=True,
            grow_grad2d=0.0008,  # higher for absgrad
            grow_scale3d=0.01,
            prune_opa=max(prune_threshold, 0.005),
            refine_start_iter=200, refine_stop_iter=int(iterations * 0.85),
            refine_every=100, reset_every=max(1000, iterations // 3),
            revised_opacity=True,
        )
        strategy_state = gsplat_strategy.initialize_state(scene_scale=scene_scale)
        strategy_state['splat_normals'] = splat_normals
        gsplat_strategy.check_sanity(splats, optimizers)
        use_absgrad = True
        print(f"  Strategy: AbsGrad/IGS+")

    elif strategy_name == 'mrnf':
        print(f"  Strategy: MRNF (organic opacity decay)")

    elif strategy_name == 'adaptive':
        print(f"  Strategy: Adaptive (frequency-aware spawning)")

    else:
        print(f"  Strategy: Simple (scale-weighted)")

    # Prepare image tensors
    for img in images:
        img['pixels_t'] = torch.from_numpy(img['pixels']).float().to(device)
        img['K_t'] = torch.from_numpy(img['K']).float().to(device)
        img['w2c_t'] = torch.from_numpy(img['w2c'].astype(np.float32)).to(device)

    # Try gsplat (fast CUDA), fall back to pure PyTorch rasterizer
    try:
        from gsplat.rendering import rasterization as _test_gsplat
        use_gsplat = True
        CROP = 0  # gsplat is fast enough for full-frame rendering
        print("  Using gsplat CUDA rasterizer")
    except Exception:
        use_gsplat = False
        CROP = 128  # small crops for pure PyTorch rasterizer
        print("  Using pure PyTorch rasterizer (gsplat not available)")
    ssim_lambda = 0.2

    # Precompute camera neighborhoods for multi-view grouping
    MAX_VIEWS_PER_STEP = max(2, multi_view_count)
    if len(images) > MAX_VIEWS_PER_STEP:
        cam_positions = np.array([img['c2w'][:3, 3] for img in images])
        from scipy.spatial import cKDTree
        cam_tree = cKDTree(cam_positions)
        # For each camera, store indices of its nearest neighbors
        _, cam_neighbors = cam_tree.query(cam_positions, k=min(MAX_VIEWS_PER_STEP, len(images)))
        print(f"  Multi-view groups: {MAX_VIEWS_PER_STEP} nearest cameras per step")
    t0 = time.time()

    # Yield initial state so viewport shows splats immediately
    with torch.no_grad():
        m = splats['means'].detach().cpu().numpy()
        c = (splats['sh0'].squeeze(1).detach() * C0 + 0.5).clamp(0, 1).cpu().numpy()
        s = torch.exp(splats['scales'].detach()).max(dim=-1).values.cpu().numpy()
    yield {
        'step': 0, 'total': iterations,
        'loss': 0.0, 'n_splats': len(m),
        'means': m, 'colors': c, 'scales': s,
        'done': False,
    }

    print(f"  Training for {iterations} iterations...")
    import sys; sys.stdout.flush()

    for step in range(iterations):
        if stop_flag and stop_flag():
            break

        # Anchor weight decay
        t_frac = step / max(iterations - 1, 1)
        anchor_w = anchor_weight_start + (anchor_weight_end - anchor_weight_start) * t_frac

        # Select views
        if multi_view:
            if len(images) <= MAX_VIEWS_PER_STEP:
                # Few cameras — render all
                view_indices = list(range(len(images)))
            else:
                # Many cameras — pick a random anchor and its nearest neighbors
                anchor_cam = torch.randint(len(images), (1,)).item()
                view_indices = list(cam_neighbors[anchor_cam])
        else:
            view_indices = [torch.randint(len(images), (1,)).item()]

        # Activations (shared across views)
        means = splats['means']
        quats = splats['quats']
        scales = torch.exp(splats['scales'])
        opacities = torch.sigmoid(splats['opacities'])

        # Clamp scales
        with torch.no_grad():
            splats['scales'].data.clamp_(min=-10.0, max=math.log(max_scale_abs))

        bg = torch.rand(3, device=device)

        # Accumulate losses across all selected views
        color_loss = torch.tensor(0.0, device=device)
        d_loss = torch.tensor(0.0, device=device)
        info = {}

        for vi, idx in enumerate(view_indices):
            img = images[idx]
            pixels_full = img['pixels_t']
            K_full = img['K_t']
            w2c = img['w2c_t']
            H, W = img['H'], img['W']
            depth_gt_full = depth_maps[idx]

            # Random crop (only for non-gsplat single-view)
            if not multi_view and CROP > 0 and min(H, W) > CROP and step % 50 != 0:
                cy = torch.randint(0, H - CROP, (1,)).item()
                cx = torch.randint(0, W - CROP, (1,)).item()
                pixels = pixels_full[cy:cy+CROP, cx:cx+CROP]
                depth_gt = depth_gt_full[cy:cy+CROP, cx:cx+CROP]
                K = K_full.clone()
                K[0, 2] -= cx
                K[1, 2] -= cy
                rH, rW = CROP, CROP
            else:
                pixels = pixels_full
                K = K_full
                depth_gt = depth_gt_full
                rH, rW = H, W

            # Render this view
            if use_gsplat:
                sh_coeffs = torch.cat([splats['sh0'], splats['shN']], dim=1)
                sh_degree = int(math.sqrt(sh_coeffs.shape[1])) - 1
                need_meta = gsplat_strategy is not None
                result = render_gaussians_gsplat(
                    means3d=means, scales=scales, quats=quats,
                    opacities=opacities, colors=None,
                    viewmat=w2c, K=K, W=rW, H=rH, bg_color=bg,
                    sh_degree=sh_degree, sh_coeffs=sh_coeffs,
                    absgrad=use_absgrad, return_meta=need_meta,
                )
                if need_meta:
                    colors_pred, depth_pred, alpha_pred, view_info = result
                    if vi == 0:
                        info = view_info  # use first view's info for strategy
                else:
                    colors_pred, depth_pred, alpha_pred = result
            else:
                colors_rgb = (splats['sh0'].squeeze(1) * C0 + 0.5).clamp(0, 1)
                colors_pred, depth_pred, alpha_pred = render_gaussians(
                    means3d=means, scales=scales, quats=quats,
                    opacities=opacities, colors=colors_rgb,
                    viewmat=w2c, K=K, W=rW, H=rH, bg_color=bg)

            # Strategy pre-backward (only first view to retain gradients)
            if vi == 0 and gsplat_strategy is not None:
                gsplat_strategy.step_pre_backward(
                    splats, optimizers, strategy_state, step, info)

            # Per-view losses
            l1 = F.l1_loss(colors_pred, pixels)
            pred_p = colors_pred.permute(2, 0, 1).unsqueeze(0)
            gt_p = pixels.permute(2, 0, 1).unsqueeze(0)
            ssim_val = _ssim(pred_p, gt_p)
            color_loss = color_loss + (1 - ssim_lambda) * l1 + ssim_lambda * (1 - ssim_val)
            d_loss = d_loss + depth_loss_fn(depth_pred, depth_gt, alpha_pred)

        # Average across views
        n_views = len(view_indices)
        color_loss = color_loss / n_views
        d_loss = d_loss / n_views

        # Opacity decay
        with torch.no_grad():
            if strategy_name == 'mrnf':
                opa_sig = torch.sigmoid(splats['opacities'].data)
                decay_factor = 1.0 - opacity_decay * 0.1
                opa_sig *= decay_factor
                splats['opacities'].data.copy_(torch.log(opa_sig / (1.0 - opa_sig + 1e-8)))
            elif opacity_decay > 0:
                splats['opacities'].data -= opacity_decay * optimizers['opacities'].defaults['lr']

        # Coverage loss: only image-based loss besides color
        coverage_loss = F.relu(0.95 - alpha_pred).mean()

        # Only color + coverage — no regularization losses fighting the optimizer
        loss = color_loss + depth_lambda * d_loss + coverage_lambda * coverage_loss

        # Neighbor smoothing every 100 steps
        if smooth_strength > 0 and step > 0 and step % 100 == 0:
            smooth_splat_field(splats, strength=smooth_strength)

        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        means_scheduler.step()

        # ── Direct constraints (applied AFTER optimizer step) ──
        # The optimizer runs free on color loss, then we project parameters
        # back into allowed ranges. No loss terms fighting each other.
        with torch.no_grad():
            splat_normals_cur = strategy_state.get('splat_normals', splat_normals)

            # Anchor: lerp positions toward nearest surface point
            if anchor_w > 0 and step % 10 == 0:
                pts_np = splats['means'].data.cpu().numpy()
                _, nn_idx = anchor_tree.query(pts_np)
                nearest = anchor_pts_t[torch.from_numpy(nn_idx.astype(np.int64)).to(device)]
                splats['means'].data.lerp_(nearest, anchor_w)
                # Update cache for other uses
                nn_cache = surface_anchor_loss.__defaults__[0]
                nn_cache['idx'] = torch.from_numpy(nn_idx.astype(np.int64)).to(device)
                nn_cache['N'] = splats['means'].shape[0]

            # Normal alignment: slerp quaternion toward surface-aligned orientation
            if normal_lambda > 0 and len(splat_normals_cur) == splats['quats'].shape[0]:
                R_cur = _quat_to_rotmat(splats['quats'].data)
                exp_s = torch.exp(splats['scales'].data)
                thin_idx = exp_s.argmin(dim=-1)
                thin_axis = R_cur[torch.arange(len(R_cur), device=device), :, thin_idx]

                # How misaligned is each splat? (0 = perfect, 1 = perpendicular)
                alignment = (thin_axis * splat_normals_cur).sum(dim=-1).abs()
                # Correction: rotate thin axis toward normal
                # Use small rotation via cross product
                cross = torch.cross(thin_axis, splat_normals_cur, dim=-1)
                cross_norm = cross.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                cross = cross / cross_norm
                # Small angle correction (proportional to misalignment and slider)
                angle = (1.0 - alignment) * normal_lambda  # slider 1 = full correction
                # Apply as delta quaternion: [cos(a/2), sin(a/2)*axis]
                half_a = angle * 0.5
                dq = torch.cat([half_a.cos().unsqueeze(-1), half_a.sin().unsqueeze(-1) * cross], dim=-1)
                # Quaternion multiply: q_new = dq * q_old
                q = splats['quats'].data
                w0, x0, y0, z0 = dq[:, 0], dq[:, 1], dq[:, 2], dq[:, 3]
                w1, x1, y1, z1 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
                splats['quats'].data = F.normalize(torch.stack([
                    w0*w1 - x0*x1 - y0*y1 - z0*z1,
                    w0*x1 + x0*w1 + y0*z1 - z0*y1,
                    w0*y1 - x0*z1 + y0*w1 + z0*x1,
                    w0*z1 + x0*y1 - y0*x1 + z0*w1,
                ], dim=-1), dim=-1)

            # Anisotropy: hard clamp max/min scale ratio
            if aniso_lambda > 0:
                # slider 0 = ratio 20 allowed, slider 1 = ratio 1 (isotropic)
                max_ratio = 1.0 + (20.0 - 1.0) * (1.0 - aniso_lambda)
                s = splats['scales'].data
                exp_s = torch.exp(s)
                s_max = exp_s.max(dim=-1, keepdim=True).values
                s_min = exp_s.min(dim=-1, keepdim=True).values.clamp(min=1e-8)
                ratio = s_max / s_min
                need_fix = (ratio > max_ratio).squeeze(-1)
                if need_fix.any():
                    # Hard clamp: shrink max and grow min to meet the allowed ratio
                    mid = exp_s[need_fix].median(dim=-1, keepdim=True).values
                    allowed_max = mid * math.sqrt(max_ratio)
                    allowed_min = mid / math.sqrt(max_ratio)
                    clamped = exp_s[need_fix].clamp(min=allowed_min, max=allowed_max)
                    splats['scales'].data[need_fix] = torch.log(clamped.clamp(min=1e-8))

            # Flatness: push thin axis thinner relative to other axes
            if flatness_lambda > 0:
                s = splats['scales'].data
                exp_s = torch.exp(s)
                min_idx = exp_s.argmin(dim=-1)
                mid_val = exp_s.median(dim=-1).values
                # Target: thin axis = mid_val * (1 - flatness)
                target_thin = torch.log((mid_val * (1.0 - flatness_lambda * 0.9)).clamp(min=1e-7))
                current_thin = s[torch.arange(len(s), device=device), min_idx]
                # Only push thinner, don't inflate
                new_thin = torch.min(current_thin, current_thin.lerp(target_thin, flatness_lambda))
                s[torch.arange(len(s), device=device), min_idx] = new_thin

        # ── Strategy post-backward (densification/pruning) ──
        if gsplat_strategy is not None:
            n_before = splats['means'].shape[0]
            post_kwargs = {}
            if strategy_name == 'mcmc':
                post_kwargs['lr'] = optimizers['means'].param_groups[0]['lr']
            post_kwargs_full = dict(**post_kwargs)
            if strategy_name != 'mcmc':
                post_kwargs_full['packed'] = True
            gsplat_strategy.step_post_backward(
                splats, optimizers, strategy_state, step, info,
                **post_kwargs_full)
            n_after = splats['means'].shape[0]
            if n_after != n_before:
                # Strategy changed splat count — update normals reference
                splat_normals = strategy_state.get('splat_normals', splat_normals)
                if len(splat_normals) != n_after:
                    # Fallback: requery normals from anchor tree
                    with torch.no_grad():
                        pts_np = splats['means'].detach().cpu().numpy()
                        _, nn_idx = anchor_tree.query(pts_np)
                        n_anch = anchor_pts_t.cpu().numpy()
                        new_normals = n_anch[nn_idx] - pts_np
                        nrm = np.linalg.norm(new_normals, axis=-1, keepdims=True)
                        nrm[nrm < 1e-8] = 1.0
                        # Use anchor point normals instead
                        splat_normals = torch.from_numpy(
                            (normals[nn_idx % len(normals)] if len(normals) > 0
                             else np.zeros_like(pts_np))).float().to(device)
                        strategy_state['splat_normals'] = splat_normals
                print(f"    [{strategy_name}] {n_before:,d} -> {n_after:,d} splats")

        # ── Adaptive frequency-aware cell division ──
        elif strategy_name == 'adaptive' and use_gsplat:
            target = target_splats if target_splats > 0 else 500000
            cur_n = splats['means'].shape[0]

            # Cell division every 50 steps during first 85% of training
            if step > 100 and step % 50 == 0 and cur_n < target and step < iterations * 85 // 100:
                with torch.no_grad():
                    # Score each splat: how much frequency detail is it missing?
                    # Project each splat center into the last rendered view
                    R_cam = w2c[:3, :3]
                    t_cam = w2c[:3, 3]
                    means_cam = splats['means'].data @ R_cam.T + t_cam[None, :]
                    z = means_cam[:, 2].clamp(min=0.1)
                    u = (means_cam[:, 0] / z * K[0, 0] + K[0, 2]).long()
                    v = (means_cam[:, 1] / z * K[1, 1] + K[1, 2]).long()

                    # Compute frequency gap map
                    freq_gap = _compute_freq_gap(colors_pred, pixels)  # (H, W)

                    # Score each splat by the freq gap at its projected pixel
                    in_bounds = (u >= 0) & (u < rW) & (v >= 0) & (v < rH) & (z > 0.1)
                    splat_scores = torch.zeros(cur_n, device=device)
                    valid_mask = in_bounds
                    splat_scores[valid_mask] = freq_gap[
                        v[valid_mask].clamp(0, rH - 1),
                        u[valid_mask].clamp(0, rW - 1)]

                    # Also boost score for splats covering alpha holes
                    alpha_gap = F.relu(0.5 - alpha_pred.squeeze(-1))  # (H, W)
                    splat_scores[valid_mask] += alpha_gap[
                        v[valid_mask].clamp(0, rH - 1),
                        u[valid_mask].clamp(0, rW - 1)] * 2.0

                    # Select top-scoring splats to split (up to budget)
                    n_split = min(cur_n // 10, target - cur_n, 5000)  # max 10% per step
                    if n_split > 0 and splat_scores.max() > 0:
                        _, split_idx = torch.topk(splat_scores, min(n_split, cur_n))

                        # Cell division: split along longest axis
                        parent_scales = splats['scales'].data[split_idx]  # (K, 3)
                        parent_means = splats['means'].data[split_idx]    # (K, 3)
                        parent_quats = splats['quats'].data[split_idx]    # (K, 4)

                        # Find longest axis per splat
                        longest_ax = parent_scales.argmax(dim=-1)  # (K,)

                        # Get rotation matrix to find world-space direction of longest axis
                        R_splat = _quat_to_rotmat(parent_quats)  # (K, 3, 3)
                        # Extract the longest axis direction for each splat
                        ax_dir = R_splat[torch.arange(len(split_idx), device=device), :, longest_ax]  # (K, 3)

                        # Offset distance = half the scale along longest axis
                        offset_dist = torch.exp(parent_scales[
                            torch.arange(len(split_idx), device=device), longest_ax]) * 0.5

                        # Child positions: parent ± offset along longest axis
                        offset = ax_dir * offset_dist.unsqueeze(-1)  # (K, 3)

                        # Build children — all params copied, then modify
                        for key in splats:
                            parent_data = splats[key].data[split_idx].clone()
                            child_data = parent_data.clone()

                            if key == 'means':
                                # Parent shifts one way, child the other
                                splats[key].data[split_idx] = parent_data + offset
                                child_data = parent_data - offset
                            elif key == 'scales':
                                # Halve the longest axis (log space: subtract log(2))
                                new_scales = parent_data.clone()
                                new_scales[torch.arange(len(split_idx)), longest_ax] -= 0.693
                                splats[key].data[split_idx] = new_scales
                                child_data = new_scales.clone()

                            splats[key] = torch.nn.Parameter(
                                torch.cat([splats[key].data, child_data], dim=0))

                        # Inherit normals
                        splat_normals = torch.cat([splat_normals, splat_normals[split_idx]], dim=0)

                        print(f"    [adaptive] split {len(split_idx)} -> {splats['means'].shape[0]:,d} splats")

                        optimizers = create_optimizers(splats, scene_scale)
                        means_scheduler = optim.lr_scheduler.ExponentialLR(
                            optimizers['means'],
                            gamma=0.01 ** (1.0 / max(1, iterations - step)))

            # Prune every 200 steps
            if step > 0 and step % 200 == 0 and prune_threshold > 0:
                with torch.no_grad():
                    cur_n = splats['means'].shape[0]
                    alive = torch.sigmoid(splats['opacities'].data) > prune_threshold
                    if alive.sum() < cur_n:
                        n_pruned = cur_n - alive.sum().item()
                        for key in splats:
                            splats[key] = torch.nn.Parameter(splats[key].data[alive])
                        splat_normals = splat_normals[alive]
                        print(f"    [adaptive] pruned {n_pruned} -> {splats['means'].shape[0]:,d} splats")
                        optimizers = create_optimizers(splats, scene_scale)
                        means_scheduler = optim.lr_scheduler.ExponentialLR(
                            optimizers['means'],
                            gamma=0.01 ** (1.0 / max(1, iterations - step)))

        # ── Simple/MRNF manual densification ──
        elif (strategy_name in ('simple', 'mrnf')) and step > 0 and step % 200 == 0:
            target = target_splats
            with torch.no_grad():
                cur_n = splats['means'].shape[0]

                # Prune dead splats
                alive = torch.sigmoid(splats['opacities'].data) > prune_threshold
                if alive.sum() < cur_n:
                    n_pruned = cur_n - alive.sum().item()
                    for key in splats:
                        splats[key] = torch.nn.Parameter(splats[key].data[alive])
                    splat_normals = splat_normals[alive]
                    cur_n = splats['means'].shape[0]
                    print(f"    Pruned {n_pruned} -> {cur_n:,d} splats")

                # Densify: grow toward target
                if target > 0 and cur_n < target and step < iterations * 3 // 4:
                    n_add = min(target - cur_n, cur_n)
                    if n_add > 0:
                        scale_mag = torch.exp(splats['scales'].data).max(dim=-1).values
                        probs = scale_mag / scale_mag.sum()
                        idx_s = torch.multinomial(probs, n_add, replacement=True)
                        for key in splats:
                            new_data = splats[key].data[idx_s].clone()
                            if key == 'means':
                                offset_scale = torch.exp(splats['scales'].data[idx_s]).max(dim=-1).values
                                noise = torch.randn_like(new_data)
                                noise = noise / (noise.norm(dim=-1, keepdim=True) + 1e-8)
                                new_data = new_data + noise * offset_scale[:, None] * 0.5
                            elif key == 'opacities':
                                # Child starts slightly transparent, parent unchanged
                                new_data -= 0.5
                            elif key == 'scales':
                                # Child starts slightly smaller, parent unchanged
                                new_data -= 0.3
                            splats[key] = torch.nn.Parameter(
                                torch.cat([splats[key].data, new_data], dim=0))
                        splat_normals = torch.cat([splat_normals, splat_normals[idx_s]], dim=0)
                        print(f"    Densified +{n_add:,d} -> {splats['means'].shape[0]:,d} splats")

                # Final trimming
                if target > 0 and cur_n > target and step >= iterations * 3 // 4:
                    n_remove = cur_n - target
                    opa = torch.sigmoid(splats['opacities'].data)
                    _, remove_idx = torch.topk(opa, n_remove, largest=False)
                    keep = torch.ones(cur_n, dtype=torch.bool, device=device)
                    keep[remove_idx] = False
                    for key in splats:
                        splats[key] = torch.nn.Parameter(splats[key].data[keep])
                    splat_normals = splat_normals[keep]
                    print(f"    Trimmed -> {splats['means'].shape[0]:,d} splats")

                optimizers = create_optimizers(splats, scene_scale)
                means_scheduler = optim.lr_scheduler.ExponentialLR(
                    optimizers['means'],
                    gamma=0.01 ** (1.0 / max(1, iterations - step)))

        # Yield progress every 50 steps (+ step 1 to confirm start)
        if step <= 1 or step % 50 == 0 or step == iterations - 1:
            loss_val = loss.item()
            elapsed = time.time() - t0
            n_gs = splats['means'].shape[0]
            print(f"  [{step:5d}/{iterations}] loss={loss_val:.4f} "
                  f"#gs={n_gs:,d} "
                  f"{elapsed:.1f}s ({elapsed/(step+1):.2f}s/it)")
            sys.stdout.flush()

            with torch.no_grad():
                m = splats['means'].detach().cpu().numpy()
                c = (splats['sh0'].squeeze(1).detach() * C0 + 0.5).clamp(0, 1).cpu().numpy()
                s = torch.exp(splats['scales'].detach()).max(dim=-1).values.cpu().numpy()
                q = splats['quats'].detach().cpu().numpy()
                sc = splats['scales'].detach().cpu().numpy()
                op = splats['opacities'].detach().cpu().numpy()
                sh = splats['sh0'].detach().cpu().numpy()

            yield {
                'step': step, 'total': iterations,
                'loss': loss_val, 'n_splats': n_gs,
                'means': m, 'colors': c, 'scales': s,
                'quats': q, 'scales_log': sc, 'opacities_logit': op, 'sh0': sh,
                'done': False,
            }

    # Final yield
    with torch.no_grad():
        m = splats['means'].detach().cpu().numpy()
        c = (splats['sh0'].squeeze(1).detach() * C0 + 0.5).clamp(0, 1).cpu().numpy()
        s = torch.exp(splats['scales'].detach()).max(dim=-1).values.cpu().numpy()

    yield {
        'step': iterations, 'total': iterations,
        'loss': loss.item() if 'loss' in dir() else 0,
        'n_splats': splats['means'].shape[0],
        'means': m, 'colors': c, 'scales': s,
        'done': True, 'splats': splats,
    }
    print(f"  Training complete: {splats['means'].shape[0]:,d} splats")
