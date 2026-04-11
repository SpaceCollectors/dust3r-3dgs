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

    # Scale from KNN distances
    avg_dist = knn_distances(points, k=3)
    scales_init = np.log(np.clip(avg_dist * 0.5, 1e-7, None))
    scales = np.stack([scales_init, scales_init, scales_init], axis=-1)
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

    # Opacity
    opacities = torch.full((N,), -1.0, device=device)  # sigmoid(-1) ~ 0.27

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


def surface_anchor_loss(means, anchor_tree, anchor_pts_t):
    """L2 distance from each splat to nearest mesh point."""
    with torch.no_grad():
        pts_np = means.detach().cpu().numpy()
        _, idx = anchor_tree.query(pts_np)
    nearest = anchor_pts_t[idx]  # (N, 3) fixed target
    return F.mse_loss(means, nearest)


def train_surface_splats(
    mesh_data=None, point_cloud=None, colmap_dir=None,
    device='cuda', iterations=3000, max_resolution=256,
    n_samples=50000,
    anchor_weight_start=0.1, anchor_weight_end=0.001,
    depth_lambda=0.5, aniso_lambda=0.01, scale_lambda=0.001,
    max_scale=2.0,
    target_splats=0,
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
    print(f"  Initializing {len(points):,d} surface splats...")
    splats = init_surface_splats(points, normals, scolors, device)
    optimizers = create_optimizers(splats, scene_scale)
    means_scheduler = optim.lr_scheduler.ExponentialLR(
        optimizers['means'], gamma=0.01 ** (1.0 / iterations))

    # Prepare image tensors
    for img in images:
        img['pixels_t'] = torch.from_numpy(img['pixels']).float().to(device)
        img['K_t'] = torch.from_numpy(img['K']).float().to(device)
        img['w2c_t'] = torch.from_numpy(img['w2c'].astype(np.float32)).to(device)

    CROP = 128  # small crops for faster training with pure PyTorch rasterizer
    ssim_lambda = 0.2
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

        # Random view
        idx = torch.randint(len(images), (1,)).item()
        img = images[idx]
        pixels_full = img['pixels_t']
        K_full = img['K_t']
        w2c = img['w2c_t']
        H, W = img['H'], img['W']
        depth_gt_full = depth_maps[idx]

        # Random crop
        is_full = (step % 50 == 0)
        if is_full or min(H, W) <= CROP:
            pixels = pixels_full
            K = K_full
            depth_gt = depth_gt_full
            rH, rW = H, W
        else:
            cy = torch.randint(0, H - CROP, (1,)).item()
            cx = torch.randint(0, W - CROP, (1,)).item()
            pixels = pixels_full[cy:cy+CROP, cx:cx+CROP]
            depth_gt = depth_gt_full[cy:cy+CROP, cx:cx+CROP]
            K = K_full.clone()
            K[0, 2] -= cx
            K[1, 2] -= cy
            rH, rW = CROP, CROP

        # Activations
        means = splats['means']
        quats = splats['quats']
        scales = torch.exp(splats['scales'])
        opacities = torch.sigmoid(splats['opacities'])
        colors_rgb = (splats['sh0'].squeeze(1) * C0 + 0.5).clamp(0, 1)

        # Clamp scales
        with torch.no_grad():
            splats['scales'].data.clamp_(max=math.log(max_scale * scene_scale))

        bg = torch.rand(3, device=device)

        # Render
        colors_pred, depth_pred, alpha_pred = render_gaussians(
            means3d=means, scales=scales, quats=quats,
            opacities=opacities, colors=colors_rgb,
            viewmat=w2c, K=K, W=rW, H=rH, bg_color=bg)

        # Losses
        l1 = F.l1_loss(colors_pred, pixels)
        pred_p = colors_pred.permute(2, 0, 1).unsqueeze(0)
        gt_p = pixels.permute(2, 0, 1).unsqueeze(0)
        ssim_val = _ssim(pred_p, gt_p)
        color_loss = (1 - ssim_lambda) * l1 + ssim_lambda * (1 - ssim_val)

        d_loss = depth_loss_fn(depth_pred, depth_gt, alpha_pred)
        a_loss = anisotropy_loss_fn(splats['scales'])
        s_loss = scale_loss_fn(splats['scales'])

        # Surface anchor loss
        anchor_loss = surface_anchor_loss(means, anchor_tree, anchor_pts_t)

        # Opacity decay
        with torch.no_grad():
            splats['opacities'].data -= 0.004 * optimizers['opacities'].defaults['lr']

        loss = (color_loss + depth_lambda * d_loss + aniso_lambda * a_loss
                + scale_lambda * s_loss + anchor_w * anchor_loss)

        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        means_scheduler.step()

        # Densification & Pruning
        if step > 0 and step % 500 == 0:
            target = target_splats
            with torch.no_grad():
                cur_n = splats['means'].shape[0]
                alive = torch.sigmoid(splats['opacities'].data) > 0.005
                if alive.sum() < cur_n:
                    n_pruned = cur_n - alive.sum().item()
                    for key in splats:
                        splats[key] = torch.nn.Parameter(splats[key].data[alive])
                    cur_n = splats['means'].shape[0]

                if target > 0 and cur_n < target and step < iterations * 3 // 4:
                    n_add = min(target - cur_n, cur_n // 4)
                    if n_add > 0:
                        scale_mag = torch.exp(splats['scales'].data).max(dim=-1).values
                        probs = scale_mag / scale_mag.sum()
                        idx_s = torch.multinomial(probs, n_add, replacement=True)
                        for key in splats:
                            new_data = splats[key].data[idx_s].clone()
                            if key == 'means':
                                new_data += torch.randn_like(new_data) * 0.01 * scene_scale
                            elif key == 'opacities':
                                new_data -= 0.7
                                splats[key].data[idx_s] -= 0.7
                            elif key == 'scales':
                                new_data -= 0.5
                                splats[key].data[idx_s] -= 0.5
                            splats[key] = torch.nn.Parameter(
                                torch.cat([splats[key].data, new_data], dim=0))

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
                  f"anchor={anchor_loss.item():.4f} #gs={n_gs:,d} "
                  f"{elapsed:.1f}s ({elapsed/(step+1):.2f}s/it)")
            sys.stdout.flush()

            with torch.no_grad():
                m = splats['means'].detach().cpu().numpy()
                c = (splats['sh0'].squeeze(1).detach() * C0 + 0.5).clamp(0, 1).cpu().numpy()
                s = torch.exp(splats['scales'].detach()).max(dim=-1).values.cpu().numpy()

            yield {
                'step': step, 'total': iterations,
                'loss': loss_val, 'n_splats': n_gs,
                'means': m, 'colors': c, 'scales': s,
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
