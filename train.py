"""
Depth-Regularized Gaussian Splatting for Few-View Reconstruction

Custom training loop that uses depth priors from DUSt3R/MASt3R/VGGT
to constrain gaussian placement and prevent degenerate elongated splats.

Key differences from vanilla 3DGS:
  - Depth rendering loss (disparity space) from reconstruction depth maps
  - Anisotropy regularization to keep gaussians compact
  - Conservative densification for few-view scenarios
  - Scale clamping to prevent runaway gaussian growth

Usage:
  python train.py --data_dir path/to/colmap_export --output_dir output/
"""

import os
import sys
import math
import json
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from PIL import Image
from scipy.spatial.transform import Rotation

# Pure PyTorch rasterizer — no CUDA compilation needed
from rasterizer import render_gaussians


# ── COLMAP Parser ────────────────────────────────────────────────────────────

def parse_colmap_cameras(path):
    """Parse cameras.txt → dict of {camera_id: (W, H, fx, fy, cx, cy)}"""
    cameras = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split()
            cam_id = int(parts[0])
            model = parts[1]
            W, H = int(parts[2]), int(parts[3])
            if model == 'PINHOLE':
                fx, fy, cx, cy = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            elif model == 'SIMPLE_PINHOLE':
                f_val = float(parts[4])
                fx = fy = f_val
                cx, cy = float(parts[5]), float(parts[6])
            else:
                raise ValueError(f"Unsupported camera model: {model}")
            cameras[cam_id] = (W, H, fx, fy, cx, cy)
    return cameras


def parse_colmap_images(path):
    """Parse images.txt → list of dicts with pose info."""
    images = []
    with open(path) as f:
        lines = [l.strip() for l in f if not l.startswith('#')]
    # images.txt has 2 lines per image: pose line + points2d line (may be empty)
    # Filter: pose lines have 9+ fields, points2d lines are empty or have x,y,id triples
    pose_lines = [l for l in lines if l and len(l.split()) >= 9]
    for line in pose_lines:
        parts = line.split()
        img_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        cam_id = int(parts[8])
        name = parts[9]

        # COLMAP stores w2c as quaternion + translation
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = [tx, ty, tz]
        c2w = np.linalg.inv(w2c)

        images.append({
            'id': img_id, 'name': name, 'cam_id': cam_id,
            'w2c': w2c, 'c2w': c2w,
        })
    return images


def parse_colmap_points3d(path):
    """Parse points3D.txt → (points [M,3], colors [M,3])."""
    pts, cols = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split()
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
            pts.append([x, y, z])
            cols.append([r, g, b])
    return np.array(pts, dtype=np.float32), np.array(cols, dtype=np.uint8)


def load_colmap_dataset(data_dir, max_resolution=512):
    """Load COLMAP dataset, downscaling images for faster training."""
    sparse_dir = os.path.join(data_dir, 'sparse', '0')
    images_dir = os.path.join(data_dir, 'images')

    cameras = parse_colmap_cameras(os.path.join(sparse_dir, 'cameras.txt'))
    images = parse_colmap_images(os.path.join(sparse_dir, 'images.txt'))
    points, colors = parse_colmap_points3d(os.path.join(sparse_dir, 'points3D.txt'))

    # Load images, downscale for training speed
    for img in images:
        img_path = os.path.join(images_dir, img['name'])
        pil_img = Image.open(img_path).convert('RGB')

        cam = cameras[img['cam_id']]
        W_cam, H_cam, fx, fy, cx, cy = cam

        # Downscale if needed
        scale = min(1.0, max_resolution / max(W_cam, H_cam))
        if scale < 1.0:
            new_W = int(W_cam * scale)
            new_H = int(H_cam * scale)
            pil_img = pil_img.resize((new_W, new_H), Image.LANCZOS)
            fx, fy, cx, cy = fx * scale, fy * scale, cx * scale, cy * scale
            W_cam, H_cam = new_W, new_H

        img['pixels'] = np.array(pil_img, dtype=np.float32) / 255.0
        img['K'] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        img['W'] = W_cam
        img['H'] = H_cam

    return images, points, colors


# ── SH Utilities ─────────────────────────────────────────────────────────────

C0 = 0.28209479177387814

def rgb_to_sh(rgb):
    """Convert RGB [0,1] to 0th-order SH coefficient."""
    return (rgb - 0.5) / C0


# ── Gaussian Initialization ─────────────────────────────────────────────────

def knn_distances(points, k=4):
    """Compute average distance to k nearest neighbors."""
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1)  # includes self
    return dists[:, 1:].mean(axis=1)  # exclude self


def init_gaussians(points, colors, device='cuda'):
    """Initialize gaussian parameters from point cloud."""
    N = len(points)
    means = torch.from_numpy(points).float().to(device)

    # Scale from KNN distances
    avg_dist = knn_distances(points, k=3)
    scales_init = np.log(np.clip(avg_dist * 0.5, 1e-7, None))
    scales = torch.from_numpy(scales_init).float().unsqueeze(-1).repeat(1, 3).to(device)

    # Random quaternions (identity-ish)
    quats = torch.zeros(N, 4, device=device)
    quats[:, 0] = 1.0  # w=1, identity rotation
    quats += torch.randn_like(quats) * 0.01

    # Low initial opacity
    opacities = torch.full((N,), -1.0, device=device)  # sigmoid(-1) ≈ 0.27

    # SH colors (DC component only to start)
    rgb = torch.from_numpy(colors / 255.0).float().to(device)
    sh0 = rgb_to_sh(rgb).unsqueeze(1)  # (N, 1, 3)

    # Higher order SH = 0
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


def create_optimizers(splats, scene_scale=1.0):
    """Create per-parameter optimizers with appropriate learning rates."""
    return {
        'means': optim.Adam([splats['means']], lr=1.6e-4 * scene_scale, eps=1e-15),
        'scales': optim.Adam([splats['scales']], lr=5e-3, eps=1e-15),
        'quats': optim.Adam([splats['quats']], lr=1e-3, eps=1e-15),
        'opacities': optim.Adam([splats['opacities']], lr=5e-2, eps=1e-15),
        'sh0': optim.Adam([splats['sh0']], lr=2.5e-3, eps=1e-15),
        'shN': optim.Adam([splats['shN']], lr=2.5e-3 / 20, eps=1e-15),
    }


# ── Depth from Scene Reconstruction ─────────────────────────────────────────

def render_depth_from_pts3d(pts3d, w2c, K, H, W):
    """
    Render a pseudo-depth map by projecting 3D points into the camera.
    This gives us the depth prior from dust3r/mast3r/vggt for supervision.
    Returns (H, W) depth map (0 where no points project).
    """
    pts = pts3d.reshape(-1, 3)
    # Transform to camera space
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    pts_cam = (R @ pts.T).T + t  # (M, 3)

    # Project to image
    z = pts_cam[:, 2]
    valid = z > 0.01
    pts_cam = pts_cam[valid]
    z = z[valid]

    u = (K[0, 0] * pts_cam[:, 0] / z + K[0, 2]).astype(np.int32)
    v = (K[1, 1] * pts_cam[:, 1] / z + K[1, 2]).astype(np.int32)

    mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[mask], v[mask], z[mask]

    depth = np.zeros((H, W), dtype=np.float32)
    # Z-buffer: keep closest
    for j in range(len(u)):
        if depth[v[j], u[j]] == 0 or z[j] < depth[v[j], u[j]]:
            depth[v[j], u[j]] = z[j]

    return depth


# ── Loss Functions ───────────────────────────────────────────────────────────

def depth_loss_fn(depth_pred, depth_gt, alpha_pred):
    """
    Depth loss in disparity space (1/depth).
    Only computed where we have valid depth from reconstruction.
    """
    valid = (depth_gt > 0.01) & (depth_pred.squeeze(-1) > 0.01) & (alpha_pred.squeeze(-1) > 0.5)
    if valid.sum() < 100:
        return torch.tensor(0.0, device=depth_pred.device)

    pred = depth_pred.squeeze(-1)[valid]
    gt = depth_gt[valid]

    # Disparity space: more stable, emphasizes nearby geometry
    disp_pred = 1.0 / pred
    disp_gt = 1.0 / gt

    # Normalize to handle scale ambiguity
    disp_pred_norm = disp_pred / (disp_pred.median() + 1e-8)
    disp_gt_norm = disp_gt / (disp_gt.median() + 1e-8)

    return F.l1_loss(disp_pred_norm, disp_gt_norm)


def anisotropy_loss_fn(scales):
    """
    Penalize elongated gaussians. Ratio of max/min scale should be close to 1.
    This prevents the "paint stroke" effect where gaussians stretch to cover pixels.
    """
    exp_scales = torch.exp(scales)  # (N, 3)
    max_scale = exp_scales.max(dim=-1).values
    min_scale = exp_scales.min(dim=-1).values.clamp(min=1e-8)
    ratio = max_scale / min_scale
    # Penalize ratios > 3 (soft threshold)
    return F.relu(ratio - 3.0).mean()


def scale_loss_fn(scales):
    """Penalize overly large gaussians."""
    return torch.exp(scales).max(dim=-1).values.clamp(min=0.01).mean()


# ── Training ─────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load dataset
    print(f"Loading COLMAP dataset from {args.data_dir}")
    images, points, colors = load_colmap_dataset(args.data_dir)
    print(f"  {len(images)} images, {len(points)} points")

    if len(points) == 0:
        raise ValueError("No 3D points found in COLMAP dataset")

    # Compute scene scale (median distance between camera centers)
    cam_centers = np.array([img['c2w'][:3, 3] for img in images])
    if len(cam_centers) > 1:
        from scipy.spatial.distance import pdist
        scene_scale = float(np.median(pdist(cam_centers)))
    else:
        scene_scale = 1.0
    print(f"  Scene scale: {scene_scale:.4f}")

    # Precompute depth maps from 3D points for each view
    print("Computing depth priors from point cloud...")
    depth_maps = []
    for img in images:
        dm = render_depth_from_pts3d(points, img['w2c'], img['K'], img['H'], img['W'])
        depth_maps.append(torch.from_numpy(dm).float().to(device))
    print(f"  Depth maps computed for {len(depth_maps)} views")

    # Initialize gaussians
    print(f"Initializing {len(points)} gaussians...")
    splats = init_gaussians(points, colors, device)
    optimizers = create_optimizers(splats, scene_scale)

    # LR scheduler for means
    means_scheduler = optim.lr_scheduler.ExponentialLR(
        optimizers['means'], gamma=0.01 ** (1.0 / args.iterations)
    )

    # Prepare image tensors
    for img in images:
        img['pixels_t'] = torch.from_numpy(img['pixels']).float().to(device)
        img['K_t'] = torch.from_numpy(img['K']).float().to(device)
        img['w2c_t'] = torch.from_numpy(img['w2c'].astype(np.float32)).to(device)

    # Training config
    ssim_lambda = 0.2
    depth_lambda = args.depth_lambda
    aniso_lambda = args.aniso_lambda
    scale_lambda = args.scale_lambda
    max_scale = args.max_scale

    # Live preview directory
    preview_dir = os.path.join(args.output_dir, 'preview')
    os.makedirs(preview_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Crop size for training (render patches, not full images)
    CROP = 256  # render 256x256 crops — ~4x faster than full 512x384

    print(f"\nStarting training for {args.iterations} iterations...")
    print(f"  depth_lambda={depth_lambda}, aniso_lambda={aniso_lambda}, scale_lambda={scale_lambda}")
    print(f"  crop_size={CROP}, random_bg=True")
    t0 = time.time()
    log_loss = torch.tensor(0.0, device=device)  # GPU-resident, no .item() in hot loop

    for step in range(args.iterations):
        # Random view
        idx = torch.randint(len(images), (1,)).item()
        img = images[idx]
        pixels_full = img['pixels_t']
        K_full = img['K_t']
        w2c = img['w2c_t']
        H, W = img['H'], img['W']
        depth_gt_full = depth_maps[idx]

        # Random crop for fast training (full image every 50 steps for preview)
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
            # Shift principal point for crop
            K = K_full.clone()
            K[0, 2] -= cx
            K[1, 2] -= cy
            rH, rW = CROP, CROP

        # Gaussian params with activations
        means = splats['means']
        quats = splats['quats']
        scales = torch.exp(splats['scales'])
        opacities = torch.sigmoid(splats['opacities'])
        colors_rgb = (splats['sh0'].squeeze(1) * C0 + 0.5).clamp(0, 1)

        # Clamp scales
        with torch.no_grad():
            splats['scales'].data.clamp_(max=math.log(max_scale * scene_scale))

        # Random background color (prevents color baking)
        bg = torch.rand(3, device=device)

        # Rasterize
        colors_pred, depth_pred, alpha_pred = render_gaussians(
            means3d=means, scales=scales, quats=quats,
            opacities=opacities, colors=colors_rgb,
            viewmat=w2c, K=K, W=rW, H=rH, bg_color=bg,
        )

        # GT with same random background
        gt_with_bg = pixels  # photos have no alpha, so no bg needed

        # ── Losses ──
        l1 = F.l1_loss(colors_pred, gt_with_bg)
        pred_p = colors_pred.permute(2, 0, 1).unsqueeze(0)
        gt_p = gt_with_bg.permute(2, 0, 1).unsqueeze(0)
        ssim_val = _ssim(pred_p, gt_p)
        color_loss = (1 - ssim_lambda) * l1 + ssim_lambda * (1 - ssim_val)

        d_loss = depth_loss_fn(depth_pred, depth_gt, alpha_pred)
        a_loss = anisotropy_loss_fn(splats['scales'])
        s_loss = scale_loss_fn(splats['scales'])

        # Opacity decay (from LichtFeld's MRNF strategy)
        with torch.no_grad():
            splats['opacities'].data -= 0.004 * optimizers['opacities'].defaults['lr']

        loss = color_loss + depth_lambda * d_loss + aniso_lambda * a_loss + scale_lambda * s_loss

        loss.backward()

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        means_scheduler.step()

        # ── Densification & Pruning (every 500 steps, between iter 500-5000) ──
        if step > 0 and step % 500 == 0:
            target = args.target_splats
            with torch.no_grad():
                cur_n = splats['means'].shape[0]

                # Prune dead gaussians
                alive = torch.sigmoid(splats['opacities'].data) > 0.005
                if alive.sum() < cur_n:
                    n_pruned = cur_n - alive.sum().item()
                    for key in splats:
                        splats[key] = torch.nn.Parameter(splats[key].data[alive])
                    print(f"  Pruned {n_pruned} dead gaussians")
                    cur_n = splats['means'].shape[0]

                # Densify: grow toward target by duplicating high-scale gaussians
                if target > 0 and cur_n < target and step < args.iterations * 3 // 4:
                    n_add = min(target - cur_n, cur_n // 4)  # grow by at most 25% per step
                    if n_add > 0:
                        # Sample gaussians weighted by scale (larger ones get split)
                        scale_mag = torch.exp(splats['scales'].data).max(dim=-1).values
                        probs = scale_mag / scale_mag.sum()
                        idx_sample = torch.multinomial(probs, n_add, replacement=True)

                        for key in splats:
                            new_data = splats[key].data[idx_sample].clone()
                            if key == 'means':
                                # Offset new positions slightly
                                noise = torch.randn_like(new_data) * 0.01 * scene_scale
                                new_data = new_data + noise
                            elif key == 'opacities':
                                # Split opacity: each gets half (in logit space)
                                new_data = new_data - 0.7  # roughly halves sigmoid
                                splats[key].data[idx_sample] -= 0.7
                            elif key == 'scales':
                                # Shrink both parent and child
                                new_data = new_data - 0.5
                                splats[key].data[idx_sample] -= 0.5
                            splats[key] = torch.nn.Parameter(
                                torch.cat([splats[key].data, new_data], dim=0))

                        print(f"  Densified: added {n_add} gaussians -> {splats['means'].shape[0]:,d}")

                # Final pruning to hit target exactly (last 25% of training)
                if target > 0 and cur_n > target and step >= args.iterations * 3 // 4:
                    n_remove = cur_n - target
                    opa = torch.sigmoid(splats['opacities'].data)
                    _, remove_idx = torch.topk(opa, n_remove, largest=False)
                    keep = torch.ones(cur_n, dtype=torch.bool, device=device)
                    keep[remove_idx] = False
                    for key in splats:
                        splats[key] = torch.nn.Parameter(splats[key].data[keep])
                    print(f"  Trimmed to target: {splats['means'].shape[0]:,d}")

                # Rebuild optimizers after any structural change
                optimizers = create_optimizers(splats, scene_scale)
                means_scheduler = optim.lr_scheduler.ExponentialLR(
                    optimizers['means'], gamma=0.01 ** (1.0 / max(1, args.iterations - step))
                )

        # Logging + live preview (only sync to CPU here)
        if step % 100 == 0 or step == args.iterations - 1:
            n_gs = splats['means'].shape[0]
            elapsed = time.time() - t0
            loss_val = loss.item()
            color_val = color_loss.item()
            depth_val = d_loss.item()
            aniso_val = a_loss.item()
            print(f"[{step:5d}/{args.iterations}] "
                  f"loss={loss_val:.4f} color={color_val:.4f} "
                  f"depth={depth_val:.4f} aniso={aniso_val:.4f} "
                  f"#gs={n_gs:,d} time={elapsed:.1f}s")

        # Save live preview every 200 steps
        if step % 200 == 0:
            with torch.no_grad():
                prev_img = colors_pred.detach().clamp(0, 1).cpu().numpy()
                prev_img = (prev_img * 255).astype(np.uint8)
                Image.fromarray(prev_img).save(os.path.join(preview_dir, 'latest.png'))

        # Save checkpoints
        if (step + 1) % args.save_every == 0 or step == args.iterations - 1:
            save_ply(splats, os.path.join(args.output_dir, f'point_cloud_{step+1}.ply'))

    # Final save
    save_ply(splats, os.path.join(args.output_dir, 'point_cloud.ply'))
    print(f"\nTraining complete. Output saved to {args.output_dir}")
    print(f"  Final gaussian count: {splats['means'].shape[0]:,d}")
    print(f"  Total time: {time.time() - t0:.1f}s")


# ── SSIM ─────────────────────────────────────────────────────────────────────

def _ssim(pred, gt, window_size=11):
    """Simplified SSIM computation."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - window_size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)
    window = window.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)

    pad = window_size // 2
    mu1 = F.conv2d(pred, window, padding=pad, groups=3)
    mu2 = F.conv2d(gt, window, padding=pad, groups=3)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(pred ** 2, window, padding=pad, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(gt ** 2, window, padding=pad, groups=3) - mu2_sq
    sigma12 = F.conv2d(pred * gt, window, padding=pad, groups=3) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


# ── PLY Export ───────────────────────────────────────────────────────────────

def save_ply(splats, path):
    """Save gaussians as PLY file (compatible with gaussian splatting viewers)."""
    means = splats['means'].detach().cpu().numpy()
    scales = splats['scales'].detach().cpu().numpy()
    quats = splats['quats'].detach().cpu().numpy()
    opacities = splats['opacities'].detach().cpu().numpy()
    sh0 = splats['sh0'].detach().cpu().numpy()
    shN = splats['shN'].detach().cpu().numpy()

    N = means.shape[0]
    sh_all = np.concatenate([sh0, shN], axis=1)  # (N, K, 3)

    # Normalize quaternions
    quats = quats / (np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-8)

    # Build PLY header
    num_sh = sh_all.shape[1]
    header = f"""ply
format binary_little_endian 1.0
element vertex {N}
property float x
property float y
property float z
property float nx
property float ny
property float nz
"""
    # SH coefficients as f_dc_0..2 and f_rest_0..N
    for i in range(3):
        header += f"property float f_dc_{i}\n"
    for i in range((num_sh - 1) * 3):
        header += f"property float f_rest_{i}\n"

    header += "property float opacity\n"
    for i in range(3):
        header += f"property float scale_{i}\n"
    for i in range(4):
        header += f"property float rot_{i}\n"
    header += "end_header\n"

    # Build contiguous binary data (vectorized)
    normals = np.zeros((N, 3), dtype=np.float32)
    sh_dc = sh_all[:, 0, :].astype(np.float32)  # (N, 3)
    # SH rest: group by channel (all R, all G, all B) per splat
    sh_rest = sh_all[:, 1:, :]  # (N, K-1, 3)
    sh_rest_interleaved = sh_rest.transpose(0, 2, 1).reshape(N, -1).astype(np.float32)  # (N, 3*(K-1))

    # Pack per-vertex: pos(3) + normal(3) + dc(3) + rest(3*(K-1)) + opacity(1) + scale(3) + quat(4)
    row = np.concatenate([
        means.astype(np.float32),
        normals,
        sh_dc,
        sh_rest_interleaved,
        opacities.reshape(N, 1).astype(np.float32),
        scales.astype(np.float32),
        quats.astype(np.float32),
    ], axis=1)

    with open(path, 'wb') as f:
        f.write(header.encode())
        f.write(row.tobytes())

    print(f"  Saved {N:,d} gaussians to {path}")


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Depth-Regularized Gaussian Splatting")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to COLMAP dataset (from our exporter)')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory for trained model')
    parser.add_argument('--iterations', type=int, default=7000,
                        help='Training iterations')
    parser.add_argument('--device', type=str, default='cuda')

    # Loss weights
    parser.add_argument('--depth_lambda', type=float, default=0.5,
                        help='Depth supervision weight (0=disabled)')
    parser.add_argument('--aniso_lambda', type=float, default=0.01,
                        help='Anisotropy regularization weight')
    parser.add_argument('--scale_lambda', type=float, default=0.001,
                        help='Scale regularization weight')
    parser.add_argument('--max_scale', type=float, default=2.0,
                        help='Maximum gaussian scale relative to scene scale')

    parser.add_argument('--save_every', type=int, default=1000,
                        help='Save checkpoint every N iterations')
    parser.add_argument('--target_splats', type=int, default=0,
                        help='Target gaussian count (0=no densification, keep initial count)')

    args = parser.parse_args()
    train(args)
