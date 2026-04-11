"""
Pure PyTorch differentiable Gaussian Splatting rasterizer — vectorized.
No CUDA compilation needed — runs on any GPU with PyTorch.

This version eliminates Python loops by:
1. Computing all 2D gaussians in parallel
2. Rendering full image in chunks (row-bands) with vectorized alpha compositing
3. Using cumulative product for transmittance instead of sequential loop
"""

import torch
import torch.nn.functional as F
import math


def quaternion_to_rotation_matrix(q):
    """Convert quaternions (N, 4) [w,x,y,z] to rotation matrices (N, 3, 3)."""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.stack([
        1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y),
            2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x),
            2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)

    return R


def project_gaussians(means3d, scales, quats, viewmat, K):
    """
    Project 3D gaussians to 2D. Returns all needed 2D params.
    """
    N = means3d.shape[0]
    device = means3d.device

    # Transform to camera space
    R_cam = viewmat[:3, :3]
    t_cam = viewmat[:3, 3]
    means_cam = means3d @ R_cam.T + t_cam[None, :]

    depths = means_cam[:, 2]
    valid = depths > 0.1

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    tz = depths.clamp(min=0.1)
    means2d_x = means_cam[:, 0] / tz * fx + cx
    means2d_y = means_cam[:, 1] / tz * fy + cy
    means2d = torch.stack([means2d_x, means2d_y], dim=-1)

    # 3D covariance
    R_gauss = quaternion_to_rotation_matrix(quats)
    S = torch.diag_embed(scales)
    M = R_gauss @ S
    cov3d = M @ M.transpose(-1, -2)

    # Jacobian of projection
    tx = means_cam[:, 0]
    ty = means_cam[:, 1]
    J = torch.zeros(N, 2, 3, device=device)
    J[:, 0, 0] = fx / tz
    J[:, 0, 2] = -fx * tx / (tz * tz)
    J[:, 1, 1] = fy / tz
    J[:, 1, 2] = -fy * ty / (tz * tz)

    cov3d_cam = R_cam[None] @ cov3d @ R_cam.T[None]
    cov2d = J @ cov3d_cam @ J.transpose(-1, -2)
    cov2d[:, 0, 0] += 0.3
    cov2d[:, 1, 1] += 0.3

    # Inverse 2x2 covariance
    a, b, c, d = cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 0], cov2d[:, 1, 1]
    det = (a * d - b * c).clamp(min=1e-8)
    inv_cov = torch.stack([d, -b, -c, a], dim=-1).reshape(N, 2, 2) / det[:, None, None]

    # Radius (3-sigma of largest eigenvalue)
    eigenvalues = 0.5 * (a + d) + 0.5 * torch.sqrt(((a - d) ** 2 + 4 * b * b).clamp(min=0))
    radius = torch.sqrt(eigenvalues.clamp(min=0)) * 3.0

    return means2d, inv_cov, depths, radius, valid


def render_gaussians(means3d, scales, quats, opacities, colors,
                     viewmat, K, W, H, bg_color=None):
    """
    Fully vectorized gaussian splatting renderer.
    Renders in row-bands to manage memory while avoiding Python per-gaussian loops.

    Args:
        means3d: (N, 3)
        scales: (N, 3) already exponentiated
        quats: (N, 4)
        opacities: (N,) already sigmoided
        colors: (N, 3) RGB
        viewmat: (4, 4) world-to-camera
        K: (3, 3) intrinsics
        W, H: image dimensions
        bg_color: (3,)

    Returns:
        image: (H, W, 3)
        depth: (H, W, 1)
        alpha: (H, W, 1)
    """
    if bg_color is None:
        bg_color = torch.ones(3, device=means3d.device)

    device = means3d.device

    # Project all gaussians
    means2d, inv_cov, depths, radius, valid = project_gaussians(
        means3d, scales, quats, viewmat, K
    )

    # Filter
    valid_idx = torch.where(valid)[0]
    if len(valid_idx) == 0:
        return (bg_color.view(1, 1, 3).expand(H, W, 3),
                torch.zeros(H, W, 1, device=device),
                torch.zeros(H, W, 1, device=device))

    means2d = means2d[valid_idx]
    inv_cov = inv_cov[valid_idx]
    depths_v = depths[valid_idx]
    opacities_v = opacities[valid_idx]
    colors_v = colors[valid_idx]
    radius_v = radius[valid_idx]

    # Sort by depth
    sort_idx = torch.argsort(depths_v)
    means2d = means2d[sort_idx]
    inv_cov = inv_cov[sort_idx]
    depths_v = depths_v[sort_idx]
    opacities_v = opacities_v[sort_idx]
    colors_v = colors_v[sort_idx]
    radius_v = radius_v[sort_idx]

    M = len(means2d)

    # Render in row-bands to limit memory usage
    # Each band: (BAND_H, W) pixels evaluated against overlapping gaussians
    BAND_H = 32  # rows per band
    image = torch.zeros(H, W, 3, device=device)
    depth_map = torch.zeros(H, W, 1, device=device)
    alpha_map = torch.zeros(H, W, 1, device=device)

    for y_start in range(0, H, BAND_H):
        y_end = min(y_start + BAND_H, H)
        bh = y_end - y_start

        # Find gaussians overlapping this band
        g_top = means2d[:, 1] - radius_v
        g_bot = means2d[:, 1] + radius_v
        band_overlap = (g_bot >= y_start) & (g_top < y_end)

        # Also filter by x range
        g_left = means2d[:, 0] - radius_v
        g_right = means2d[:, 0] + radius_v
        x_overlap = (g_right >= 0) & (g_left < W)

        overlap = band_overlap & x_overlap
        oidx = torch.where(overlap)[0]

        if len(oidx) == 0:
            image[y_start:y_end] = bg_color.view(1, 1, 3)
            continue

        # Cap per-band gaussians to prevent OOM (keep closest ones)
        MAX_PER_BAND = 8192
        if len(oidx) > MAX_PER_BAND:
            oidx = oidx[:MAX_PER_BAND]  # already depth-sorted

        K_gs = len(oidx)
        m = means2d[oidx]          # (K, 2)
        ic = inv_cov[oidx]         # (K, 2, 2)
        opa = opacities_v[oidx]    # (K,)
        col = colors_v[oidx]       # (K, 3)
        dep = depths_v[oidx]       # (K,)

        # Create pixel grid for this band: (bh, W, 2)
        py = torch.arange(y_start, y_end, device=device, dtype=torch.float32)
        px = torch.arange(0, W, device=device, dtype=torch.float32)
        gy, gx = torch.meshgrid(py, px, indexing='ij')
        pixels = torch.stack([gx, gy], dim=-1)  # (bh, W, 2)

        # diff: (bh, W, K, 2)
        diff = pixels.unsqueeze(2) - m.view(1, 1, K_gs, 2)

        # Mahalanobis: einsum for batched quadratic form
        # ic: (K, 2, 2), diff: (bh, W, K, 2)
        diff_ic = torch.einsum('hwki,kij->hwkj', diff, ic)
        maha = (diff_ic * diff).sum(dim=-1)  # (bh, W, K)

        # Gaussian weights * opacity → alpha
        gauss = torch.exp(-0.5 * maha.clamp(max=30))
        alpha_k = (gauss * opa.view(1, 1, K_gs)).clamp(max=0.99)  # (bh, W, K)

        # Vectorized front-to-back alpha compositing using cumprod
        one_minus_alpha = 1.0 - alpha_k  # (bh, W, K)

        # Exclusive cumprod: T_0=1, T_k = prod_{j<k}(1 - alpha_j)
        # Shift one_minus_alpha right and prepend 1.0
        ones = torch.ones(bh, W, 1, device=device)
        T = torch.cumprod(torch.cat([ones, one_minus_alpha[:, :, :-1]], dim=2), dim=2)

        # Transmittance cutoff: zero out contributions where T < 1e-4
        active = (T > 1e-4).float()
        w = alpha_k * T * active  # (bh, W, K)

        # Composite color and depth
        band_color = torch.einsum('hwk,kc->hwc', w, col)
        band_depth = torch.einsum('hwk,k->hw', w, dep).unsqueeze(-1)
        total_alpha = w.sum(dim=-1, keepdim=True)

        # Remaining transmittance for background
        T_final = (T[:, :, -1:] * one_minus_alpha[:, :, -1:])
        band_color = band_color + T_final * bg_color.view(1, 1, 3)

        image[y_start:y_end] = band_color
        depth_map[y_start:y_end] = band_depth
        alpha_map[y_start:y_end] = total_alpha

    return image, depth_map, alpha_map
