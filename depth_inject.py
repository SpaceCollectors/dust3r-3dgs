"""
Depth Injection: Replace/blend dust3r's depth maps with AI depth.

After dust3r has solved camera poses and initial depth, we:
1. Extract dust3r's per-view depth maps
2. Run DepthAnything v2 on each image
3. Build a transfer curve to align AI depth to dust3r's scale
4. Blend: final = (1-mix) * dust3r + mix * aligned_ai
5. Inject back into the scene object
6. Re-generate point cloud from the blended depth

The result: dust3r's multi-view consistent poses + AI depth's smoothness.
"""

import numpy as np
import torch
import os
from PIL import Image


def inject_ai_depth(scene, images, mix=0.5, highpass_sigma=10.0, device='cuda', progress_fn=None):
    """
    Enhance dust3r depth maps with AI high-frequency detail.

    Like Photoshop high-pass overlay:
    1. Get dust3r depth (base) and AI depth (detail source)
    2. Normalize AI depth to same range as dust3r
    3. Compute difference: AI - dust3r (mid-grey = no change)
    4. High-pass filter the difference (remove low-freq drift, keep sharp edges)
    5. Apply via linear light: result = dust3r + high_freq_detail * mix

    Result: dust3r's macro geometry + AI's sharp edges/details.

    Args:
        scene: dust3r PointCloudOptimizer scene (after alignment)
        images: list of (H,W,3) float [0,1] numpy arrays
        mix: 0.0 = pure dust3r, 1.0 = full AI detail overlay
        device: cuda or cpu
        progress_fn: optional callback(frac, msg)

    Returns:
        new_pts3d: list of (H,W,3) numpy arrays
    """
    from mono_depth import predict_depth
    from dust3r.utils.device import to_numpy
    from scipy.ndimage import gaussian_filter

    n_views = len(images)
    print(f"  Injecting AI depth detail into {n_views} views (mix={mix:.2f})")

    dust3r_depths = to_numpy(scene.get_depthmaps())  # list of (H,W)

    debug_dir = os.path.join(os.path.dirname(__file__), 'refine_debug')
    os.makedirs(debug_dir, exist_ok=True)

    # High-pass radius: controls what counts as "detail"
    # Smaller = more detail transferred, larger = only finest edges
    print(f"  Highpass radius: {highpass_sigma:.1f}px")

    for i in range(n_views):
        if progress_fn:
            progress_fn(i / n_views, f"AI depth detail: view {i+1}/{n_views}")
        print(f"  View {i+1}/{n_views}...")

        img = images[i]
        H, W = img.shape[:2]
        dust3r_depth = dust3r_depths[i]  # (H, W)

        # 1. Predict AI depth
        mono_depth = predict_depth(img, device=device)

        # Resize to match dust3r
        if mono_depth.shape != dust3r_depth.shape:
            mono_depth = np.array(Image.fromarray(mono_depth).resize(
                (dust3r_depth.shape[1], dust3r_depth.shape[0]), Image.BILINEAR))

        # 2. Check if inverted
        valid_both = (mono_depth > 0.01) & (dust3r_depth > 0.01)
        if valid_both.sum() > 100:
            corr = np.corrcoef(mono_depth[valid_both].ravel(),
                               dust3r_depth[valid_both].ravel())[0, 1]
            print(f"    Correlation: {corr:.4f}")
            if corr < -0.3:
                print(f"    Inverting monocular depth")
                mono_max = mono_depth[mono_depth > 0.01].max()
                mono_depth = mono_max - mono_depth
                mono_depth = np.clip(mono_depth, 0.001, None)

        # 3. Normalize AI depth to same range as dust3r
        # (just match min/max — we only care about the relative detail, not absolute values)
        valid = (mono_depth > 0.01) & (dust3r_depth > 0.01)
        if valid.sum() > 100:
            d3r_min, d3r_max = dust3r_depth[valid].min(), dust3r_depth[valid].max()
            ai_min, ai_max = mono_depth[valid].min(), mono_depth[valid].max()
            # Normalize AI to [0,1] then scale to dust3r range
            mono_normalized = (mono_depth - ai_min) / (ai_max - ai_min + 1e-8)
            mono_normalized = mono_normalized * (d3r_max - d3r_min) + d3r_min
        else:
            mono_normalized = mono_depth

        # 4. Compute difference (like Photoshop: invert AI at 50% opacity over dust3r)
        # difference = AI_normalized - dust3r
        # positive = AI thinks it's farther, negative = AI thinks it's closer
        difference = mono_normalized - dust3r_depth

        # 5. High-pass filter: remove low-frequency drift, keep sharp edges
        # High-pass = original - blur(original)
        low_freq = gaussian_filter(difference, sigma=highpass_sigma)
        high_freq_detail = difference - low_freq

        # 6. Apply via linear light: result = dust3r + detail * mix
        # (linear light in Photoshop is: base + 2*(overlay - 0.5),
        #  but since our detail is already centered at 0, just add it)
        enhanced = dust3r_depth + high_freq_detail * mix
        enhanced = np.clip(enhanced, 0.001, None).astype(np.float32)

        # Debug saves
        def _save_depth(d, name):
            valid_d = d > 0.01 if d.min() >= 0 else np.ones_like(d, dtype=bool)
            if valid_d.any():
                d_vis = d.copy()
                d_min, d_max = d[valid_d].min(), d[valid_d].max()
                d_vis = ((d - d_min) / (d_max - d_min + 1e-8) * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(d_vis).save(os.path.join(debug_dir, f'{name}_view{i}.png'))

        def _save_detail(d, name):
            # Center at 128 (grey = no change)
            d_max = max(abs(d.min()), abs(d.max()), 1e-8)
            d_vis = ((d / d_max) * 127 + 128).clip(0, 255).astype(np.uint8)
            Image.fromarray(d_vis).save(os.path.join(debug_dir, f'{name}_view{i}.png'))

        _save_depth(dust3r_depth, 'inject_dust3r')
        _save_depth(mono_normalized, 'inject_ai_normalized')
        _save_detail(difference, 'inject_difference')
        _save_detail(high_freq_detail, 'inject_highfreq_detail')
        _save_depth(enhanced, 'inject_enhanced')

        # 7. Inject back into scene
        enhanced_tensor = torch.from_numpy(enhanced).to(scene.im_depthmaps[0].device)
        scene._set_depthmap(i, enhanced_tensor, force=True)

        detail_strength = np.abs(high_freq_detail).mean()
        print(f"    Detail strength: {detail_strength:.6f}, "
              f"dust3r range: [{dust3r_depth.min():.4f}, {dust3r_depth.max():.4f}]")

    # Re-generate point cloud
    if progress_fn:
        progress_fn(0.9, "Regenerating point cloud...")

    new_pts3d = to_numpy(scene.get_pts3d())
    print(f"  Done. {n_views} views enhanced with AI detail (mix={mix:.2f})")

    return new_pts3d


def merge_overlapping_points(points, colors, merge_radius=None, cam_poses=None,
                             view_ids=None, confidences=None):
    """
    Merge points from DIFFERENT cameras that overlap in 3D space.
    Points from the same camera are never merged.

    Simple and correct approach:
    - Process one view at a time
    - For each point in view i, check if there's a nearby point from any other view
    - If yes: average them, mark the other-view point as consumed
    - If no: keep as-is

    Args:
        points: (N, 3) numpy
        colors: (N, 3) numpy uint8
        view_ids: (N,) int — which camera each point came from
        merge_radius: distance threshold
        confidences: optional (N,) float

    Returns:
        merged_points, merged_colors
    """
    from scipy.spatial import cKDTree

    N = len(points)
    if N == 0:
        return points, colors

    if view_ids is None:
        print("    No view_ids — skipping merge")
        return points, colors

    if confidences is None:
        confidences = np.ones(N, dtype=np.float32)

    # Auto radius
    if merge_radius is None:
        sample_idx = np.random.choice(N, min(10000, N), replace=False)
        tree_s = cKDTree(points[sample_idx])
        dists_s, _ = tree_s.query(points[sample_idx], k=2)
        merge_radius = float(np.median(dists_s[:, 1])) * 1.5
        print(f"    Auto merge radius: {merge_radius:.6f}")

    unique_views = np.unique(view_ids)
    n_views = len(unique_views)
    print(f"    Merging {N:,d} points from {n_views} views (radius={merge_radius:.6f})...")

    # Strategy: keep view 0 as the base, merge other views into it
    # For each subsequent view, find matches and average

    # Start with all points, mark which are "alive"
    alive = np.ones(N, dtype=bool)
    merged_pos = points.copy().astype(np.float64)
    merged_col = colors.copy().astype(np.float64)
    merged_weight = confidences.copy().astype(np.float64)

    # Process view pairs
    for vi in range(n_views):
        view_a = unique_views[vi]
        mask_a = (view_ids == view_a) & alive
        idx_a = np.where(mask_a)[0]

        if len(idx_a) == 0:
            continue

        # Build tree from view A's alive points
        tree_a = cKDTree(merged_pos[idx_a])

        for vj in range(vi + 1, n_views):
            view_b = unique_views[vj]
            mask_b = (view_ids == view_b) & alive
            idx_b = np.where(mask_b)[0]

            if len(idx_b) == 0:
                continue

            # For each point in view B, find nearest in view A
            dists, nearest_in_a = tree_a.query(merged_pos[idx_b])

            # Points within merge radius
            close = dists < merge_radius
            if not close.any():
                continue

            # Merge: average the matched pairs
            b_close = idx_b[close]
            a_close = idx_a[nearest_in_a[close]]

            # Weighted average into A's point
            w_a = merged_weight[a_close]
            w_b = merged_weight[b_close]
            w_total = w_a + w_b + 1e-8

            merged_pos[a_close] = (merged_pos[a_close] * w_a[:, None] +
                                   merged_pos[b_close] * w_b[:, None]) / w_total[:, None]
            merged_col[a_close] = (merged_col[a_close] * w_a[:, None] +
                                   merged_col[b_close] * w_b[:, None]) / w_total[:, None]
            merged_weight[a_close] = w_total

            # Kill the B points that were merged
            alive[b_close] = False

    # Collect survivors
    final_points = merged_pos[alive].astype(np.float32)
    final_colors = merged_col[alive].clip(0, 255).astype(np.uint8)

    n_merged = N - alive.sum()
    print(f"    Merged: {N:,d} -> {len(final_points):,d} points ({n_merged:,d} merged)")
    return final_points, final_colors


def refine_poses_with_ai_depth(scene, niter=100, lr=0.005, progress_fn=None):
    """
    Re-optimize camera poses using the (already injected) AI depth maps.

    Freezes depth maps and only optimizes poses + focals.
    This adjusts cameras to be consistent with the smoother AI depth.

    Args:
        scene: dust3r PointCloudOptimizer (after depth injection)
        niter: optimization iterations
        lr: learning rate

    Returns:
        loss: final alignment loss
    """
    print(f"  Refining poses with frozen AI depth ({niter} iterations, lr={lr})...")

    # Freeze depth maps — only optimize poses and focals
    scene.im_depthmaps.requires_grad_(False)

    # Make sure poses and focals are trainable
    scene.im_poses.requires_grad_(True)
    scene.im_focals.requires_grad_(True)

    # Run alignment loop
    params = [p for p in scene.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.9))

    for step in range(niter):
        optimizer.zero_grad()
        loss = scene()
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            loss_val = float(loss)
            print(f"    [{step:3d}/{niter}] loss={loss_val:.6f}")
            if progress_fn:
                progress_fn(step / niter, f"Pose refinement: step {step}/{niter}, loss={loss_val:.4f}")

    # Unfreeze depth maps for future use
    scene.im_depthmaps.requires_grad_(True)

    final_loss = float(loss)
    print(f"  Pose refinement done. Final loss: {final_loss:.6f}")
    return final_loss
