"""
Monocular Depth + Normals → Dense Point Cloud

Uses DepthAnything v2 for per-pixel depth and DSINE for normals,
combined with camera poses from dust3r/mast3r/vggt to generate
a high-quality dense point cloud and mesh.

Pipeline:
  1. Camera poses from dust3r/mast3r/vggt (they're great at this)
  2. Monocular depth per image (DepthAnything v2 — smooth, consistent)
  3. Monocular normals per image (DSINE — sharp edges)
  4. Unproject depth → dense 3D points per view
  5. Multi-view consistency: align scales, filter outliers
  6. Poisson reconstruction with oriented normals → clean mesh
"""

import numpy as np
from PIL import Image
import os


# ── Monocular Depth ──────────────────────────────────────────────────────────

_depth_pipe = None

def load_depth_model(device='cuda'):
    global _depth_pipe
    if _depth_pipe is not None:
        return _depth_pipe

    print("  Loading DepthAnything v2...")
    from transformers import pipeline
    _depth_pipe = pipeline(
        'depth-estimation',
        model='depth-anything/Depth-Anything-V2-Small-hf',
        device=device
    )
    print("  DepthAnything v2 loaded")
    return _depth_pipe


def predict_depth(image, device='cuda'):
    """
    Predict relative depth from an RGB image.

    Args:
        image: (H, W, 3) float [0,1] numpy

    Returns:
        depth: (H, W) float numpy — relative depth (not metric, needs scale alignment)
    """
    pipe = load_depth_model(device)
    img_pil = Image.fromarray((image * 255).clip(0, 255).astype(np.uint8))
    result = pipe(img_pil)

    depth = np.array(result['predicted_depth'])
    # Resize to match input if needed
    if depth.shape[:2] != image.shape[:2]:
        depth = np.array(Image.fromarray(depth).resize(
            (image.shape[1], image.shape[0]), Image.BILINEAR))

    return depth.astype(np.float32)


# ── Unproject Depth to 3D ────────────────────────────────────────────────────

def unproject_depth(depth, K, c2w):
    """
    Unproject a depth map to 3D world points.

    Args:
        depth: (H, W) float — depth values
        K: (3, 3) intrinsics matrix
        c2w: (4, 4) camera-to-world matrix

    Returns:
        points: (H, W, 3) world-space 3D points
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Pixel grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    # Camera-space 3D points
    x_cam = (u - cx) / fx * depth
    y_cam = (v - cy) / fy * depth
    z_cam = depth

    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (H, W, 3)

    # Transform to world space
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    pts_world = (pts_cam @ R.T) + t[None, None, :]

    return pts_world


# ── Scale Alignment ──────────────────────────────────────────────────────────

def align_mono_depth_to_reconstruction(mono_depth, recon_pts3d, K, w2c, conf=None, conf_thr=1.0):
    """
    Align monocular depth to reconstruction depth using a nonlinear transfer curve.

    The mono depth has the right relative ordering but the wrong gamma/curve.
    We fit a monotonic piecewise-linear transfer function:
      real_depth = transfer(mono_depth)

    Built by:
    1. Collect (mono_val, dust3r_val) pairs from all valid pixels
    2. Sort by mono_val, bin into N buckets
    3. For each bucket: median of dust3r values → one control point
    4. Interpolate → smooth transfer curve
    5. Apply to all pixels

    Args:
        mono_depth: (H, W) monocular depth (already inverted if needed)
        recon_pts3d: (H, W, 3) reconstruction 3D points
        K, w2c: camera matrices
        conf: optional (H, W) confidence

    Returns:
        aligned_depth: (H, W) corrected depth
        transfer_curve: (N, 2) array of [mono_val, real_val] control points
    """
    H, W = mono_depth.shape

    # Get reconstruction depth
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    pts_flat = recon_pts3d.reshape(-1, 3)
    pts_cam = (R @ pts_flat.T).T + t
    recon_depth = pts_cam[:, 2].reshape(H, W)

    # Valid correspondences
    valid = (mono_depth > 0.01) & (recon_depth > 0.01)
    if conf is not None:
        valid &= conf > conf_thr

    n_valid = valid.sum()
    print(f"    Transfer curve: {n_valid:,d} valid correspondences")

    if n_valid < 50:
        med_mono = np.median(mono_depth[mono_depth > 0.01]) + 1e-8
        med_recon = np.median(recon_depth[recon_depth > 0.01]) if (recon_depth > 0.01).any() else 1.0
        scale = med_recon / med_mono
        curve = np.array([[0, 0], [1, scale]])
        return mono_depth * scale, curve

    # Collect pairs
    mono_vals = mono_depth[valid].ravel()
    recon_vals = recon_depth[valid].ravel()

    # Remove outliers
    p2, p98 = np.percentile(recon_vals, [2, 98])
    good = (recon_vals > p2) & (recon_vals < p98)
    mono_vals = mono_vals[good]
    recon_vals = recon_vals[good]

    # Sort by mono value
    sort_idx = np.argsort(mono_vals)
    mono_sorted = mono_vals[sort_idx]
    recon_sorted = recon_vals[sort_idx]

    # Bin into N buckets, take median of each
    N_BINS = 64
    bin_size = max(1, len(mono_sorted) // N_BINS)
    control_mono = []
    control_recon = []

    for bi in range(0, len(mono_sorted), bin_size):
        chunk_m = mono_sorted[bi:bi + bin_size]
        chunk_r = recon_sorted[bi:bi + bin_size]
        if len(chunk_m) > 0:
            control_mono.append(np.median(chunk_m))
            control_recon.append(np.median(chunk_r))

    control_mono = np.array(control_mono, dtype=np.float64)
    control_recon = np.array(control_recon, dtype=np.float64)

    # Ensure monotonic (both should increase together)
    # If not, enforce by taking cumulative max
    for i in range(1, len(control_recon)):
        if control_recon[i] < control_recon[i - 1]:
            control_recon[i] = control_recon[i - 1]

    print(f"    Transfer curve: {len(control_mono)} control points")
    print(f"    Mono range: [{control_mono[0]:.4f}, {control_mono[-1]:.4f}]")
    print(f"    Recon range: [{control_recon[0]:.4f}, {control_recon[-1]:.4f}]")

    # Apply transfer curve via interpolation
    aligned = np.interp(mono_depth.ravel(), control_mono, control_recon).reshape(H, W)
    aligned = np.clip(aligned, 0.001, None).astype(np.float32)

    curve = np.stack([control_mono, control_recon], axis=-1)
    return aligned, curve


# ── Full Pipeline ────────────────────────────────────────────────────────────

def generate_enhanced_pointcloud(scene, image_paths, progress_fn=None, device='cuda'):
    """
    Generate an enhanced dense point cloud using:
    - Camera poses from the reconstruction scene
    - Monocular depth (DepthAnything v2) aligned to reconstruction scale
    - Monocular normals (DSINE) for surface orientation

    Args:
        scene: reconstruction scene (dust3r/mast3r/vggt)
        image_paths: list of image file paths
        progress_fn: optional callback(fraction, message)

    Returns:
        points: (N, 3) numpy
        colors: (N, 3) numpy uint8
        normals: (N, 3) numpy float — oriented surface normals
    """
    from dust3r.utils.device import to_numpy
    from normal_estimator import predict_normals
    import torch

    imgs = scene.imgs
    n_views = len(imgs)

    # Get camera poses
    c2w_all = to_numpy(scene.get_im_poses().cpu())  # (N, 4, 4)

    # Get intrinsics
    if hasattr(scene, '_is_vggt'):
        intrinsics = [scene._intrinsic[i] for i in range(n_views)]
        recon_pts3d = scene._pts3d
        recon_confs = scene._depth_conf
    elif hasattr(scene, 'canonical_paths'):
        intrinsics = [to_numpy(K.cpu()) for K in scene.intrinsics]
        pts_raw, _, confs_raw = scene.get_dense_pts3d(clean_depth=True)
        recon_pts3d = [to_numpy(pts_raw[i]).reshape(imgs[i].shape[0], imgs[i].shape[1], 3)
                       for i in range(n_views)]
        recon_confs = [to_numpy(confs_raw[i]) for i in range(n_views)]
    else:
        intrinsics = [to_numpy(K) for K in scene.get_intrinsics().cpu()]
        recon_pts3d = to_numpy(scene.get_pts3d())
        recon_confs = [None] * n_views

    all_points = []
    all_colors = []
    all_normals = []

    for i in range(n_views):
        if progress_fn:
            progress_fn(i / n_views, f"Processing view {i+1}/{n_views}")
        print(f"  View {i+1}/{n_views}...")

        img = imgs[i]
        H, W = img.shape[:2]
        K = intrinsics[i]
        c2w = c2w_all[i]
        w2c = np.linalg.inv(c2w)

        # 1. Predict monocular depth
        mono_depth = predict_depth(img, device=device)

        # 2. Get dust3r depth for this view (project recon points to camera)
        R_w2c = w2c[:3, :3]
        t_w2c = w2c[:3, 3]
        recon_pts_cam = (R_w2c @ recon_pts3d[i].reshape(-1, 3).T).T + t_w2c
        dust3r_depth = recon_pts_cam[:, 2].reshape(H, W)

        # Debug: save both depth maps for comparison
        debug_dir = os.path.join(os.path.dirname(__file__), 'refine_debug')
        os.makedirs(debug_dir, exist_ok=True)
        from PIL import Image as _PILImg

        # Normalize both to [0, 255] for visualization
        def _depth_to_vis(d, name):
            valid = d > 0.01
            if valid.any():
                d_vis = d.copy()
                d_vis[~valid] = 0
                d_min, d_max = d_vis[valid].min(), d_vis[valid].max()
                d_vis = ((d_vis - d_min) / (d_max - d_min + 1e-8) * 255).clip(0, 255).astype(np.uint8)
                d_vis[~valid] = 0
                _PILImg.fromarray(d_vis).save(os.path.join(debug_dir, f'{name}_view{i}.png'))
                print(f"    {name}: range [{d_min:.4f}, {d_max:.4f}]")

        _depth_to_vis(mono_depth, 'mono_depth')
        _depth_to_vis(dust3r_depth, 'dust3r_depth')

        # Check if monocular depth is inverted relative to dust3r
        valid_both = (mono_depth > 0.01) & (dust3r_depth > 0.01)
        if valid_both.sum() > 100:
            corr = np.corrcoef(mono_depth[valid_both].ravel(), dust3r_depth[valid_both].ravel())[0, 1]
            print(f"    Correlation mono vs dust3r: {corr:.4f}")
            if corr < -0.3:
                print(f"    *** INVERTING monocular depth (negative correlation detected) ***")
                mono_max = mono_depth[mono_depth > 0.01].max()
                mono_depth = mono_max - mono_depth
                mono_depth = np.clip(mono_depth, 0.001, None)
                _depth_to_vis(mono_depth, 'mono_depth_inverted')

        # 3. Align to reconstruction scale using per-pixel LUT
        if recon_pts3d[i] is not None:
            conf = recon_confs[i] if recon_confs[i] is not None else None
            aligned_depth, correction_map = align_mono_depth_to_reconstruction(
                mono_depth, recon_pts3d[i], K, w2c, conf=conf)
            _depth_to_vis(aligned_depth, 'aligned_depth')
        else:
            aligned_depth = mono_depth

        # 3. Predict normals
        normals_cam = predict_normals(img, intrinsics=K, device=device)

        # Convert normals from DSINE camera space to world space
        # DSINE: X=right, Y=up, Z=toward camera
        # OpenCV: X=right, Y=down, Z=away from camera
        normals_opencv = normals_cam.copy()
        normals_opencv[:, :, 1] *= -1  # Y: up -> down
        normals_opencv[:, :, 2] *= -1  # Z: toward cam -> away
        R = c2w[:3, :3]
        normals_flat = normals_opencv.reshape(-1, 3)
        normals_world = (R @ normals_flat.T).T.reshape(H, W, 3)

        # 4. Unproject depth to 3D
        pts_world = unproject_depth(aligned_depth, K, c2w)

        # 5. Filter: remove invalid / low-confidence points
        valid = aligned_depth > 0.01

        # Also filter by comparing to reconstruction depth if available
        if recon_pts3d[i] is not None and recon_confs[i] is not None:
            # Keep points where reconstruction has reasonable confidence
            conf_mask = recon_confs[i] > np.median(recon_confs[i]) * 0.3 if recon_confs[i] is not None else np.ones((H, W), dtype=bool)
            valid &= conf_mask

        # Subsample to limit total points per view
        max_per_view = 200000
        valid_idx = np.where(valid.ravel())[0]
        if len(valid_idx) > max_per_view:
            valid_idx = np.random.choice(valid_idx, max_per_view, replace=False)

        pts = pts_world.reshape(-1, 3)[valid_idx]
        cols = (np.clip(img.reshape(-1, 3)[valid_idx], 0, 1) * 255).astype(np.uint8)
        nrms = normals_world.reshape(-1, 3)[valid_idx]

        # Normalize normals
        n_len = np.linalg.norm(nrms, axis=-1, keepdims=True) + 1e-8
        nrms /= n_len

        all_points.append(pts)
        all_colors.append(cols)
        all_normals.append(nrms)

        print(f"    {len(pts):,d} points from this view")

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    normals = np.concatenate(all_normals, axis=0)

    print(f"  Total enhanced cloud: {len(points):,d} points")

    if progress_fn:
        progress_fn(1.0, "Enhanced point cloud complete")

    return points, colors, normals


def enhanced_cloud_to_mesh(points, colors, normals, target_faces=100000):
    """
    Convert oriented point cloud to mesh using Poisson reconstruction.

    Args:
        points: (N, 3)
        colors: (N, 3) uint8
        normals: (N, 3) float — oriented normals
        target_faces: decimate to this count

    Returns:
        verts, faces, vert_colors
    """
    try:
        import open3d as o3d

        print(f"  Building mesh from {len(points):,d} oriented points...")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))

        # Poisson reconstruction (normals make this much better)
        print("  Running Poisson reconstruction with oriented normals...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9, width=0, scale=1.1, linear_fit=False)

        # Remove low-density faces
        densities = np.asarray(densities)
        threshold = np.quantile(densities, 0.05)
        mesh.remove_vertices_by_mask(densities < threshold)

        # Decimate
        if len(mesh.triangles) > target_faces:
            print(f"  Decimating {len(mesh.triangles):,d} -> {target_faces:,d} faces...")
            mesh = mesh.simplify_quadric_decimation(target_faces)

        # Transfer colors from point cloud
        mesh_verts = np.asarray(mesh.vertices).astype(np.float32)
        tree = o3d.geometry.KDTreeFlann(pcd)
        mesh_colors = np.zeros((len(mesh_verts), 3), dtype=np.float64)
        for vi in range(len(mesh_verts)):
            _, idx, _ = tree.search_knn_vector_3d(mesh_verts[vi], 1)
            mesh_colors[vi] = colors[idx[0]] / 255.0
        mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)

        out_v = np.asarray(mesh.vertices, dtype=np.float32)
        out_f = np.asarray(mesh.triangles, dtype=np.int32)
        out_c = (np.asarray(mesh.vertex_colors) * 255).clip(0, 255).astype(np.uint8)

        print(f"  Enhanced mesh: {len(out_v):,d} verts, {len(out_f):,d} faces")
        return out_v, out_f, out_c

    except ImportError:
        print("  Open3D not available, returning point cloud only")
        return points, np.zeros((0, 3), dtype=np.int32), colors
