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
    Align monocular (relative) depth to the reconstruction's (metric) depth.
    Finds scale and shift: aligned = scale * mono + shift

    Args:
        mono_depth: (H, W) monocular relative depth
        recon_pts3d: (H, W, 3) reconstruction 3D points (from dust3r etc.)
        K: (3, 3) intrinsics
        w2c: (4, 4) world-to-camera
        conf: optional (H, W) confidence mask

    Returns:
        aligned_depth: (H, W) depth in metric scale
        scale, shift: alignment parameters
    """
    H, W = mono_depth.shape

    # Get reconstruction depth by projecting recon points to camera
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    pts_flat = recon_pts3d.reshape(-1, 3)
    pts_cam = (R @ pts_flat.T).T + t
    recon_depth = pts_cam[:, 2].reshape(H, W)

    # Valid pixels: positive depth in both, and within confidence
    valid = (mono_depth > 0.01) & (recon_depth > 0.01)
    if conf is not None:
        valid &= conf > conf_thr

    if valid.sum() < 100:
        # Not enough overlap, use median ratio
        med_mono = np.median(mono_depth[mono_depth > 0.01])
        med_recon = np.median(recon_depth[recon_depth > 0.01]) if (recon_depth > 0.01).any() else 1.0
        scale = med_recon / (med_mono + 1e-8)
        shift = 0.0
    else:
        # Least squares: recon_depth = scale * mono_depth + shift
        m = mono_depth[valid].ravel()
        r = recon_depth[valid].ravel()
        A = np.stack([m, np.ones_like(m)], axis=-1)
        result = np.linalg.lstsq(A, r, rcond=None)
        scale, shift = result[0]

    aligned = scale * mono_depth + shift
    aligned = np.clip(aligned, 0.001, None)
    return aligned, scale, shift


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

        # 2. Align to reconstruction scale
        if recon_pts3d[i] is not None:
            conf = recon_confs[i] if recon_confs[i] is not None else None
            aligned_depth, scale, shift = align_mono_depth_to_reconstruction(
                mono_depth, recon_pts3d[i], K, w2c, conf=conf)
            print(f"    Depth aligned: scale={scale:.4f}, shift={shift:.4f}")
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
