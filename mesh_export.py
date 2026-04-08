"""
Dense mesh generation from depth maps + camera poses.
Two methods:
  1. Open3D TSDF (if available) — proper volumetric fusion
  2. Poisson reconstruction from oriented point cloud — fallback
  3. Ball-pivoting from dense points — last resort
"""

import numpy as np
import struct


def create_dense_mesh(imgs, pts3d_list, confs_list, cam2world_list=None,
                      intrinsics_list=None, min_conf=2.0):
    """
    Create a dense mesh from reconstruction point maps.

    Args:
        imgs: list of (H,W,3) float [0,1]
        pts3d_list: list of (H,W,3) numpy — world-frame points
        confs_list: list of (H,W) numpy — confidence
        cam2world_list: optional list of (4,4) for oriented normals
        min_conf: confidence threshold

    Returns:
        vertices: (M, 3) numpy
        faces: (F, 3) numpy int32
        colors: (M, 3) numpy uint8
    """
    # Gather all valid points with colors and normals
    all_pts = []
    all_colors = []
    all_normals = []

    for i in range(len(imgs)):
        pts = pts3d_list[i]
        conf = confs_list[i]
        img = imgs[i]

        if pts.ndim == 2:
            # Flat array, can't compute normals from grid
            mask = np.ones(len(pts), dtype=bool) if conf is None else conf.ravel() > min_conf
            all_pts.append(pts[mask] if pts.ndim == 2 else pts.reshape(-1, 3)[mask])
            all_colors.append((np.clip(img.reshape(-1, 3)[mask] if img.ndim == 3 and img.shape[-1] == 3
                                       else img[mask], 0, 1) * 255).astype(np.uint8))
            continue

        H, W = pts.shape[:2]
        mask = conf > min_conf

        valid_pts = pts[mask]
        valid_colors = (np.clip(img[mask], 0, 1) * 255).astype(np.uint8)

        # Compute normals from the depth map grid (cross product of neighbors)
        normals = np.zeros((H, W, 3), dtype=np.float32)
        dx = np.zeros_like(pts)
        dy = np.zeros_like(pts)
        dx[:-1, :] = pts[1:, :] - pts[:-1, :]
        dy[:, :-1] = pts[:, 1:] - pts[:, :-1]
        normals = np.cross(dx, dy)
        n_len = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = np.where(n_len > 1e-8, normals / n_len, 0)

        # Orient normals toward camera if we have poses
        if cam2world_list is not None and i < len(cam2world_list):
            cam_center = cam2world_list[i][:3, 3]
            view_dirs = cam_center[None, None, :] - pts
            dot = (normals * view_dirs).sum(axis=-1)
            flip = dot < 0
            normals[flip] *= -1

        valid_normals = normals[mask]

        all_pts.append(valid_pts)
        all_colors.append(valid_colors)
        all_normals.append(valid_normals)

    if not all_pts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32), np.zeros((0, 3), dtype=np.uint8)

    points = np.concatenate(all_pts, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    normals = np.concatenate(all_normals, axis=0) if all_normals else None

    print(f"  Dense cloud: {len(points):,d} points")

    # Subsample if too many (for speed)
    max_pts = 500_000
    if len(points) > max_pts:
        idx = np.random.choice(len(points), max_pts, replace=False)
        points = points[idx]
        colors = colors[idx]
        if normals is not None:
            normals = normals[idx]

    # Try Open3D methods
    try:
        return _mesh_open3d(points, colors, normals)
    except ImportError:
        print("  Open3D not available, falling back to simple mesh")
    except Exception as e:
        print(f"  Open3D meshing failed: {e}, falling back")

    # Fallback: simple Delaunay-like mesh from point cloud
    return _mesh_from_points(points, colors)


def _mesh_open3d(points, colors, normals):
    """Create mesh using Open3D's Poisson reconstruction."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

    if normals is not None and len(normals) == len(points):
        pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    else:
        print("  Estimating normals...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=np.linalg.norm(np.asarray(pcd.points).max(0) - np.asarray(pcd.points).min(0)) * 0.02,
            max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=15)

    print("  Running Poisson reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9, width=0, scale=1.1, linear_fit=False)

    # Remove low-density vertices (outlier faces)
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.05)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # Transfer colors: color each vertex from nearest input point
    mesh_verts = np.asarray(mesh.vertices).astype(np.float32)
    tree = o3d.geometry.KDTreeFlann(pcd)
    mesh_colors = np.zeros((len(mesh_verts), 3), dtype=np.float64)
    for i in range(len(mesh_verts)):
        _, idx, _ = tree.search_knn_vector_3d(mesh_verts[i], 1)
        mesh_colors[i] = colors[idx[0]] / 255.0
    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)

    out_v = np.asarray(mesh.vertices, dtype=np.float32)
    out_f = np.asarray(mesh.triangles, dtype=np.int32)
    out_c = (np.asarray(mesh.vertex_colors) * 255).clip(0, 255).astype(np.uint8)

    print(f"  Poisson mesh: {len(out_v):,d} vertices, {len(out_f):,d} faces")
    return out_v, out_f, out_c


def _mesh_from_points(points, colors):
    """Simple mesh from point cloud using scipy Delaunay on 2D projection."""
    from scipy.spatial import Delaunay

    # Project to dominant plane for 2D triangulation
    center = points.mean(axis=0)
    pts_centered = points - center

    # PCA to find dominant plane
    cov = pts_centered.T @ pts_centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Project onto the two largest eigenvectors
    proj_2d = pts_centered @ eigvecs[:, 1:]  # (N, 2)

    print("  Running Delaunay triangulation...")
    try:
        tri = Delaunay(proj_2d)
        faces = tri.simplices

        # Filter out very long triangles (outliers)
        v0 = points[faces[:, 0]]
        v1 = points[faces[:, 1]]
        v2 = points[faces[:, 2]]
        edge_max = np.maximum(
            np.linalg.norm(v1 - v0, axis=-1),
            np.maximum(np.linalg.norm(v2 - v1, axis=-1),
                       np.linalg.norm(v0 - v2, axis=-1)))
        median_edge = np.median(edge_max)
        good = edge_max < median_edge * 5
        faces = faces[good]

        print(f"  Delaunay mesh: {len(points):,d} vertices, {len(faces):,d} faces")
        return points, faces.astype(np.int32), colors
    except Exception as e:
        print(f"  Delaunay failed: {e}")
        return points, np.zeros((0, 3), dtype=np.int32), colors


# Backwards compatibility alias
def tsdf_fusion(imgs, pts3d_list, confs_list, min_conf=2.0, **kwargs):
    return create_dense_mesh(imgs, pts3d_list, confs_list, min_conf=min_conf)


def save_mesh_ply(path, vertices, faces, colors):
    """Save mesh as PLY."""
    n_verts = len(vertices)
    n_faces = len(faces)
    header = f"""ply
format binary_little_endian 1.0
element vertex {n_verts}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face {n_faces}
property list uchar int vertex_indices
end_header
"""
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        for i in range(n_verts):
            f.write(struct.pack('<3f', *vertices[i].astype(np.float32)))
            f.write(struct.pack('<3B', *colors[i].astype(np.uint8)))
        for i in range(n_faces):
            f.write(struct.pack('<B3i', 3, *faces[i].astype(np.int32)))
    print(f"  Saved mesh: {n_verts:,d} verts, {n_faces:,d} faces -> {path}")


def save_dense_ply(path, pts, colors):
    """Save dense point cloud as PLY."""
    n = len(pts)
    header = f"""ply
format binary_little_endian 1.0
element vertex {n}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        for i in range(n):
            f.write(struct.pack('<3f', *pts[i].astype(np.float32)))
            f.write(struct.pack('<3B', *colors[i].astype(np.uint8)))
    print(f"  Saved dense PLY: {n:,d} points -> {path}")
