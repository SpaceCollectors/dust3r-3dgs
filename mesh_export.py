"""
Dense mesh generation from depth maps + camera poses.

Triangulates each camera's depth map directly in pixel-grid space,
preserving every original 3D point as a mesh vertex. Long edges
(depth discontinuities) are rejected to avoid connecting foreground
to background.
"""

import os
import numpy as np
import struct


def create_dense_mesh(imgs, pts3d_list, confs_list, cam2world_list=None,
                      intrinsics_list=None, min_conf=2.0,
                      poisson_depth=8, normal_radius=0.03, trim_percentile=5,
                      mode='reprojected', hole_cap_size=50, dense_colors=None,
                      bp_radius_mult=4.0, delaunay_edge_mult=8.0):
    """Create mesh from multi-view point clouds.

    mode='reprojected': Voxel-merge all points, shared vertex grid triangulation.
    mode='ballpivot':   Voxel-dedup, ball-pivot.
    mode='delaunay':    Voxel-dedup, local tangent-plane Delaunay.
    hole_cap_size:      Max boundary edges to close (0=disable, via PyMeshLab).
    """
    if mode == 'poisson':
        all_points, all_colors = _collect_points(imgs, pts3d_list, confs_list, min_conf, dense_colors=dense_colors)
        if len(all_points) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32), np.zeros((0, 3), dtype=np.uint8)
        import open3d as o3d
        print(f"  Poisson reconstruction on {len(all_points):,d} points...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(all_colors.astype(np.float64) / 255.0)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        # Voxel downsample to merge overlapping points from different views
        # This prevents Poisson from creating double surfaces
        dists = np.asarray(pcd.compute_nearest_neighbor_distance())
        median_spacing = float(np.median(dists))
        voxel_size = median_spacing * 2.0
        n_before = len(pcd.points)
        pcd = pcd.voxel_down_sample(voxel_size)
        print(f"  Voxel merge: {n_before:,d} -> {len(pcd.points):,d} points (voxel={voxel_size:.6f})")

        # Recompute spacing after merge
        dists = np.asarray(pcd.compute_nearest_neighbor_distance())
        median_spacing = float(np.median(dists))
        print(f"  Median point spacing: {median_spacing:.6f}")
        print(f"  Scene extent: {np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound())):.4f}")

        # Normal estimation with radius based on actual point spacing
        normal_radius = median_spacing * 5
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=30))
        if cam2world_list:
            cam_center = np.mean([c[:3, 3] for c in cam2world_list], axis=0)
            pcd.orient_normals_towards_camera_location(cam_center)
        else:
            pcd.orient_normals_consistent_tangent_plane(k=15)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=poisson_depth, width=0, scale=1.1, linear_fit=True)
        # Trim low-density regions
        densities = np.asarray(densities)
        if trim_percentile > 0:
            mesh.remove_vertices_by_mask(densities < np.quantile(densities, trim_percentile / 100.0))
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_vertices()
        # Transfer colors
        mesh_verts = np.asarray(mesh.vertices, dtype=np.float64)
        tree = o3d.geometry.KDTreeFlann(pcd)
        pcd_colors = np.asarray(pcd.colors)
        mesh_colors = np.zeros((len(mesh_verts), 3), dtype=np.float64)
        for vi in range(len(mesh_verts)):
            _, idx, _ = tree.search_knn_vector_3d(mesh_verts[vi], 3)
            mesh_colors[vi] = np.mean([pcd_colors[j] for j in idx], axis=0)
        v = mesh_verts.astype(np.float32)
        f = np.asarray(mesh.triangles, dtype=np.int32)
        c = (mesh_colors * 255).clip(0, 255).astype(np.uint8)
        print(f"  Poisson mesh: {len(v):,d} verts, {len(f):,d} faces")
    elif mode == 'delaunay':
        all_points, all_colors = _collect_points(imgs, pts3d_list, confs_list, min_conf, dense_colors=dense_colors)
        if len(all_points) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32), np.zeros((0, 3), dtype=np.uint8)
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(all_colors.astype(np.float64) / 255.0)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        extent = np.linalg.norm(np.asarray(pcd.points).max(0) - np.asarray(pcd.points).min(0))
        pcd = pcd.voxel_down_sample(extent / 800)
        pts = np.asarray(pcd.points, dtype=np.float32)
        cols = (np.asarray(pcd.colors) * 255).clip(0, 255).astype(np.uint8)
        cam_center = np.mean([c[:3, 3] for c in cam2world_list], axis=0) if cam2world_list else None
        v, f, c = _mesh_local_delaunay(pts, cols, cam_center=cam_center,
                                       radius_mult=delaunay_edge_mult)
    elif mode == 'reprojected' and cam2world_list is not None:
        v, f, c = _mesh_reprojected_grid(imgs, pts3d_list, confs_list,
                                          cam2world_list, min_conf)
    else:
        v, f, c = _mesh_ball_pivot(imgs, pts3d_list, confs_list,
                                    cam2world_list, min_conf, dense_colors=dense_colors,
                                    radius_mult=bp_radius_mult)

    # Close holes using PyMeshLab
    if hole_cap_size > 0 and len(f) > 0:
        v, f, c = _close_holes_pymeshlab(v, f, c, max_hole_size=hole_cap_size)

    return v, f, c


def _collect_points(imgs, pts3d_list, confs_list, min_conf, dense_colors=None):
    """Gather all confidence-filtered points + colors from all views."""
    all_pts, all_colors = [], []
    for i in range(len(pts3d_list)):  # iterate ALL entries, not just len(imgs)
        pts = pts3d_list[i]
        conf = confs_list[i] if i < len(confs_list) else None
        img = imgs[i] if i < len(imgs) else None
        if pts.ndim == 2 or (pts.ndim == 1 and len(pts) > 0):
            flat = pts.reshape(-1, 3)
            mask = conf.ravel() > min_conf if conf is not None else np.ones(len(flat), dtype=bool)
            all_pts.append(flat[mask])
            # Try dense_colors first (matches COLMAP dense cloud)
            if dense_colors is not None and len(dense_colors) == len(flat):
                all_colors.append(dense_colors[mask])
            elif img is not None and len(img.reshape(-1, 3)) == len(flat):
                all_colors.append((np.clip(img.reshape(-1, 3)[mask], 0, 1) * 255).astype(np.uint8))
            else:
                all_colors.append(np.full((mask.sum(), 3), 180, dtype=np.uint8))
        else:
            H, W = pts.shape[:2]
            mask = conf.reshape(H, W) > min_conf if conf is not None else np.ones((H, W), dtype=bool)
            all_pts.append(pts[mask])
            if img is not None and img.shape[:2] == (H, W):
                all_colors.append((np.clip(img[mask], 0, 1) * 255).astype(np.uint8))
            else:
                all_colors.append(np.full((mask.sum(), 3), 180, dtype=np.uint8))
    if not all_pts:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    return (np.concatenate(all_pts, axis=0).astype(np.float32),
            np.concatenate(all_colors, axis=0))


def _mesh_reprojected_grid(imgs, pts3d_list, confs_list, cam2world_list, min_conf):
    """Multi-view consensus meshing via voxel-based vertex sharing.

    1. Collect all valid points from all views
    2. Voxel-grid them → each voxel averages all points that fall in it (multi-view fusion)
    3. For each view, map each pixel to its nearest voxel vertex
    4. Grid-triangulate each view using shared voxel vertex IDs
    5. Deduplicate faces — overlapping views produce identical faces (same vertex IDs)
    """
    import open3d as o3d

    # Step 1: Collect all points with view/pixel tracking
    all_pts, all_cols = [], []
    pixel_info = []  # (view_idx, row, col) for each point

    for vi in range(len(imgs)):
        pts = pts3d_list[vi]
        if pts.ndim != 3:
            continue
        H, W = pts.shape[:2]
        conf = confs_list[vi]
        mask = conf.reshape(H, W) > min_conf if conf is not None else np.ones((H, W), dtype=bool)
        img = imgs[vi]

        rows, cols_px = np.where(mask)
        p = pts[rows, cols_px].astype(np.float32)
        c = (np.clip(img[rows, cols_px], 0, 1) * 255).astype(np.uint8)

        all_pts.append(p)
        all_cols.append(c)
        info = np.zeros((len(rows), 3), dtype=np.int32)
        info[:, 0] = vi; info[:, 1] = rows; info[:, 2] = cols_px
        pixel_info.append(info)
        print(f"  View {vi+1}: {len(rows):,d} valid points")

    if not all_pts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32), np.zeros((0, 3), dtype=np.uint8)

    points = np.concatenate(all_pts, axis=0)
    colors = np.concatenate(all_cols, axis=0)
    pinfo = np.concatenate(pixel_info, axis=0)
    print(f"  Total: {len(points):,d} points")

    # Step 2: Voxel grid — merge overlapping points from different views
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

    extent = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    voxel_size = extent / 800
    downsampled, _, idx_map_list = pcd.voxel_down_sample_and_trace(
        voxel_size, pcd.get_min_bound(), pcd.get_max_bound())

    # Build mapping: original point index → merged vertex index
    merged_verts = np.asarray(downsampled.points, dtype=np.float32)
    merged_colors = (np.asarray(downsampled.colors) * 255).clip(0, 255).astype(np.uint8)
    n_merged = len(merged_verts)
    print(f"  Voxel merge: {len(points):,d} -> {n_merged:,d} vertices (voxel={voxel_size:.6f})")

    # idx_map_list[i] = list of original point indices that merged into vertex i
    # Build reverse: original_idx → merged_vertex_idx
    orig_to_merged = np.full(len(points), -1, dtype=np.int32)
    for merged_idx, orig_indices in enumerate(idx_map_list):
        for oi in orig_indices:
            orig_to_merged[int(oi)] = merged_idx

    # Step 3: Build per-view pixel→vertex index maps
    view_shapes = {}
    view_vidx = {}
    for vi in range(len(imgs)):
        pts = pts3d_list[vi]
        if pts.ndim != 3:
            continue
        H, W = pts.shape[:2]
        view_shapes[vi] = (H, W)
        view_vidx[vi] = np.full((H, W), -1, dtype=np.int32)

    # Scatter merged vertex IDs back to each view's pixel grid
    for i in range(len(pinfo)):
        mid = orig_to_merged[i]
        if mid < 0:
            continue
        vi, r, c = int(pinfo[i, 0]), int(pinfo[i, 1]), int(pinfo[i, 2])
        if vi in view_vidx:
            view_vidx[vi][r, c] = mid

    # Step 4: Grid-triangulate each view using shared vertex indices
    all_faces = []
    for vi, (H, W) in view_shapes.items():
        idx_map = view_vidx[vi]
        filled = idx_map >= 0

        # Edge length threshold
        pts_grid = np.zeros((H, W, 3), dtype=np.float32)
        pts_grid[filled] = merged_verts[idx_map[filled]]
        dists_h = np.linalg.norm(pts_grid[:-1, :] - pts_grid[1:, :], axis=-1)
        dists_w = np.linalg.norm(pts_grid[:, :-1] - pts_grid[:, 1:], axis=-1)
        valid_edges = np.concatenate([
            dists_h[filled[:-1, :] & filled[1:, :]].ravel(),
            dists_w[filled[:, :-1] & filled[:, 1:]].ravel()])
        if len(valid_edges) < 10:
            continue
        max_edge = np.percentile(valid_edges, 95) * 5.0

        i00 = idx_map[:-1, :-1]; i10 = idx_map[1:, :-1]
        i01 = idx_map[:-1, 1:]; i11 = idx_map[1:, 1:]
        valid_t1 = (i00 >= 0) & (i10 >= 0) & (i01 >= 0)
        valid_t2 = (i10 >= 0) & (i11 >= 0) & (i01 >= 0)

        d00_10 = np.linalg.norm(pts_grid[:-1, :-1] - pts_grid[1:, :-1], axis=-1)
        d00_01 = np.linalg.norm(pts_grid[:-1, :-1] - pts_grid[:-1, 1:], axis=-1)
        d10_01 = np.linalg.norm(pts_grid[1:, :-1] - pts_grid[:-1, 1:], axis=-1)
        d10_11 = np.linalg.norm(pts_grid[1:, :-1] - pts_grid[1:, 1:], axis=-1)
        d01_11 = np.linalg.norm(pts_grid[:-1, 1:] - pts_grid[1:, 1:], axis=-1)
        valid_t1 &= (d00_10 < max_edge) & (d00_01 < max_edge) & (d10_01 < max_edge)
        valid_t2 &= (d10_11 < max_edge) & (d01_11 < max_edge) & (d10_01 < max_edge)

        tri1 = np.stack([i00[valid_t1], i10[valid_t1], i01[valid_t1]], axis=-1)
        tri2 = np.stack([i10[valid_t2], i11[valid_t2], i01[valid_t2]], axis=-1)
        if len(tri1) + len(tri2) > 0:
            vf = np.concatenate([tri1, tri2], axis=0).astype(np.int32)
            # Remove degenerate (where voxel merge collapsed vertices)
            non_degen = (vf[:, 0] != vf[:, 1]) & (vf[:, 1] != vf[:, 2]) & (vf[:, 0] != vf[:, 2])
            vf = vf[non_degen]
            all_faces.append(vf)
            print(f"    View {vi+1}: {len(vf):,d} faces")

    if not all_faces:
        return merged_verts, np.zeros((0, 3), dtype=np.int32), merged_colors

    faces = np.concatenate(all_faces, axis=0)

    # Step 5: Deduplicate faces across views (same 3 vertex IDs = same face)
    n_before = len(faces)
    face_keys = np.sort(faces, axis=1)
    _, unique_idx = np.unique(face_keys, axis=0, return_index=True)
    faces = faces[unique_idx]
    print(f"  Dedup: {n_before:,d} -> {len(faces):,d} faces")

    # Remove unreferenced vertices
    used = np.unique(faces.ravel())
    remap = np.full(n_merged, -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    merged_verts = merged_verts[used]
    merged_colors = merged_colors[used]
    faces = remap[faces]

    print(f"  Final mesh: {len(merged_verts):,d} verts, {len(faces):,d} faces")
    return merged_verts, faces, merged_colors


def _estimate_intrinsics(pts_orig, pts_cam, H, W):
    """Estimate fx, fy, cx, cy from original grid points and their camera-space coords."""
    # Use a grid of sample pixels to estimate focal length
    cy_px, cx_px = H // 2, W // 2
    # Sample a band around center
    band = 10
    r0, r1 = max(cy_px - band, 0), min(cy_px + band, H)
    c0, c1 = max(cx_px - band, 0), min(cx_px + band, W)
    z_center = pts_cam[r0:r1, c0:c1, 2]
    valid = z_center > 0.01
    if valid.sum() < 4:
        # Fallback: assume 45° FOV
        fx = fy = W / (2 * np.tan(np.radians(22.5)))
        return np.array([fx, fy, W / 2.0, H / 2.0])

    # For pixels (r, c), the projection is: u = fx * X/Z + cx, v = fy * Y/Z + cy
    # We know (r, c) are pixel coords and (X, Y, Z) are camera-space coords
    # Estimate fx from horizontal: fx = (u - cx) * Z / X
    rows = np.arange(r0, r1)[:, None] * np.ones((1, c1 - c0))
    cols = np.ones((r1 - r0, 1)) * np.arange(c0, c1)[None, :]
    X = pts_cam[r0:r1, c0:c1, 0]
    Y = pts_cam[r0:r1, c0:c1, 1]
    Z = pts_cam[r0:r1, c0:c1, 2]

    ok = valid & (np.abs(X) > 1e-6) & (np.abs(Y) > 1e-6)
    if ok.sum() < 4:
        fx = fy = W / (2 * np.tan(np.radians(22.5)))
        return np.array([fx, fy, W / 2.0, H / 2.0])

    # fx = (col - cx) * Z / X → solve for fx and cx
    # Simple: assume cx = W/2, cy = H/2 and solve for fx, fy
    cx, cy = W / 2.0, H / 2.0
    fx_samples = (cols[ok] - cx) * Z[ok] / X[ok]
    fy_samples = (rows[ok] - cy) * Z[ok] / Y[ok]
    fx = float(np.median(fx_samples))
    fy = float(np.median(fy_samples))
    # Sanity clamp
    fx = np.clip(fx, W * 0.3, W * 5)
    fy = np.clip(fy, H * 0.3, H * 5)
    return np.array([fx, fy, cx, cy])


def _smooth_cloud(verts, colors, radius_mult=1.5, view_ids=None, iterations=3):
    """Voxel merge first (fast reduction), then cross-view attraction on the reduced set.

    1. Voxel downsample at user radius — fast, reduces point count massively
    2. Cross-view attraction on the reduced cloud — pulls remaining layers together
    3. Final voxel merge to collapse converged points
    """
    import open3d as o3d
    from scipy.spatial import cKDTree

    n_before = len(verts)

    # Compute median point spacing
    sample_idx = np.random.choice(len(verts), min(10000, len(verts)), replace=False)
    tree0 = cKDTree(verts[sample_idx])
    d0, _ = tree0.query(verts[sample_idx], k=2)
    median_dist = float(np.median(d0[:, 1]))
    voxel_size = median_dist * max(radius_mult, 0.1)

    # Step 1: Voxel merge — fast point reduction
    print(f"  Voxel merge (size={voxel_size:.6f})...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

    if view_ids is not None:
        pcd_merged, _, idx_lists = pcd.voxel_down_sample_and_trace(
            voxel_size, pcd.get_min_bound(), pcd.get_max_bound())
        # Track: does this voxel contain points from multiple views?
        merged_vids = np.zeros(len(idx_lists), dtype=np.int32)
        multi_view = np.zeros(len(idx_lists), dtype=bool)
        for mi, orig_indices in enumerate(idx_lists):
            src_views = view_ids[[int(oi) for oi in orig_indices]]
            merged_vids[mi] = np.bincount(src_views).argmax()
            multi_view[mi] = len(np.unique(src_views)) > 1
        pcd = pcd_merged
        print(f"  After merge: {n_before:,d} -> {len(pcd.points):,d} ({multi_view.sum():,d} multi-view)")
    else:
        pcd = pcd.voxel_down_sample(voxel_size)
        merged_vids = None
        print(f"  After merge: {n_before:,d} -> {len(pcd.points):,d}")

    # Step 2: Cross-view attraction on the reduced cloud (fast now)
    pts = np.asarray(pcd.points).copy()
    N = len(pts)
    attract_radius = voxel_size * 3.0
    if merged_vids is not None and len(np.unique(merged_vids)) > 1 and N < 500000:
        print(f"  Cross-view attraction ({iterations} iters, {N:,d} pts)...")
        for it in range(iterations):
            tree = cKDTree(pts)
            pairs = tree.query_pairs(r=attract_radius, output_type='ndarray')
            if len(pairs) == 0:
                break

            cross = merged_vids[pairs[:, 0]] != merged_vids[pairs[:, 1]]
            pairs = pairs[cross]
            if len(pairs) == 0:
                break

            shift_sum = np.zeros_like(pts)
            shift_count = np.zeros(N, dtype=np.float64)
            np.add.at(shift_sum, pairs[:, 0], pts[pairs[:, 1]])
            np.add.at(shift_count, pairs[:, 0], 1.0)
            np.add.at(shift_sum, pairs[:, 1], pts[pairs[:, 0]])
            np.add.at(shift_count, pairs[:, 1], 1.0)

            moved = shift_count > 0
            avg_neighbor = shift_sum[moved] / shift_count[moved, None]
            pts[moved] += (avg_neighbor - pts[moved]) * 0.4

            print(f"    Iter {it+1}: {moved.sum():,d} points, {len(pairs):,d} cross-view pairs")

        # Step 3: Re-merge to collapse attracted points
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd = pcd.voxel_down_sample(voxel_size)

    out_v = np.asarray(pcd.points, dtype=np.float32)
    out_c = (np.asarray(pcd.colors) * 255).clip(0, 255).astype(np.uint8)

    print(f"  Smooth: {n_before:,d} -> {len(out_v):,d} points")
    return out_v, out_c


def _mesh_local_delaunay(points, colors, cam_center=None, radius_mult=8.0, max_edge_mult=0.0):
    """Surface reconstruction via local 2D Delaunay on tangent planes.

    For each point:
      - Compute adaptive selection radius = local_density * radius_mult
      - Project all neighbours inside that radius onto the tangent plane
      - Run 2D Delaunay
      - Keep only triangles that touch the center point

    Radius-based selection adapts to varying point density automatically.

    radius_mult: selection radius = local_spacing * this (default 8.0).
    max_edge_mult: 0 = no edge filtering (default). >0 = reject edges longer
                   than local_spacing * mult.
    """
    import open3d as o3d
    from scipy.spatial import Delaunay, cKDTree

    V = len(points)
    if V < 3:
        return points, np.zeros((0, 3), dtype=np.int32), colors

    filter_edges = max_edge_mult > 0
    print(f"  Local radius-based Delaunay on {V:,d} points (radius_mult={radius_mult:.1f})...")

    # Estimate normals
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0, max_nn=30))
    if cam_center is not None:
        pcd.orient_normals_towards_camera_location(cam_center)
    else:
        pcd.orient_normals_consistent_tangent_plane(k=15)
    normals = np.asarray(pcd.normals, dtype=np.float64)

    # KD-tree
    tree = cKDTree(points)

    # Local density estimate — use k=30 to get local spacing
    small_k = min(30, V)
    nn_dists_all, _ = tree.query(points, k=small_k)
    local_scale = nn_dists_all[:, -1]  # distance to 30th neighbour
    median_dist = float(np.median(local_scale))
    local_scale = np.clip(local_scale, median_dist * 0.05, median_dist * 20.0)

    # Compute adaptive k from radius: query enough neighbors to cover the radius
    # For roughly uniform density, points in a circle of radius R ~ pi*R^2 / spacing^2
    max_radius = float(np.max(local_scale)) * radius_mult
    median_radius = median_dist * radius_mult
    # Estimate k needed to cover the radius for the densest regions
    adaptive_k = min(int(3.14 * (radius_mult ** 2)) + 10, V, 200)
    print(f"    adaptive k={adaptive_k} for radius_mult={radius_mult:.1f}")

    # Single batch query with adaptive_k — much faster than per-point query_ball
    nn_dists_full, nn_idx_full = tree.query(points, k=adaptive_k)

    all_tris = set()
    batch_log = max(1, V // 10)

    for pi in range(V):
        if pi % batch_log == 0:
            print(f"    {pi:,d} / {V:,d} ({pi * 100 // V}%)")

        center = points[pi]
        n = normals[pi]
        selection_radius = local_scale[pi] * radius_mult

        # Filter the k-nearest to those within the selection radius
        dists = nn_dists_full[pi]
        within = dists <= selection_radius
        local_idx = nn_idx_full[pi][within]

        if len(local_idx) < 4:
            continue

        # Ensure self is at index 0
        self_mask = local_idx == pi
        if self_mask.any():
            others = local_idx[~self_mask]
            local_idx = np.concatenate(([pi], others))
        else:
            local_idx = np.concatenate(([pi], local_idx))

        if len(local_idx) < 4:
            continue

        # Local tangent basis
        if abs(n[0]) < 0.9:
            tangent = np.cross(n, [1, 0, 0])
        else:
            tangent = np.cross(n, [0, 1, 0])
        tangent /= max(np.linalg.norm(tangent), 1e-12)
        bitangent = np.cross(n, tangent)

        # Project to 2D tangent plane
        rel = points[local_idx] - center
        pts_2d = np.column_stack([rel @ tangent, rel @ bitangent])

        try:
            tri = Delaunay(pts_2d, qhull_options="Qbb Qc Qz")
        except Exception:
            continue

        # Keep only triangles touching the center point (local index 0)
        for simplex in tri.simplices:
            if 0 not in simplex:
                continue
            a, b, c = int(local_idx[simplex[0]]), int(local_idx[simplex[1]]), int(local_idx[simplex[2]])

            if filter_edges:
                max_edge = local_scale[pi] * max_edge_mult
                d0 = np.linalg.norm(points[a] - points[b])
                d1 = np.linalg.norm(points[b] - points[c])
                d2 = np.linalg.norm(points[a] - points[c])
                if d0 > max_edge or d1 > max_edge or d2 > max_edge:
                    continue

            all_tris.add(tuple(sorted([a, b, c])))

    faces = np.array(list(all_tris), dtype=np.int32) if all_tris else np.zeros((0, 3), dtype=np.int32)
    print(f"  Local Delaunay: {V:,d} verts, {len(faces):,d} faces")
    return points.astype(np.float32), faces, colors


def _mesh_ball_pivot_from_cloud(points, colors, cam_center=None):
    """Ball-pivot a pre-processed point cloud into a mesh."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0, max_nn=30))
    if cam_center is not None:
        pcd.orient_normals_towards_camera_location(cam_center)
    else:
        pcd.orient_normals_consistent_tangent_plane(k=15)

    dists = np.asarray(pcd.compute_nearest_neighbor_distance())
    avg_dist = np.mean(dists)
    radii = [avg_dist * 1.0, avg_dist * 2.0, avg_dist * 4.0, avg_dist * 8.0]
    print(f"  Ball-pivoting (radii={[f'{r:.5f}' for r in radii]})...")

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()

    out_v = np.asarray(mesh.vertices, dtype=np.float32)
    out_f = np.asarray(mesh.triangles, dtype=np.int32)
    out_c = ((np.asarray(mesh.vertex_colors) * 255).clip(0, 255).astype(np.uint8)
             if mesh.has_vertex_colors() else np.full((len(out_v), 3), 128, dtype=np.uint8))
    print(f"  Ball-pivot: {len(out_v):,d} verts, {len(out_f):,d} faces")
    return out_v, out_f, out_c


def _mesh_ball_pivot(imgs, pts3d_list, confs_list, cam2world_list, min_conf, dense_colors=None, radius_mult=4.0):
    """Voxel-dedup all points, then ball-pivot into a single mesh."""
    import open3d as o3d

    all_points, all_colors = _collect_points(imgs, pts3d_list, confs_list, min_conf, dense_colors=dense_colors)
    if len(all_points) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32), np.zeros((0, 3), dtype=np.uint8)
    print(f"  Combined: {len(all_points):,d} points")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(all_colors.astype(np.float64) / 255.0)

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Compute median spacing — only downsample if there are actual duplicates
    dists_raw = np.asarray(pcd.compute_nearest_neighbor_distance())
    median_spacing = float(np.median(dists_raw))
    extent = np.linalg.norm(np.asarray(pcd.points).max(0) - np.asarray(pcd.points).min(0))
    print(f"  Median spacing: {median_spacing:.6f}, extent: {extent:.4f}")

    # Voxel downsample to merge overlapping views into a single surface
    # 1.0x median spacing merges near-duplicate points from different cameras
    voxel_size = median_spacing * 1.0
    n_before = len(pcd.points)
    pcd = pcd.voxel_down_sample(voxel_size)
    print(f"  Dedup: {n_before:,d} -> {len(pcd.points):,d} (voxel={voxel_size:.6f})")

    # Normal estimation at actual point scale
    normal_radius = median_spacing * 4
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=normal_radius, max_nn=30))
    if cam2world_list:
        cam_center = np.mean([c[:3, 3] for c in cam2world_list], axis=0)
        pcd.orient_normals_towards_camera_location(cam_center)
    else:
        pcd.orient_normals_consistent_tangent_plane(k=15)

    # Ball-pivoting with user-controlled max radius
    dists = np.asarray(pcd.compute_nearest_neighbor_distance())
    avg_dist = np.mean(dists)
    max_r = avg_dist * radius_mult
    radii = sorted(set([avg_dist, avg_dist * 2, max_r * 0.5, max_r]))
    radii = [r for r in radii if r > 0]
    print(f"  Ball-pivoting (radii={[f'{r:.6f}' for r in radii]}, avg_dist={avg_dist:.6f})...")

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()

    out_v = np.asarray(mesh.vertices, dtype=np.float32)
    out_f = np.asarray(mesh.triangles, dtype=np.int32)
    out_c = ((np.asarray(mesh.vertex_colors) * 255).clip(0, 255).astype(np.uint8)
             if mesh.has_vertex_colors() else np.full((len(out_v), 3), 128, dtype=np.uint8))

    print(f"  Ball-pivot mesh: {len(out_v):,d} verts, {len(out_f):,d} faces")
    return out_v, out_f, out_c


def _close_holes_pymeshlab(verts, faces, colors, max_hole_size=50):
    """Close holes using PyMeshLab's battle-tested hole closing filter."""
    try:
        import pymeshlab
    except ImportError:
        print("  PyMeshLab not available, skipping hole closing")
        return verts, faces, colors

    if len(faces) == 0:
        return verts, faces, colors

    n_before = len(faces)

    ms = pymeshlab.MeshSet()
    m = pymeshlab.Mesh(
        vertex_matrix=verts.astype(np.float64),
        face_matrix=faces.astype(np.int32),
        v_color_matrix=np.column_stack([
            colors.astype(np.float64) / 255.0,
            np.ones((len(colors), 1), dtype=np.float64)  # alpha
        ])
    )
    ms.add_mesh(m)

    # Close holes up to max_hole_size edges
    try:
        ms.meshing_close_holes(maxholesize=max_hole_size)
    except Exception as e:
        print(f"  Hole closing failed: {e}")
        return verts, faces, colors

    mesh_out = ms.current_mesh()
    out_v = mesh_out.vertex_matrix().astype(np.float32)
    out_f = mesh_out.face_matrix().astype(np.int32)

    # Get colors back
    if mesh_out.has_vertex_color():
        vc = mesh_out.vertex_color_matrix()
        out_c = (vc[:, :3] * 255).clip(0, 255).astype(np.uint8)
    else:
        # New vertices from hole filling won't have colors — use nearest neighbor
        out_c = np.full((len(out_v), 3), 128, dtype=np.uint8)
        n_orig = min(len(colors), len(out_v))
        out_c[:n_orig] = colors[:n_orig]

    n_closed = len(out_f) - n_before
    if n_closed > 0:
        print(f"  Closed holes: {n_before:,d} -> {len(out_f):,d} faces (+{n_closed:,d})")
    else:
        print(f"  No holes closed (none within {max_hole_size} edge limit)")

    return out_v, out_f, out_c


def tsdf_fusion(imgs, pts3d_list, confs_list, min_conf=2.0, **kwargs):
    return create_dense_mesh(imgs, pts3d_list, confs_list, min_conf=min_conf)


def save_mesh_ply(path, vertices, faces, colors):
    n_v, n_f = len(vertices), len(faces)
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {n_v}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face {n_f}\nproperty list uchar int vertex_indices\nend_header\n"
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        for i in range(n_v):
            f.write(struct.pack('<3f', *vertices[i].astype(np.float32)))
            f.write(struct.pack('<3B', *colors[i].astype(np.uint8)))
        for i in range(n_f):
            f.write(struct.pack('<B3i', 3, *faces[i].astype(np.int32)))


def _find_colmap_exe():
    """Find COLMAP executable — check PATH, project folder, common locations."""
    import shutil
    # Check PATH
    exe = shutil.which('colmap')
    if exe:
        return exe
    # Check project subfolder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for subdir in ['colmap', 'COLMAP', 'colmap/bin', 'COLMAP/bin']:
        candidate = os.path.join(script_dir, subdir, 'colmap.exe')
        if os.path.exists(candidate):
            return candidate
    # Check Program Files
    for pf in [os.environ.get('ProgramFiles', ''), os.environ.get('ProgramFiles(x86)', '')]:
        if pf:
            candidate = os.path.join(pf, 'COLMAP', 'bin', 'colmap.exe')
            if os.path.exists(candidate):
                return candidate
    return None


def _write_colmap_sparse_model(sparse_dir, image_paths, c2w_list, K_list,
                               scene_pts=None):
    """Write a COLMAP sparse model in text format from known cameras.
    If scene_pts is provided, projects them into cameras to create proper
    observations for PatchMatch depth range estimation."""
    from scipy.spatial.transform import Rotation
    from PIL import Image as PILImage

    os.makedirs(sparse_dir, exist_ok=True)
    n = min(len(image_paths), len(c2w_list), len(K_list))

    # cameras.txt
    img_sizes = []
    with open(os.path.join(sparse_dir, 'cameras.txt'), 'w') as f:
        f.write("# Camera list\n# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i in range(n):
            img = PILImage.open(image_paths[i])
            W, H = img.size
            img_sizes.append((W, H))
            K = K_list[i]
            f.write(f"{i+1} PINHOLE {W} {H} {K[0,0]:.6f} {K[1,1]:.6f} {K[0,2]:.6f} {K[1,2]:.6f}\n")

    # images.txt
    w2c_list = []
    with open(os.path.join(sparse_dir, 'images.txt'), 'w') as f:
        f.write("# Image list\n# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        for i in range(n):
            c2w = c2w_list[i].astype(np.float64)
            w2c = np.linalg.inv(c2w)
            w2c_list.append(w2c)
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            rot = Rotation.from_matrix(R)
            quat = rot.as_quat()  # (x, y, z, w)
            qw, qx, qy, qz = float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])
            name = os.path.basename(image_paths[i])
            f.write(f"{i+1} {qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f} {t[0]:.10f} {t[1]:.10f} {t[2]:.10f} {i+1} {name}\n")
            f.write("\n")

    # points3D.txt — use scene points if available, otherwise dummy points
    n_pts = 0
    with open(os.path.join(sparse_dir, 'points3D.txt'), 'w') as f:
        f.write("# 3D point list\n")
        if scene_pts is not None and len(scene_pts) > 0:
            # Subsample to reasonable count
            pts = scene_pts
            if len(pts) > 10000:
                idx = np.random.choice(len(pts), 10000, replace=False)
                pts = pts[idx]
            # For each point, find which cameras see it
            pid = 1
            for pt in pts:
                observers = []
                for i in range(n):
                    w2c = w2c_list[i]
                    K = K_list[i]
                    W, H = img_sizes[i]
                    pt_cam = w2c[:3, :3] @ pt + w2c[:3, 3]
                    if pt_cam[2] < 0.01:
                        continue
                    u = K[0, 0] * pt_cam[0] / pt_cam[2] + K[0, 2]
                    v = K[1, 1] * pt_cam[1] / pt_cam[2] + K[1, 2]
                    if 0 <= u < W and 0 <= v < H:
                        observers.append((i + 1, int(u), int(v)))
                if len(observers) >= 2:
                    track = " ".join(f"{img_id} {px}" for img_id, px, py in observers)
                    f.write(f"{pid} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} 128 128 128 0.5 {track}\n")
                    pid += 1
            n_pts = pid - 1
        if n_pts == 0:
            # Fallback: dummy points at camera centers
            pid = 1
            for i in range(n):
                center = c2w_list[i][:3, 3]
                j = (i + 1) % n
                f.write(f"{pid} {center[0]:.6f} {center[1]:.6f} {center[2]:.6f} 128 128 128 0.5 "
                        f"{i+1} 1 {j+1} 1\n")
                pid += 1
            n_pts = n

    print(f"  Wrote COLMAP sparse model: {n} cameras, {n_pts} points")
    return n


def densify_colmap(image_paths, c2w_list=None, K_list=None, progress_fn=None,
                   max_image_size=-1, num_iterations=5, window_radius=5,
                   min_consistent=2, geom_consistency=True, filter_min_ncc=0.1,
                   existing_pts=None, colmap_workdir=None):
    """Dense MVS via full COLMAP pipeline (SfM + PatchMatch + fusion).

    Runs COLMAP's own SfM to get reliable cameras, then PatchMatch dense stereo.
    This avoids issues with mixing DUSt3R cameras with COLMAP's fusion.
    """
    import tempfile, shutil
    from PIL import Image as PILImage

    colmap_exe = _find_colmap_exe()
    if not colmap_exe:
        raise RuntimeError(
            "COLMAP executable not found. Download from:\n"
            "  https://github.com/colmap/colmap/releases\n"
            "  Get: colmap-x64-windows-cuda.zip\n"
            "  Extract to project folder as colmap/")

    # Prevent COLMAP/glog from trying to create log files (fails on Windows)
    colmap_env = os.environ.copy()
    colmap_env['GLOG_logtostderr'] = '1'

    n_imgs = len(image_paths)
    import subprocess, re

    # Reuse existing COLMAP workspace if available (same coordinate frame)
    reuse = False
    if colmap_workdir and os.path.isdir(colmap_workdir):
        sparse_sub = None
        for d in ['sparse/0', 'sparse/1', 'sparse']:
            candidate = os.path.join(colmap_workdir, d)
            if os.path.isdir(candidate) and (
                os.path.exists(os.path.join(candidate, 'cameras.bin')) or
                os.path.exists(os.path.join(candidate, 'cameras.txt'))):
                sparse_sub = candidate
                break
        if sparse_sub:
            workdir = colmap_workdir
            img_dir = os.path.join(workdir, "images")
            reuse = True
            print(f"  Reusing COLMAP workspace: {workdir}")
            print(f"  Sparse model: {sparse_sub}")

    if not reuse:
        workdir = tempfile.mkdtemp(prefix="colmap_dense_")

    try:
        if not reuse:
            # Copy images
            img_dir = os.path.join(workdir, "images")
            os.makedirs(img_dir, exist_ok=True)
            for p in image_paths:
                shutil.copy2(p, os.path.join(img_dir, os.path.basename(p)))

            db_path = os.path.join(workdir, "database.db")
            sparse_dir = os.path.join(workdir, "sparse")
            os.makedirs(sparse_dir, exist_ok=True)

            # Step 1: Feature extraction
            if progress_fn: progress_fn("COLMAP: extracting features...")
            print("  Feature extraction...")
            subprocess.run([colmap_exe, 'feature_extractor',
                           '--database_path', db_path,
                           '--image_path', img_dir],
                          capture_output=True, timeout=600,
                          env=colmap_env)

            # Step 2: Feature matching
            if progress_fn: progress_fn("COLMAP: matching features...")
            print("  Feature matching...")
            subprocess.run([colmap_exe, 'exhaustive_matcher',
                           '--database_path', db_path],
                          capture_output=True, timeout=600,
                          env=colmap_env)

            # Step 3: Sparse reconstruction
            # If we have prior cameras (from DUSt3R etc.), use them as initialization
            # Run point_triangulator + bundle_adjuster instead of full mapper
            if c2w_list is not None and K_list is not None and len(c2w_list) == n_imgs:
                if progress_fn: progress_fn("COLMAP: mapping with prior cameras...")
                print("  Using prior cameras for initialization...")
                prior_dir = os.path.join(sparse_dir, 'prior')
                _write_colmap_sparse_model(prior_dir, image_paths, c2w_list, K_list,
                                          scene_pts=existing_pts)

                # Use point_triangulator to keep cameras FIXED from DUSt3R
                # This triangulates 2D matches using our cameras without moving them
                print("  Triangulating points with fixed cameras...")
                tri_output = os.path.join(sparse_dir, '0')
                os.makedirs(tri_output, exist_ok=True)
                # Copy prior model as the input
                import shutil as _sh
                for f in os.listdir(prior_dir):
                    _sh.copy2(os.path.join(prior_dir, f), tri_output)
                r = subprocess.run([colmap_exe, 'point_triangulator',
                               '--database_path', db_path,
                               '--image_path', img_dir,
                               '--input_path', tri_output,
                               '--output_path', tri_output],
                              capture_output=True, text=True, timeout=600,
                              env=colmap_env)
                if r.stderr:
                    for line in r.stderr.split('\n'):
                        if any(k in line for k in ['Registering', 'Bundle', 'points', 'images']):
                            print(f"    {line.strip()}")

                # Find best reconstruction
                sparse_sub = None
                for d in ['0', '1', '2']:
                    candidate = os.path.join(sparse_dir, d)
                    if os.path.isdir(candidate):
                        sparse_sub = candidate
                        break

                if sparse_sub is None:
                    # Mapper with priors failed — fall back to full COLMAP mapper
                    print("  Mapper with priors failed, running full COLMAP SfM...")
                    if progress_fn: progress_fn("COLMAP: full sparse reconstruction...")
                    subprocess.run([colmap_exe, 'mapper',
                                   '--database_path', db_path,
                                   '--image_path', img_dir,
                                   '--output_path', sparse_dir],
                                  capture_output=True, timeout=600,
                                  env=colmap_env)
                    for d in ['0', '1', '2']:
                        candidate = os.path.join(sparse_dir, d)
                        if os.path.isdir(candidate):
                            sparse_sub = candidate
                            break
                    if sparse_sub is None:
                        raise RuntimeError("COLMAP reconstruction failed — no model produced")

                print(f"  Sparse model: {sparse_sub}")
            else:
                # Full COLMAP mapper (no prior cameras)
                if progress_fn: progress_fn("COLMAP: sparse reconstruction...")
                print("  Sparse reconstruction (full mapper)...")
                subprocess.run([colmap_exe, 'mapper',
                               '--database_path', db_path,
                               '--image_path', img_dir,
                               '--output_path', sparse_dir],
                              capture_output=True, timeout=600,
                              env=colmap_env)

                # Find the reconstruction
                sparse_sub = os.path.join(sparse_dir, '0')
                if not os.path.isdir(sparse_sub):
                    for d in os.listdir(sparse_dir):
                        if os.path.isdir(os.path.join(sparse_dir, d)):
                            sparse_sub = os.path.join(sparse_dir, d)
                            break
                    else:
                        raise RuntimeError("COLMAP sparse reconstruction failed")

            print(f"  Sparse model: {sparse_sub}")

        # Step 4: Undistort
        if progress_fn: progress_fn("COLMAP: undistorting images...")
        dense_dir = os.path.join(workdir, "dense")
        os.makedirs(dense_dir, exist_ok=True)
        subprocess.run([colmap_exe, 'image_undistorter',
                       '--image_path', img_dir,
                       '--input_path', sparse_sub,
                       '--output_path', dense_dir,
                       '--output_type', 'COLMAP'],
                      capture_output=True, timeout=600,
                      env=colmap_env)
        print("  Images undistorted")

        # Step 5: PatchMatch stereo
        if progress_fn: progress_fn("COLMAP: PatchMatch stereo...")
        print("  Running PatchMatch stereo...")

        cmd = [colmap_exe, 'patch_match_stereo',
               '--workspace_path', dense_dir,
               '--PatchMatchStereo.geom_consistency', '1' if geom_consistency else '0',
               '--PatchMatchStereo.filter', '1',
               '--PatchMatchStereo.num_iterations', str(num_iterations),
               '--PatchMatchStereo.window_radius', str(window_radius),
               '--PatchMatchStereo.filter_min_ncc', str(filter_min_ncc),
               '--PatchMatchStereo.filter_min_num_consistent', str(min_consistent)]
        if max_image_size > 0:
            cmd += ['--PatchMatchStereo.max_image_size', str(max_image_size)]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1, env=colmap_env)
        for line in proc.stdout:
            line = line.strip()
            m = re.search(r'Processing view (\d+) / (\d+) for (.+)', line)
            if m:
                cur, total, name = int(m.group(1)), int(m.group(2)), m.group(3)
                if progress_fn:
                    progress_fn(f"PatchMatch {cur}/{total}: {name}")
                print(f"  PatchMatch view {cur}/{total}: {name}")
            elif 'Total:' in line:
                t = re.search(r'Total: ([\d.]+)s', line)
                if t:
                    print(f"    {t.group(1)}s")
            elif 'Error' in line or 'Check failed' in line or 'FATAL' in line:
                print(f"  ERROR: {line}")
        proc.wait(timeout=1800)
        if proc.returncode != 0:
            raise RuntimeError(f"PatchMatch failed (code {proc.returncode})")

        # Step 6: Stereo fusion
        output_ply = os.path.join(dense_dir, "fused.ply")
        if progress_fn: progress_fn("COLMAP: fusing depth maps...")
        print("  Fusing...")
        r = subprocess.run([colmap_exe, 'stereo_fusion',
                           '--workspace_path', dense_dir,
                           '--output_path', output_ply,
                           '--StereoFusion.min_num_pixels', str(max(min_consistent, 2))],
                          capture_output=True, text=True, timeout=600,
                          env=colmap_env)
        if r.stderr:
            for line in r.stderr.split('\n'):
                if any(k in line for k in ['Fusing', 'points', 'Number']):
                    print(f"  {line.strip()}")
        print(f"  Fused to {output_ply}")

        # Step 5: Load the fused point cloud
        if not os.path.exists(output_ply):
            print("  ERROR: fused.ply not found")
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8), None

        import open3d as o3d
        pcd = o3d.io.read_point_cloud(output_ply)
        pts = np.asarray(pcd.points, dtype=np.float32)
        if pcd.has_colors():
            cols = (np.asarray(pcd.colors) * 255).clip(0, 255).astype(np.uint8)
        else:
            cols = np.full((len(pts), 3), 180, dtype=np.uint8)

        print(f"  Dense cloud: {len(pts):,d} points")

        # Read COLMAP cameras so caller can update scene
        colmap_cameras = None
        try:
            import pycolmap
            rec = pycolmap.Reconstruction(sparse_sub)
            colmap_cameras = []
            name_to_idx = {os.path.basename(p): i for i, p in enumerate(image_paths)}
            for img_id, image in sorted(rec.images.items()):
                name = image.name
                if name not in name_to_idx:
                    continue
                cam = rec.cameras[image.camera_id]
                K = cam.calibration_matrix().astype(np.float32)
                # Get c2w
                try:
                    cfw = image.cam_from_world
                    if callable(cfw):
                        w2c_34 = np.array(cfw().matrix())
                    elif hasattr(cfw, 'matrix'):
                        w2c_34 = np.array(cfw.matrix())
                    else:
                        w2c_34 = np.array(cfw)
                except Exception:
                    w2c_34 = np.eye(3, 4)
                w2c = np.eye(4, dtype=np.float32)
                w2c[:3, :] = w2c_34.astype(np.float32)
                c2w = np.linalg.inv(w2c).astype(np.float32)
                colmap_cameras.append((name_to_idx[name], c2w, K, cam.width, cam.height))
            # Sort by original image index
            colmap_cameras.sort(key=lambda x: x[0])
            colmap_cameras = [(c2w, K, W, H) for _, c2w, K, W, H in colmap_cameras]
            print(f"  Read {len(colmap_cameras)} COLMAP cameras")
        except Exception as e:
            print(f"  Could not read COLMAP cameras: {e}")

        return pts, cols, colmap_cameras

    except Exception as e:
        print(f"  COLMAP dense failed: {e}")
        import traceback; traceback.print_exc()
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8), None
    finally:
        if not reuse:
            shutil.rmtree(workdir, ignore_errors=True)


def bundle_adjust_depth_consistency(c2w_list, K_list, pts3d_list, confs_list,
                                    image_paths=None,
                                    refine_focal=False, stride=4, min_conf=2.0,
                                    n_iters=200, max_shift=0.2, huber_scale=0.1,
                                    progress_fn=None):
    """Bundle adjustment via multi-view depth consistency.

    For each pair of overlapping cameras, projects camera A's pointmap into
    camera B's image and compares depths. Optimizes pose (6 DOF) + optional
    focal length (1 DOF) per camera to minimize depth disagreement.

    No feature matching needed — works directly on the pointmaps.

    Returns list of refined (c2w, K, W, H) per image, or None on failure.
    """
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    n_imgs = len(c2w_list)
    if n_imgs < 2:
        print("  Need at least 2 images")
        return None

    try:
        if progress_fn:
            progress_fn("BA: Preparing pointmaps...")
        print("  Depth-consistency BA: preparing data...")

        # Subsample pointmaps for speed
        # pts3d_list[i] is (H, W, 3) in world space, confs_list[i] is (H, W)
        cam_pts = []   # world-space 3D points per camera (N_i, 3)
        cam_confs = []  # confidence per point
        cam_pixels = []  # pixel coords (u, v) in model resolution
        cam_HW = []     # (H, W) of each pointmap

        for i in range(n_imgs):
            p = pts3d_list[i]
            c = confs_list[i]
            if p.ndim != 3:
                print(f"  Skipping camera {i}: pointmap not (H,W,3)")
                cam_pts.append(None)
                cam_confs.append(None)
                cam_pixels.append(None)
                cam_HW.append((0, 0))
                continue

            H, W = p.shape[:2]
            cam_HW.append((H, W))

            # Subsample with stride
            ys = np.arange(0, H, stride)
            xs = np.arange(0, W, stride)
            yy, xx = np.meshgrid(ys, xs, indexing='ij')
            yy = yy.ravel()
            xx = xx.ravel()

            pts = p[yy, xx]  # (N, 3)
            conf = c.reshape(H, W)[yy, xx] if c is not None else np.ones(len(yy), dtype=np.float32) * 10
            pixels = np.stack([xx.astype(np.float64), yy.astype(np.float64)], axis=1)  # (N, 2) as (u, v)

            # Filter by confidence and valid points
            valid = (conf > min_conf) & np.isfinite(pts).all(axis=1)
            cam_pts.append(pts[valid].astype(np.float64))
            cam_confs.append(conf[valid].astype(np.float64))
            cam_pixels.append(pixels[valid])

        # Build overlapping pairs: project A's points into B, check how many land inside
        if progress_fn:
            progress_fn("BA: Finding overlapping pairs...")
        print("  Finding overlapping camera pairs...")

        # Scale intrinsics to model resolution for projection/indexing into pointmaps.
        # The caller may pass full-resolution K but pointmaps are at model resolution.
        K_f64 = []
        K_full = [K.astype(np.float64) for K in K_list]  # keep originals for output
        for i in range(n_imgs):
            K = K_list[i].astype(np.float64).copy()
            H_m, W_m = cam_HW[i]
            if H_m > 0 and W_m > 0:
                # Detect scale: if principal point is far from model center, K is at full res
                cx, cy = K[0, 2], K[1, 2]
                if cx > W_m * 1.5 or cy > H_m * 1.5:
                    sx = W_m / (2.0 * cx)  # cx ≈ W_full/2, so sx ≈ W_m/W_full
                    sy = H_m / (2.0 * cy)
                    K[0, 0] *= sx; K[0, 2] = W_m / 2.0
                    K[1, 1] *= sy; K[1, 2] = H_m / 2.0
                    if i == 0:
                        print(f"    Rescaled intrinsics to model resolution "
                              f"(scale ~{sx:.3f}x{sy:.3f})")
            K_f64.append(K)
        w2c_init = [np.linalg.inv(c2w.astype(np.float64)) for c2w in c2w_list]

        pairs = []  # (i, j) pairs with overlap
        pair_data = []  # precomputed data per pair

        for i in range(n_imgs):
            if cam_pts[i] is None or len(cam_pts[i]) == 0:
                continue
            for j in range(n_imgs):
                if i == j or cam_pts[j] is None or len(cam_pts[j]) == 0:
                    continue

                # Project camera i's points into camera j
                pts_w = cam_pts[i]  # (N, 3) world space
                H_j, W_j = cam_HW[j]

                # Transform to camera j space
                p_cam = (w2c_init[j][:3, :3] @ pts_w.T).T + w2c_init[j][:3, 3]
                depth_proj = p_cam[:, 2]

                # Project to pixel coords in j
                u_proj = K_f64[j][0, 0] * p_cam[:, 0] / p_cam[:, 2] + K_f64[j][0, 2]
                v_proj = K_f64[j][1, 1] * p_cam[:, 1] / p_cam[:, 2] + K_f64[j][1, 2]

                # Check which points land inside j's image and are in front
                inside = ((depth_proj > 0.01) &
                          (u_proj >= 0) & (u_proj < W_j) &
                          (v_proj >= 0) & (v_proj < H_j))

                n_overlap = inside.sum()
                if n_overlap < 20:
                    continue

                # For overlapping points, get camera j's depth at those pixel locations
                # Use nearest-neighbor lookup into j's pointmap
                u_int = np.clip(u_proj[inside].astype(int), 0, W_j - 1)
                v_int = np.clip(v_proj[inside].astype(int), 0, H_j - 1)

                # Camera j's points at those pixels, transformed to j's camera space
                pts_j_world = pts3d_list[j][v_int, u_int]  # (M, 3)
                p_j_cam = (w2c_init[j][:3, :3] @ pts_j_world.astype(np.float64).T).T + w2c_init[j][:3, 3]
                depth_j = p_j_cam[:, 2]
                depth_i_in_j = depth_proj[inside]

                # Filter: both depths positive, j's confidence at those pixels
                conf_j_at = confs_list[j].reshape(H_j, W_j)[v_int, u_int].astype(np.float64)
                conf_i_at = cam_confs[i][inside]
                valid = (depth_j > 0.01) & (depth_i_in_j > 0.01) & (conf_j_at > min_conf)

                if valid.sum() < 10:
                    continue

                # Store indices of i's points that overlap with j
                inside_idx = np.where(inside)[0]
                overlap_idx = inside_idx[valid]
                weights = np.sqrt(conf_i_at[valid] * conf_j_at[valid])

                pairs.append((i, j))
                pair_data.append({
                    'pts_i_idx': overlap_idx,  # indices into cam_pts[i]
                    'pix_j': np.stack([u_int[valid], v_int[valid]], axis=1),  # pixel coords in j
                    'weights': weights,
                })
                print(f"    Pair ({i}->{j}): {valid.sum()} overlapping points")

        if not pairs:
            print("  No overlapping pairs found")
            return None

        total_overlaps = sum(len(pd['weights']) for pd in pair_data)
        print(f"  {len(pairs)} directed pairs, {total_overlaps} overlap points")

        # Pack camera parameters: 6 per camera (rvec + tvec), optionally +1 focal
        if progress_fn:
            progress_fn("BA: Optimizing cameras...")
        print("  Optimizing camera poses via depth consistency...")

        params_per_cam = 7 if refine_focal else 6
        x0 = np.zeros(n_imgs * params_per_cam)

        for i in range(n_imgs):
            rvec = Rotation.from_matrix(w2c_init[i][:3, :3]).as_rotvec()
            tvec = w2c_init[i][:3, 3]
            off = i * params_per_cam
            x0[off:off + 3] = rvec
            x0[off + 3:off + 6] = tvec
            if refine_focal:
                x0[off + 6] = K_f64[i][0, 0]

        # Compute scene scale for bounds
        all_centers = np.array([c2w_list[i][:3, 3].astype(np.float64) for i in range(n_imgs)])
        scene_scale = max(np.linalg.norm(all_centers.max(0) - all_centers.min(0)), 1e-3)

        # Normalize weights: mean=1 so cost is interpretable regardless of scene
        all_weights = np.concatenate([pd['weights'] for pd in pair_data])
        w_mean = max(all_weights.mean(), 1e-8)
        for pd in pair_data:
            pd['weights'] = pd['weights'] / w_mean

        # Best-solution tracking for early stopping
        best_state = {'cost': np.inf, 'x': x0.copy(), 'eval': 0}
        n_eval = [0]

        def residuals(x):
            res_list = []

            for pi, (i, j) in enumerate(pairs):
                pd = pair_data[pi]
                pts_w = cam_pts[i][pd['pts_i_idx']]  # (M, 3) world space

                # Camera j's current w2c
                off_j = j * params_per_cam
                R_j = Rotation.from_rotvec(x[off_j:off_j + 3]).as_matrix()
                t_j = x[off_j + 3:off_j + 6]

                # Project i's points into j's camera
                p_cam_j = (R_j @ pts_w.T).T + t_j  # (M, 3)
                depth_i_in_j = p_cam_j[:, 2]

                # Camera j's own points at the overlap pixels
                pix_j = pd['pix_j']  # (M, 2) as (u, v)
                pts_j_w = pts3d_list[j][pix_j[:, 1], pix_j[:, 0]].astype(np.float64)
                p_j_cam = (R_j @ pts_j_w.T).T + t_j
                depth_j = p_j_cam[:, 2]

                # Depth ratio residual: log(depth_i / depth_j)
                # Using log makes it scale-invariant and symmetric
                valid = (depth_i_in_j > 0.01) & (depth_j > 0.01)
                ratio = np.where(valid, depth_i_in_j / depth_j, 1.0)
                log_ratio = np.where(valid, np.log(ratio), 0.0)

                res_list.append(log_ratio * pd['weights'])

            res = np.concatenate(res_list)
            cost = np.sum(res ** 2)

            n_eval[0] += 1

            # Track best solution
            if cost < best_state['cost']:
                best_state['cost'] = cost
                best_state['x'] = x.copy()
                best_state['eval'] = n_eval[0]

            if n_eval[0] % 20 == 0:
                print(f"    iter {n_eval[0]}: cost={cost:.4f} (best={best_state['cost']:.4f})")
                if progress_fn:
                    progress_fn(f"BA: iter {n_eval[0]}, cost={cost:.4f}")

            return res

        # Set bounds: tight bounds prevent divergence
        lb = np.full_like(x0, -np.inf)
        ub = np.full_like(x0, np.inf)
        max_t_shift = scene_scale * max_shift
        max_r_shift = np.radians(max_shift * 75)  # scale rotation proportionally
        for i in range(n_imgs):
            off = i * params_per_cam
            lb[off:off + 3] = x0[off:off + 3] - max_r_shift
            ub[off:off + 3] = x0[off:off + 3] + max_r_shift
            lb[off + 3:off + 6] = x0[off + 3:off + 6] - max_t_shift
            ub[off + 3:off + 6] = x0[off + 3:off + 6] + max_t_shift

        initial_cost = np.sum(residuals(x0) ** 2)
        print(f"  Initial cost: {initial_cost:.4f}")
        print(f"  Scene scale: {scene_scale:.4f}, max translation shift: {max_t_shift:.4f}")

        result = least_squares(residuals, x0, method='trf',
                               bounds=(lb, ub),
                               loss='huber', f_scale=huber_scale,
                               max_nfev=n_iters, verbose=0)

        # Use the best solution seen, not the final one (optimizer can overshoot)
        final_cost_raw = np.sum(result.fun ** 2)
        if best_state['cost'] < final_cost_raw:
            print(f"  Using best solution from iter {best_state['eval']} "
                  f"(cost={best_state['cost']:.4f} vs final={final_cost_raw:.4f})")
            result_x = best_state['x']
            final_cost = best_state['cost']
        else:
            result_x = result.x
            final_cost = final_cost_raw

        print(f"  Final cost: {final_cost:.4f} ({result.message})")

        if final_cost > initial_cost * 0.95:
            print("  BA did not improve enough — keeping original cameras")
            return None

        # Extract refined cameras, reject any that moved too far
        refined = []
        reject_threshold = max_t_shift * 0.75  # reject cameras that used >75% of the allowed shift
        for i in range(n_imgs):
            off = i * params_per_cam
            R = Rotation.from_rotvec(result_x[off:off + 3]).as_matrix()
            t = result_x[off + 3:off + 6]

            # Check how far this camera moved
            t_shift = np.linalg.norm(t - x0[off + 3:off + 6])
            r_shift = np.linalg.norm(result_x[off:off + 3] - x0[off:off + 3])
            if t_shift > reject_threshold:
                print(f"    Camera {i}: translation shift {t_shift:.4f} exceeds threshold "
                      f"{reject_threshold:.4f} — keeping original")
                # Use original pose
                R = Rotation.from_rotvec(x0[off:off + 3]).as_matrix()
                t = x0[off + 3:off + 6]
            elif t_shift > reject_threshold * 0.5:
                print(f"    Camera {i}: translation shift {t_shift:.4f} (large)")

            w2c = np.eye(4)
            w2c[:3, :3] = R
            w2c[:3, 3] = t
            c2w = np.linalg.inv(w2c).astype(np.float32)

            # Return full-resolution K (K_full), not model-resolution K_f64
            K_out = K_full[i].copy()
            if refine_focal:
                # Optimized focal is at model resolution; scale back to full res
                f_model = result_x[off + 6]
                f_scale = K_full[i][0, 0] / K_f64[i][0, 0] if K_f64[i][0, 0] > 0 else 1.0
                K_out[0, 0] = K_out[1, 1] = f_model * f_scale

            if image_paths is not None:
                from PIL import Image as PILImage
                img = PILImage.open(image_paths[i])
                W, H = img.size
            else:
                H, W = cam_HW[i]
            refined.append((c2w, K_out.astype(np.float32), W, H))

        print(f"  Depth-consistency BA done: {n_imgs} cameras refined, "
              f"cost {initial_cost:.4f} -> {final_cost:.4f}")
        return refined

    except Exception as e:
        print(f"  Depth-consistency BA failed: {e}")
        import traceback; traceback.print_exc()
        return None


def _write_ba_model(model_dir, image_paths, c2w_list, K_list,
                    pts3d_list, confs_list, imgs, min_conf=2.0,
                    max_points=50000, stride=4):
    """Write a COLMAP sparse model with proper 2D-3D tracks for bundle adjustment.

    Uses DUSt3R/MASt3R/VGGT per-pixel pointmaps directly: each pixel in image i
    is a 2D observation of its corresponding 3D point. Points visible in multiple
    images (via reprojection) get multi-view tracks.
    """
    from scipy.spatial.transform import Rotation
    from PIL import Image as PILImage

    os.makedirs(model_dir, exist_ok=True)
    n = min(len(image_paths), len(c2w_list), len(K_list))

    # Get image sizes
    img_sizes = []
    for p in image_paths[:n]:
        im = PILImage.open(p)
        img_sizes.append(im.size)  # (W, H)

    # Precompute w2c matrices
    w2c_list = []
    for i in range(n):
        c2w = c2w_list[i].astype(np.float64)
        w2c_list.append(np.linalg.inv(c2w))

    # Sample 3D points from pointmaps with their source pixel coordinates
    # Each point knows: (3D xyz, source_image_id, source_pixel_u, source_pixel_v)
    sampled_pts = []  # (x, y, z)
    sampled_obs = []  # list of lists: [(img_id, u, v), ...]
    for i in range(min(n, len(pts3d_list))):
        pts = pts3d_list[i]
        conf = confs_list[i] if i < len(confs_list) else None
        if pts is None or pts.ndim != 3:
            continue
        H_m, W_m = pts.shape[:2]
        if conf is not None:
            mask = conf > min_conf
        else:
            mask = np.ones((H_m, W_m), dtype=bool)

        # Scale factor from model resolution to full image resolution
        W_full, H_full = img_sizes[i]
        sx = W_full / W_m
        sy = H_full / H_m

        # Sample on a grid with stride, only confident points
        for v in range(0, H_m, stride):
            for u in range(0, W_m, stride):
                if not mask[v, u]:
                    continue
                pt3d = pts[v, u]
                if np.any(np.isnan(pt3d)) or np.any(np.isinf(pt3d)):
                    continue

                # Source observation in full-res pixel coords
                u_full = u * sx + sx / 2
                v_full = v * sy + sy / 2
                observations = [(i + 1, u_full, v_full)]

                # Project into other cameras for multi-view tracks
                for j in range(n):
                    if j == i:
                        continue
                    w2c = w2c_list[j]
                    K = K_list[j]
                    pt_cam = w2c[:3, :3] @ pt3d + w2c[:3, 3]
                    if pt_cam[2] < 0.01:
                        continue
                    pu = float(K[0, 0]) * pt_cam[0] / pt_cam[2] + float(K[0, 2])
                    pv = float(K[1, 1]) * pt_cam[1] / pt_cam[2] + float(K[1, 2])
                    Wj, Hj = img_sizes[j]
                    if 0 <= pu < Wj and 0 <= pv < Hj:
                        observations.append((j + 1, pu, pv))

                if len(observations) >= 2:
                    sampled_pts.append(pt3d)
                    sampled_obs.append(observations)

    # Subsample if too many
    if len(sampled_pts) > max_points:
        idx = np.random.choice(len(sampled_pts), max_points, replace=False)
        sampled_pts = [sampled_pts[i] for i in idx]
        sampled_obs = [sampled_obs[i] for i in idx]

    n_pts = len(sampled_pts)
    print(f"  BA model: {n_pts} points with multi-view tracks")

    # Build per-image 2D point lists
    # img_points[img_id] = [(u, v, point3d_id), ...]
    img_points = {i + 1: [] for i in range(n)}
    for pid, obs_list in enumerate(sampled_obs):
        point3d_id = pid + 1
        for (img_id, u, v) in obs_list:
            pt2d_idx = len(img_points[img_id])
            img_points[img_id].append((u, v, point3d_id))

    # Write cameras.txt
    with open(os.path.join(model_dir, 'cameras.txt'), 'w') as f:
        f.write("# Camera list\n# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i in range(n):
            W, H = img_sizes[i]
            K = K_list[i]
            f.write(f"{i+1} PINHOLE {W} {H} {K[0,0]:.6f} {K[1,1]:.6f} {K[0,2]:.6f} {K[1,2]:.6f}\n")

    # Write images.txt with proper POINTS2D lines
    with open(os.path.join(model_dir, 'images.txt'), 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i in range(n):
            w2c = w2c_list[i]
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            rot = Rotation.from_matrix(R)
            quat = rot.as_quat()  # (x, y, z, w)
            qw, qx, qy, qz = float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])
            name = os.path.basename(image_paths[i])
            f.write(f"{i+1} {qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f} "
                    f"{t[0]:.10f} {t[1]:.10f} {t[2]:.10f} {i+1} {name}\n")
            # POINTS2D line: X Y POINT3D_ID for each observation
            pts2d = img_points[i + 1]
            if pts2d:
                parts = " ".join(f"{u:.2f} {v:.2f} {pid}" for u, v, pid in pts2d)
                f.write(f"{parts}\n")
            else:
                f.write("\n")

    # Write points3D.txt with tracks
    with open(os.path.join(model_dir, 'points3D.txt'), 'w') as f:
        f.write("# 3D point list\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        # Build POINT2D_IDX lookup: for each image, the index of each point in that image's list
        img_pt2d_idx = {i + 1: {} for i in range(n)}  # img_id -> {point3d_id: idx}
        for img_id in range(1, n + 1):
            for idx, (u, v, pid) in enumerate(img_points[img_id]):
                img_pt2d_idx[img_id][pid] = idx

        for pid in range(1, n_pts + 1):
            pt = sampled_pts[pid - 1]
            obs = sampled_obs[pid - 1]
            track_parts = []
            for (img_id, u, v) in obs:
                pt2d_idx = img_pt2d_idx[img_id].get(pid)
                if pt2d_idx is not None:
                    track_parts.append(f"{img_id} {pt2d_idx}")
            track_str = " ".join(track_parts)
            f.write(f"{pid} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} 128 128 128 0.5 {track_str}\n")

    print(f"  Wrote BA model: {n} cameras, {n_pts} points with tracks")
    return n_pts


def bundle_adjust(image_paths, c2w_list, K_list,
                  pts3d_list, confs_list, imgs,
                  min_conf=2.0, refine_focal=False, refine_pp=False,
                  progress_fn=None):
    """Run COLMAP bundle adjustment on DUSt3R/MASt3R/VGGT scene data.

    Takes cameras and per-pixel pointmaps directly — no feature extraction or
    matching needed. Builds 2D-3D tracks from the pointmaps and runs BA.
    Returns list of refined (c2w, K, W, H) per image, or None on failure.
    """
    import tempfile, shutil, subprocess

    colmap_exe = _find_colmap_exe()
    if not colmap_exe:
        raise RuntimeError(
            "COLMAP executable not found. Download from:\n"
            "  https://github.com/colmap/colmap/releases\n"
            "  Get: colmap-x64-windows-cuda.zip\n"
            "  Extract to project folder as colmap/")

    colmap_env = os.environ.copy()
    colmap_env['GLOG_logtostderr'] = '1'

    n_imgs = len(image_paths)
    workdir = tempfile.mkdtemp(prefix='colmap_ba_')
    print(f"  Bundle adjust workdir: {workdir}")

    try:
        # Step 1: Write COLMAP model with proper tracks from pointmaps
        if progress_fn:
            progress_fn("Bundle adjust: building tracks...")
        model_dir = os.path.join(workdir, "sparse", "0")
        n_pts = _write_ba_model(model_dir, image_paths, c2w_list, K_list,
                                pts3d_list, confs_list, imgs,
                                min_conf=min_conf)
        if n_pts == 0:
            raise RuntimeError("No multi-view points found for bundle adjustment")

        # Step 2: Run bundle adjustment directly on the model
        if progress_fn:
            progress_fn("Bundle adjust: optimizing...")
        print("  Running bundle adjustment...")
        ba_output = os.path.join(workdir, "sparse", "ba")
        os.makedirs(ba_output, exist_ok=True)
        r = subprocess.run([colmap_exe, 'bundle_adjuster',
                        '--input_path', model_dir,
                        '--output_path', ba_output,
                        '--BundleAdjustment.refine_focal_length',
                        '1' if refine_focal else '0',
                        '--BundleAdjustment.refine_principal_point',
                        '1' if refine_pp else '0',
                        '--BundleAdjustment.refine_extra_params', '0'],
                       capture_output=True, text=True, timeout=600,
                       env=colmap_env)
        if r.stdout:
            for line in r.stdout.split('\n'):
                if line.strip():
                    print(f"    {line.strip()}")
        if r.stderr:
            for line in r.stderr.split('\n'):
                if any(k in line.lower() for k in ['cost', 'residual', 'iteration', 'points', 'images']):
                    print(f"    {line.strip()}")

        # Step 3: Convert binary output to text for reading
        subprocess.run([colmap_exe, 'model_converter',
                        '--input_path', ba_output,
                        '--output_path', ba_output,
                        '--output_type', 'TXT'],
                       capture_output=True, timeout=60, env=colmap_env)

        # Step 4: Read back refined cameras
        cameras_path = os.path.join(ba_output, 'cameras.txt')
        images_path = os.path.join(ba_output, 'images.txt')
        if not os.path.exists(cameras_path) or not os.path.exists(images_path):
            # Fall back to input dir (BA may not have produced output)
            cameras_path = os.path.join(model_dir, 'cameras.txt')
            images_path = os.path.join(model_dir, 'images.txt')
            if not os.path.exists(cameras_path):
                raise RuntimeError("Bundle adjustment produced no output model")

        from train import parse_colmap_cameras, parse_colmap_images
        cameras = parse_colmap_cameras(cameras_path)
        col_images = parse_colmap_images(images_path)

        # Match refined cameras back to input images by filename
        name_to_idx = {}
        for i, p in enumerate(image_paths):
            name_to_idx[os.path.basename(p)] = i

        refined = [None] * n_imgs
        for ci in col_images:
            idx = name_to_idx.get(ci['name'])
            if idx is not None:
                cam = cameras.get(ci['cam_id'])
                if cam:
                    W, H, fx, fy, cx, cy = cam
                    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                                 dtype=np.float64)
                    refined[idx] = (ci['c2w'].astype(np.float32), K.astype(np.float32), W, H)

        # Fill any missing with originals
        from PIL import Image as PILImage
        for i in range(n_imgs):
            if refined[i] is None:
                img = PILImage.open(image_paths[i])
                W, H = img.size
                refined[i] = (c2w_list[i], K_list[i], W, H)

        n_refined = sum(1 for r in refined if r is not None)
        print(f"  Bundle adjustment done: {n_refined}/{n_imgs} cameras refined")
        return refined

    except Exception as e:
        print(f"  Bundle adjustment failed: {e}")
        import traceback; traceback.print_exc()
        return None
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def save_dense_ply(path, pts, colors):
    n = len(pts)
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        for i in range(n):
            f.write(struct.pack('<3f', *pts[i].astype(np.float32)))
            f.write(struct.pack('<3B', *colors[i].astype(np.uint8)))
