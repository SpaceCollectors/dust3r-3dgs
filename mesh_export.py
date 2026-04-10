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
                      mode='reprojected', hole_cap_size=50, dense_colors=None):
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

        # Compute and print median point spacing
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
        v, f, c = _mesh_local_delaunay(pts, cols, cam_center=cam_center)
    elif mode == 'reprojected' and cam2world_list is not None:
        v, f, c = _mesh_reprojected_grid(imgs, pts3d_list, confs_list,
                                          cam2world_list, min_conf)
    else:
        v, f, c = _mesh_ball_pivot(imgs, pts3d_list, confs_list,
                                    cam2world_list, min_conf)

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
            if img is not None and len(img.reshape(-1, 3)) == len(flat):
                all_colors.append((np.clip(img.reshape(-1, 3)[mask], 0, 1) * 255).astype(np.uint8))
            elif dense_colors is not None and i >= len(imgs) and len(dense_colors) == len(flat):
                all_colors.append(dense_colors[mask])
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


def _mesh_local_delaunay(points, colors, cam_center=None, k=20):
    """Surface reconstruction via local 2D Delaunay on tangent planes.

    For each point, project its K nearest neighbors onto the local tangent plane
    (defined by the point's estimated normal), run 2D Delaunay, keep triangles
    that touch the center point. Produces a clean surface following local geometry.
    """
    import open3d as o3d
    from scipy.spatial import Delaunay, cKDTree

    V = len(points)
    if V < 3:
        return points, np.zeros((0, 3), dtype=np.int32), colors

    print(f"  Local Delaunay on {V:,d} points (k={k})...")

    # Estimate normals
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0, max_nn=k))
    if cam_center is not None:
        pcd.orient_normals_towards_camera_location(cam_center)
    else:
        pcd.orient_normals_consistent_tangent_plane(k=min(k, 15))
    normals = np.asarray(pcd.normals, dtype=np.float64)

    # KD-tree for neighbor queries
    tree = cKDTree(points)
    _, nn_idx = tree.query(points, k=k)  # (V, k) indices

    # Compute median NN distance for edge length filtering
    nn_dists = np.linalg.norm(points[nn_idx[:, 1]] - points, axis=-1)
    median_dist = np.median(nn_dists)
    max_edge = median_dist * 5.0

    # For each point: project neighbors to tangent plane, local Delaunay
    all_tris = set()
    batch = max(1, V // 10)

    for pi in range(V):
        if pi % batch == 0:
            print(f"    {pi:,d} / {V:,d} ({pi * 100 // V}%)")

        n = normals[pi]
        center = points[pi]
        neighbors = nn_idx[pi]  # k indices including self

        # Build local tangent basis
        # Pick any vector not parallel to n for cross product
        if abs(n[0]) < 0.9:
            tangent = np.cross(n, [1, 0, 0])
        else:
            tangent = np.cross(n, [0, 1, 0])
        tangent /= max(np.linalg.norm(tangent), 1e-12)
        bitangent = np.cross(n, tangent)

        # Project neighbors to 2D tangent plane
        rel = points[neighbors] - center
        u = rel @ tangent
        v = rel @ bitangent
        pts_2d = np.column_stack([u, v])

        # 2D Delaunay
        if len(pts_2d) < 3:
            continue
        try:
            tri = Delaunay(pts_2d)
        except Exception:
            continue

        # Keep triangles that include point 0 (the center point = self)
        for simplex in tri.simplices:
            if 0 not in simplex:
                continue
            # Map local indices back to global
            a, b, c = int(neighbors[simplex[0]]), int(neighbors[simplex[1]]), int(neighbors[simplex[2]])

            # Edge length check
            d0 = np.linalg.norm(points[a] - points[b])
            d1 = np.linalg.norm(points[b] - points[c])
            d2 = np.linalg.norm(points[a] - points[c])
            if d0 > max_edge or d1 > max_edge or d2 > max_edge:
                continue

            # Canonical face key for dedup
            key = tuple(sorted([a, b, c]))
            all_tris.add(key)

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


def _mesh_ball_pivot(imgs, pts3d_list, confs_list, cam2world_list, min_conf):
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

    # Only voxel downsample to remove near-exact duplicates (0.5x median spacing)
    voxel_size = median_spacing * 0.5
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

    # Ball-pivoting with wide radii range
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


def densify_sgm(image_paths, c2w_list, K_list, existing_pts=None,
                 num_disparities=128, block_size=5, max_pairs=20, progress_fn=None):
    """Dense multi-view stereo via OpenCV StereoSGBM.

    For each image pair with sufficient baseline:
    1. Compute relative pose from known cameras
    2. Stereo-rectify both images
    3. Run StereoSGBM for dense disparity
    4. Reproject to 3D in world frame
    5. Filter and merge all pairs

    Returns: (points, colors) numpy arrays
    """
    import cv2
    from PIL import Image as PILImage

    n_imgs = min(len(image_paths), len(c2w_list), len(K_list))
    all_pts, all_cols = [], []

    # Build pairs: select pairs with good baseline
    pairs = []
    for i in range(n_imgs):
        for j in range(i + 1, n_imgs):
            # Baseline = distance between camera centers
            ci = c2w_list[i][:3, 3]
            cj = c2w_list[j][:3, 3]
            baseline = np.linalg.norm(ci - cj)
            pairs.append((i, j, baseline))

    # Sort by baseline, take pairs with moderate baseline (not too small, not too large)
    pairs.sort(key=lambda x: x[2])
    if len(pairs) > 3:
        # Skip very small baselines (bottom 20%) and very large (top 20%)
        n = len(pairs)
        pairs = pairs[n // 5 : n * 4 // 5]
    pairs = pairs[:max_pairs]

    print(f"  SGM densification: {len(pairs)} pairs from {n_imgs} images")

    for pi, (i, j, baseline) in enumerate(pairs):
        if progress_fn:
            progress_fn(f"SGM pair {pi+1}/{len(pairs)} (imgs {i+1}-{j+1})")
        print(f"    Pair {i+1}-{j+1} (baseline={baseline:.4f})...")

        try:
            # Load full-res images
            img_i = np.array(PILImage.open(image_paths[i]).convert('RGB'))
            img_j = np.array(PILImage.open(image_paths[j]).convert('RGB'))

            # Resize to manageable size if too large (SGM is CPU-heavy)
            max_dim = 2000
            h_i, w_i = img_i.shape[:2]
            h_j, w_j = img_j.shape[:2]
            scale_i = min(max_dim / max(h_i, w_i), 1.0)
            scale_j = min(max_dim / max(h_j, w_j), 1.0)

            if scale_i < 1.0:
                img_i = cv2.resize(img_i, None, fx=scale_i, fy=scale_i)
                K_i = K_list[i].copy(); K_i[:2] *= scale_i
            else:
                K_i = K_list[i].copy()

            if scale_j < 1.0:
                img_j = cv2.resize(img_j, None, fx=scale_j, fy=scale_j)
                K_j = K_list[j].copy(); K_j[:2] *= scale_j
            else:
                K_j = K_list[j].copy()

            h_i, w_i = img_i.shape[:2]
            h_j, w_j = img_j.shape[:2]

            # Relative pose: R, t from camera i to camera j
            w2c_i = np.linalg.inv(c2w_list[i])
            w2c_j = np.linalg.inv(c2w_list[j])
            R_rel = (w2c_j[:3, :3] @ np.linalg.inv(w2c_i[:3, :3])).astype(np.float64)
            t_rel = (w2c_j[:3, 3] - R_rel @ w2c_i[:3, 3]).astype(np.float64)

            # Use the same image size for both (take the smaller)
            h = min(h_i, h_j)
            w = min(w_i, w_j)
            img_i = cv2.resize(img_i, (w, h))
            img_j = cv2.resize(img_j, (w, h))
            K_i[0, 2] = w / 2; K_i[1, 2] = h / 2
            K_j[0, 2] = w / 2; K_j[1, 2] = h / 2

            dist = np.zeros(5)

            # Stereo rectification
            R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
                K_i.astype(np.float64), dist,
                K_j.astype(np.float64), dist,
                (w, h), R_rel, t_rel,
                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

            map1x, map1y = cv2.initUndistortRectifyMap(K_i.astype(np.float64), dist, R1, P1, (w, h), cv2.CV_32FC1)
            map2x, map2y = cv2.initUndistortRectifyMap(K_j.astype(np.float64), dist, R2, P2, (w, h), cv2.CV_32FC1)

            rect_i = cv2.remap(img_i, map1x, map1y, cv2.INTER_LINEAR)
            rect_j = cv2.remap(img_j, map2x, map2y, cv2.INTER_LINEAR)

            # Convert to grayscale for SGM
            gray_i = cv2.cvtColor(rect_i, cv2.COLOR_RGB2GRAY)
            gray_j = cv2.cvtColor(rect_j, cv2.COLOR_RGB2GRAY)

            # StereoSGBM
            sgm = cv2.StereoSGBM.create(
                minDisparity=0,
                numDisparities=num_disparities,
                blockSize=block_size,
                P1=8 * block_size ** 2,
                P2=32 * block_size ** 2,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

            disparity = sgm.compute(gray_i, gray_j).astype(np.float32) / 16.0

            # Filter invalid disparities
            valid = disparity > 0

            # Reproject to 3D (in rectified camera i frame)
            pts_3d = cv2.reprojectImageTo3D(disparity, Q)

            # Filter by valid disparity and reasonable depth
            pts = pts_3d[valid]
            cols = rect_i[valid]

            if len(pts) == 0:
                continue

            # Filter outliers by depth (remove very far points)
            z = pts[:, 2]
            z_med = np.median(z[z > 0]) if (z > 0).any() else 1.0
            depth_ok = (z > 0) & (z < z_med * 10)
            pts = pts[depth_ok]
            cols = cols[depth_ok]

            # Transform from rectified camera i frame to world frame
            # Rectified frame: R1 @ K_i_inv @ pixel → rectified camera
            # World: c2w_i @ inv(R1) @ pts_rect
            R1_inv = np.linalg.inv(R1[:3, :3]) if R1.shape[0] >= 3 else np.eye(3)
            pts_cam = (R1_inv @ pts.T).T  # back to original camera i frame
            pts_world = (c2w_list[i][:3, :3] @ pts_cam.T).T + c2w_list[i][:3, 3]

            all_pts.append(pts_world.astype(np.float32))
            all_cols.append(cols.astype(np.uint8))

            print(f"      {len(pts_world):,d} points")

        except Exception as e:
            print(f"      Failed: {e}")
            continue

    if not all_pts:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    points = np.concatenate(all_pts, axis=0)
    colors = np.concatenate(all_cols, axis=0)
    print(f"  SGM total: {len(points):,d} dense points from {len(pairs)} pairs")
    return points, colors


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


def _write_colmap_sparse_model(sparse_dir, image_paths, c2w_list, K_list):
    """Write a COLMAP sparse model in text format from known cameras.
    Also creates dummy 3D points from triangulation of camera centers
    so that COLMAP's tools recognize images as connected."""
    from scipy.spatial.transform import Rotation
    from PIL import Image as PILImage

    os.makedirs(sparse_dir, exist_ok=True)
    n = min(len(image_paths), len(c2w_list), len(K_list))

    # cameras.txt
    with open(os.path.join(sparse_dir, 'cameras.txt'), 'w') as f:
        f.write("# Camera list\n# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i in range(n):
            img = PILImage.open(image_paths[i])
            W, H = img.size
            K = K_list[i]
            f.write(f"{i+1} PINHOLE {W} {H} {K[0,0]:.6f} {K[1,1]:.6f} {K[0,2]:.6f} {K[1,2]:.6f}\n")

    # images.txt
    with open(os.path.join(sparse_dir, 'images.txt'), 'w') as f:
        f.write("# Image list\n# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        for i in range(n):
            c2w = c2w_list[i].astype(np.float64)
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            rot = Rotation.from_matrix(R)
            quat = rot.as_quat()  # (x, y, z, w)
            qw, qx, qy, qz = float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])
            name = os.path.basename(image_paths[i])
            f.write(f"{i+1} {qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f} {t[0]:.10f} {t[1]:.10f} {t[2]:.10f} {i+1} {name}\n")
            f.write("\n")

    # points3D.txt — create some dummy points at camera centers so images appear connected
    with open(os.path.join(sparse_dir, 'points3D.txt'), 'w') as f:
        f.write("# 3D point list\n")
        pid = 1
        for i in range(n):
            center = c2w_list[i][:3, 3]
            # Each point "seen" by this image and one neighbor
            j = (i + 1) % n
            f.write(f"{pid} {center[0]:.6f} {center[1]:.6f} {center[2]:.6f} 128 128 128 0.5 "
                    f"{i+1} 1 {j+1} 1\n")
            pid += 1

    print(f"  Wrote COLMAP sparse model: {n} cameras, {n} points")
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
                          capture_output=True, timeout=600)

            # Step 2: Feature matching
            if progress_fn: progress_fn("COLMAP: matching features...")
            print("  Feature matching...")
            subprocess.run([colmap_exe, 'exhaustive_matcher',
                           '--database_path', db_path],
                          capture_output=True, timeout=600)

            # Step 3: Sparse reconstruction
            # If we have prior cameras (from DUSt3R etc.), use them as initialization
            # Run point_triangulator + bundle_adjuster instead of full mapper
            if c2w_list is not None and K_list is not None and len(c2w_list) == n_imgs:
                if progress_fn: progress_fn("COLMAP: triangulating with prior cameras...")
                print("  Using prior cameras for initialization...")
                prior_dir = os.path.join(sparse_dir, '0')
                _write_colmap_sparse_model(prior_dir, image_paths, c2w_list, K_list)

                # Triangulate points using prior cameras + matched features
                print("  Point triangulation...")
                subprocess.run([colmap_exe, 'point_triangulator',
                               '--database_path', db_path,
                               '--image_path', img_dir,
                               '--input_path', prior_dir,
                               '--output_path', prior_dir],
                              capture_output=True, timeout=600)

                # Refine cameras via bundle adjustment
                if progress_fn: progress_fn("COLMAP: bundle adjustment (refining cameras)...")
                print("  Bundle adjustment...")
                refined_dir = os.path.join(sparse_dir, 'refined')
                os.makedirs(refined_dir, exist_ok=True)
                subprocess.run([colmap_exe, 'bundle_adjuster',
                               '--input_path', prior_dir,
                               '--output_path', refined_dir],
                              capture_output=True, timeout=600)

                # Use refined if it exists, else fall back to prior
                if os.path.exists(os.path.join(refined_dir, 'cameras.bin')) or \
                   os.path.exists(os.path.join(refined_dir, 'cameras.txt')):
                    sparse_sub = refined_dir
                    print("  Using bundle-adjusted cameras")
                else:
                    sparse_sub = prior_dir
                    print("  Bundle adjustment failed, using prior cameras")
            else:
                # Full COLMAP mapper (no prior cameras)
                if progress_fn: progress_fn("COLMAP: sparse reconstruction...")
                print("  Sparse reconstruction (full mapper)...")
                subprocess.run([colmap_exe, 'mapper',
                               '--database_path', db_path,
                               '--image_path', img_dir,
                               '--output_path', sparse_dir],
                              capture_output=True, timeout=600)

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
                      capture_output=True, timeout=600)
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
                               text=True, bufsize=1)
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
                          capture_output=True, text=True, timeout=600)
        if r.stderr:
            for line in r.stderr.split('\n'):
                if any(k in line for k in ['Fusing', 'points', 'Number']):
                    print(f"  {line.strip()}")
        print(f"  Fused to {output_ply}")

        # Step 5: Load the fused point cloud
        if not os.path.exists(output_ply):
            print("  ERROR: fused.ply not found")
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

        import open3d as o3d
        pcd = o3d.io.read_point_cloud(output_ply)
        pts = np.asarray(pcd.points, dtype=np.float32)
        if pcd.has_colors():
            cols = (np.asarray(pcd.colors) * 255).clip(0, 255).astype(np.uint8)
        else:
            cols = np.full((len(pts), 3), 180, dtype=np.uint8)

        print(f"  Dense cloud: {len(pts):,d} points")
        return pts, cols

    except Exception as e:
        print(f"  COLMAP dense failed: {e}")
        import traceback; traceback.print_exc()
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    finally:
        if not reuse:
            shutil.rmtree(workdir, ignore_errors=True)


def save_dense_ply(path, pts, colors):
    n = len(pts)
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        for i in range(n):
            f.write(struct.pack('<3f', *pts[i].astype(np.float32)))
            f.write(struct.pack('<3B', *colors[i].astype(np.uint8)))
