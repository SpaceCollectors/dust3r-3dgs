"""
Dense mesh generation from depth maps + camera poses.

Triangulates each camera's depth map directly in pixel-grid space,
preserving every original 3D point as a mesh vertex. Long edges
(depth discontinuities) are rejected to avoid connecting foreground
to background.
"""

import numpy as np
import struct


def create_dense_mesh(imgs, pts3d_list, confs_list, cam2world_list=None,
                      intrinsics_list=None, min_conf=2.0,
                      poisson_depth=8, normal_radius=0.03, trim_percentile=5,
                      mode='reprojected', hole_cap_size=50):
    """Create mesh from multi-view point clouds.

    mode='reprojected': Voxel-merge all points, shared vertex grid triangulation.
    mode='ballpivot':   Voxel-dedup, ball-pivot, hole capping.
    hole_cap_size:      Max boundary edges to cap a hole (0=disable, higher=close bigger holes).
    """
    if mode == 'reprojected' and cam2world_list is not None:
        v, f, c = _mesh_reprojected_grid(imgs, pts3d_list, confs_list,
                                          cam2world_list, min_conf)
    else:
        v, f, c = _mesh_ball_pivot(imgs, pts3d_list, confs_list,
                                    cam2world_list, min_conf)

    # Cap holes on either method
    if hole_cap_size > 0 and len(f) > 0:
        v, f, c = _cap_holes(v, f, c, max_hole_edges=hole_cap_size)

    return v, f, c


def _collect_points(imgs, pts3d_list, confs_list, min_conf):
    """Gather all confidence-filtered points + colors from all views."""
    all_pts, all_colors = [], []
    for i in range(len(imgs)):
        pts = pts3d_list[i]
        conf = confs_list[i]
        img = imgs[i]
        if pts.ndim == 2:
            mask = conf.ravel() > min_conf if conf is not None else np.ones(len(pts), dtype=bool)
            all_pts.append(pts[mask])
            all_colors.append((np.clip(img.reshape(-1, 3)[mask], 0, 1) * 255).astype(np.uint8))
        else:
            H, W = pts.shape[:2]
            mask = conf.reshape(H, W) > min_conf if conf is not None else np.ones((H, W), dtype=bool)
            all_pts.append(pts[mask])
            all_colors.append((np.clip(img[mask], 0, 1) * 255).astype(np.uint8))
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


def _mesh_ball_pivot(imgs, pts3d_list, confs_list, cam2world_list, min_conf):
    """Voxel-dedup all points, then ball-pivot into a single mesh."""
    import open3d as o3d

    all_points, all_colors = _collect_points(imgs, pts3d_list, confs_list, min_conf)
    if len(all_points) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32), np.zeros((0, 3), dtype=np.uint8)
    print(f"  Combined: {len(all_points):,d} points")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(all_colors.astype(np.float64) / 255.0)

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    extent = np.linalg.norm(np.asarray(pcd.points).max(0) - np.asarray(pcd.points).min(0))
    voxel_size = extent / 800
    pcd = pcd.voxel_down_sample(voxel_size)
    print(f"  After dedup: {len(pcd.points):,d} (voxel={voxel_size:.6f})")

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel_size * 4, max_nn=30))
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


def _cap_holes(verts, faces, colors, max_hole_edges=60):
    """Find boundary loops (holes) and fill with fan triangulation from centroid."""
    V, F = len(verts), len(faces)
    if F == 0:
        return verts, faces, colors

    # Build all half-edges and find boundary edges (appear only once)
    # A half-edge (a,b) from face winding. Boundary = half-edge with no twin (b,a).
    half_edges = set()
    for fi in range(F):
        a, b, c = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
        half_edges.add((a, b))
        half_edges.add((b, c))
        half_edges.add((c, a))

    # Boundary half-edges: (a,b) exists but (b,a) does not
    # The hole runs in the REVERSE direction of the face winding
    # So the hole boundary goes b -> a for each boundary half-edge (a,b)
    boundary_next = {}  # maps vertex -> next vertex along hole boundary
    for (a, b) in half_edges:
        if (b, a) not in half_edges:
            # (a,b) is boundary. Hole direction is b->a.
            # But we need: for each vertex on the hole, which is the NEXT vertex?
            # The hole walks: ... -> b -> a -> ...
            # So boundary_next[b] = a
            if b not in boundary_next:
                boundary_next[b] = a

    n_boundary = len(boundary_next)
    if n_boundary == 0:
        print(f"  No boundary edges — mesh is watertight")
        return verts, faces, colors

    print(f"  {n_boundary} boundary edges found")

    # Trace closed loops
    visited = set()
    loops = []
    for start in boundary_next:
        if start in visited:
            continue
        loop = []
        cur = start
        for _ in range(100000):  # safety limit
            if cur in visited or cur not in boundary_next:
                break
            visited.add(cur)
            loop.append(cur)
            cur = boundary_next[cur]
        if len(loop) >= 3 and cur == start:
            loops.append(loop)

    print(f"  Found {len(loops)} boundary loops")

    # Fill holes
    new_verts = list(verts)
    new_colors = list(colors)
    new_faces = list(faces)
    n_capped = 0

    for loop in loops:
        if len(loop) > max_hole_edges:
            continue

        lv = verts[loop]

        # Centroid vertex
        centroid = lv.mean(axis=0).astype(np.float32)
        centroid_col = colors[loop].mean(axis=0).clip(0, 255).astype(np.uint8)
        ci = len(new_verts)
        new_verts.append(centroid)
        new_colors.append(centroid_col)

        for j in range(len(loop)):
            new_faces.append(np.array([ci, loop[j], loop[(j + 1) % len(loop)]], dtype=np.int32))
        n_capped += 1

    if n_capped > 0:
        print(f"  Capped {n_capped} holes (skipped {len(loops) - n_capped} too large)")
        return (np.array(new_verts, dtype=np.float32),
                np.array(new_faces, dtype=np.int32),
                np.array(new_colors, dtype=np.uint8))

    print(f"  {len(loops)} loops found, none within {max_hole_edges} edge limit")
    return verts, faces, colors


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


def save_dense_ply(path, pts, colors):
    n = len(pts)
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        for i in range(n):
            f.write(struct.pack('<3f', *pts[i].astype(np.float32)))
            f.write(struct.pack('<3B', *colors[i].astype(np.uint8)))
