"""
Image-first UV texture baking.

For each camera image, rasterize the mesh to get a face-ID per pixel.
Each pixel that hits a face → barycentrics → UV coord → write color to texture.
Per-pixel weights: incidence angle + border falloff.
Composite per-view UV maps into final texture.
"""

import os
import numpy as np
from PIL import Image, ImageDraw


# ── Public API ──────────────────────────────────────────────────────────────

def create_uvs(verts, faces, views=None):
    """Create UV unwrap and return (uvs, uv_faces, debug_texture).
    If views are provided, uses camera directions for projection.
    debug_texture is a checkerboard for visual verification."""
    uvs, uv_faces = _unwrap_uvs(verts, faces, views)
    debug = _make_debug_texture(1024)
    return uvs, uv_faces, debug


def bake_texture(verts, faces, uvs, uv_faces, views, texture_size=4096):
    """Bake texture from camera images using image-first projection.

    For each view: rasterize mesh → per-pixel face ID + barycentrics →
    interpolate UV → write source pixel color into UV map.
    Composite all views with incidence-angle weighting.
    """
    F = len(faces)
    n_cams = len(views)
    print(f"  Baking {F:,d} faces, {n_cams} cameras, {texture_size}px texture")

    # Precompute face normals
    fv0 = verts[faces[:, 0]]; fv1 = verts[faces[:, 1]]; fv2 = verts[faces[:, 2]]
    face_normals = np.cross(fv1 - fv0, fv2 - fv0)
    face_normals /= (np.linalg.norm(face_normals, axis=-1, keepdims=True) + 1e-8)

    # Precompute UV coords per face vertex
    face_uv0 = uvs[uv_faces[:, 0]]  # (F, 2)
    face_uv1 = uvs[uv_faces[:, 1]]
    face_uv2 = uvs[uv_faces[:, 2]]

    # Accumulate weighted colors across views
    color_accum = np.zeros((texture_size, texture_size, 3), dtype=np.float64)
    weight_accum = np.zeros((texture_size, texture_size), dtype=np.float64)

    for ci, view in enumerate(views):
        print(f"    Camera {ci+1}/{n_cams}...", end=' ')
        view_color, view_weight = _project_view(
            verts, faces, face_normals, face_uv0, face_uv1, face_uv2,
            view, texture_size)

        mask = view_weight > 0
        color_accum[mask] += view_color[mask] * view_weight[mask, None]
        weight_accum[mask] += view_weight[mask]
        n_px = mask.sum()
        print(f"{n_px:,d} texels written")

    # Normalize
    has_color = weight_accum > 0
    color_accum[has_color] /= weight_accum[has_color, None]
    texture = (color_accum * 255).clip(0, 255).astype(np.uint8)

    n_filled = has_color.sum()
    n_total = (texture_size * texture_size)
    print(f"  {n_filled:,d} texels filled before dilation")

    texture = _dilate_texture(texture, iterations=32)
    return texture


def create_textured_mesh(verts, faces, colors, views, output_dir,
                         texture_size=4096, return_data=False):
    """Full pipeline: UV unwrap → bake → save OBJ."""
    os.makedirs(output_dir, exist_ok=True)

    V, F = len(verts), len(faces)
    print(f"  Creating texture for {V:,d} verts, {F:,d} faces, {texture_size}px")

    uvs, uv_faces = _unwrap_uvs(verts, faces)
    texture = bake_texture(verts, faces, uvs, uv_faces, views, texture_size)

    tex_path = os.path.join(output_dir, 'texture.png')
    Image.fromarray(texture).save(tex_path, quality=95)

    mtl_path = os.path.join(output_dir, 'mesh.mtl')
    with open(mtl_path, 'w') as f:
        f.write("newmtl material0\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\n"
                "d 1\nillum 1\nmap_Kd texture.png\n")

    uvs_obj = uvs.copy()
    uvs_obj[:, 1] = 1.0 - uvs_obj[:, 1]

    obj_path = os.path.join(output_dir, 'mesh.obj')
    _write_obj(obj_path, verts, faces, uvs_obj, uv_faces)
    print(f"  Saved: {obj_path}")

    if return_data:
        return obj_path, uvs, uv_faces, texture
    return obj_path


# ── Core: per-view image→UV projection ─────────────────────────────────────

def _project_view(verts, faces, face_normals, face_uv0, face_uv1, face_uv2,
                  view, tex_size):
    """For one camera view, rasterize mesh and write image pixels into UV space.

    Returns (color_map, weight_map) both (tex_size, tex_size, ...).
    """
    R = view['w2c'][:3, :3]
    t = view['w2c'][:3, 3]
    K = view['K']
    W_img, H_img = int(view['W']), int(view['H'])
    pixels = view['pixels']  # (H, W, 3) float [0,1]
    c2w = np.linalg.inv(view['w2c'])
    cam_center = c2w[:3, 3]

    # Use actual image dimensions (may differ from camera model)
    actual_H, actual_W = pixels.shape[:2]
    if actual_W != W_img or actual_H != H_img:
        sx, sy = actual_W / W_img, actual_H / H_img
        K = K.copy()
        K[0, 0] *= sx; K[0, 2] *= sx
        K[1, 1] *= sy; K[1, 2] *= sy
        W_img, H_img = actual_W, actual_H

    # Project all vertices to image space
    cam_pts = (R @ verts.T).T + t
    z_vert = cam_pts[:, 2]
    safe_z = np.maximum(z_vert, 0.01)
    u_vert = cam_pts[:, 0] / safe_z * K[0, 0] + K[0, 2]
    v_vert = cam_pts[:, 1] / safe_z * K[1, 1] + K[1, 2]

    # Rasterize: painter's algorithm (back-to-front), face ID per pixel
    fu = u_vert[faces]  # (F, 3)
    fv = v_vert[faces]
    fz = z_vert[faces]

    in_front = fz.min(axis=1) > 0.01
    in_frame = ((fu.max(axis=1) >= 0) & (fu.min(axis=1) < W_img) &
                (fv.max(axis=1) >= 0) & (fv.min(axis=1) < H_img))
    candidates = np.where(in_front & in_frame)[0]

    if len(candidates) == 0:
        return (np.zeros((tex_size, tex_size, 3), dtype=np.float64),
                np.zeros((tex_size, tex_size), dtype=np.float64))

    # Sort back-to-front so closest face overwrites
    centroid_z = fz[candidates].mean(axis=1)
    order = np.argsort(-centroid_z)
    sorted_fc = candidates[order]

    face_id_img = Image.new('I', (W_img, H_img), 0)
    draw = ImageDraw.Draw(face_id_img)
    for fi in sorted_fc:
        px = [(int(round(fu[fi, j])), int(round(fv[fi, j]))) for j in range(3)]
        draw.polygon(px, fill=int(fi) + 1)  # 1-indexed

    face_id_map = np.array(face_id_img, dtype=np.int32) - 1  # (H, W), -1=no hit

    # Get all pixels that hit a face
    rows, cols = np.where(face_id_map >= 0)
    if len(rows) == 0:
        return (np.zeros((tex_size, tex_size, 3), dtype=np.float64),
                np.zeros((tex_size, tex_size), dtype=np.float64))

    fids = face_id_map[rows, cols]

    # Compute barycentrics from projected 2D triangle vertices
    pixel_pts = np.stack([cols.astype(np.float64) + 0.5,
                          rows.astype(np.float64) + 0.5], axis=-1)
    tri_a = np.stack([u_vert[faces[fids, 0]], v_vert[faces[fids, 0]]], axis=-1)
    tri_b = np.stack([u_vert[faces[fids, 1]], v_vert[faces[fids, 1]]], axis=-1)
    tri_c = np.stack([u_vert[faces[fids, 2]], v_vert[faces[fids, 2]]], axis=-1)
    bary = _barycentric_batch(pixel_pts, tri_a, tri_b, tri_c)
    bary = np.clip(bary, 0.0, None)
    bary /= (bary.sum(axis=-1, keepdims=True) + 1e-10)

    # Interpolate UV coordinates using barycentrics
    hit_uv = (bary[:, 0:1] * face_uv0[fids] +
              bary[:, 1:2] * face_uv1[fids] +
              bary[:, 2:3] * face_uv2[fids])

    # Convert UV to texel coordinates
    tex_u = np.clip((hit_uv[:, 0] * tex_size).astype(int), 0, tex_size - 1)
    tex_v = np.clip((hit_uv[:, 1] * tex_size).astype(int), 0, tex_size - 1)

    # Compute per-pixel weight: incidence angle
    # Recover 3D position via barycentrics for normal dot product
    fv0 = verts[faces[fids, 0]]
    fv1 = verts[faces[fids, 1]]
    fv2 = verts[faces[fids, 2]]
    pos_3d = bary[:, 0:1] * fv0 + bary[:, 1:2] * fv1 + bary[:, 2:3] * fv2

    view_dir = cam_center[None, :] - pos_3d
    view_dir /= (np.linalg.norm(view_dir, axis=-1, keepdims=True) + 1e-8)
    ndot = (face_normals[fids] * view_dir).sum(axis=-1)
    ndot = np.maximum(ndot, 0.0)
    weight = ndot ** 2  # strongly favor head-on views

    # Border falloff: reduce weight near image edges
    margin_u = np.minimum(cols.astype(float), W_img - 1.0 - cols) / (W_img * 0.2)
    margin_v = np.minimum(rows.astype(float), H_img - 1.0 - rows) / (H_img * 0.2)
    border = np.clip(np.minimum(margin_u, margin_v), 0.01, 1.0)
    weight *= border

    # Sample source image colors
    src_colors = pixels[rows, cols].astype(np.float64)  # (N, 3)

    # Write into UV-space maps
    # Multiple image pixels may map to the same texel — accumulate with weights
    color_map = np.zeros((tex_size, tex_size, 3), dtype=np.float64)
    weight_map = np.zeros((tex_size, tex_size), dtype=np.float64)

    # Use np.add.at for proper accumulation of duplicates
    np.add.at(color_map, (tex_v, tex_u), src_colors * weight[:, None])
    np.add.at(weight_map, (tex_v, tex_u), weight)

    # Normalize per-view map (so each view contributes its average, not sum)
    has = weight_map > 0
    color_map[has] /= weight_map[has, None]
    # Set weight to 1 where we have data (actual blending weight across views
    # is the max weight at that texel, representing quality)
    avg_weight = np.zeros_like(weight_map)
    count_map = np.zeros_like(weight_map)
    np.add.at(count_map, (tex_v, tex_u), np.ones(len(weight)))
    np.add.at(avg_weight, (tex_v, tex_u), weight)
    avg_weight[has] /= count_map[has]
    weight_map = avg_weight

    return color_map, weight_map


# ── UV unwrapping ───────────────────────────────────────────────────────────

def _unwrap_uvs(verts, faces, views=None):
    """Project-based UV unwrap.

    If views are provided, uses camera view directions as projection axes
    (one chart per camera). Otherwise falls back to 6 cube-map axes.
    Neighbor smoothing eliminates isolated 1-2 face islands.
    """
    import time
    t0 = time.time()
    F = len(faces)

    # Face normals
    v0 = verts[faces[:, 0]]; v1 = verts[faces[:, 1]]; v2 = verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn /= (np.linalg.norm(fn, axis=-1, keepdims=True) + 1e-8)
    face_centroids = (v0 + v1 + v2) / 3.0

    # Build projection directions from views or cube-map
    if views and len(views) > 0:
        directions = []
        for view in views:
            c2w = np.linalg.inv(view['w2c'])
            cam_dir = -c2w[:3, 2]  # camera looks along -Z
            directions.append(cam_dir / (np.linalg.norm(cam_dir) + 1e-8))
        directions = np.array(directions, dtype=np.float32)
        label = f"{len(directions)} camera views"
    else:
        directions = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ], dtype=np.float32)
        label = "6 cube-map axes"

    n_dirs = len(directions)

    # Initial assignment: best-aligned direction per face normal
    dots = fn @ directions.T  # (F, n_dirs)
    face_dir = np.argmax(dots, axis=1)

    # ── Neighbor smoothing to remove isolated face islands ──
    # Build face adjacency
    from collections import defaultdict
    edge_to_faces = defaultdict(list)
    for fi in range(F):
        f = faces[fi]
        for j in range(3):
            e = (min(f[j], f[(j+1) % 3]), max(f[j], f[(j+1) % 3]))
            edge_to_faces[e].append(fi)

    neighbors = [[] for _ in range(F)]
    for flist in edge_to_faces.values():
        if len(flist) == 2:
            neighbors[flist[0]].append(flist[1])
            neighbors[flist[1]].append(flist[0])

    # Iterative majority vote: each face adopts the most common direction
    # among itself + neighbors, unless its own normal strongly disagrees
    for iteration in range(5):
        new_dir = face_dir.copy()
        changed = 0
        for fi in range(F):
            if not neighbors[fi]:
                continue
            # Count direction votes from neighbors
            votes = np.zeros(n_dirs, dtype=np.int32)
            votes[face_dir[fi]] += 2  # self gets double vote
            for ni in neighbors[fi]:
                votes[face_dir[ni]] += 1
            winner = int(np.argmax(votes))
            # Only switch if this face's normal is at least somewhat compatible
            # (dot product > 0 means face roughly faces the direction)
            if winner != face_dir[fi] and dots[fi, winner] > -0.1:
                new_dir[fi] = winner
                changed += 1
        face_dir = new_dir
        if changed == 0:
            break

    # ── Project UVs from assigned direction ──
    uvs = np.zeros((F * 3, 2), dtype=np.float32)
    uv_faces = np.arange(F * 3, dtype=np.int32).reshape(F, 3)

    # Precompute projection axes per direction
    proj_axes = []
    for d in directions:
        if abs(d[2]) < 0.9:
            u_ax = np.cross(d, np.array([0, 0, 1], dtype=np.float32))
        else:
            u_ax = np.cross(d, np.array([0, 1, 0], dtype=np.float32))
        u_ax = u_ax / (np.linalg.norm(u_ax) + 1e-8)
        v_ax = np.cross(d, u_ax)
        v_ax = v_ax / (np.linalg.norm(v_ax) + 1e-8)
        proj_axes.append((u_ax.astype(np.float32), v_ax.astype(np.float32)))

    # Vectorized projection per direction group
    for di in range(n_dirs):
        mask = face_dir == di
        if not mask.any():
            continue
        u_ax, v_ax = proj_axes[di]
        face_idx = np.where(mask)[0]
        fv = verts[faces[face_idx]]  # (N, 3, 3)
        proj_u = fv @ u_ax
        proj_v = fv @ v_ax
        for vi in range(3):
            uvs[face_idx * 3 + vi, 0] = proj_u[:, vi]
            uvs[face_idx * 3 + vi, 1] = proj_v[:, vi]

    # Pack into grid atlas
    grid_cols = int(np.ceil(np.sqrt(n_dirs)))
    grid_rows = int(np.ceil(n_dirs / grid_cols))
    cell_w, cell_h = 1.0 / grid_cols, 1.0 / grid_rows
    padding = 0.003

    for di in range(n_dirs):
        mask = face_dir == di
        if not mask.any():
            continue
        face_idx = np.where(mask)[0]
        uv_idx = np.concatenate([face_idx * 3, face_idx * 3 + 1, face_idx * 3 + 2])

        sub_uv = uvs[uv_idx]
        mn = sub_uv.min(axis=0)
        mx = sub_uv.max(axis=0)
        rng = mx - mn
        rng[rng < 1e-8] = 1.0
        sub_uv = (sub_uv - mn) / rng

        gc = di % grid_cols
        gr = di // grid_cols
        sub_uv[:, 0] = gc * cell_w + padding + sub_uv[:, 0] * (cell_w - 2 * padding)
        sub_uv[:, 1] = gr * cell_h + padding + sub_uv[:, 1] * (cell_h - 2 * padding)
        uvs[uv_idx] = sub_uv

    elapsed = time.time() - t0
    counts = [int((face_dir == i).sum()) for i in range(n_dirs)]
    print(f"  UV projection: {F:,d} faces -> {n_dirs} charts ({label}) in {elapsed:.2f}s")
    for di, c in enumerate(counts):
        if c > 0:
            print(f"    chart {di}: {c:,d} faces")
    return uvs, uv_faces




# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_debug_texture(size=1024):
    """Checkerboard + UV gradient for verifying unwrap."""
    grid = 32
    coords = np.indices((size, size))
    check = ((coords[0] // grid) + (coords[1] // grid)) % 2
    tex = np.zeros((size, size, 3), dtype=np.uint8)
    # R = U, G = V, modulated by checkerboard
    u_grad = np.linspace(0, 255, size, dtype=np.uint8)
    v_grad = np.linspace(0, 255, size, dtype=np.uint8)
    tex[:, :, 0] = u_grad[None, :]  # R = horizontal
    tex[:, :, 1] = v_grad[:, None]  # G = vertical
    tex[:, :, 2] = 80
    # Darken checkerboard cells
    tex[check == 1] = (tex[check == 1] * 0.6).astype(np.uint8)
    return tex


def _barycentric_batch(pts, a, b, c):
    v0 = b - a; v1 = c - a; v2 = pts - a
    d00 = (v0 * v0).sum(axis=-1)
    d01 = (v0 * v1).sum(axis=-1)
    d11 = (v1 * v1).sum(axis=-1)
    d20 = (v2 * v0).sum(axis=-1)
    d21 = (v2 * v1).sum(axis=-1)
    denom = d00 * d11 - d01 * d01
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.stack([u, v, w], axis=-1)


def _dilate_texture(texture, iterations=32):
    from scipy.ndimage import binary_dilation, uniform_filter
    filled = (texture.sum(axis=-1) > 0)
    for _ in range(iterations):
        dilated = binary_dilation(filled)
        new_px = dilated & ~filled
        if not new_px.any():
            break
        for c in range(3):
            ch = texture[:, :, c].astype(np.float32)
            ch[~filled] = 0
            blurred = uniform_filter(ch, size=3)
            cnt = uniform_filter(filled.astype(np.float32), size=3)
            valid = (cnt > 0) & new_px
            texture[valid, c] = (blurred[valid] / (cnt[valid] + 1e-8)).clip(0, 255).astype(np.uint8)
        filled = dilated
    return texture


def _write_obj(path, verts, faces, uvs, uv_faces):
    with open(path, 'w') as f:
        f.write("mtllib mesh.mtl\nusemtl material0\n")
        for i in range(len(verts)):
            f.write(f"v {verts[i,0]:.6f} {verts[i,1]:.6f} {verts[i,2]:.6f}\n")
        for i in range(len(uvs)):
            f.write(f"vt {uvs[i,0]:.6f} {uvs[i,1]:.6f}\n")
        for i in range(len(faces)):
            a, b, c = faces[i] + 1
            ua, ub, uc = uv_faces[i] + 1
            f.write(f"f {a}/{ua} {b}/{ub} {c}/{uc}\n")
