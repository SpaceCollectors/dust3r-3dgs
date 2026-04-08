"""
UV Mapping + Texture Projection

Uses xatlas for UV unwrapping. Texture baking:
1. Rasterize face IDs into UV space (PIL polygon fill — no gaps)
2. For each texel, compute 3D position via barycentric interpolation
3. Project into cameras, sample image color with incidence weighting
4. Dilate to fill edge pixels
"""

import os
import numpy as np
from PIL import Image, ImageDraw


def create_textured_mesh(verts, faces, colors, views, output_dir, texture_size=2048):
    os.makedirs(output_dir, exist_ok=True)
    V, F = len(verts), len(faces)
    print(f"  Creating texture for {V:,d} verts, {F:,d} faces, {texture_size}px")

    uvs, uv_faces = _unwrap_uvs(verts, faces)
    texture = _bake_texture(verts, faces, uvs, uv_faces, views, texture_size)
    texture = _dilate_texture(texture)

    tex_path = os.path.join(output_dir, 'texture.png')
    Image.fromarray(texture).save(tex_path, quality=95)

    mtl_path = os.path.join(output_dir, 'mesh.mtl')
    with open(mtl_path, 'w') as f:
        f.write("newmtl material0\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\nmap_Kd texture.png\n")

    uvs_obj = uvs.copy()
    uvs_obj[:, 1] = 1.0 - uvs_obj[:, 1]

    obj_path = os.path.join(output_dir, 'mesh.obj')
    _write_obj(obj_path, verts, faces, uvs_obj, uv_faces)
    print(f"  Saved: {obj_path}")
    return obj_path


def _unwrap_uvs(verts, faces):
    try:
        import xatlas
        print("  UV unwrapping with xatlas...")
        vmapping, uv_faces, uvs = xatlas.parametrize(
            verts.astype(np.float32), faces.astype(np.uint32))
        print(f"  UV atlas: {len(uvs):,d} UV coords")
        return uvs.astype(np.float32), uv_faces.astype(np.int32)
    except ImportError:
        print("  xatlas not available, using simple atlas")
        return _simple_atlas(faces)


def _simple_atlas(faces):
    F = len(faces)
    cols = int(np.ceil(np.sqrt(F)))
    cell = 1.0 / cols
    margin = cell * 0.05
    uvs = np.zeros((F * 3, 2), dtype=np.float32)
    uv_faces = np.zeros((F, 3), dtype=np.int32)
    for fi in range(F):
        r, c = fi // cols, fi % cols
        base = fi * 3
        uvs[base] = [c * cell + margin, r * cell + margin]
        uvs[base + 1] = [(c + 1) * cell - margin, r * cell + margin]
        uvs[base + 2] = [c * cell + cell * 0.5, (r + 1) * cell - margin]
        uv_faces[fi] = [base, base + 1, base + 2]
    return uvs, uv_faces


def _bake_texture(verts, faces, uvs, uv_faces, views, tex_size):
    """
    Bake camera images into UV texture.

    Step 1: Rasterize face IDs into UV space using PIL (guaranteed no gaps)
    Step 2: For each texel with a face ID, compute 3D position
    Step 3: Project into cameras and sample
    """
    F = len(faces)

    # Precompute face data
    face_verts = verts[faces]  # (F, 3, 3)
    face_uvs = np.zeros((F, 3, 2), dtype=np.float32)
    for fi in range(F):
        face_uvs[fi, 0] = uvs[uv_faces[fi, 0]]
        face_uvs[fi, 1] = uvs[uv_faces[fi, 1]]
        face_uvs[fi, 2] = uvs[uv_faces[fi, 2]]

    # Face normals
    v0 = face_verts[:, 0]; v1 = face_verts[:, 1]; v2 = face_verts[:, 2]
    face_normals = np.cross(v1 - v0, v2 - v0)
    face_normals /= (np.linalg.norm(face_normals, axis=-1, keepdims=True) + 1e-8)

    # Step 1: Rasterize face IDs into UV space
    print(f"  Rasterizing face IDs into UV space...")
    face_id_map = np.full((tex_size, tex_size), -1, dtype=np.int32)
    img = Image.new('I', (tex_size, tex_size), 0)  # 32-bit int image
    draw = ImageDraw.Draw(img)

    for fi in range(F):
        uv0 = face_uvs[fi, 0]
        uv1 = face_uvs[fi, 1]
        uv2 = face_uvs[fi, 2]
        px = [(int(uv0[0] * tex_size), int(uv0[1] * tex_size)),
              (int(uv1[0] * tex_size), int(uv1[1] * tex_size)),
              (int(uv2[0] * tex_size), int(uv2[1] * tex_size))]
        draw.polygon(px, fill=fi + 1)  # +1 so 0 = background

    face_id_map = np.array(img, dtype=np.int32) - 1  # back to 0-indexed, -1 = bg

    n_covered = (face_id_map >= 0).sum()
    print(f"  {n_covered:,d} texels covered ({n_covered * 100 // (tex_size*tex_size)}%)")

    # Step 2: For each covered texel, compute 3D position via barycentric
    print(f"  Computing 3D positions for texels...")
    texture = np.zeros((tex_size, tex_size, 3), dtype=np.uint8)

    # Get all covered texel coordinates
    rows, cols = np.where(face_id_map >= 0)
    fids = face_id_map[rows, cols]
    n_texels = len(rows)
    print(f"  {n_texels:,d} texels to shade")

    # UV position of each texel
    texel_u = (cols + 0.5) / tex_size
    texel_v = (rows + 0.5) / tex_size
    texel_uv = np.stack([texel_u, texel_v], axis=-1)  # (N, 2)

    # Barycentric coords for each texel in its face
    fuv0 = face_uvs[fids, 0]  # (N, 2)
    fuv1 = face_uvs[fids, 1]
    fuv2 = face_uvs[fids, 2]

    bary = _barycentric_batch(texel_uv, fuv0, fuv1, fuv2)

    # 3D positions
    fv0 = face_verts[fids, 0]  # (N, 3)
    fv1 = face_verts[fids, 1]
    fv2 = face_verts[fids, 2]
    pos_3d = bary[:, 0:1] * fv0 + bary[:, 1:2] * fv1 + bary[:, 2:3] * fv2  # (N, 3)
    normals = face_normals[fids]  # (N, 3)

    # Step 3: Project into cameras and sample
    print(f"  Sampling from {len(views)} cameras...")
    color_accum = np.zeros((n_texels, 3), dtype=np.float64)
    weight_accum = np.zeros(n_texels, dtype=np.float64)

    for ci, view in enumerate(views):
        c2w = np.linalg.inv(view['w2c'])
        cam_center = c2w[:3, 3]
        R = view['w2c'][:3, :3]
        t = view['w2c'][:3, 3]

        # View direction
        view_dirs = cam_center[None, :] - pos_3d
        view_dists = np.linalg.norm(view_dirs, axis=-1) + 1e-8
        view_dirs /= view_dists[:, None]

        # Facing: use abs dot for two-sided
        dots = np.abs((normals * view_dirs).sum(axis=-1))

        # Project to image
        pts_cam = (R @ pos_3d.T).T + t
        z = pts_cam[:, 2]
        good = (z > 0.01) & (dots > 0.01)

        if not good.any():
            continue

        u_px = pts_cam[good, 0] / z[good] * view['K'][0, 0] + view['K'][0, 2]
        v_px = pts_cam[good, 1] / z[good] * view['K'][1, 1] + view['K'][1, 2]

        in_frame = ((u_px >= 0) & (u_px < view['W'] - 1) &
                    (v_px >= 0) & (v_px < view['H'] - 1))

        if not in_frame.any():
            continue

        # Sample
        u_i = u_px[in_frame].astype(int)
        v_i = v_px[in_frame].astype(int)
        sampled = view['pixels'][v_i, u_i]

        # Weight: incidence^2 * border falloff
        w = dots[good][in_frame] ** 2
        margin_u = np.minimum(u_px[in_frame], view['W'] - 1 - u_px[in_frame]) / (view['W'] * 0.3)
        margin_v = np.minimum(v_px[in_frame], view['H'] - 1 - v_px[in_frame]) / (view['H'] * 0.3)
        w *= np.clip(np.minimum(margin_u, margin_v), 0.01, 1.0)

        idx_good = np.where(good)[0]
        idx_final = idx_good[in_frame]

        color_accum[idx_final] += sampled * w[:, None]
        weight_accum[idx_final] += w

    # Normalize
    has = weight_accum > 0.001
    colors_out = np.full((n_texels, 3), 128, dtype=np.uint8)
    colors_out[has] = (color_accum[has] / weight_accum[has, None] * 255).clip(0, 255).astype(np.uint8)

    # Write to texture
    texture[rows, cols] = colors_out

    n_colored = has.sum()
    print(f"  {n_colored:,d} / {n_texels:,d} texels colored ({n_colored * 100 // max(n_texels, 1)}%)")

    return texture


def _barycentric_batch(pts, a, b, c):
    """Barycentric coords for batch. Returns (N, 3) with u,v,w."""
    v0 = b - a
    v1 = c - a
    v2 = pts - a

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


def _dilate_texture(texture, iterations=10):
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
