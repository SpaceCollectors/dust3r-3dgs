"""
UV Mapping + Texture Projection — Fast Vectorized Version

Uses xatlas for UV unwrapping and vectorized projection for texture baking.
"""

import os
import numpy as np
from PIL import Image


def create_textured_mesh(verts, faces, colors, views, output_dir, texture_size=2048):
    """
    Create a UV-mapped textured mesh from camera projections.

    Returns:
        obj_path: path to the OBJ file
    """
    os.makedirs(output_dir, exist_ok=True)
    V, F = len(verts), len(faces)
    print(f"  Creating texture for {V:,d} verts, {F:,d} faces")

    # Step 1: UV unwrap
    uvs, uv_faces = _unwrap_uvs(verts, faces)

    # Step 2: Bake texture from camera projections
    texture = _bake_texture_fast(verts, faces, uvs, uv_faces, views, texture_size)

    # Step 3: Save OBJ + MTL + texture
    tex_path = os.path.join(output_dir, 'texture.png')
    Image.fromarray(texture).save(tex_path, quality=95)

    mtl_path = os.path.join(output_dir, 'mesh.mtl')
    with open(mtl_path, 'w') as f:
        f.write("newmtl material0\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\nmap_Kd texture.png\n")

    # Flip texture vertically (OBJ UV convention: V=0 is bottom)
    texture = texture[::-1].copy()

    obj_path = os.path.join(output_dir, 'mesh.obj')
    _write_obj(obj_path, verts, faces, uvs, uv_faces)

    print(f"  Saved: {obj_path}")
    return obj_path


def _unwrap_uvs(verts, faces):
    """UV unwrap using xatlas."""
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
    """Fallback: pack each face as a small triangle in a grid."""
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


def _bake_texture_fast(verts, faces, uvs, uv_faces, views, tex_size):
    """
    Fast texture baking: rasterize each face into UV space,
    project the 3D position into cameras, pick best color.

    Vectorized over texels within each face.
    """
    texture = np.zeros((tex_size, tex_size, 3), dtype=np.float32)
    weight_map = np.zeros((tex_size, tex_size), dtype=np.float32)
    F = len(faces)

    # Precompute face normals
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    fn_len = np.linalg.norm(face_normals, axis=-1, keepdims=True) + 1e-8
    face_normals /= fn_len

    # Precompute per-camera data
    cam_data = []
    for view in views:
        c2w = np.linalg.inv(view['w2c'])
        cam_data.append({
            'center': c2w[:3, 3],
            'R': view['w2c'][:3, :3],
            't': view['w2c'][:3, 3],
            'K': view['K'],
            'W': view['W'], 'H': view['H'],
            'pixels': view['pixels'],
        })

    # Process faces in batches
    BATCH = 5000
    for fi_start in range(0, F, BATCH):
        fi_end = min(fi_start + BATCH, F)
        if fi_start % 20000 == 0:
            print(f"    Baking faces {fi_start:,d}-{fi_end:,d} / {F:,d}")

        for fi in range(fi_start, fi_end):
            # UV coords of this face
            uv0 = uvs[uv_faces[fi, 0]]
            uv1 = uvs[uv_faces[fi, 1]]
            uv2 = uvs[uv_faces[fi, 2]]

            # 3D positions
            p0 = verts[faces[fi, 0]]
            p1 = verts[faces[fi, 1]]
            p2 = verts[faces[fi, 2]]
            fn = face_normals[fi]

            # Bounding box in texel space
            u_min = max(0, int(min(uv0[0], uv1[0], uv2[0]) * tex_size))
            u_max = min(tex_size - 1, int(max(uv0[0], uv1[0], uv2[0]) * tex_size) + 1)
            v_min = max(0, int(min(uv0[1], uv1[1], uv2[1]) * tex_size))
            v_max = min(tex_size - 1, int(max(uv0[1], uv1[1], uv2[1]) * tex_size) + 1)

            if u_max <= u_min or v_max <= v_min:
                continue

            # Generate texel grid within bbox
            tu = np.arange(u_min, u_max + 1)
            tv = np.arange(v_min, v_max + 1)
            tgu, tgv = np.meshgrid(tu, tv)
            texels_uv = np.stack([(tgu + 0.5) / tex_size, (tgv + 0.5) / tex_size], axis=-1)
            # (th, tw, 2)
            th, tw = texels_uv.shape[:2]

            # Barycentric coords for all texels at once
            bary = _barycentric_batch(texels_uv.reshape(-1, 2), uv0, uv1, uv2)
            if bary is None:
                continue
            bu, bv, bw = bary  # (N,) each
            inside = (bu >= -0.001) & (bv >= -0.001) & (bw >= -0.001)

            if not inside.any():
                continue

            # 3D positions of valid texels
            pos_3d = bu[inside, None] * p0 + bv[inside, None] * p1 + bw[inside, None] * p2
            n_valid = len(pos_3d)

            # Blend ALL cameras weighted by incidence angle (dot product)
            # This avoids hard seams between camera contributions
            blended_colors = np.zeros((n_valid, 3), dtype=np.float64)
            total_weights = np.zeros(n_valid, dtype=np.float64)

            for cd in cam_data:
                view_dirs = cd['center'][None, :] - pos_3d
                view_dists = np.linalg.norm(view_dirs, axis=-1) + 1e-8
                view_dirs /= view_dists[:, None]

                # Incidence angle: more frontal = higher weight
                dots = (fn[None, :] * view_dirs).sum(axis=-1)
                facing = dots > 0.1

                if not facing.any():
                    continue

                pts_cam = (cd['R'] @ pos_3d[facing].T).T + cd['t']
                z = pts_cam[:, 2]
                good_z = z > 0.01

                if not good_z.any():
                    continue

                u_px = pts_cam[good_z, 0] / z[good_z] * cd['K'][0, 0] + cd['K'][0, 2]
                v_px = pts_cam[good_z, 1] / z[good_z] * cd['K'][1, 1] + cd['K'][1, 2]

                # Margin from image edges (lower weight near borders)
                margin_u = np.minimum(u_px, cd['W'] - 1 - u_px) / (cd['W'] * 0.5)
                margin_v = np.minimum(v_px, cd['H'] - 1 - v_px) / (cd['H'] * 0.5)
                border_weight = np.clip(np.minimum(margin_u, margin_v), 0, 1)

                in_frame = ((u_px >= 0) & (u_px < cd['W'] - 1) &
                            (v_px >= 0) & (v_px < cd['H'] - 1))

                if not in_frame.any():
                    continue

                u_i = u_px[in_frame].astype(int)
                v_i = v_px[in_frame].astype(int)
                sampled = cd['pixels'][v_i, u_i]

                # Weight = incidence angle^2 * border_weight (penalize grazing + edges)
                w = (dots[facing][good_z][in_frame] ** 2) * border_weight[in_frame]

                idx_facing = np.where(facing)[0]
                idx_goodz = idx_facing[good_z]
                idx_final = idx_goodz[in_frame]

                blended_colors[idx_final] += sampled * w[:, None]
                total_weights[idx_final] += w

            # Normalize
            has_weight = total_weights > 0.001
            best_colors = np.zeros((n_valid, 3), dtype=np.float32)
            best_weights = np.zeros(n_valid, dtype=np.float32)
            best_colors[has_weight] = (blended_colors[has_weight] / total_weights[has_weight, None]).astype(np.float32)
            best_weights[has_weight] = total_weights[has_weight]

            # Write to texture
            # Map valid texels back to texture pixel coordinates
            inside_flat = np.where(inside)[0]
            has_color = best_weights > 0
            if has_color.any():
                # inside_flat indexes into the flattened (th*tw) texel grid
                # Map back to row/col within the bbox
                colored_flat = inside_flat[has_color]
                tex_rows = colored_flat // tw  # row within bbox
                tex_cols = colored_flat % tw   # col within bbox
                abs_rows = tex_rows + v_min
                abs_cols = tex_cols + u_min

                valid_tex = ((abs_rows >= 0) & (abs_rows < tex_size) &
                             (abs_cols >= 0) & (abs_cols < tex_size))

                r = abs_rows[valid_tex]
                c_idx = abs_cols[valid_tex]
                c_val = (best_colors[has_color][valid_tex] * 255).clip(0, 255)
                w_val = best_weights[has_color][valid_tex]

                for k in range(len(r)):
                    if w_val[k] > weight_map[r[k], c_idx[k]]:
                        texture[r[k], c_idx[k]] = c_val[k]
                        weight_map[r[k], c_idx[k]] = w_val[k]

    # Fill holes
    texture = _dilate_texture(texture, weight_map)
    return texture.clip(0, 255).astype(np.uint8)


def _barycentric_batch(pts, a, b, c):
    """Barycentric coords for batch of points. Returns (u,v,w) arrays or None."""
    v0 = b - a
    v1 = c - a
    v2 = pts - a[None, :]

    d00 = v0 @ v0
    d01 = v0 @ v1
    d11 = v1 @ v1
    d20 = v2 @ v0
    d21 = v2 @ v1

    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-10:
        return None

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w


def _dilate_texture(texture, weight_map, iterations=10):
    """Fill texture holes by dilating from filled pixels."""
    from scipy.ndimage import binary_dilation, uniform_filter

    filled = weight_map > 0
    for _ in range(iterations):
        dilated = binary_dilation(filled)
        new_pixels = dilated & ~filled
        if not new_pixels.any():
            break
        for c in range(3):
            ch = texture[:, :, c].copy()
            ch[~filled] = 0
            blurred = uniform_filter(ch, size=3)
            cnt = uniform_filter(filled.astype(np.float32), size=3)
            valid = (cnt > 0) & new_pixels
            texture[valid, c] = blurred[valid] / (cnt[valid] + 1e-8)
        filled = dilated
    return texture


def _write_obj(path, verts, faces, uvs, uv_faces):
    """Write OBJ with UV coordinates."""
    with open(path, 'w') as f:
        f.write("mtllib mesh.mtl\nusemtl material0\n")
        for i in range(len(verts)):
            f.write(f"v {verts[i,0]:.6f} {verts[i,1]:.6f} {verts[i,2]:.6f}\n")
        for i in range(len(uvs)):
            f.write(f"vt {uvs[i,0]:.6f} {uvs[i,1]:.6f}\n")
        for i in range(len(faces)):
            a, b, c = faces[i] + 1  # 1-indexed
            ua, ub, uc = uv_faces[i] + 1
            f.write(f"f {a}/{ua} {b}/{ub} {c}/{uc}\n")
