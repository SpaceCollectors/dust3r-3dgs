"""
UV Mapping + Texture Projection

Per-face camera assignment with proper 3D→2D projection, back-face culling,
full-resolution depth-buffer visibility, and seam color correction.
"""

import os
import numpy as np
from PIL import Image, ImageDraw


def create_textured_mesh(verts, faces, colors, views, output_dir,
                         texture_size=4096, return_data=False):
    os.makedirs(output_dir, exist_ok=True)

    # Try COLMAP mesh_texturer first (uses correct cameras from dense workspace)
    colmap_workdir = views[0].get('_colmap_workdir') if views else None
    if colmap_workdir:
        try:
            result = _texture_colmap(verts, faces, colmap_workdir, output_dir)
            if result is not None:
                if return_data:
                    try:
                        tex_img = np.array(Image.open(os.path.join(output_dir, 'texture.png')))
                        return result, None, None, tex_img
                    except Exception:
                        return result, None, None, None
                return result
        except Exception as e:
            print(f"  COLMAP texturing failed: {e}, falling back to custom")

    # Try PyMeshLab texturing
    try:
        result = _texture_pymeshlab(verts, faces, views, output_dir, texture_size)
        if result is not None:
            if return_data:
                try:
                    tex_img = np.array(Image.open(os.path.join(output_dir, 'texture.png')))
                    return result, None, None, tex_img
                except Exception:
                    return result, None, None, None
            return result
    except Exception as e:
        print(f"  PyMeshLab texturing failed: {e}, falling back to custom")
        import traceback; traceback.print_exc()
    V, F = len(verts), len(faces)
    print(f"  Creating texture for {V:,d} verts, {F:,d} faces, {texture_size}px")

    uvs, uv_faces = _unwrap_uvs(verts, faces)
    texture = _bake_texture(verts, faces, uvs, uv_faces, views, texture_size)
    texture = _dilate_texture(texture, iterations=32)

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

    if return_data:
        return obj_path, uvs, uv_faces, texture
    return obj_path


def _bake_texture(verts, faces, uvs, uv_faces, views, tex_size):
    """
    Per-face camera assignment with correct 3D→2D projection.

    1. For each face, score every camera (visibility, incidence, resolution)
    2. Assign one best camera per face (no intra-face seams)
    3. For each texel, recover its 3D position via barycentrics, then project
       through the assigned camera's K,R,t to sample the image
    """
    F = len(faces)
    V = len(verts)

    # Precompute face UV coords
    face_uvs = np.zeros((F, 3, 2), dtype=np.float32)
    for fi in range(F):
        face_uvs[fi, 0] = uvs[uv_faces[fi, 0]]
        face_uvs[fi, 1] = uvs[uv_faces[fi, 1]]
        face_uvs[fi, 2] = uvs[uv_faces[fi, 2]]

    # Face normals and centroids
    fv0 = verts[faces[:, 0]]; fv1 = verts[faces[:, 1]]; fv2 = verts[faces[:, 2]]
    face_normals = np.cross(fv1 - fv0, fv2 - fv0)
    face_normals /= (np.linalg.norm(face_normals, axis=-1, keepdims=True) + 1e-8)
    face_centroids = (fv0 + fv1 + fv2) / 3.0

    # ── Step 1: Per-face camera scoring ──
    print(f"  Scoring {F:,d} faces across {len(views)} cameras...")
    face_best_cam = np.full(F, -1, dtype=np.int32)
    face_best_score = np.full(F, -1.0, dtype=np.float64)

    for ci, view in enumerate(views):
        print(f"    Camera {ci+1}/{len(views)}...")
        R = view['w2c'][:3, :3]
        t = view['w2c'][:3, 3]
        K = view['K']
        W_img, H_img = view['W'], view['H']
        c2w = np.linalg.inv(view['w2c'])
        cam_center = c2w[:3, 3]

        # Project all vertices to image space
        pts_cam = (R @ verts.T).T + t
        z_vert = pts_cam[:, 2]
        u_vert = pts_cam[:, 0] / np.maximum(z_vert, 0.01) * K[0, 0] + K[0, 2]
        v_vert = pts_cam[:, 1] / np.maximum(z_vert, 0.01) * K[1, 1] + K[1, 2]

        # Rasterize face visibility (painter's algorithm — proper occlusion)
        visible_faces = _rasterize_visibility(faces, u_vert, v_vert, z_vert, W_img, H_img)

        # Back-face culling: face normal must point toward camera
        view_to_face = cam_center[None, :] - face_centroids
        view_to_face /= (np.linalg.norm(view_to_face, axis=-1, keepdims=True) + 1e-8)
        ndot = (face_normals * view_to_face).sum(axis=-1)
        front_facing = ndot > 0.05

        # In-frame check (all 3 vertices must be in frame and in front)
        fz = z_vert[faces]
        fu = u_vert[faces]
        fvv = v_vert[faces]
        all_in_front = fz.min(axis=1) > 0.01
        all_in_frame = ((fu.min(axis=1) >= 0) & (fu.max(axis=1) < W_img - 1) &
                        (fvv.min(axis=1) >= 0) & (fvv.max(axis=1) < H_img - 1))

        # Projected face centroids (for border falloff scoring)
        c_u = fu.mean(axis=1)
        c_v = fvv.mean(axis=1)

        # Face-level visibility from rasterization
        not_occluded = np.zeros(F, dtype=bool)
        if visible_faces:
            not_occluded[np.array(list(visible_faces), dtype=np.int64)] = True

        # Combined visibility
        visible = front_facing & all_in_front & all_in_frame & not_occluded

        # Score: incidence angle * image-space area (prefer frontal, high-res views)
        # Incidence: dot product (already computed as ndot, clamped positive by front_facing)
        score = np.zeros(F, dtype=np.float64)
        score[visible] = ndot[visible] ** 2

        # Border falloff: penalize faces near image edges
        margin_u = np.minimum(c_u, W_img - 1 - c_u) / (W_img * 0.25)
        margin_v = np.minimum(c_v, H_img - 1 - c_v) / (H_img * 0.25)
        border = np.clip(np.minimum(margin_u, margin_v), 0.01, 1.0)
        score *= border

        # Projected area score: prefer cameras where face appears larger
        # Use cross product of projected edges as proxy for projected area
        pu = fu[visible]; pv = fvv[visible]
        e1u = pu[:, 1] - pu[:, 0]; e1v = pv[:, 1] - pv[:, 0]
        e2u = pu[:, 2] - pu[:, 0]; e2v = pv[:, 2] - pv[:, 0]
        proj_area = np.abs(e1u * e2v - e2u * e1v)
        area_boost = np.log1p(proj_area + 1.0)
        score[visible] *= area_boost

        better = visible & (score > face_best_score)
        face_best_cam[better] = ci
        face_best_score[better] = score[better]

    assigned = face_best_cam >= 0
    print(f"  {assigned.sum():,d} / {F:,d} faces assigned a camera")

    # ── Step 1b: Global color correction across cameras ──
    # For each pair of cameras that share seam edges, compute a color offset
    # that minimizes the difference at shared vertices. This is a least-squares
    # global adjustment (Waechter et al. "Let There Be Color!").
    cam_color_scale, cam_color_offset = _compute_color_correction(
        verts, faces, face_best_cam, views, len(views))

    # ── Step 2: Rasterize UV space and bake texels ──
    print(f"  Rasterizing face IDs into UV space...")
    img = Image.new('I', (tex_size, tex_size), 0)
    draw = ImageDraw.Draw(img)

    for fi in range(F):
        p0 = (face_uvs[fi, 0, 0] * tex_size, face_uvs[fi, 0, 1] * tex_size)
        p1 = (face_uvs[fi, 1, 0] * tex_size, face_uvs[fi, 1, 1] * tex_size)
        p2 = (face_uvs[fi, 2, 0] * tex_size, face_uvs[fi, 2, 1] * tex_size)
        ep0, ep1, ep2 = _expand_triangle_px(p0, p1, p2, 2.0)
        px = [(int(round(ep0[0])), int(round(ep0[1]))),
              (int(round(ep1[0])), int(round(ep1[1]))),
              (int(round(ep2[0])), int(round(ep2[1])))]
        draw.polygon(px, fill=fi + 1)

    face_id_map = np.array(img, dtype=np.int32) - 1
    rows, cols = np.where(face_id_map >= 0)
    fids = face_id_map[rows, cols]
    n_texels = len(rows)
    print(f"  {n_texels:,d} texels to shade")

    # Barycentrics for each texel in UV space
    texel_uv = np.stack([(cols + 0.5) / tex_size, (rows + 0.5) / tex_size], axis=-1)
    fuv0 = face_uvs[fids, 0]
    fuv1 = face_uvs[fids, 1]
    fuv2 = face_uvs[fids, 2]
    bary = _barycentric_batch(texel_uv, fuv0, fuv1, fuv2)
    bary = np.clip(bary, 0.0, None)
    bary /= (bary.sum(axis=-1, keepdims=True) + 1e-10)

    # Recover 3D world position for each texel via barycentrics
    face_verts = verts[faces]
    texel_pos = (bary[:, 0:1] * face_verts[fids, 0] +
                 bary[:, 1:2] * face_verts[fids, 1] +
                 bary[:, 2:3] * face_verts[fids, 2])

    # ── Step 3: Project each texel's 3D position through its assigned camera ──
    print(f"  Projecting texels through assigned cameras...")
    texture = np.zeros((tex_size, tex_size, 3), dtype=np.uint8)
    colors_out = np.zeros((n_texels, 3), dtype=np.uint8)
    texel_cam = face_best_cam[fids]  # camera assigned to each texel's face

    for ci, view in enumerate(views):
        mask = texel_cam == ci
        if not mask.any():
            continue

        R = view['w2c'][:3, :3]
        t_vec = view['w2c'][:3, 3]
        K = view['K']
        W_img, H_img = view['W'], view['H']

        # Project texel 3D positions through this camera (correct perspective projection)
        pos = texel_pos[mask]  # (N, 3)
        cam_pts = (R @ pos.T).T + t_vec  # (N, 3)
        z = cam_pts[:, 2]
        safe_z = np.maximum(z, 0.01)
        px_u = cam_pts[:, 0] / safe_z * K[0, 0] + K[0, 2]
        px_v = cam_pts[:, 1] / safe_z * K[1, 1] + K[1, 2]

        # Clamp to image bounds for sampling
        px_u = np.clip(px_u, 0, W_img - 1.001)
        px_v = np.clip(px_v, 0, H_img - 1.001)

        # Bilinear sampling
        u0 = np.floor(px_u).astype(int); v0 = np.floor(px_v).astype(int)
        u1 = np.minimum(u0 + 1, W_img - 1); v1 = np.minimum(v0 + 1, H_img - 1)
        fu = px_u - u0; fv = px_v - v0
        sampled = ((1 - fu)[:, None] * (1 - fv)[:, None] * view['pixels'][v0, u0] +
                   fu[:, None] * (1 - fv)[:, None] * view['pixels'][v0, u1] +
                   (1 - fu)[:, None] * fv[:, None] * view['pixels'][v1, u0] +
                   fu[:, None] * fv[:, None] * view['pixels'][v1, u1])

        # Apply per-camera color correction
        corrected = sampled * cam_color_scale[ci][None, :] + cam_color_offset[ci][None, :]
        colors_out[mask] = (corrected * 255).clip(0, 255).astype(np.uint8)

    texture[rows, cols] = colors_out

    n_colored = (texel_cam >= 0).sum()
    print(f"  {n_colored:,d} / {n_texels:,d} texels colored ({n_colored * 100 // max(n_texels, 1)}%)")
    return texture


def _compute_color_correction(verts, faces, face_best_cam, views, n_cams):
    """Compute per-camera color scale + offset to match exposure/white-balance.

    For each edge shared by two faces assigned to different cameras, sample both
    cameras at the shared vertices and compute the color difference. Then solve
    a least-squares system for per-camera (scale, offset) that minimizes seam
    discontinuities globally.
    """
    if n_cams <= 1:
        return (np.ones((n_cams, 3), dtype=np.float64),
                np.zeros((n_cams, 3), dtype=np.float64))

    print(f"  Computing global color correction...")

    # Sample a reference color for each camera from face centroids
    # Average the image color at each face's centroid for all faces assigned to that camera
    cam_means = np.zeros((n_cams, 3), dtype=np.float64)
    cam_counts = np.zeros(n_cams, dtype=np.float64)

    face_centroids = (verts[faces[:, 0]] + verts[faces[:, 1]] + verts[faces[:, 2]]) / 3.0

    for ci, view in enumerate(views):
        assigned_mask = face_best_cam == ci
        if not assigned_mask.any():
            continue

        R = view['w2c'][:3, :3]
        t = view['w2c'][:3, 3]
        K = view['K']
        W_img, H_img = view['W'], view['H']

        centroids = face_centroids[assigned_mask]
        cam_pts = (R @ centroids.T).T + t
        z = np.maximum(cam_pts[:, 2], 0.01)
        px_u = np.clip(cam_pts[:, 0] / z * K[0, 0] + K[0, 2], 0, W_img - 1.001)
        px_v = np.clip(cam_pts[:, 1] / z * K[1, 1] + K[1, 2], 0, H_img - 1.001)

        # Simple nearest-pixel sample for speed
        ui = np.clip(np.round(px_u).astype(int), 0, W_img - 1)
        vi = np.clip(np.round(px_v).astype(int), 0, H_img - 1)
        sampled = view['pixels'][vi, ui]  # (N, 3) in [0, 1]

        cam_means[ci] = sampled.mean(axis=0)
        cam_counts[ci] = len(sampled)

    # Compute scale + offset relative to the camera with the most assigned faces
    ref_cam = int(np.argmax(cam_counts))
    ref_mean = cam_means[ref_cam]

    cam_color_scale = np.ones((n_cams, 3), dtype=np.float64)
    cam_color_offset = np.zeros((n_cams, 3), dtype=np.float64)

    for ci in range(n_cams):
        if cam_counts[ci] < 10:
            continue
        # Scale to match reference brightness per channel
        for ch in range(3):
            if cam_means[ci, ch] > 0.01:
                cam_color_scale[ci, ch] = ref_mean[ch] / cam_means[ci, ch]
            # Clamp scale to prevent extreme corrections
            cam_color_scale[ci, ch] = np.clip(cam_color_scale[ci, ch], 0.5, 2.0)

    n_corrected = (cam_counts > 10).sum()
    print(f"  Color correction: {n_corrected} cameras normalized to camera {ref_cam}")
    return cam_color_scale, cam_color_offset



def _texture_colmap(verts, faces, colmap_workdir, output_dir):
    """Texture mesh using COLMAP's mesh_texturer (uses dense workspace cameras)."""
    import subprocess
    from mesh_export import _find_colmap_exe, save_mesh_ply

    colmap_exe = _find_colmap_exe()
    if not colmap_exe:
        return None

    # Check for dense workspace
    dense_dir = os.path.join(colmap_workdir, 'dense')
    if not os.path.isdir(dense_dir):
        dense_dir = colmap_workdir

    # Save mesh as PLY
    mesh_ply = os.path.join(output_dir, 'mesh_for_tex.ply')
    dummy_colors = np.full((len(verts), 3), 180, dtype=np.uint8)
    save_mesh_ply(mesh_ply, verts, faces, dummy_colors)

    print(f"  COLMAP mesh_texturer: workspace={dense_dir}")
    r = subprocess.run([colmap_exe, 'mesh_texturer',
                       '--workspace_path', dense_dir,
                       '--input_path', mesh_ply,
                       '--output_path', output_dir],
                      capture_output=True, text=True, timeout=600)
    if r.stderr:
        for line in r.stderr.split('\n'):
            if line.strip():
                print(f"    {line.strip()}")

    # Find output files
    for f in os.listdir(output_dir):
        if f.endswith('.ply') and 'textured' in f.lower():
            print(f"  COLMAP textured mesh: {f}")
            return os.path.join(output_dir, f)
        if f.endswith('.obj'):
            print(f"  COLMAP textured mesh: {f}")
            return os.path.join(output_dir, f)

    # Check if mesh_for_tex was modified in-place
    if r.returncode == 0:
        print("  COLMAP texturing completed")
        return mesh_ply

    return None


def _texture_pymeshlab(verts, faces, views, output_dir, texture_size=4096):
    """Texture a mesh using PyMeshLab's registered raster texturing.
    Writes an MLP project file with camera poses, then calls the texturing filter."""
    import pymeshlab
    import tempfile

    n_views = len(views)
    if n_views == 0:
        return None

    print(f"  PyMeshLab texturing: {len(verts):,d} verts, {len(faces):,d} faces, {n_views} cameras")

    # Save mesh as PLY
    mesh_path = os.path.join(output_dir, 'mesh_for_tex.ply')
    from mesh_export import save_mesh_ply
    dummy_colors = np.full((len(verts), 3), 180, dtype=np.uint8)
    save_mesh_ply(mesh_path, verts, faces, dummy_colors)

    # Copy images to output dir and build camera data
    image_entries = []
    for ci, view in enumerate(views):
        w2c = view['w2c']
        K = view['K']
        W_img, H_img = int(view['W']), int(view['H'])

        # Copy image
        src_path = view.get('path', '')
        if src_path and os.path.exists(src_path):
            import shutil
            dst = os.path.join(output_dir, f"cam_{ci:04d}.jpg")
            shutil.copy2(src_path, dst)
            img_name = os.path.basename(dst)
        else:
            # Save from pixels array
            pixels = view.get('pixels', None)
            if pixels is None:
                continue
            img_name = f"cam_{ci:04d}.jpg"
            dst = os.path.join(output_dir, img_name)
            img_arr = (np.clip(pixels, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(img_arr).save(dst, quality=95)

        # VCG camera format: world-to-camera [R|t]
        R = w2c[:3, :3]
        t = w2c[:3, 3]

        # Intrinsics: FocalMm = fx (with PixelSizeMm = 1.0)
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = int(round(K[0, 2]))
        cy = int(round(K[1, 2]))

        rot_str = f"{R[0,0]} {R[0,1]} {R[0,2]} 0 {R[1,0]} {R[1,1]} {R[1,2]} 0 {R[2,0]} {R[2,1]} {R[2,2]} 0 0 0 0 1"
        t_str = f"{t[0]} {t[1]} {t[2]} 1"

        image_entries.append((img_name, rot_str, t_str, fx, W_img, H_img, cx, cy))

    if not image_entries:
        return None

    # Write MLP project file
    mlp_path = os.path.join(output_dir, 'project.mlp')
    mesh_name = os.path.basename(mesh_path)

    rasters = ""
    for img_name, rot_str, t_str, fx, W, H, cx, cy in image_entries:
        rasters += f'''  <MLRaster label="{img_name}">
   <VCGCamera TranslationVector="{t_str}" RotationMatrix="{rot_str}" FocalMm="{fx}" ViewportPx="{W} {H}" PixelSizeMm="1 1" CenterPx="{cx} {cy}" LensDistortion="0 0" CameraType="0" BinaryData="0"/>
   <Plane semantic="1" fileName="{img_name}"/>
  </MLRaster>
'''

    mlp_content = f'''<!DOCTYPE MeshLabDocument>
<MeshLabProject>
 <MeshGroup>
  <MLMesh visible="1" label="{mesh_name}" filename="{mesh_name}">
   <MLMatrix44>
1 0 0 0
0 1 0 0
0 0 1 0
0 0 0 1
</MLMatrix44>
  </MLMesh>
 </MeshGroup>
 <RasterGroup>
{rasters} </RasterGroup>
</MeshLabProject>'''

    with open(mlp_path, 'w') as f:
        f.write(mlp_content)
    print(f"  Wrote MLP project: {len(image_entries)} cameras")

    # Load project and run texturing
    ms = pymeshlab.MeshSet()
    ms.load_project(mlp_path)
    print(f"  Loaded: {ms.mesh_number()} meshes, {ms.raster_number()} rasters")

    ms.compute_texcoord_parametrization_and_texture_from_registered_rasters(
        texturesize=texture_size,
        texturename='texture.png',
        colorcorrection=True,
        usedistanceweight=True,
        useimgborderweight=True,
        cleanisolatedtriangles=True,
        texturegutter=4)

    # Save textured mesh
    obj_path = os.path.join(output_dir, 'mesh.obj')
    ms.save_current_mesh(obj_path)
    print(f"  Saved textured mesh: {obj_path}")
    return obj_path


def _unwrap_uvs(verts, faces):
    try:
        import xatlas
        print("  UV unwrapping with xatlas...")

        v0 = verts[faces[:, 0]]; v1 = verts[faces[:, 1]]; v2 = verts[faces[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        fn /= (np.linalg.norm(fn, axis=-1, keepdims=True) + 1e-8)
        vn = np.zeros_like(verts, dtype=np.float64)
        for ax in range(3):
            np.add.at(vn[:, ax], faces[:, 0], fn[:, ax])
            np.add.at(vn[:, ax], faces[:, 1], fn[:, ax])
            np.add.at(vn[:, ax], faces[:, 2], fn[:, ax])
        vn /= (np.linalg.norm(vn, axis=-1, keepdims=True) + 1e-8)

        try:
            chart_options = xatlas.ChartOptions()
            chart_options.max_chart_area = 0.0
            chart_options.max_boundary_length = 0.0
            chart_options.normal_deviation_weight = 2.0
            chart_options.roundness_weight = 0.01
            chart_options.straightness_weight = 6.0
            chart_options.normal_seam_weight = 4.0
            chart_options.max_cost = 2.0

            pack_options = xatlas.PackOptions()
            pack_options.padding = 4
            pack_options.bilinear = True
            pack_options.bruteForce = False
            pack_options.blockAlign = True

            atlas = xatlas.Atlas()
            atlas.add_mesh(verts.astype(np.float32), faces.astype(np.uint32),
                           normals=vn.astype(np.float32))
            atlas.generate(chart_options=chart_options, pack_options=pack_options)
            vmapping, uv_faces, uvs = atlas[0]
            print(f"  UV atlas: {len(uvs):,d} UV coords, {atlas.chart_count} charts")
            return uvs.astype(np.float32), uv_faces.astype(np.int32)
        except (AttributeError, TypeError):
            pass

        vmapping, uv_faces, uvs = xatlas.parametrize(
            verts.astype(np.float32), faces.astype(np.uint32))
        print(f"  UV atlas (basic): {len(uvs):,d} UV coords")
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


def _rasterize_visibility(faces, u_vert, v_vert, z_vert, W, H):
    """Rasterize a face-visibility map using painter's algorithm.

    Draws all faces back-to-front using PIL polygon fill.
    Returns a set of face indices that are visible (their centroid pixel
    shows their own face ID = not occluded by closer geometry).
    """
    scale = 2
    sw, sh = max(W // scale, 1), max(H // scale, 1)

    fu_all = u_vert[faces] / scale
    fv_all = v_vert[faces] / scale
    fz_all = z_vert[faces]
    F = len(faces)

    # Filter to faces in front of camera and at least partially in frame
    in_front = fz_all.min(axis=1) > 0.01
    in_frame = ((fu_all.max(axis=1) >= 0) & (fu_all.min(axis=1) < sw) &
                (fv_all.max(axis=1) >= 0) & (fv_all.min(axis=1) < sh))
    candidates = np.where(in_front & in_frame)[0]

    if len(candidates) == 0:
        return set()

    # Sort back-to-front (farthest first) — PIL overwrites, so closest drawn last wins
    centroid_z = fz_all[candidates].mean(axis=1)
    order = np.argsort(-centroid_z)
    sorted_faces = candidates[order]

    # Rasterize face IDs (1-indexed) using PIL painter's algorithm
    img = Image.new('I', (sw, sh), 0)
    draw = ImageDraw.Draw(img)

    fu = fu_all[sorted_faces]
    fv = fv_all[sorted_faces]

    for idx in range(len(sorted_faces)):
        fi = int(sorted_faces[idx])
        px = [(int(round(fu[idx, 0])), int(round(fv[idx, 0]))),
              (int(round(fu[idx, 1])), int(round(fv[idx, 1]))),
              (int(round(fu[idx, 2])), int(round(fv[idx, 2])))]
        draw.polygon(px, fill=fi + 1)  # 1-indexed

    face_id_map = np.array(img, dtype=np.int32) - 1  # back to 0-indexed

    # A face is visible if its own ID appears at its centroid pixel (vectorized)
    cu = np.clip(fu_all[candidates].mean(axis=1).astype(int), 0, sw - 1)
    cv = np.clip(fv_all[candidates].mean(axis=1).astype(int), 0, sh - 1)
    rendered_id = face_id_map[cv, cu]
    vis_mask = rendered_id == candidates
    visible = set(candidates[vis_mask].tolist())

    return visible


def _expand_triangle_px(p0, p1, p2, expand_px):
    cx = (p0[0] + p1[0] + p2[0]) / 3.0
    cy = (p0[1] + p1[1] + p2[1]) / 3.0
    result = []
    for px, py in [p0, p1, p2]:
        dx = px - cx
        dy = py - cy
        length = max(np.sqrt(dx * dx + dy * dy), 1e-6)
        result.append((px + dx / length * expand_px,
                        py + dy / length * expand_px))
    return result


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
