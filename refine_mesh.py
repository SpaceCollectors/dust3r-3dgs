"""
Multi-View Mesh Refinement via OpenGL Render-and-Compare

Uses the GPU's hardware rasterizer (OpenGL) for instant mesh rendering,
then computes analytical gradients to move vertices and reduce photometric error.

Pipeline per iteration:
  1. OpenGL renders mesh → color image + face_id + barycentric (instant, hardware)
  2. Compare rendered vs photo → per-pixel error map
  3. Scatter pixel errors back to vertices via barycentric weights (analytical gradient)
  4. Move vertices to reduce error + regularization

No differentiable renderer. No CUDA compilation. Just OpenGL + numpy math.
"""

import os
import sys
import time
import argparse
import struct
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation

import moderngl


# ── Mesh I/O ─────────────────────────────────────────────────────────────────

def load_ply_mesh(path):
    import re
    with open(path, 'rb') as f:
        header = b''
        while True:
            line = f.readline()
            header += line
            if b'end_header' in line:
                break
        header_str = header.decode('ascii')
        n_verts = int(re.search(r'element vertex (\d+)', header_str).group(1))
        face_match = re.search(r'element face (\d+)', header_str)
        n_faces = int(face_match.group(1)) if face_match else 0

        verts = np.zeros((n_verts, 3), dtype=np.float32)
        colors = np.zeros((n_verts, 3), dtype=np.uint8)
        for i in range(n_verts):
            data = struct.unpack('<3f3B', f.read(15))
            verts[i] = data[:3]
            colors[i] = data[3:6]

        faces = np.zeros((n_faces, 3), dtype=np.int32)
        for i in range(n_faces):
            n = struct.unpack('<B', f.read(1))[0]
            idx = struct.unpack(f'<{n}i', f.read(4 * n))
            if n >= 3:
                faces[i] = idx[:3]
    return verts, faces, colors


def save_ply_mesh(path, verts, faces, colors):
    n_verts, n_faces = len(verts), len(faces)
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
            f.write(struct.pack('<3f', *verts[i].astype(np.float32)))
            f.write(struct.pack('<3B', *colors[i].astype(np.uint8)))
        for i in range(n_faces):
            f.write(struct.pack('<B3i', 3, *faces[i].astype(np.int32)))


# ── COLMAP Loading ───────────────────────────────────────────────────────────

def load_cameras(data_dir):
    sparse_dir = os.path.join(data_dir, 'sparse', '0')
    images_dir = os.path.join(data_dir, 'images')

    cameras = {}
    with open(os.path.join(sparse_dir, 'cameras.txt')) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            p = line.strip().split()
            cameras[int(p[0])] = (int(p[2]), int(p[3]),
                                   float(p[4]), float(p[5]), float(p[6]), float(p[7]))

    views = []
    with open(os.path.join(sparse_dir, 'images.txt')) as f:
        lines = [l.strip() for l in f if not l.startswith('#')]
    for line in [l for l in lines if l and len(l.split()) >= 9]:
        p = line.split()
        qw, qx, qy, qz = float(p[1]), float(p[2]), float(p[3]), float(p[4])
        tx, ty, tz = float(p[5]), float(p[6]), float(p[7])
        cam = cameras[int(p[8])]
        W, H, fx, fy, cx, cy = cam

        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R
        w2c[:3, 3] = [tx, ty, tz]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        img_path = os.path.join(images_dir, p[9])
        img = np.array(Image.open(img_path).convert('RGB'),
                       dtype=np.float32) / 255.0
        views.append({'w2c': w2c, 'K': K, 'W': W, 'H': H, 'pixels': img, 'path': img_path})
    return views


# ── OpenGL Renderer ──────────────────────────────────────────────────────────

VERTEX_SHADER = """
#version 330
uniform mat4 mvp;
in vec3 in_position;
in vec3 in_color;
in float in_vert_id;
out vec3 v_color;
out float v_vert_id;
void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    v_color = in_color;
    v_vert_id = in_vert_id;
}
"""

# Renders color to attachment 0, vertex IDs + barycentric to attachment 1
FRAGMENT_SHADER = """
#version 330
in vec3 v_color;
in float v_vert_id;
layout(location = 0) out vec4 frag_color;
layout(location = 1) out vec4 frag_info;
void main() {
    frag_color = vec4(v_color, 1.0);
    frag_info = vec4(v_vert_id, 0.0, 0.0, 1.0);
}
"""


class GLRenderer:
    def __init__(self, max_w=1024, max_h=1024):
        self.ctx = moderngl.create_standalone_context()
        self.prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER,
        )
        self.max_w = max_w
        self.max_h = max_h

        # Create FBO with 2 color attachments + depth
        self.color_tex = self.ctx.texture((max_w, max_h), 4, dtype='f4')
        self.info_tex = self.ctx.texture((max_w, max_h), 4, dtype='f4')
        self.depth_rb = self.ctx.depth_renderbuffer((max_w, max_h))
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.color_tex, self.info_tex],
            depth_attachment=self.depth_rb,
        )

    def render(self, verts, faces, colors, w2c, K, W, H):
        """
        Render mesh and return color image + per-pixel vertex IDs.

        Returns:
            color_img: (H, W, 3) float32
            vert_ids: (H, W) int32 — vertex ID at each pixel (-1 = background)
        """
        # Build MVP matrix from w2c + K
        # OpenGL uses column-major, clip space [-1,1]
        near, far = 0.01, 100.0
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Projection matrix (OpenGL NDC)
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = 2 * fx / W
        proj[1, 1] = 2 * fy / H
        proj[0, 2] = 1 - 2 * cx / W
        proj[1, 2] = 2 * cy / H - 1
        proj[2, 2] = -(far + near) / (far - near)
        proj[2, 3] = -2 * far * near / (far - near)
        proj[3, 2] = -1.0

        # OpenGL flips Y compared to OpenCV
        flip = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
        view = flip @ w2c

        mvp = (proj @ view).T  # column-major for OpenGL

        # Prepare vertex data: position(3) + color(3) + vert_id(1) per vertex, indexed by faces
        V = len(verts)
        vert_ids = np.arange(V, dtype=np.float32)

        # Flatten faces to get per-triangle vertex data
        face_verts = verts[faces.ravel()]  # (F*3, 3)
        face_colors = (colors[faces.ravel()] / 255.0).astype(np.float32)  # (F*3, 3)
        face_ids = vert_ids[faces.ravel()]  # (F*3,)

        # Interleave: pos(3) + color(3) + id(1) = 7 floats per vertex
        data = np.empty((len(face_verts), 7), dtype=np.float32)
        data[:, :3] = face_verts
        data[:, 3:6] = face_colors
        data[:, 6] = face_ids

        vbo = self.ctx.buffer(data.tobytes())
        vao = self.ctx.vertex_array(self.prog, [(vbo, '3f 3f 1f', 'in_position', 'in_color', 'in_vert_id')])

        # Render
        self.fbo.use()
        self.ctx.viewport = (0, 0, W, H)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.fbo.clear(0, 0, 0, 0)
        self.prog['mvp'].write(mvp.astype(np.float32).tobytes())
        vao.render()

        # Read back
        color_raw = np.frombuffer(self.color_tex.read(), dtype=np.float32).reshape(H, W, 4)
        info_raw = np.frombuffer(self.info_tex.read(), dtype=np.float32).reshape(H, W, 4)

        # Flip vertically (OpenGL origin is bottom-left)
        color_img = color_raw[::-1, :, :3].copy()
        vert_id_img = info_raw[::-1, :, 0].copy()

        # Where alpha=0 → background → vert_id = -1
        bg_mask = color_raw[::-1, :, 3] < 0.5
        vert_id_img[bg_mask] = -1

        vbo.release()
        vao.release()

        return color_img, vert_id_img.astype(np.int32)


# ── Image Comparison Modes ────────────────────────────────────────────────────

def extract_edges(img):
    """Sobel edge detection on a color image. Returns (H, W) edge magnitude."""
    gray = img.mean(axis=-1)  # (H, W)
    # Sobel kernels
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = kx.T
    from scipy.ndimage import convolve
    gx = convolve(gray, kx)
    gy = convolve(gray, ky)
    return np.sqrt(gx**2 + gy**2)


def extract_high_freq(img, sigma=2.0):
    """High-pass filter: original minus gaussian blur. Returns (H, W, 3)."""
    from scipy.ndimage import gaussian_filter
    blurred = gaussian_filter(img, sigma=(sigma, sigma, 0))
    return img - blurred


def compare_images(rendered, gt, mode='edges'):
    """
    Compare rendered image vs ground truth.

    Modes:
      'color'     - raw RGB L1 difference (includes texture, lighting)
      'edges'     - Sobel edge comparison (geometry-sensitive, ignores flat color)
      'highfreq'  - High-frequency detail comparison (textures + edges)
      'both'      - 50/50 blend of edges + color

    Returns:
      error_map: (H, W) per-pixel error magnitude
    """
    if mode == 'color':
        return np.abs(rendered - gt).mean(axis=-1)
    elif mode == 'edges':
        edges_render = extract_edges(rendered)
        edges_gt = extract_edges(gt)
        return np.abs(edges_render - edges_gt)
    elif mode == 'highfreq':
        hf_render = extract_high_freq(rendered)
        hf_gt = extract_high_freq(gt)
        return np.abs(hf_render - hf_gt).mean(axis=-1)
    elif mode == 'both':
        color_err = np.abs(rendered - gt).mean(axis=-1)
        edge_err = np.abs(extract_edges(rendered) - extract_edges(gt))
        # Normalize both to [0,1] range before blending
        c_max = color_err.max() + 1e-8
        e_max = edge_err.max() + 1e-8
        return 0.5 * (color_err / c_max) + 0.5 * (edge_err / e_max)
    else:
        return np.abs(rendered - gt).mean(axis=-1)


# ── Analytical Gradient Computation ──────────────────────────────────────────

def compute_vertex_gradients(verts, colors, faces, views, renderer, scene_scale,
                             compare_mode='edges'):
    """
    For each camera:
      1. Hardware-render the mesh (instant)
      2. Compare against GT photo → error map
      3. For each pixel, find which vertex is closest (from vert_id buffer)
      4. Accumulate: vertex gets gradient = sum of pixel errors it contributes to

    Returns:
      position_grad: (V, 3) gradient for vertex positions
      color_update: (V, 3) better colors from photos
      avg_error: scalar
    """
    V = len(verts)
    grad_accum = np.zeros((V, 3), dtype=np.float64)
    color_accum = np.zeros((V, 3), dtype=np.float64)
    color_counts = np.zeros(V, dtype=np.float64)
    total_error = 0.0
    total_pixels = 0

    for view in views:
        W, H = view['W'], view['H']
        w2c = view['w2c']
        K = view['K']
        gt = view['pixels']  # (H, W, 3)

        # 1. Hardware render (instant)
        color_img, vert_ids = renderer.render(verts, faces, colors, w2c, K, W, H)

        # 2. Error map (mode-dependent)
        error_mag = compare_images(color_img, gt, mode=compare_mode)  # (H, W)

        # 3. Scatter errors to vertices
        valid = vert_ids >= 0
        if not valid.any():
            continue

        valid_ids = vert_ids[valid]  # flat array of vertex IDs
        valid_errors = error_mag[valid]

        # Accumulate error per vertex
        np.add.at(grad_accum[:, 0], valid_ids, valid_errors)
        np.add.at(color_counts, valid_ids, 1.0)

        # Accumulate GT colors for vertex color update
        valid_gt = gt[valid]  # (N, 3)
        np.add.at(color_accum, (valid_ids, slice(None)), valid_gt)

        total_error += valid_errors.sum()
        total_pixels += valid.sum()

        # 4. Per-vertex error accumulation (no directional gradient here —
        #    direction is computed via finite differences in the main loop)

    avg_error = total_error / max(total_pixels, 1)

    # Normalize color accumulation
    has_color = color_counts > 0
    color_result = colors.copy().astype(np.float64) / 255.0
    color_result[has_color] = color_accum[has_color] / color_counts[has_color, None]

    return grad_accum, (color_result * 255).clip(0, 255).astype(np.uint8), avg_error


# ── Mesh Decimation ──────────────────────────────────────────────────────────

def decimate_mesh(verts, faces, colors, target_faces):
    if len(faces) <= target_faces:
        return verts, faces, colors
    try:
        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        mesh = mesh.simplify_quadric_decimation(target_faces)
        out_v = np.asarray(mesh.vertices, dtype=np.float32)
        out_f = np.asarray(mesh.triangles, dtype=np.int32)
        out_c = (np.asarray(mesh.vertex_colors) * 255).clip(0, 255).astype(np.uint8)
        print(f"  Decimated: {len(verts):,d} -> {len(out_v):,d} verts, "
              f"{len(faces):,d} -> {len(out_f):,d} faces")
        return out_v, out_f, out_c
    except ImportError:
        print("  Warning: open3d not available, skipping decimation")
        return verts, faces, colors


# ── Adaptive Subdivision ─────────────────────────────────────────────────────

def subdivide_high_error(verts, faces, colors, per_vertex_error):
    threshold = np.percentile(per_vertex_error[per_vertex_error > 0], 75) if (per_vertex_error > 0).any() else 0

    edges = {}
    for f in faces:
        for a, b in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]:
            key = (min(a, b), max(a, b))
            err = max(per_vertex_error[a], per_vertex_error[b])
            if key not in edges or err > edges[key]:
                edges[key] = err

    split_edges = {k for k, v in edges.items() if v > threshold}
    if not split_edges:
        return verts, faces, colors, 0

    new_verts = list(verts)
    new_colors = list(colors)
    edge_to_mid = {}

    for (vi, vj) in split_edges:
        mid = (verts[vi] + verts[vj]) / 2
        mid_c = ((colors[vi].astype(np.int32) + colors[vj].astype(np.int32)) // 2).astype(np.uint8)
        edge_to_mid[(vi, vj)] = len(new_verts)
        new_verts.append(mid)
        new_colors.append(mid_c)

    new_faces = []
    for f in faces:
        a, b, c = f
        e_ab = edge_to_mid.get((min(a, b), max(a, b)))
        e_bc = edge_to_mid.get((min(b, c), max(b, c)))
        e_ca = edge_to_mid.get((min(c, a), max(c, a)))
        splits = (e_ab is not None) + (e_bc is not None) + (e_ca is not None)

        if splits == 0:
            new_faces.append([a, b, c])
        elif splits == 3:
            new_faces += [[a, e_ab, e_ca], [e_ab, b, e_bc], [e_ca, e_bc, c], [e_ab, e_bc, e_ca]]
        elif splits == 1:
            if e_ab is not None: new_faces += [[a, e_ab, c], [e_ab, b, c]]
            elif e_bc is not None: new_faces += [[a, b, e_bc], [a, e_bc, c]]
            else: new_faces += [[a, b, e_ca], [b, c, e_ca]]
        else:
            if e_ab is not None and e_bc is not None:
                new_faces += [[a, e_ab, e_bc], [a, e_bc, c], [e_ab, b, e_bc]]
            elif e_bc is not None and e_ca is not None:
                new_faces += [[a, b, e_bc], [a, e_bc, e_ca], [e_bc, c, e_ca]]
            else:
                new_faces += [[e_ab, b, e_ca], [a, e_ab, e_ca], [b, c, e_ca]]

    n_added = len(new_verts) - len(verts)
    return (np.array(new_verts, dtype=np.float32),
            np.array(new_faces, dtype=np.int32),
            np.array(new_colors, dtype=np.uint8), n_added)


# ── Laplacian Smoothing (numpy) ──────────────────────────────────────────────

def laplacian_smooth(verts, faces, strength=0.1):
    """One step of Laplacian smoothing."""
    V = len(verts)
    nbr_sum = np.zeros_like(verts)
    nbr_count = np.zeros(V, dtype=np.float32)

    for f in faces:
        for i in range(3):
            for j in range(3):
                if i != j:
                    nbr_sum[f[i]] += verts[f[j]]
                    nbr_count[f[i]] += 1

    has = nbr_count > 0
    nbr_mean = np.zeros_like(verts)
    nbr_mean[has] = nbr_sum[has] / nbr_count[has, None]

    delta = nbr_mean - verts
    verts[has] += delta[has] * strength
    return verts


# ── Main Refinement ──────────────────────────────────────────────────────────

def refine_mesh(data_dir, mesh_path, output_path, iterations=300,
                lr=0.0005, depth_reg=0.1, smooth_reg=0.01,
                compare_mode='edges', device='cuda'):

    print(f"Loading cameras from {data_dir}")
    views = load_cameras(data_dir)
    C = len(views)
    print(f"  {C} cameras")

    print(f"Loading mesh from {mesh_path}")
    verts, faces, colors = load_ply_mesh(mesh_path)
    print(f"  Original: {len(verts):,d} vertices, {len(faces):,d} faces")

    if len(verts) == 0 or len(faces) == 0:
        save_ply_mesh(output_path, verts, faces, colors)
        return

    # Decimate for fast start
    if len(faces) > 7500:
        verts, faces, colors = decimate_mesh(verts, faces, colors, 5000)

    # Save initial positions for depth regularization
    verts_init = verts.copy()

    # Init OpenGL renderer
    max_w = max(v['W'] for v in views)
    max_h = max(v['H'] for v in views)
    renderer = GLRenderer(max_w, max_h)
    print(f"  OpenGL renderer initialized ({max_w}x{max_h})")

    # Scene scale
    cam_centers = np.array([-np.linalg.inv(v['w2c'])[:3, 3] for v in views])
    from scipy.spatial.distance import pdist
    scene_scale = float(np.median(pdist(cam_centers))) if len(cam_centers) > 1 else 1.0

    preview_dir = os.path.join(os.path.dirname(output_path), 'refine_preview')
    os.makedirs(preview_dir, exist_ok=True)

    # Launch live 3D viewer
    viewer = None
    mesh_handle = None
    try:
        import viser
        viewer = viser.ViserServer(host="127.0.0.1", port=8890)
        print(f"  Live 3D viewer: http://127.0.0.1:8890")

        # Add camera frustums
        for ci, view in enumerate(views):
            c2w = np.linalg.inv(view['w2c'])
            pos = c2w[:3, 3]
            R_c2w = c2w[:3, :3]
            quat = Rotation.from_matrix(R_c2w).as_quat()  # xyzw
            wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])
            viewer.scene.add_frame(f"/cameras/cam_{ci}", wxyz=wxyz, position=pos, axes_length=scene_scale * 0.1)
    except Exception as e:
        print(f"  Viser viewer not available: {e}")

    def update_viewer(verts, faces, colors):
        nonlocal mesh_handle
        if viewer is None:
            return
        try:
            if mesh_handle is not None:
                mesh_handle.remove()
            mesh_handle = viewer.scene.add_mesh_simple(
                "/mesh",
                vertices=verts.astype(np.float32),
                faces=faces.astype(np.int32),
                color=(200, 200, 200),
                vertex_colors=colors.astype(np.uint8),
                flat_shading=True,
            )
        except Exception:
            pass

    t0 = time.time()
    n_passes = 3
    iters_per_pass = iterations // n_passes

    for pass_idx in range(n_passes):
        V = len(verts)
        print(f"\n=== Pass {pass_idx + 1}/{n_passes}: {V:,d} verts, {len(faces):,d} faces, "
              f"{iters_per_pass} iters ===")
        update_viewer(verts, faces, colors)

        for step in range(iters_per_pass):
            global_step = pass_idx * iters_per_pass + step

            # ── Image-space gradient optimization ──
            # For each camera: render, compare, compute per-vertex pixel offset
            # Move vertices along camera's local right/up axes proportional to error

            ci = step % C
            view = views[ci]
            W_v, H_v = view['W'], view['H']
            w2c = view['w2c']
            K = view['K']
            fx, fy = K[0, 0], K[1, 1]

            # Camera local axes in world space
            R_cam = w2c[:3, :3]  # world-to-camera rotation
            cam_right = R_cam[0, :]  # world-space right direction
            cam_up = -R_cam[1, :]    # world-space up (negated because y is down in OpenCV)
            cam_fwd = R_cam[2, :]    # world-space forward (into screen)

            # Render current mesh
            color_img, vert_ids = renderer.render(verts, faces, colors, w2c, K, W_v, H_v)
            gt = view['pixels']
            error_map = compare_images(color_img, gt, mode=compare_mode)

            # For each visible vertex, compute error-weighted pixel offset
            # This tells us: "where in the image is the error, relative to where
            # this vertex projects?" → that offset, mapped to world space, is the gradient
            valid = vert_ids >= 0
            if not valid.any():
                continue

            pixel_rows, pixel_cols = np.where(valid)
            pixel_vids = vert_ids[pixel_rows, pixel_cols]
            pixel_errs = error_map[pixel_rows, pixel_cols]

            # Project vertices to get their expected pixel position
            pts_cam = (R_cam @ verts.T).T + w2c[:3, 3]
            z = np.clip(pts_cam[:, 2], 0.01, None)
            u_proj = pts_cam[:, 0] / z * fx + K[0, 2]
            v_proj = pts_cam[:, 1] / z * fy + K[1, 2]

            # Per-vertex: error-weighted centroid of pixel offsets
            # offset = (pixel_pos - vertex_projection) * error
            # This points from vertex toward where the error is concentrated
            dx_accum = np.zeros(V, dtype=np.float64)  # horizontal pixel offset
            dy_accum = np.zeros(V, dtype=np.float64)  # vertical pixel offset
            err_accum = np.zeros(V, dtype=np.float64)
            count = np.zeros(V, dtype=np.float64)

            dx_px = pixel_cols.astype(np.float64) - u_proj[pixel_vids]  # pixel offset from projection
            dy_px = pixel_rows.astype(np.float64) - v_proj[pixel_vids]

            np.add.at(dx_accum, pixel_vids, dx_px * pixel_errs)
            np.add.at(dy_accum, pixel_vids, dy_px * pixel_errs)
            np.add.at(err_accum, pixel_vids, pixel_errs)
            np.add.at(count, pixel_vids, 1.0)

            has = count > 0
            avg_error = err_accum[has].sum() / count[has].sum() if has.any() else 0.0

            # Normalize
            dx_accum[has] /= (count[has] + 1e-8)
            dy_accum[has] /= (count[has] + 1e-8)

            # Convert pixel offsets to world-space movement
            # dx pixels → move along cam_right by (dx / fx * z) meters
            # dy pixels → move along cam_up by (dy / fy * z) meters
            move_right = dx_accum / fx * z  # (V,) meters along camera right
            move_up = dy_accum / fy * z     # (V,) meters along camera up

            # Scale by error magnitude (larger error = larger movement)
            err_scale = np.clip(err_accum / (count + 1e-8), 0, None)

            # World-space displacement: proportional to pixel offset AND error
            displacement = (move_right[:, None] * cam_right[None, :] +
                           move_up[:, None] * cam_up[None, :]) * lr

            # Clamp displacement to prevent explosions
            max_move = scene_scale * 0.01
            disp_mag = np.linalg.norm(displacement, axis=-1, keepdims=True)
            displacement = np.where(disp_mag > max_move,
                                     displacement * max_move / (disp_mag + 1e-8),
                                     displacement)

            verts[has] += displacement[has].astype(np.float32)

            # Depth regularization
            verts += (depth_reg * lr * (verts_init - verts)).astype(np.float32)

            # Laplacian smoothing
            if smooth_reg > 0:
                verts = laplacian_smooth(verts, faces, strength=smooth_reg)

            # Update vertex colors periodically
            if step % 50 == 0:
                _, new_colors, _ = compute_vertex_gradients(
                    verts, colors, faces, views, renderer, scene_scale, compare_mode)
                colors = new_colors

            if step % 10 == 0 or step == iters_per_pass - 1:
                print(f"[{global_step:4d}/{iterations}] error={avg_error:.6f} "
                      f"verts={V:,d} time={time.time() - t0:.1f}s")
                update_viewer(verts, faces, colors)

            if step % 25 == 0:
                err_vis = (error_map / (error_map.max() + 1e-8) * 255).astype(np.uint8)
                import matplotlib.cm as cm
                Image.fromarray((cm.jet(err_vis)[:, :, :3] * 255).astype(np.uint8)).save(
                    os.path.join(preview_dir, 'error_heatmap.png'))
                Image.fromarray((color_img * 255).clip(0, 255).astype(np.uint8)).save(
                    os.path.join(preview_dir, 'render.png'))

        # Adaptive subdivision between passes
        if pass_idx < n_passes - 1:
            print("  Computing error for subdivision...")
            _, _, per_vert_err_map = compute_vertex_gradients(verts, colors, faces, views, renderer, scene_scale, compare_mode)

            # Per-vertex error from the grad accumulator
            per_vert_error = np.zeros(len(verts))
            for view in views:
                _, vert_ids = renderer.render(verts, faces, colors, view['w2c'], view['K'], view['W'], view['H'])
                gt = view['pixels']
                color_img, _ = renderer.render(verts, faces, colors, view['w2c'], view['K'], view['W'], view['H'])
                error_mag = np.abs(color_img - gt).mean(axis=-1)
                valid = vert_ids >= 0
                if valid.any():
                    np.add.at(per_vert_error, vert_ids[valid], error_mag[valid])

            verts, faces, colors, n_added = subdivide_high_error(verts, faces, colors, per_vert_error)
            verts_init = np.concatenate([verts_init, verts[len(verts_init):]], axis=0) if len(verts) > len(verts_init) else verts_init
            # Extend verts_init for new vertices
            if len(verts) > len(verts_init):
                extra = verts[len(verts_init):]
                verts_init = np.concatenate([verts_init, extra], axis=0)
            print(f"  +{n_added:,d} vertices -> {len(verts):,d} total, {len(faces):,d} faces")

    save_ply_mesh(output_path, verts, faces, colors)
    print(f"\nDone. {time.time() - t0:.1f}s. Saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--mesh_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--depth_reg', type=float, default=0.1)
    parser.add_argument('--smooth_reg', type=float, default=0.01)
    parser.add_argument('--compare_mode', type=str, default='edges',
                        choices=['color', 'edges', 'highfreq', 'both'])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    refine_mesh(**vars(args))
