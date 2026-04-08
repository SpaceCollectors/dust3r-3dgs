"""
Desktop 3D Reconstruction App — Dear ImGui + OpenGL

Native-feeling desktop app with:
  - 3D viewport with orbit camera (point cloud / mesh live view)
  - Side panel with pipeline controls
  - Real-time progress during reconstruction and refinement
  - Direct integration with DUSt3R/MASt3R/VGGT backends
"""

import os
import sys
import time
import math
import threading
import numpy as np
from pathlib import Path

import glfw
import OpenGL.GL as gl
import imgui
from imgui.integrations.glfw import GlfwRenderer

# Add our repos to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAST3R_DIR = os.path.join(SCRIPT_DIR, 'mast3r')
VGGT_DIR = os.path.join(SCRIPT_DIR, 'vggt')
sys.path.insert(0, MAST3R_DIR)
sys.path.insert(0, os.path.join(MAST3R_DIR, 'dust3r'))
sys.path.insert(0, VGGT_DIR)
sys.path.insert(0, SCRIPT_DIR)


# ── 3D Camera ────────────────────────────────────────────────────────────────

class OrbitCamera:
    def __init__(self):
        self.target = np.array([0, 0, 0], dtype=np.float32)
        self.distance = 2.0
        self.yaw = 0.0      # radians
        self.pitch = 0.3     # radians
        self.fov = 60.0

    def get_position(self):
        """Get camera eye position in world space."""
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        return self.target + self.distance * np.array([
            sy * cp, sp, cy * cp
        ], dtype=np.float32)

    def get_view_matrix(self):
        """Compute 4x4 view matrix."""
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)

        eye = self.target + self.distance * np.array([
            sy * cp, sp, cy * cp
        ], dtype=np.float32)

        forward = self.target - eye
        forward /= np.linalg.norm(forward) + 1e-8

        world_up = np.array([0, 1, 0], dtype=np.float32)
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right) + 1e-8
        up = np.cross(right, forward)

        view = np.eye(4, dtype=np.float32)
        view[0, :3] = right
        view[1, :3] = up
        view[2, :3] = -forward
        view[0, 3] = -np.dot(right, eye)
        view[1, 3] = -np.dot(up, eye)
        view[2, 3] = np.dot(forward, eye)
        return view

    def get_projection_matrix(self, aspect):
        """Compute 4x4 perspective projection matrix."""
        near, far = 0.01, 100.0
        f = 1.0 / math.tan(math.radians(self.fov) / 2.0)
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = 2 * far * near / (near - far)
        proj[3, 2] = -1.0
        return proj

    def orbit(self, dx, dy):
        self.yaw -= dx * 0.01
        self.pitch += dy * 0.01
        self.pitch = max(-math.pi / 2 + 0.01, min(math.pi / 2 - 0.01, self.pitch))

    def zoom(self, delta):
        self.distance *= (1.0 - delta * 0.1)
        self.distance = max(0.01, self.distance)

    def pan(self, dx, dy):
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        right = np.array([cy, 0, -sy], dtype=np.float32)
        up = np.array([0, 1, 0], dtype=np.float32)
        scale = self.distance * 0.002
        self.target += right * dx * scale + up * dy * scale


# ── OpenGL Point Cloud / Mesh Renderer ───────────────────────────────────────

VERT_SHADER = """
#version 330
uniform mat4 mvp;
in vec3 position;
in vec3 color;
out vec3 v_color;
void main() {
    gl_Position = mvp * vec4(position, 1.0);
    gl_PointSize = 3.0;
    v_color = color;
}
"""

FRAG_SHADER = """
#version 330
in vec3 v_color;
out vec4 frag_color;
void main() {
    frag_color = vec4(v_color, 1.0);
}
"""


class DebugImages:
    """Manages debug images displayed in ImGui via OpenGL textures."""

    def __init__(self):
        self._images = {}       # name -> numpy (H,W,3) uint8
        self._textures = {}     # name -> GL texture ID
        self._pending = {}      # name -> numpy (to upload on main thread)
        import threading
        self._lock = threading.Lock()

    def set_image(self, name, img_np):
        """Queue an image for display (thread-safe). img_np: (H,W,3) uint8."""
        with self._lock:
            self._pending[name] = img_np.copy()

    def clear(self):
        with self._lock:
            self._pending.clear()
            self._images.clear()

    def flush(self):
        """Upload pending images to GL textures (main thread only)."""
        with self._lock:
            pending = dict(self._pending)
            self._pending.clear()

        for name, img in pending.items():
            self._images[name] = img
            h, w = img.shape[:2]

            if name not in self._textures:
                tex_id = gl.glGenTextures(1)
                self._textures[name] = tex_id
            else:
                tex_id = self._textures[name]

            gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, w, h, 0,
                           gl.GL_RGB, gl.GL_UNSIGNED_BYTE, img)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def draw_window(self, title="Debug Views"):
        """Draw an ImGui window showing all debug images."""
        if not self._textures:
            return

        imgui.set_next_window_size(600, 400, condition=imgui.FIRST_USE_EVER)
        expanded, opened = imgui.begin(title, True)
        if not opened:
            imgui.end()
            return

        for name, tex_id in self._textures.items():
            if name in self._images:
                img = self._images[name]
                h, w = img.shape[:2]
                # Scale to fit width
                avail_w = imgui.get_content_region_available_width()
                scale = min(1.0, avail_w / w)
                imgui.text(name)
                imgui.image(tex_id, w * scale, h * scale)
                imgui.separator()

        imgui.end()


class GLScene:
    """Manages OpenGL buffers for point clouds and meshes.
    Thread-safe: background threads call set_points/set_mesh which queue data.
    Main thread calls flush_pending() to upload to GL."""

    def __init__(self):
        self.program = self._compile_shader()
        self.mvp_loc = gl.glGetUniformLocation(self.program, "mvp")
        self.point_vao = None
        self.point_vbo = None
        self.point_count = 0
        self.mesh_vao = None
        self.mesh_vbo = None
        self.mesh_ebo = None
        self.mesh_vertex_count = 0
        self.mesh_face_count = 0
        self.cam_vao = None
        self.cam_vbo = None
        self.cam_line_count = 0
        self.grid_vao = None
        self.grid_vbo = None
        self.grid_line_count = 0
        # Pending uploads from background threads
        self._pending_points = None
        self._pending_mesh = None
        self._pending_cams = None
        self._pending_grid = True  # build grid on first flush
        import threading
        self._lock = threading.Lock()

    def _compile_shader(self):
        vert = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        gl.glShaderSource(vert, VERT_SHADER)
        gl.glCompileShader(vert)

        frag = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        gl.glShaderSource(frag, FRAG_SHADER)
        gl.glCompileShader(frag)

        prog = gl.glCreateProgram()
        gl.glAttachShader(prog, vert)
        gl.glAttachShader(prog, frag)
        gl.glLinkProgram(prog)
        gl.glDeleteShader(vert)
        gl.glDeleteShader(frag)
        return prog

    def set_points(self, points, colors):
        """Queue point cloud for upload (thread-safe)."""
        with self._lock:
            self._pending_points = (points.copy(), colors.copy())

    def set_mesh(self, verts, faces, colors):
        """Queue mesh for upload (thread-safe). Also computes normals/shading."""
        verts = verts.copy()
        faces = faces.copy()
        colors = colors.copy()

        # Compute vertex normals for normal-map and shaded views
        V = len(verts)
        vert_normals = np.zeros((V, 3), dtype=np.float32)
        if len(faces) > 0:
            v0 = verts[faces[:, 0]]; v1 = verts[faces[:, 1]]; v2 = verts[faces[:, 2]]
            fn = np.cross(v1 - v0, v2 - v0)
            fn /= (np.linalg.norm(fn, axis=-1, keepdims=True) + 1e-8)
            for ax in range(3):
                np.add.at(vert_normals[:, ax], faces[:, 0], fn[:, ax].astype(np.float32))
                np.add.at(vert_normals[:, ax], faces[:, 1], fn[:, ax].astype(np.float32))
                np.add.at(vert_normals[:, ax], faces[:, 2], fn[:, ax].astype(np.float32))
            vert_normals /= (np.linalg.norm(vert_normals, axis=-1, keepdims=True) + 1e-8)

        # Normal-map colors: encode normal as RGB (n+1)/2 * 255
        normal_colors = ((vert_normals + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)

        # Shaded colors: simple directional light from above-right
        light_dir = np.array([0.3, 0.7, 0.5], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)
        ndotl = (vert_normals * light_dir[None, :]).sum(axis=-1).clip(0, 1)
        ambient = 0.2
        shading = (ambient + (1 - ambient) * ndotl)
        # Pure grey Lambert shading (no texture/vertex color)
        grey = 200.0
        shaded_colors = (grey * shading[:, None]).clip(0, 255).astype(np.uint8)
        shaded_colors = np.broadcast_to(shaded_colors, (V, 3)).copy()

        with self._lock:
            self._pending_mesh = (verts, faces, colors, normal_colors, shaded_colors)


    def _upload_points(self, points, colors):
        if len(points) == 0:
            self.point_count = 0
            return
        data = np.empty((len(points), 6), dtype=np.float32)
        data[:, :3] = points.astype(np.float32)
        data[:, 3:6] = colors.astype(np.float32) / 255.0

        if self.point_vao is None:
            self.point_vao = gl.glGenVertexArrays(1)
            self.point_vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.point_vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.point_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, 24, None)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, False, 24, gl.ctypes.c_void_p(12))
        gl.glBindVertexArray(0)
        self.point_count = len(points)

    def _upload_mesh(self, verts, faces, colors):
        if len(verts) == 0 or len(faces) == 0:
            self.mesh_face_count = 0
            return
        data = np.empty((len(verts), 6), dtype=np.float32)
        data[:, :3] = verts.astype(np.float32)
        data[:, 3:6] = colors.astype(np.float32) / 255.0
        indices = faces.astype(np.uint32).ravel()

        if self.mesh_vao is None:
            self.mesh_vao = gl.glGenVertexArrays(1)
            self.mesh_vbo = gl.glGenBuffers(1)
            self.mesh_ebo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.mesh_vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.mesh_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, 24, None)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, False, 24, gl.ctypes.c_void_p(12))
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.mesh_ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_DYNAMIC_DRAW)
        gl.glBindVertexArray(0)
        self.mesh_vertex_count = len(verts)
        self.mesh_face_count = len(faces)

    def draw(self, mvp_grid, mvp_scene, draw_mode='points', camera_pos=None):
        gl.glUseProgram(self.program)

        # Grid + axes: fixed orientation
        gl.glUniformMatrix4fv(self.mvp_loc, 1, gl.GL_TRUE, mvp_grid)
        if self.grid_line_count > 0:
            gl.glBindVertexArray(self.grid_vao)
            gl.glDrawArrays(gl.GL_LINES, 0, self.grid_line_count)

        # Swap mesh colors based on draw mode
        if self.mesh_face_count > 0 and getattr(self, '_mesh_verts', None) is not None:
            if draw_mode == 'normals':
                alt = getattr(self, '_normal_colors', None)
                if alt is not None:
                    self._upload_mesh(self._mesh_verts, self._mesh_faces, alt)
                draw_mode = 'mesh'
            elif draw_mode == 'shaded':
                # Compute shading from current camera position (headlamp effect)
                self._compute_shaded_from_camera(camera_pos)
                draw_mode = 'mesh'
            elif draw_mode in ('mesh', 'wireframe'):
                base = getattr(self, '_base_colors', None)
                if base is not None:
                    self._upload_mesh(self._mesh_verts, self._mesh_faces, base)

        # Scene content: rotated
        gl.glUniformMatrix4fv(self.mvp_loc, 1, gl.GL_TRUE, mvp_scene)

        if draw_mode == 'points' and self.point_count > 0:
            gl.glBindVertexArray(self.point_vao)
            gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
            gl.glDrawArrays(gl.GL_POINTS, 0, self.point_count)

        if draw_mode == 'mesh' and self.mesh_face_count > 0:
            gl.glBindVertexArray(self.mesh_vao)
            gl.glDrawElements(gl.GL_TRIANGLES, self.mesh_face_count * 3,
                              gl.GL_UNSIGNED_INT, None)

        if draw_mode == 'wireframe' and self.mesh_face_count > 0:
            gl.glBindVertexArray(self.mesh_vao)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            gl.glDrawElements(gl.GL_TRIANGLES, self.mesh_face_count * 3,
                              gl.GL_UNSIGNED_INT, None)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

        # Cameras: same rotation as scene
        if self.cam_line_count > 0:
            gl.glBindVertexArray(self.cam_vao)
            gl.glDrawArrays(gl.GL_LINES, 0, self.cam_line_count)

        gl.glBindVertexArray(0)

    def set_cameras(self, cam_poses, scale=0.1):
        """Queue camera frustums for display. cam_poses: list of 4x4 c2w matrices."""
        data = []
        for c2w in cam_poses:
            o = c2w[:3, 3].astype(np.float32)
            r = c2w[:3, 0].astype(np.float32) * scale
            u = c2w[:3, 1].astype(np.float32) * scale
            f = c2w[:3, 2].astype(np.float32) * scale

            tl = o + f - r * 0.5 + u * 0.4
            tr = o + f + r * 0.5 + u * 0.4
            bl = o + f - r * 0.5 - u * 0.4
            br = o + f + r * 0.5 - u * 0.4

            green = np.array([0, 1, 0], dtype=np.float32)
            for p in [tl, tr, bl, br]:
                data.append(np.concatenate([o, green]))
                data.append(np.concatenate([p, green]))
            for a, b in [(tl, tr), (tr, br), (br, bl), (bl, tl)]:
                data.append(np.concatenate([a, green * 0.8]))
                data.append(np.concatenate([b, green * 0.8]))

        if not data:
            return
        print(f"  Camera frustums: {len(cam_poses)} cameras, {len(data)} line verts")
        with self._lock:
            self._pending_cams = np.array(data, dtype=np.float32)

    def flush_pending(self):
        """Upload pending data to GL (must be called from main/GL thread)."""
        # Build grid on first call
        if self._pending_grid:
            self._pending_grid = False
            self._build_grid()

        with self._lock:
            pts = self._pending_points
            msh = self._pending_mesh
            cams = getattr(self, '_pending_cams', None)
            self._pending_points = None
            self._pending_mesh = None
            self._pending_cams = None

        if pts is not None:
            self._upload_points(pts[0], pts[1])
        if msh is not None:
            self._upload_mesh(msh[0], msh[1], msh[2])
            # Store alternate color buffers
            if len(msh) >= 5:
                self._normal_colors = msh[3]
                self._shaded_colors = msh[4]
                self._base_colors = msh[2]
                self._mesh_verts = msh[0]
                self._mesh_faces = msh[1]
        if cams is not None:
            self._upload_cams(cams)

    def _build_grid(self, size=2.0, divisions=20):
        """Build ground grid + RGB axis lines."""
        lines = []
        half = size / 2.0
        step = size / divisions
        grid_color = [0.3, 0.3, 0.3]

        # Ground grid (XZ plane, Y=0)
        for i in range(divisions + 1):
            t = -half + i * step
            # Lines along X
            lines.append(([-half, 0, t], grid_color))
            lines.append(([half, 0, t], grid_color))
            # Lines along Z
            lines.append(([t, 0, -half], grid_color))
            lines.append(([t, 0, half], grid_color))

        # Axis lines (thicker effect via slight offset)
        axis_len = size * 0.4
        # X axis = red
        lines.append(([0, 0, 0], [1, 0, 0]))
        lines.append(([axis_len, 0, 0], [1, 0, 0]))
        # Y axis = green
        lines.append(([0, 0, 0], [0, 1, 0]))
        lines.append(([0, axis_len, 0], [0, 1, 0]))
        # Z axis = blue
        lines.append(([0, 0, 0], [0, 0, 1]))
        lines.append(([0, 0, axis_len], [0, 0, 1]))

        data = np.array([(list(p) + list(c)) for p, c in lines], dtype=np.float32)
        self.grid_vao = gl.glGenVertexArrays(1)
        self.grid_vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.grid_vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.grid_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, 24, None)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, False, 24, gl.ctypes.c_void_p(12))
        gl.glBindVertexArray(0)
        self.grid_line_count = len(data)

    def _compute_shaded_from_camera(self, camera_pos):
        """Compute grey Lambert shading with light at camera position."""
        verts = getattr(self, '_mesh_verts', None)
        faces = getattr(self, '_mesh_faces', None)
        if verts is None or faces is None or len(faces) == 0:
            return

        V = len(verts)

        # Vertex normals
        v0 = verts[faces[:, 0]]; v1 = verts[faces[:, 1]]; v2 = verts[faces[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        fn /= (np.linalg.norm(fn, axis=-1, keepdims=True) + 1e-8)
        vert_normals = np.zeros((V, 3), dtype=np.float32)
        for ax in range(3):
            np.add.at(vert_normals[:, ax], faces[:, 0], fn[:, ax].astype(np.float32))
            np.add.at(vert_normals[:, ax], faces[:, 1], fn[:, ax].astype(np.float32))
            np.add.at(vert_normals[:, ax], faces[:, 2], fn[:, ax].astype(np.float32))
        vert_normals /= (np.linalg.norm(vert_normals, axis=-1, keepdims=True) + 1e-8)

        # Light direction = from vertex toward camera
        if camera_pos is not None:
            light_dirs = camera_pos[None, :] - verts
            light_dirs /= (np.linalg.norm(light_dirs, axis=-1, keepdims=True) + 1e-8)
            ndotl = (vert_normals * light_dirs).sum(axis=-1).clip(0, 1)
        else:
            ndotl = np.full(V, 0.7, dtype=np.float32)

        ambient = 0.15
        brightness = ambient + (1.0 - ambient) * ndotl
        grey = (brightness * 220).clip(0, 255).astype(np.uint8)
        shaded = np.stack([grey, grey, grey], axis=-1)

        self._upload_mesh(verts, faces, shaded)

    def _upload_cams(self, data):
        # data is already (N, 6) float32 numpy array
        if not hasattr(self, 'cam_vao') or self.cam_vao is None:
            self.cam_vao = gl.glGenVertexArrays(1)
            self.cam_vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.cam_vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.cam_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, 24, None)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, False, 24, gl.ctypes.c_void_p(12))
        gl.glBindVertexArray(0)
        self.cam_line_count = len(data)


# ── App State ────────────────────────────────────────────────────────────────

class AppState:
    def __init__(self):
        # Images
        self.image_paths = []
        self.image_dir = ""

        # Backend
        self.backend_idx = 1  # 0=dust3r, 1=mast3r, 2=vggt
        self.backends = ['dust3r', 'mast3r', 'vggt']

        # Reconstruction
        self.scene = None
        self.reconstructing = False
        self.recon_progress = ""
        self.recon_thread = None

        # Point cloud / Mesh
        self.has_points = False
        self.has_mesh = False
        self.draw_mode = 0  # 0=points, 1=mesh, 2=wireframe, 3=normals, 4=shaded
        self.draw_modes = ['points', 'mesh', 'wireframe', 'normals', 'shaded']

        # Refinement
        self.refining = False
        self.refine_progress = ""
        self.refine_thread = None

        # Export
        self.export_path = ""

        # Reconstruction params
        self.niter1 = 300
        self.niter2 = 300
        self.min_conf = 2.0
        self.optim_level = 2  # 0=coarse, 1=refine, 2=refine+depth

        # Refine params
        self.refine_iters = 300
        self.refine_lr = 0.0005
        self.refine_depth_reg = 0.1
        self.refine_smooth_reg = 0.01
        self.compare_mode = 0  # 0=edges, 1=highfreq, 2=color, 3=both
        self.compare_modes = ['edges', 'highfreq', 'color', 'both', 'normals']

        # Mesh data (numpy, for saving)
        self.mesh_data = None  # (verts, faces, colors)
        self.target_faces = 100000  # target face count for dense mesh

        # Scene orientation transform (applied in shader)
        self.scene_rot_x = 0.0  # degrees
        self.scene_rot_y = 0.0
        self.scene_rot_z = 0.0
        self.scene_flip_y = False

        # Stop flag
        self.stop_requested = False

        # Status
        self.status = "Ready. Load images to begin."
        self.error_msg = ""


# ── Pipeline Integration ─────────────────────────────────────────────────────

def run_reconstruction(state, scene_gl):
    """Run reconstruction in background thread."""
    import torch
    import copy

    state.reconstructing = True
    state.status = "Loading model..."

    try:
        backend = state.backends[state.backend_idx]

        if backend == 'vggt':
            from vggt.models.vggt import VGGT
            from vggt.utils.load_fn import load_and_preprocess_images
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
            from vggt.utils.geometry import unproject_depth_map_to_point_map

            state.status = "Loading VGGT model..."
            model = VGGT.from_pretrained("facebook/VGGT-1B").to('cuda')
            model.eval()

            state.status = "Running VGGT inference..."
            images = load_and_preprocess_images(state.image_paths).to('cuda')
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=dtype):
                    predictions = model(images)

            pose_enc = predictions["pose_enc"]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            depth_map = predictions["depth"].squeeze(0).cpu().numpy()
            depth_conf = predictions["depth_conf"].squeeze(0).cpu().numpy()
            extrinsic = extrinsic.squeeze(0).cpu().numpy()
            intrinsic = intrinsic.squeeze(0).cpu().numpy()

            if depth_map.ndim == 3:
                depth_map = depth_map[..., None]
            pts3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

            imgs_np = images.cpu().numpy().transpose(0, 2, 3, 1)

            # Build simple scene data
            all_pts = []
            all_colors = []
            for i in range(len(state.image_paths)):
                mask = depth_conf[i] > state.min_conf
                all_pts.append(pts3d[i][mask])
                all_colors.append((imgs_np[i][mask] * 255).astype(np.uint8))

            if all_pts:
                points = np.concatenate(all_pts, axis=0)
                colors = np.concatenate(all_colors, axis=0)
                if len(points) > 200000:
                    idx = np.random.choice(len(points), 200000, replace=False)
                    points, colors = points[idx], colors[idx]
                scene_gl.set_points(points, colors)
                state.has_points = True

            # Build VGGTScene for downstream use (mesh gen, export)
            from app import VGGTScene
            from PIL import Image as PILImage
            from PIL.ImageOps import exif_transpose
            orig_sizes = []
            for p in state.image_paths:
                im = exif_transpose(PILImage.open(p)).convert('RGB')
                orig_sizes.append(im.size)

            imgs_list = [imgs_np[i] for i in range(len(state.image_paths))]
            pts3d_all = [pts3d[i] for i in range(len(state.image_paths))]
            conf_all = [depth_conf[i] for i in range(len(state.image_paths))]
            state.scene = VGGTScene(imgs_list, extrinsic, intrinsic,
                                    pts3d_all, conf_all, orig_sizes)

            # VGGT cameras: extrinsic is w2c (3x4), invert to c2w
            cam_poses = []
            for i in range(len(extrinsic)):
                w2c_44 = np.eye(4, dtype=np.float32)
                w2c_44[:3, :] = extrinsic[i]
                cam_poses.append(np.linalg.inv(w2c_44))
            if cam_poses:
                ext = np.linalg.norm(points.max(axis=0) - points.min(axis=0)) if len(points) > 0 else 1.0
                scene_gl.set_cameras(cam_poses, scale=float(ext) * 0.05)

            del model, predictions
            torch.cuda.empty_cache()

        else:
            # DUSt3R / MASt3R
            from dust3r.utils.image import load_images as dust3r_load_images
            from dust3r.utils.device import to_numpy

            if backend == 'dust3r':
                from dust3r.model import AsymmetricCroCo3DStereo
                from dust3r.inference import inference as dust3r_inference
                from dust3r.image_pairs import make_pairs
                from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

                state.status = "Loading DUSt3R model..."
                model = AsymmetricCroCo3DStereo.from_pretrained(
                    "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt").to('cuda')
                model.eval()

                state.status = "Running DUSt3R inference..."
                imgs = dust3r_load_images(state.image_paths, size=512, verbose=True,
                                          patch_size=model.patch_size)
                if len(imgs) == 1:
                    imgs = [imgs[0], copy.deepcopy(imgs[0])]
                    imgs[1]['idx'] = 1

                pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
                output = dust3r_inference(pairs, model, 'cuda', batch_size=1)

                mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
                scene = global_aligner(output, device='cuda', mode=mode)
                if mode == GlobalAlignerMode.PointCloudOptimizer:
                    state.status = "Aligning..."
                    scene.compute_global_alignment(init='mst', niter=state.niter1,
                                                   schedule='linear', lr=0.01)
                state.scene = scene

            else:  # mast3r
                from mast3r.model import AsymmetricMASt3R
                from mast3r.image_pairs import make_pairs
                from mast3r.cloud_opt.sparse_ga import sparse_global_alignment

                state.status = "Loading MASt3R model..."
                model = AsymmetricMASt3R.from_pretrained(
                    "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric").to('cuda')
                model.eval()

                state.status = "Running MASt3R inference..."
                imgs = dust3r_load_images(state.image_paths, size=512, verbose=True,
                                          patch_size=model.patch_size)
                paths = list(state.image_paths)
                if len(imgs) == 1:
                    imgs = [imgs[0], copy.deepcopy(imgs[0])]
                    imgs[1]['idx'] = 1
                    paths = [paths[0], paths[0] + '_2']

                pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)

                import tempfile
                cache_dir = tempfile.mkdtemp()
                optim_levels = ['coarse', 'refine', 'refine+depth']
                opt = optim_levels[state.optim_level]

                state.status = "Sparse global alignment..."
                scene = sparse_global_alignment(
                    paths, pairs, cache_dir, model,
                    lr1=0.07, niter1=state.niter1,
                    lr2=0.014, niter2=0 if opt == 'coarse' else state.niter2,
                    device='cuda', opt_depth='depth' in opt,
                )
                state.scene = scene

            # Extract point cloud for display
            state.status = "Extracting point cloud..."
            scene = state.scene
            rgbimg = scene.imgs
            pts3d_raw = to_numpy(scene.get_pts3d()) if not hasattr(scene, 'canonical_paths') else None

            if hasattr(scene, 'canonical_paths'):
                # MASt3R SparseGA
                pts3d_dense, _, confs = scene.get_dense_pts3d(clean_depth=True)
                all_pts, all_colors = [], []
                for i in range(len(rgbimg)):
                    H, W = rgbimg[i].shape[:2]
                    p = to_numpy(pts3d_dense[i]).reshape(H, W, 3)
                    c = to_numpy(confs[i])
                    mask = c > state.min_conf
                    all_pts.append(p[mask])
                    all_colors.append((rgbimg[i][mask] * 255).astype(np.uint8))
            else:
                # DUSt3R
                all_pts, all_colors = [], []
                for i in range(len(rgbimg)):
                    p = pts3d_raw[i]
                    all_pts.append(p.reshape(-1, 3))
                    all_colors.append((rgbimg[i].reshape(-1, 3) * 255).astype(np.uint8))

            if all_pts:
                points = np.concatenate(all_pts, axis=0)
                colors = np.concatenate(all_colors, axis=0)

                # Subsample if too many
                if len(points) > 200000:
                    idx = np.random.choice(len(points), 200000, replace=False)
                    points, colors = points[idx], colors[idx]

                scene_gl.set_points(points, colors)
                state.has_points = True

            # Show camera frustums
            try:
                cam_poses = []
                if state.scene is not None:
                    c2w_all = state.scene.get_im_poses().cpu().numpy()
                    for i in range(len(c2w_all)):
                        cam_poses.append(c2w_all[i])
                if cam_poses:
                    ext = np.linalg.norm(points.max(axis=0) - points.min(axis=0)) if len(points) > 0 else 1.0
                    scene_gl.set_cameras(cam_poses, scale=float(ext) * 0.05)
            except Exception:
                pass

            del model
            torch.cuda.empty_cache()

        # Auto-center camera on point cloud
        if state.has_points:
            try:
                center = points.mean(axis=0)
                extent = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
                camera.target = center.astype(np.float32)
                camera.distance = float(extent * 1.5)
            except Exception:
                pass

        state.status = "Reconstruction complete!"
        state.error_msg = ""

    except Exception as e:
        state.error_msg = str(e)
        state.status = f"Error: {e}"
        import traceback
        traceback.print_exc()

    state.reconstructing = False


# ── Main App ─────────────────────────────────────────────────────────────────

def run_dense_mesh(state, scene_gl):
    """Generate dense mesh from reconstruction via TSDF fusion."""
    state.refine_progress = "Generating dense mesh..."
    try:
        from dust3r.utils.device import to_numpy

        scene = state.scene
        if scene is None:
            state.status = "No reconstruction available"
            state.refining = False
            return

        imgs = scene.imgs

        # Extract dense pts3d based on scene type
        import torch as _torch

        mesh_min_conf = state.min_conf
        if hasattr(scene, '_is_vggt'):
            # VGGT scene
            pts3d_list = scene._pts3d
            confs_list = scene._depth_conf
        elif hasattr(scene, 'canonical_paths'):
            # MASt3R SparseGA
            pts3d_raw, _, confs_raw = scene.get_dense_pts3d(clean_depth=True)
            pts3d_list = [to_numpy(pts3d_raw[i]).reshape(imgs[i].shape[0], imgs[i].shape[1], 3)
                          for i in range(len(imgs))]
            confs_list = [to_numpy(confs_raw[i]) for i in range(len(imgs))]
        else:
            # DUSt3R — use actual confidence from scene
            if hasattr(scene, 'im_conf'):
                confs_list = to_numpy([c for c in scene.im_conf])
                # DUSt3R conf values are typically 1-20, threshold relative
                mesh_min_conf = float(np.median(np.concatenate([c.ravel() for c in confs_list]))) * 0.5
            else:
                confs_list = [np.ones(imgs[i].shape[:2], dtype=np.float32) * 10 for i in range(len(imgs))]
                mesh_min_conf = 0.5
            if hasattr(scene, 'clean_pointcloud'):
                scene = scene.clean_pointcloud()
            pts3d_list = to_numpy(scene.get_pts3d())

        from mesh_export import create_dense_mesh
        state.refine_progress = "Creating mesh..."

        # Get camera poses for normal orientation
        cam_poses = None
        try:
            c2w = scene.get_im_poses().cpu().numpy()
            cam_poses = [c2w[i] for i in range(len(imgs))]
        except Exception:
            pass

        print(f"  min_conf for mesh: {mesh_min_conf:.3f}")
        print(f"  pts3d shapes: {[p.shape for p in pts3d_list]}")
        print(f"  conf ranges: {[(c.min(), c.max()) for c in confs_list]}")

        verts, faces, colors = create_dense_mesh(
            imgs, pts3d_list, confs_list,
            cam2world_list=cam_poses, min_conf=mesh_min_conf)

        if len(faces) > 0:
            # Decimate to target face count if needed
            target = state.target_faces
            if len(faces) > target * 1.2:
                state.refine_progress = f"Decimating {len(faces):,d} -> {target:,d} faces..."
                try:
                    from refine_mesh import decimate_mesh
                    verts, faces, colors = decimate_mesh(verts, faces, colors, target)
                except Exception as e:
                    print(f"  Decimation failed: {e}")

            state.mesh_data = (verts, faces, colors)
            scene_gl.set_mesh(verts, faces, colors)
            state.has_mesh = True
            state.draw_mode = 1  # switch to mesh view
            state.status = f"Dense mesh: {len(verts):,d} verts, {len(faces):,d} faces"
        else:
            state.status = "Mesh generation produced no faces"
    except Exception as e:
        state.error_msg = str(e)
        state.status = f"Mesh generation failed: {e}"
        import traceback
        traceback.print_exc()

    state.refining = False
    state.refine_progress = ""


def run_refinement(state, scene_gl, debug_imgs=None):
    """
    Run mesh refinement inline with live viewport updates.
    Uses image-space gradient: for each vertex, compute pixel offset between
    rendered and GT, move vertex along camera right/up proportional to error.
    """
    state.refine_progress = "Starting refinement..."
    try:
        from refine_mesh import (load_cameras, compare_images, decimate_mesh,
                                 subdivide_high_error, laplacian_smooth,
                                 GLRenderer, save_ply_mesh)
        import tempfile
        from colmap_export import export_scene_to_colmap

        # Export COLMAP for camera data
        tmpdir = tempfile.mkdtemp()
        export_dir = os.path.join(tmpdir, 'colmap')
        state.refine_progress = "Exporting cameras..."
        export_scene_to_colmap(
            scene=state.scene, image_paths=state.image_paths,
            output_dir=export_dir, min_conf_thr=state.min_conf)

        views = load_cameras(export_dir)
        C = len(views)

        verts, faces, colors = state.mesh_data
        verts = verts.copy()
        colors = colors.copy()

        # No decimation — refine at full resolution

        verts_init = verts.copy()

        # OpenGL renderer (in this thread)
        max_w = max(v['W'] for v in views)
        max_h = max(v['H'] for v in views)
        renderer = GLRenderer(max_w, max_h)

        from scipy.spatial.distance import pdist
        cam_centers = np.array([-np.linalg.inv(v['w2c'])[:3, 3] for v in views])
        scene_scale = float(np.median(pdist(cam_centers))) if len(cam_centers) > 1 else 1.0

        compare_mode = state.compare_modes[state.compare_mode]
        lr = state.refine_lr
        depth_reg = state.refine_depth_reg
        smooth_reg = state.refine_smooth_reg
        iterations = state.refine_iters

        max_move = scene_scale * 0.001
        V = len(verts)

        state.refine_progress = f"Computing visibility for {V:,d} verts..."
        print("  Computing multi-view visibility via OpenGL z-buffer...")

        # Precompute visibility: render from each camera, collect which vertices
        # are actually visible (frontmost, not occluded) per camera
        vis_count = np.zeros(V, dtype=np.int32)  # how many cameras see each vertex
        vis_per_cam = []  # per camera: set of visible vertex IDs

        for ci in range(C):
            view = views[ci]
            _, vert_ids = renderer.render(verts, faces, colors,
                                          view['w2c'], view['K'], view['W'], view['H'])
            visible_set = set(vert_ids[vert_ids >= 0].ravel())
            vis_per_cam.append(visible_set)
            for vid in visible_set:
                vis_count[vid] += 1

        # Only refine vertices seen by 2+ cameras
        multi_view_mask = vis_count >= 2
        n_multi = multi_view_mask.sum()
        print(f"  {n_multi:,d} / {V:,d} vertices visible from 2+ cameras")

        state.refine_progress = f"Refining {n_multi:,d} multi-view verts..."

        # Predict normals from each camera image (AI normal estimation)
        from normal_estimator import predict_normals, render_mesh_normals_gl, compare_normals

        use_normals = compare_mode == 'normals'
        if use_normals:
            state.refine_progress = "Predicting normals from images..."
            predicted_normals_per_cam = []
            for ci_n in range(C):
                pn = predict_normals(views[ci_n]['pixels'])
                predicted_normals_per_cam.append(pn)
            print(f"  Predicted normals for {C} cameras")
            # Normal mode uses normals as error weighting for the image-space approach
            # (not as a direct movement direction, which doesn't converge)

        state.stop_requested = False
        for step in range(iterations):
            if state.stop_requested:
                print(f"  Stopped at step {step}")
                state.status = f"Refinement stopped at step {step}"
                break

            ci = step % C
            view = views[ci]
            W_v, H_v = view['W'], view['H']
            w2c = view['w2c']
            K = view['K']
            fx, fy = K[0, 0], K[1, 1]
            R_cam = w2c[:3, :3]

            # Render from current camera
            color_img, vert_ids = renderer.render(verts, faces, colors, w2c, K, W_v, H_v)

            if use_normals:
                # ── NORMAL-GUIDED REFINEMENT ──
                # 1. Render mesh normals and get AI normals for this camera
                rendered_normals, _ = render_mesh_normals_gl(
                    verts, faces, w2c, K, W_v, H_v, renderer)
                pred_n = predicted_normals_per_cam[ci]

                # Resize if needed
                if pred_n.shape[:2] != rendered_normals.shape[:2]:
                    from PIL import Image as _Img
                    rH, rW = rendered_normals.shape[:2]
                    pn_r = np.array(_Img.fromarray(
                        ((pred_n + 1) * 127.5).clip(0, 255).astype(np.uint8)
                    ).resize((rW, rH), _Img.BILINEAR), dtype=np.float32) / 127.5 - 1
                    pred_n = pn_r / (np.linalg.norm(pn_r, axis=-1, keepdims=True) + 1e-8)

                error_map = compare_normals(pred_n, rendered_normals, vert_ids)

                # 2. Compute current vertex normals in WORLD space
                v0_f = verts[faces[:, 0]]; v1_f = verts[faces[:, 1]]; v2_f = verts[faces[:, 2]]
                fn = np.cross(v1_f - v0_f, v2_f - v0_f)
                fn /= (np.linalg.norm(fn, axis=-1, keepdims=True) + 1e-8)
                vert_normals_world = np.zeros((V, 3), dtype=np.float64)
                for ax in range(3):
                    np.add.at(vert_normals_world[:, ax], faces[:, 0], fn[:, ax])
                    np.add.at(vert_normals_world[:, ax], faces[:, 1], fn[:, ax])
                    np.add.at(vert_normals_world[:, ax], faces[:, 2], fn[:, ax])
                vert_normals_world /= (np.linalg.norm(vert_normals_world, axis=-1, keepdims=True) + 1e-8)

                # 3. For each visible multi-view vertex: get target normal from AI
                valid = vert_ids >= 0
                if valid.any():
                    pixel_rows, pixel_cols = np.where(valid)
                    pixel_vids = vert_ids[pixel_rows, pixel_cols]

                    multi_mask = multi_view_mask[pixel_vids]
                    pixel_rows = pixel_rows[multi_mask]
                    pixel_cols = pixel_cols[multi_mask]
                    pixel_vids = pixel_vids[multi_mask]

                    if len(pixel_vids) > 0:
                        # Sample AI normals at pixel locations (in DSINE camera space)
                        pred_n_samples = pred_n[pixel_rows, pixel_cols]  # (N, 3)

                        # Convert AI normals from DSINE camera space to world space
                        # DSINE: X=right, Y=up, Z=toward camera
                        # OpenCV w2c: X=right, Y=down, Z=away from camera
                        # To convert DSINE->OpenCV cam: flip Y and Z
                        pred_n_opencv = pred_n_samples.copy()
                        pred_n_opencv[:, 1] *= -1  # Y: up -> down
                        pred_n_opencv[:, 2] *= -1  # Z: toward cam -> away
                        # Now convert camera space to world space
                        pred_n_world = (R_cam.T @ pred_n_opencv.T).T  # (N, 3) world

                        # Per-vertex: average target normal (weighted by error)
                        pixel_errs = error_map[pixel_rows, pixel_cols]
                        target_n = np.zeros((V, 3), dtype=np.float64)
                        n_count = np.zeros(V, dtype=np.float64)
                        for ax in range(3):
                            np.add.at(target_n[:, ax], pixel_vids, pred_n_world[:, ax] * pixel_errs)
                        np.add.at(n_count, pixel_vids, pixel_errs + 1e-8)

                        has = n_count > 0.001
                        target_n[has] /= n_count[has, None]
                        tn_len = np.linalg.norm(target_n[has], axis=-1, keepdims=True) + 1e-8
                        target_n[has] /= tn_len

                        # 4. Move vertex along its CURRENT normal by an amount proportional
                        # to the dot product between current and target normal.
                        # If target normal tilts away from current → the surface needs to
                        # bulge or indent → move vertex in/out along its normal.
                        #
                        # signed_error = 1 - dot(current, target)
                        # positive = push outward, negative = push inward
                        dot = (vert_normals_world[has] * target_n[has]).sum(axis=-1)
                        # Cross product gives the rotation axis; its magnitude = sin(angle)
                        cross = np.cross(vert_normals_world[has], target_n[has])
                        cross_mag = np.linalg.norm(cross, axis=-1)

                        # Displacement along vertex normal proportional to angular error
                        # Use cross product magnitude as signed displacement indicator
                        displacement = vert_normals_world[has] * cross_mag[:, None] * lr * max_move * 5

                        disp_mag = np.linalg.norm(displacement, axis=-1, keepdims=True)
                        displacement = np.where(disp_mag > max_move,
                                                 displacement * max_move / (disp_mag + 1e-8),
                                                 displacement)

                        move_mask_full = np.zeros(V, dtype=bool)
                        move_mask_full[has] = True
                        move_mask_full &= multi_view_mask
                        verts[move_mask_full] += displacement[multi_view_mask[has]].astype(np.float32)

                        avg_error = pixel_errs.mean()
                    else:
                        avg_error = 0.0
                else:
                    avg_error = 0.0

            else:
                # ── COLOR/EDGE MODE: image-space lateral movement ──
                error_map = compare_images(color_img, view['pixels'], mode=compare_mode)

                valid = vert_ids >= 0
                if not valid.any():
                    continue

                pixel_rows, pixel_cols = np.where(valid)
                pixel_vids = vert_ids[pixel_rows, pixel_cols]
                pixel_errs = error_map[pixel_rows, pixel_cols]

                multi_mask = multi_view_mask[pixel_vids]
                pixel_rows = pixel_rows[multi_mask]
                pixel_cols = pixel_cols[multi_mask]
                pixel_vids = pixel_vids[multi_mask]
                pixel_errs = pixel_errs[multi_mask]

                if len(pixel_vids) == 0:
                    continue

                cam_right = R_cam[0, :]
                cam_up = -R_cam[1, :]

                pts_cam = (R_cam @ verts.T).T + w2c[:3, 3]
                z = np.clip(pts_cam[:, 2], 0.01, None)
                u_proj = pts_cam[:, 0] / z * fx + K[0, 2]
                v_proj = pts_cam[:, 1] / z * fy + K[1, 2]

                dx_accum = np.zeros(V, dtype=np.float64)
                dy_accum = np.zeros(V, dtype=np.float64)
                err_accum = np.zeros(V, dtype=np.float64)
                count = np.zeros(V, dtype=np.float64)

                np.add.at(dx_accum, pixel_vids,
                          (pixel_cols.astype(np.float64) - u_proj[pixel_vids]) * pixel_errs)
                np.add.at(dy_accum, pixel_vids,
                          (pixel_rows.astype(np.float64) - v_proj[pixel_vids]) * pixel_errs)
                np.add.at(err_accum, pixel_vids, pixel_errs)
                np.add.at(count, pixel_vids, 1.0)

                has = count > 0
                avg_error = err_accum[has].sum() / count[has].sum() if has.any() else 0.0
                dx_accum[has] /= (count[has] + 1e-8)
                dy_accum[has] /= (count[has] + 1e-8)

                displacement = ((dx_accum / fx * z)[:, None] * cam_right[None, :] +
                               (dy_accum / fy * z)[:, None] * cam_up[None, :]) * lr

                disp_mag = np.linalg.norm(displacement, axis=-1, keepdims=True)
                displacement = np.where(disp_mag > max_move,
                                         displacement * max_move / (disp_mag + 1e-8),
                                         displacement)

                move_mask = has & multi_view_mask
                verts[move_mask] += displacement[move_mask].astype(np.float32)

            # Depth regularization
            verts += (depth_reg * lr * (verts_init - verts)).astype(np.float32)

            # Laplacian smoothing
            if smooth_reg > 0:
                verts = laplacian_smooth(verts, faces, strength=smooth_reg)

            # Update vertex colors — blend all cameras weighted by incidence angle
            if step % 20 == 0:
                color_accum = np.zeros((V, 3), dtype=np.float64)
                weight_accum = np.zeros(V, dtype=np.float64)

                # Compute vertex normals for incidence weighting
                v0_f = verts[faces[:, 0]]; v1_f = verts[faces[:, 1]]; v2_f = verts[faces[:, 2]]
                fn_c = np.cross(v1_f - v0_f, v2_f - v0_f)
                fn_c /= (np.linalg.norm(fn_c, axis=-1, keepdims=True) + 1e-8)
                vn = np.zeros((V, 3), dtype=np.float64)
                for ax in range(3):
                    np.add.at(vn[:, ax], faces[:, 0], fn_c[:, ax])
                    np.add.at(vn[:, ax], faces[:, 1], fn_c[:, ax])
                    np.add.at(vn[:, ax], faces[:, 2], fn_c[:, ax])
                vn /= (np.linalg.norm(vn, axis=-1, keepdims=True) + 1e-8)

                for vi_cam in range(C):
                    vd = views[vi_cam]
                    _, vid_map = renderer.render(verts, faces, colors,
                                                  vd['w2c'], vd['K'], vd['W'], vd['H'])
                    vis_set = set(vid_map[vid_map >= 0].ravel())

                    R = vd['w2c'][:3, :3]; tv = vd['w2c'][:3, 3]
                    pc = (R @ verts.T).T + tv
                    zz = np.clip(pc[:, 2], 0.01, None)
                    uu = (pc[:, 0] / zz * vd['K'][0, 0] + vd['K'][0, 2]).astype(int)
                    vv = (pc[:, 1] / zz * vd['K'][1, 1] + vd['K'][1, 2]).astype(int)
                    ok = (zz > 0.01) & (uu >= 0) & (uu < vd['W']) & (vv >= 0) & (vv < vd['H'])

                    vis_arr = np.zeros(V, dtype=bool)
                    for vid in vis_set:
                        if vid < V: vis_arr[vid] = True
                    ok = ok & vis_arr

                    if ok.any():
                        # Incidence weight: dot(vertex_normal, view_direction)
                        c2w = np.linalg.inv(vd['w2c'])
                        cam_center = c2w[:3, 3]
                        view_dirs = cam_center[None, :] - verts[ok]
                        view_dirs /= (np.linalg.norm(view_dirs, axis=-1, keepdims=True) + 1e-8)
                        dots = (vn[ok] * view_dirs).sum(axis=-1).clip(0, 1)
                        w = dots ** 2  # squared = penalize grazing angles more

                        sampled = vd['pixels'][vv[ok], uu[ok]]  # (N, 3) float [0,1]
                        color_accum[ok] += sampled * w[:, None]
                        weight_accum[ok] += w

                # Normalize blended colors
                has_w = weight_accum > 0.001
                if has_w.any():
                    blended = color_accum[has_w] / weight_accum[has_w, None]
                    colors[has_w] = (blended * 255).clip(0, 255).astype(np.uint8)

            # Live viewport update
            if step % 5 == 0:
                scene_gl.set_mesh(verts, faces, colors)

            if step % 10 == 0:
                mode_str = "normals" if use_normals else compare_mode
                state.refine_progress = f"[{mode_str}] Step {step}/{iterations}, error={avg_error:.4f}"
                print(f"[{step:4d}/{iterations}] error={avg_error:.6f} mode={mode_str} verts={V:,d}")

            # Update in-app debug images every 25 steps (if debug_imgs available)
            if step % 25 == 0 and debug_imgs is not None:
                from PIL import Image as _PILImg

                # GT vs rendered
                gt_img = (view['pixels'] * 255).clip(0, 255).astype(np.uint8)
                rend_img = (color_img * 255).clip(0, 255).astype(np.uint8)
                if gt_img.shape[:2] != rend_img.shape[:2]:
                    rend_img = np.array(_PILImg.fromarray(rend_img).resize(
                        (gt_img.shape[1], gt_img.shape[0]), _PILImg.BILINEAR))
                debug_imgs.set_image("GT vs Render", np.concatenate([gt_img, rend_img], axis=1))

                # Error map
                err_vis = (error_map / (error_map.max() + 1e-8) * 255).clip(0, 255).astype(np.uint8)
                if err_vis.shape[:2] != gt_img.shape[:2]:
                    err_vis = np.array(_PILImg.fromarray(err_vis).resize(
                        (gt_img.shape[1], gt_img.shape[0]), _PILImg.BILINEAR))
                debug_imgs.set_image("Error Map", np.stack([err_vis]*3, axis=-1))

                # Normal comparison
                if use_normals:
                    rendered_normals_vis, _ = render_mesh_normals_gl(
                        verts, faces, w2c, K, W_v, H_v, renderer)
                    pred_n_vis = predicted_normals_per_cam[ci]

                    mesh_n_rgb = ((rendered_normals_vis + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)
                    pred_n_rgb = ((pred_n_vis + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)

                    if mesh_n_rgb.shape[:2] != pred_n_rgb.shape[:2]:
                        pred_n_rgb = np.array(_PILImg.fromarray(pred_n_rgb).resize(
                            (mesh_n_rgb.shape[1], mesh_n_rgb.shape[0]), _PILImg.BILINEAR))

                    debug_imgs.set_image("AI Normals", pred_n_rgb)
                    debug_imgs.set_image("Mesh Normals", mesh_n_rgb)
                    diff = np.abs(mesh_n_rgb.astype(np.int16) - pred_n_rgb.astype(np.int16)).clip(0, 255).astype(np.uint8)
                    debug_imgs.set_image("Normal Difference (x3)", (diff * 3).clip(0, 255).astype(np.uint8))

        # Final update
        state.mesh_data = (verts.copy(), faces.copy(), colors.copy())
        scene_gl.set_mesh(verts, faces, colors)
        state.status = f"Refined mesh: {len(verts):,d} verts, {len(faces):,d} faces"

    except Exception as e:
        state.error_msg = str(e)
        state.status = f"Refinement failed: {e}"
        import traceback
        traceback.print_exc()

    state.refining = False
    state.refine_progress = ""


def run_enhanced_mesh(state, scene_gl, debug_imgs=None):
    """Generate enhanced mesh using monocular depth + normals + reconstruction poses."""
    state.refine_progress = "Enhanced mesh: loading models..."
    try:
        from mono_depth import generate_enhanced_pointcloud, enhanced_cloud_to_mesh

        def progress(frac, msg):
            state.refine_progress = msg

        # Generate enhanced point cloud
        points, colors, normals = generate_enhanced_pointcloud(
            state.scene, state.image_paths,
            progress_fn=progress, device='cuda')

        # Show point cloud immediately
        if len(points) > 200000:
            idx = np.random.choice(len(points), 200000, replace=False)
            scene_gl.set_points(points[idx], colors[idx])
        else:
            scene_gl.set_points(points, colors)
        state.has_points = True
        state.draw_mode = 0  # points view

        # Build mesh with Poisson + oriented normals
        state.refine_progress = "Building mesh from oriented point cloud..."
        verts, faces, vert_colors = enhanced_cloud_to_mesh(
            points, colors, normals, target_faces=state.target_faces)

        if len(faces) > 0:
            state.mesh_data = (verts, faces, vert_colors)
            scene_gl.set_mesh(verts, faces, vert_colors)
            state.has_mesh = True
            state.draw_mode = 1
            state.status = f"Enhanced mesh: {len(verts):,d} verts, {len(faces):,d} faces"
        else:
            state.status = "Enhanced mesh: no faces generated (point cloud only)"

    except Exception as e:
        state.error_msg = str(e)
        state.status = f"Enhanced mesh failed: {e}"
        import traceback
        traceback.print_exc()

    state.refining = False
    state.refine_progress = ""


def run_texture(state, scene_gl):
    """Generate UV-mapped textured mesh."""
    state.refine_progress = "Generating texture..."
    try:
        import tempfile
        from colmap_export import export_scene_to_colmap
        from refine_mesh import load_cameras
        from texture_map import create_textured_mesh

        # Export COLMAP to get camera data
        tmpdir = tempfile.mkdtemp()
        export_dir = os.path.join(tmpdir, 'colmap')
        export_scene_to_colmap(
            scene=state.scene, image_paths=state.image_paths,
            output_dir=export_dir, min_conf_thr=state.min_conf)

        views = load_cameras(export_dir)
        verts, faces, colors = state.mesh_data

        # Decimate for texture speed if too dense
        target = state.target_faces
        if len(faces) > target * 1.2:
            state.refine_progress = f"Decimating {len(faces):,d} -> {target:,d} faces..."
            from refine_mesh import decimate_mesh
            verts, faces, colors = decimate_mesh(verts, faces, colors, target)

        output_dir = os.path.join(tmpdir, 'textured')
        state.refine_progress = f"UV unwrapping + projecting {len(faces):,d} faces..."
        obj_path = create_textured_mesh(verts, faces, colors, views, output_dir)

        # Update mesh data + viewport with the (possibly decimated) textured mesh
        state.mesh_data = (verts, faces, colors)
        scene_gl.set_mesh(verts, faces, colors)

        state.status = f"Textured mesh saved to {output_dir} ({len(faces):,d} faces)"
        state.refine_progress = ""

        # Open the folder so user can find it
        import subprocess
        subprocess.Popen(['explorer', output_dir.replace('/', '\\')])

    except Exception as e:
        state.error_msg = str(e)
        state.status = f"Texture failed: {e}"
        import traceback
        traceback.print_exc()

    state.refining = False
    state.refine_progress = ""


def main():
    if not glfw.init():
        print("Could not initialize GLFW")
        return

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(1600, 900, "3D Reconstruction Studio", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    imgui.create_context()
    impl = GlfwRenderer(window)

    camera = OrbitCamera()
    scene_gl = GLScene()
    debug_imgs = DebugImages()
    state = AppState()

    # Mouse state for viewport interaction
    mouse_down = [False, False, False]
    last_mouse = [0.0, 0.0]

    def mouse_button_callback(window, button, action, mods):
        # Let imgui handle it first
        if imgui.get_io().want_capture_mouse:
            return
        if action == glfw.PRESS:
            mouse_down[button] = True
        elif action == glfw.RELEASE:
            mouse_down[button] = False

    def scroll_callback(window, xoffset, yoffset):
        if imgui.get_io().want_capture_mouse:
            return
        camera.zoom(yoffset)

    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_scroll_callback(window, scroll_callback)

    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()

        # Mouse drag for orbit/pan
        mx, my = glfw.get_cursor_pos(window)
        dx, dy = mx - last_mouse[0], my - last_mouse[1]
        last_mouse[0], last_mouse[1] = mx, my

        if not imgui.get_io().want_capture_mouse:
            if mouse_down[0]:  # Left = orbit
                camera.orbit(dx, dy)
            if mouse_down[1]:  # Right = pan
                camera.pan(-dx, dy)
            if mouse_down[2]:  # Middle = zoom
                camera.zoom(-dy * 0.1)

        imgui.new_frame()

        # ── Side Panel ──
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(400, glfw.get_window_size(window)[1])
        imgui.begin("Controls", flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE)

        # Status
        imgui.text_colored(state.status, 0.4, 0.8, 1.0)
        if state.error_msg:
            imgui.text_colored(state.error_msg, 1.0, 0.3, 0.3)
        imgui.separator()

        # ── Load Images ──
        imgui.text("Images")
        if imgui.button("Open Folder..."):
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            folder = filedialog.askdirectory(title="Select Image Folder")
            root.destroy()
            if folder:
                exts = {'.jpg', '.jpeg', '.png', '.bmp'}
                state.image_paths = sorted([
                    os.path.join(folder, f) for f in os.listdir(folder)
                    if os.path.splitext(f)[1].lower() in exts
                ])
                state.image_dir = folder
                state.status = f"Loaded {len(state.image_paths)} images from {folder}"

        if state.image_paths:
            imgui.text(f"  {len(state.image_paths)} images")
            imgui.text(f"  {state.image_dir}")
        imgui.separator()

        # ── Backend ──
        imgui.text("Reconstruction Backend")
        _, state.backend_idx = imgui.combo("##backend",
            state.backend_idx, ["DUSt3R", "MASt3R", "VGGT"])

        if state.backends[state.backend_idx] == 'dust3r':
            _, state.niter1 = imgui.input_int("Iterations##d3r", state.niter1, 50, 100)

        if state.backends[state.backend_idx] == 'mast3r':
            _, state.optim_level = imgui.combo("Optimization##opt",
                state.optim_level, ["Coarse", "Refine", "Refine + Depth"])
            _, state.niter1 = imgui.input_int("Coarse Iters", state.niter1, 50, 100)
            _, state.niter2 = imgui.input_int("Refine Iters", state.niter2, 50, 100)

        _, state.min_conf = imgui.slider_float("Min Confidence", state.min_conf, 0.1, 20.0)

        # Reconstruct button
        if state.image_paths and not state.reconstructing:
            if imgui.button("Reconstruct", width=-1):
                state.recon_thread = threading.Thread(
                    target=run_reconstruction, args=(state, scene_gl), daemon=True)
                state.recon_thread.start()

        if state.reconstructing:
            imgui.text("Reconstructing...")
            imgui.progress_bar(-1.0 * time.time() % 1.0)  # indeterminate

        imgui.separator()

        # ── Display Options ──
        imgui.text("Display")
        _, state.draw_mode = imgui.combo("Mode##draw",
            state.draw_mode, ["Points", "Mesh", "Wireframe", "Normals", "Shaded"])

        # Scene orientation
        _, state.scene_rot_x = imgui.slider_float("Rot X", state.scene_rot_x, -180, 180)
        _, state.scene_rot_y = imgui.slider_float("Rot Y", state.scene_rot_y, -180, 180)
        _, state.scene_rot_z = imgui.slider_float("Rot Z", state.scene_rot_z, -180, 180)
        imgui.separator()

        # ── Export ──
        imgui.text("Export")
        if state.scene is not None or state.has_points:
            if imgui.button("Export COLMAP Dataset", width=-1):
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                folder = filedialog.askdirectory(title="Export To")
                root.destroy()
                if folder:
                    try:
                        from colmap_export import export_scene_to_colmap
                        export_scene_to_colmap(
                            scene=state.scene, image_paths=state.image_paths,
                            output_dir=folder, min_conf_thr=state.min_conf)
                        state.status = f"Exported to {folder}"
                    except Exception as e:
                        state.error_msg = str(e)

        imgui.separator()

        # ── Dense Mesh ──
        imgui.text("Dense Mesh")
        _, state.target_faces = imgui.input_int("Target Faces", state.target_faces, 10000, 50000)
        state.target_faces = max(1000, state.target_faces)
        if state.has_points and not state.refining:
            if imgui.button("Generate Dense Mesh (from reconstruction)", width=-1):
                state.refining = True
                state.refine_thread = threading.Thread(
                    target=run_dense_mesh, args=(state, scene_gl), daemon=True)
                state.refine_thread.start()
            if state.scene is not None:
                if imgui.button("Generate Enhanced Mesh (AI Depth + Normals)", width=-1):
                    state.refining = True
                    state.refine_thread = threading.Thread(
                        target=run_enhanced_mesh, args=(state, scene_gl, debug_imgs), daemon=True)
                    state.refine_thread.start()

        imgui.separator()

        # ── Refine ──
        imgui.text("Mesh Refinement")
        _, state.refine_iters = imgui.input_int("Iterations##ref", state.refine_iters, 50, 100)
        _, state.compare_mode = imgui.combo("Compare##cmp",
            state.compare_mode, ["Edges", "High Freq", "Color", "Both", "Normals (AI)"])
        _, state.refine_depth_reg = imgui.slider_float("Depth Reg", state.refine_depth_reg, 0, 1)
        _, state.refine_smooth_reg = imgui.slider_float("Smooth Reg", state.refine_smooth_reg, 0, 0.1)

        if state.has_mesh and not state.refining:
            if imgui.button("Refine Mesh", width=-1):
                state.refining = True
                state.refine_thread = threading.Thread(
                    target=run_refinement, args=(state, scene_gl, debug_imgs), daemon=True)
                state.refine_thread.start()

        if state.refining:
            imgui.text(state.refine_progress or "Working...")
            if imgui.button("Stop", width=-1):
                state.stop_requested = True

        imgui.separator()

        # ── Texture ──
        if state.has_mesh and state.scene is not None and not state.refining:
            if imgui.button("Generate Textured Mesh (.obj)", width=-1):
                state.refining = True
                state.refine_thread = threading.Thread(
                    target=run_texture, args=(state, scene_gl), daemon=True)
                state.refine_thread.start()

        imgui.separator()

        # ── Save ──
        if state.has_mesh:
            if imgui.button("Save Mesh (.ply)", width=-1):
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                path = filedialog.asksaveasfilename(
                    title="Save Mesh", defaultextension=".ply",
                    filetypes=[("PLY files", "*.ply")])
                root.destroy()
                if path and state.mesh_data is not None:
                    from refine_mesh import save_ply_mesh
                    v, f, c = state.mesh_data
                    save_ply_mesh(path, v, f, c)
                    state.status = f"Saved to {path}"

        imgui.end()

        # ── 3D Viewport ──
        win_w, win_h = glfw.get_window_size(window)
        vp_x = 400
        vp_w = win_w - vp_x
        vp_h = win_h

        gl.glViewport(vp_x, 0, vp_w, vp_h)
        gl.glScissor(vp_x, 0, vp_w, vp_h)
        gl.glEnable(gl.GL_SCISSOR_TEST)
        gl.glClearColor(0.15, 0.15, 0.2, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glEnable(gl.GL_DEPTH_TEST)

        # Upload any pending data from background threads
        scene_gl.flush_pending()

        if vp_w > 0 and vp_h > 0:
            aspect = vp_w / vp_h
            view = camera.get_view_matrix()
            proj = camera.get_projection_matrix(aspect)
            # Scene orientation transform
            def _rot_matrix(rx, ry, rz):
                from scipy.spatial.transform import Rotation as _R
                r = _R.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix()
                m = np.eye(4, dtype=np.float32)
                m[:3, :3] = r
                return m

            scene_tf = _rot_matrix(state.scene_rot_x, state.scene_rot_y, state.scene_rot_z)
            mvp_base = proj @ view            # for grid + axes (fixed)
            mvp_scene = proj @ view @ scene_tf  # for point cloud / mesh / cameras

            mode = state.draw_modes[state.draw_mode]
            scene_gl.draw(mvp_base, mvp_scene, draw_mode=mode, camera_pos=camera.get_position())

        gl.glDisable(gl.GL_SCISSOR_TEST)

        # ── In-app debug image viewer ──
        debug_imgs.flush()
        debug_imgs.draw_window("Debug Views")

        # Render ImGui
        gl.glViewport(0, 0, win_w, win_h)
        imgui.render()
        impl.render(imgui.get_draw_data())

        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()


if __name__ == '__main__':
    main()
