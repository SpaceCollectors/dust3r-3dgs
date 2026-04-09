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
import io
import collections
import numpy as np
from pathlib import Path


class _ConsoleCapture:
    """Captures stdout/stderr into a ring buffer for in-app display."""

    def __init__(self, maxlines=500):
        self.lines = collections.deque(maxlen=maxlines)
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._lock = threading.Lock()

    def install(self):
        sys.stdout = _TeeWriter(self._original_stdout, self)
        sys.stderr = _TeeWriter(self._original_stderr, self)

    def add(self, text):
        with self._lock:
            for line in text.splitlines():
                if line.strip():
                    self.lines.append(line)

    def get_lines(self):
        with self._lock:
            return list(self.lines)

    def clear(self):
        with self._lock:
            self.lines.clear()


class _TeeWriter:
    """Writes to both the original stream and the console capture."""

    def __init__(self, original, capture):
        self._original = original
        self._capture = capture

    def write(self, text):
        if text and text.strip():
            self._capture.add(text)
        try:
            self._original.write(text)
        except Exception:
            pass

    def flush(self):
        try:
            self._original.flush()
        except Exception:
            pass

    # Forward any other attributes to original
    def __getattr__(self, name):
        return getattr(self._original, name)


# Global console capture
_console = _ConsoleCapture()

import glfw
import OpenGL.GL as gl
import imgui
from imgui.integrations.glfw import GlfwRenderer

def _find_cached_model_path(repo_id):
    """Find the local cache path for a HuggingFace model, or None if not cached."""
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == repo_id:
                for rev in repo.revisions:
                    snapshot_path = rev.snapshot_path
                    if os.path.isdir(snapshot_path):
                        print(f"  Found cached model: {repo_id} -> {snapshot_path}")
                        return str(snapshot_path)
    except Exception:
        pass
    return None


def _load_pretrained_cached(model_class, repo_id):
    """Load a pretrained model from local cache if available, else download."""
    # Try loading from local snapshot cache (no network)
    cached_path = _find_cached_model_path(repo_id)
    if cached_path:
        try:
            model = model_class.from_pretrained(cached_path)
            print(f"  Loaded {repo_id} from cache (offline)")
            return model
        except Exception as e:
            print(f"  Cache load failed ({e})")

    # Try HF offline mode (uses cached blobs)
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        model = model_class.from_pretrained(repo_id)
        print(f"  Loaded {repo_id} in offline mode")
        return model
    except Exception:
        pass
    finally:
        os.environ.pop('HF_HUB_OFFLINE', None)

    # Not cached — download
    print(f"  Downloading {repo_id}...")
    return model_class.from_pretrained(repo_id)


# Add our repos to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAST3R_DIR = os.path.join(SCRIPT_DIR, 'mast3r')
VGGT_DIR = os.path.join(SCRIPT_DIR, 'vggt')
MVDUST3R_DIR = os.path.join(SCRIPT_DIR, 'mvdust3r')
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
in vec2 uv;
out vec3 v_color;
out vec2 v_uv;
void main() {
    gl_Position = mvp * vec4(position, 1.0);
    gl_PointSize = 3.0;
    v_color = color;
    v_uv = uv;
}
"""

FRAG_SHADER = """
#version 330
uniform sampler2D tex;
uniform int use_texture;
in vec3 v_color;
in vec2 v_uv;
out vec4 frag_color;
void main() {
    if (use_texture == 1 && v_uv.x >= 0.0) {
        frag_color = texture(tex, v_uv);
    } else {
        frag_color = vec4(v_color, 1.0);
    }
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
        self.use_tex_loc = gl.glGetUniformLocation(self.program, "use_texture")
        self.tex_loc = gl.glGetUniformLocation(self.program, "tex")
        self.point_vao = None
        self.point_vbo = None
        self.point_count = 0
        self.mesh_vao = None
        self.mesh_vbo = None
        self.mesh_ebo = None
        self.mesh_vertex_count = 0
        self.mesh_face_count = 0
        self.mesh_tex_id = None  # OpenGL texture for UV-mapped mesh
        self.mesh_has_uvs = False
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

    def _upload_mesh(self, verts, faces, colors, uvs=None):
        if len(verts) == 0 or len(faces) == 0:
            self.mesh_face_count = 0
            return
        # Layout: pos(3) + color(3) + uv(2) = 8 floats = 32 bytes per vertex
        data = np.empty((len(verts), 8), dtype=np.float32)
        data[:, :3] = verts.astype(np.float32)
        data[:, 3:6] = colors.astype(np.float32) / 255.0
        if uvs is not None and len(uvs) == len(verts):
            data[:, 6:8] = uvs.astype(np.float32)
            self.mesh_has_uvs = True
        else:
            data[:, 6:8] = -1.0  # signal "no UV" to shader
            self.mesh_has_uvs = False
        indices = faces.astype(np.uint32).ravel()

        if self.mesh_vao is None:
            self.mesh_vao = gl.glGenVertexArrays(1)
            self.mesh_vbo = gl.glGenBuffers(1)
            self.mesh_ebo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.mesh_vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.mesh_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)
        stride = 32  # 8 floats * 4 bytes
        gl.glEnableVertexAttribArray(0)  # position
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, stride, None)
        gl.glEnableVertexAttribArray(1)  # color
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, False, stride, gl.ctypes.c_void_p(12))
        gl.glEnableVertexAttribArray(2)  # uv
        gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, False, stride, gl.ctypes.c_void_p(24))
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.mesh_ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_DYNAMIC_DRAW)
        gl.glBindVertexArray(0)
        self.mesh_vertex_count = len(verts)
        self.mesh_face_count = len(faces)

    def set_texture(self, texture_image, uvs, uv_faces, verts, faces, colors):
        """Queue texture data for upload. texture_image: (H,W,3) uint8, uvs: per-uv-vertex, uv_faces: per-face UV indices."""
        with self._lock:
            self._pending_texture = (texture_image, uvs, uv_faces, verts, faces, colors)

    def _upload_texture(self, texture_image, uvs, uv_faces, verts, faces, colors):
        """Upload texture to GPU and rebuild mesh VBO with per-face-vertex UVs."""
        # Expand mesh to per-face-vertex (so each face corner can have its own UV)
        n_faces = len(faces)
        expanded_verts = np.zeros((n_faces * 3, 3), dtype=np.float32)
        expanded_colors = np.zeros((n_faces * 3, 3), dtype=np.uint8)
        expanded_uvs = np.zeros((n_faces * 3, 2), dtype=np.float32)
        expanded_faces = np.arange(n_faces * 3, dtype=np.int32).reshape(-1, 3)

        for fi in range(n_faces):
            for vi in range(3):
                expanded_verts[fi * 3 + vi] = verts[faces[fi, vi]]
                expanded_colors[fi * 3 + vi] = colors[faces[fi, vi]]
                expanded_uvs[fi * 3 + vi] = uvs[uv_faces[fi, vi]]

        # Upload mesh with UVs
        self._upload_mesh(expanded_verts, expanded_faces, expanded_colors, expanded_uvs)
        # Store expanded data for draw mode switching
        self._mesh_verts = expanded_verts
        self._mesh_faces = expanded_faces
        self._base_colors = expanded_colors
        self._mesh_uvs = expanded_uvs

        # Upload texture image to OpenGL
        H, W = texture_image.shape[:2]
        if self.mesh_tex_id is None:
            self.mesh_tex_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.mesh_tex_id)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, W, H, 0,
                        gl.GL_RGB, gl.GL_UNSIGNED_BYTE, texture_image)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        print(f"  Uploaded {W}x{H} texture to GPU")

    def draw(self, mvp_grid, mvp_scene, draw_mode='points', camera_pos=None):
        gl.glUseProgram(self.program)

        # Grid + axes: fixed orientation
        gl.glUniformMatrix4fv(self.mvp_loc, 1, gl.GL_TRUE, mvp_grid)
        if self.grid_line_count > 0:
            gl.glBindVertexArray(self.grid_vao)
            gl.glDrawArrays(gl.GL_LINES, 0, self.grid_line_count)

        # Swap mesh colors based on draw mode
        if self.mesh_face_count > 0 and getattr(self, '_mesh_verts', None) is not None:
            stored_uvs = getattr(self, '_mesh_uvs', None)
            if draw_mode == 'normals':
                alt = getattr(self, '_normal_colors', None)
                if alt is not None:
                    self._upload_mesh(self._mesh_verts, self._mesh_faces, alt, stored_uvs)
                draw_mode = 'mesh'
            elif draw_mode == 'shaded':
                # Compute shading from current camera position (headlamp effect)
                self._compute_shaded_from_camera(camera_pos)
                draw_mode = 'mesh'
            elif draw_mode in ('mesh', 'wireframe'):
                base = getattr(self, '_base_colors', None)
                if base is not None:
                    self._upload_mesh(self._mesh_verts, self._mesh_faces, base, stored_uvs)

        # Scene content: rotated
        gl.glUniformMatrix4fv(self.mvp_loc, 1, gl.GL_TRUE, mvp_scene)

        if draw_mode == 'points' and self.point_count > 0:
            gl.glBindVertexArray(self.point_vao)
            gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
            gl.glDrawArrays(gl.GL_POINTS, 0, self.point_count)

        if draw_mode == 'mesh' and self.mesh_face_count > 0:
            # Enable texture if available
            if self.mesh_tex_id is not None and self.mesh_has_uvs:
                gl.glActiveTexture(gl.GL_TEXTURE0)
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.mesh_tex_id)
                gl.glUniform1i(self.tex_loc, 0)
                gl.glUniform1i(self.use_tex_loc, 1)
            else:
                gl.glUniform1i(self.use_tex_loc, 0)
            gl.glBindVertexArray(self.mesh_vao)
            gl.glDrawElements(gl.GL_TRIANGLES, self.mesh_face_count * 3,
                              gl.GL_UNSIGNED_INT, None)
            # Disable texture after drawing
            gl.glUniform1i(self.use_tex_loc, 0)
            if self.mesh_tex_id is not None:
                gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

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
            tex = getattr(self, '_pending_texture', None)
            self._pending_points = None
            self._pending_mesh = None
            self._pending_cams = None
            self._pending_texture = None

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
        if tex is not None:
            self._upload_texture(*tex)
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

        # Light direction = from vertex toward camera (headlamp)
        if camera_pos is not None:
            light_dirs = camera_pos[None, :] - verts
            light_dirs /= (np.linalg.norm(light_dirs, axis=-1, keepdims=True) + 1e-8)
            ndotl = (vert_normals * light_dirs).sum(axis=-1)
            # Use abs(dot) so both-sided lighting works (handles flipped normals)
            ndotl = np.abs(ndotl)
        else:
            ndotl = np.full(V, 0.7, dtype=np.float32)

        ambient = 0.2
        brightness = ambient + (1.0 - ambient) * ndotl
        grey = (brightness * 220).clip(0, 255).astype(np.uint8)
        shaded = np.stack([grey, grey, grey], axis=-1)

        stored_uvs = getattr(self, '_mesh_uvs', None)
        self._upload_mesh(verts, faces, shaded, stored_uvs)

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
        self.show_console = False  # overlay console on viewport

        # Images
        self.image_paths = []
        self.image_dir = ""

        # Backend
        self.backend_idx = 0  # 0=dust3r, 1=mast3r, 2=vggt, 3=colmap, 4=pow3r
        self.backends = ['dust3r', 'mast3r', 'vggt', 'colmap', 'pow3r']
        self.camera_source_idx = 0  # 0=same, 1=COLMAP, 2=DUSt3R, 3=MASt3R, 4=VGGT
        self.camera_sources = ['same', 'colmap', 'dust3r', 'mast3r', 'vggt']
        self.camera_source_labels = ['Same as Point Cloud', 'COLMAP', 'DUSt3R', 'MASt3R', 'VGGT']
        self.cached_cameras = None  # list of (c2w, K, W, H) from camera source
        self.stack_backends = False  # run multiple backends and merge points
        self.stack_dust3r = True
        self.stack_mast3r = False
        self.stack_vggt = True
        self.stack_pow3r = False

        # Reconstruction
        self.scene = None
        self.reconstructing = False
        self.recon_progress = ""
        self.recon_frac = 0.0  # 0.0–1.0 progress fraction
        self.recon_thread = None

        # Point cloud / Mesh
        self.has_points = False
        self.points_modified = False  # True after smooth/upscale (don't reset on conf change)
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

        # Normalized scene data (backend-agnostic, set after reconstruction)
        self.pts3d_list = None   # list of (H,W,3) or (N,3) numpy arrays per view
        self.confs_list = None   # list of (H,W) or (N,) confidence arrays per view

        # Mesh data (numpy, for saving)
        self.mesh_data = None  # (verts, faces, colors)
        self.target_faces = 100000  # target face count for dense mesh
        self.mesh_mode_idx = 0     # 0=reprojected grid, 1=ball pivot, 2=local delaunay
        self.mesh_modes = ['reprojected', 'ballpivot', 'delaunay']
        self.mesh_mode_labels = ['Reprojected Grid', 'Ball Pivot', 'Local Delaunay']
        self.hole_cap_size = 50    # max boundary edges to close (higher = close bigger holes)
        self.smooth_radius = 2.0   # neighbor merge radius multiplier
        self.use_smoothing = False # whether to smooth before meshing
        self.ai_depth_mix = 0.7  # 0=pure dust3r, 1=full AI detail
        self.ai_highpass_radius = 10.0  # high-pass sigma in pixels
        self.ai_refine_poses = True  # re-optimize poses after depth injection
        self.ai_pose_iters = 100

        # Mesh generation settings (kept for compatibility)

        # Scene orientation transform (applied in shader)
        self.scene_rot_x = 180.0  # degrees (DUSt3R convention: Y-down, flip to Y-up)
        self.scene_rot_y = 0.0
        self.scene_rot_z = 0.0
        self.scene_flip_y = False

        # Stop flag
        self.stop_requested = False
        self.needs_recenter = False

        # Status
        self.status = "Ready. Load images to begin."
        self.error_msg = ""


# ── Pipeline Integration ─────────────────────────────────────────────────────

def run_reconstruction(state, scene_gl):
    """Run reconstruction in background thread."""
    import torch
    import copy

    state.reconstructing = True
    state.recon_frac = 0.0
    state.status = "Loading model..."
    # Clear cached scene data so _extract_scene_data re-extracts
    state.pts3d_list = None
    state.confs_list = None
    state.cached_cameras = None

    try:
        backend = state.backends[state.backend_idx]

        # Stack mode: run multiple backends and merge their point clouds
        if state.stack_backends:
            _run_stacked_reconstruction(state, scene_gl)
            return

        if backend == 'vggt':
            from vggt.models.vggt import VGGT
            from vggt.utils.load_fn import load_and_preprocess_images
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
            from vggt.utils.geometry import unproject_depth_map_to_point_map

            state.status = "Loading VGGT model..."
            state.recon_frac = 0.1
            model = _load_pretrained_cached(VGGT, "facebook/VGGT-1B").to('cuda')
            model.eval()

            state.status = "Running VGGT inference..."
            state.recon_frac = 0.3
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
            state.status = "Unprojecting depth maps..."
            state.recon_frac = 0.6
            pts3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

            imgs_np = images.cpu().numpy().transpose(0, 2, 3, 1)

            state.status = "Extracting point cloud..."
            state.recon_frac = 0.7
            # Build simple scene data — filter out white padding pixels
            all_pts = []
            all_colors = []
            for i in range(len(state.image_paths)):
                conf_mask = depth_conf[i] > state.min_conf
                # Mask out white padding from mixed aspect ratios
                # Padding is white (1.0, 1.0, 1.0) — detect pixels where all channels > 0.95
                not_padding = imgs_np[i].mean(axis=-1) < 0.95
                mask = conf_mask & not_padding
                all_pts.append(pts3d[i][mask])
                all_colors.append((imgs_np[i][mask] * 255).astype(np.uint8))

            if all_pts:
                points = np.concatenate(all_pts, axis=0)
                colors = np.concatenate(all_colors, axis=0)
                if len(points) > 200000:
                    idx = np.random.choice(len(points), 200000, replace=False)
                    points, colors = points[idx], colors[idx]
                scene_gl.set_points(points, colors)
                state.has_points = True; state.needs_recenter = True

            state.status = "Building scene..."
            state.recon_frac = 0.85
            # Build VGGTScene for downstream use (mesh gen, export)
            from app import VGGTScene
            from PIL import Image as PILImage
            from PIL.ImageOps import exif_transpose
            orig_sizes = []
            for p in state.image_paths:
                im = exif_transpose(PILImage.open(p)).convert('RGB')
                orig_sizes.append(im.size)

            # Zero out confidence for padding pixels so downstream ignores them
            imgs_list = []
            pts3d_all = []
            conf_all = []
            for i in range(len(state.image_paths)):
                img_i = imgs_np[i]
                conf_i = depth_conf[i].copy()
                # Mark padding as zero confidence
                padding_mask = img_i.mean(axis=-1) > 0.95
                conf_i[padding_mask] = 0
                imgs_list.append(img_i)
                pts3d_all.append(pts3d[i])
                conf_all.append(conf_i)
            # VGGT cameras: extrinsic is w2c (3x4), invert to c2w
            cam_poses = []
            for i in range(len(extrinsic)):
                w2c_44 = np.eye(4, dtype=np.float32)
                w2c_44[:3, :] = extrinsic[i]
                cam_poses.append(np.linalg.inv(w2c_44))

            # Align to cached cameras if available
            if state.cached_cameras is not None and len(state.cached_cameras) == len(cam_poses):
                print("  Aligning VGGT to cached cameras...")
                pts3d_all, cam_poses = _align_to_cached_cameras(pts3d_all, cam_poses, state.cached_cameras)
                # Rebuild extrinsics from aligned poses
                for i in range(len(cam_poses)):
                    extrinsic[i] = np.linalg.inv(cam_poses[i])[:3, :].astype(np.float32)
                # Use cached intrinsics
                intrinsic = np.stack([c[1].astype(np.float32) for c in state.cached_cameras])

            state.scene = VGGTScene(imgs_list, extrinsic, intrinsic,
                                    pts3d_all, conf_all, orig_sizes)
            if cam_poses:
                ext = np.linalg.norm(points.max(axis=0) - points.min(axis=0)) if len(points) > 0 else 1.0
                scene_gl.set_cameras(cam_poses, scale=float(ext) * 0.05)

            del model, predictions
            torch.cuda.empty_cache()

        elif backend == 'pow3r':
            # Pow3R — run in subprocess to avoid module conflicts with mast3r's dust3r
            import subprocess, pickle, tempfile
            POW3R_DIR = os.path.join(SCRIPT_DIR, 'pow3r')

            # Download checkpoint if not cached
            ckpt_dir = os.path.join(POW3R_DIR, 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, 'Pow3R_ViTLarge_BaseDecoder_512_linear.pth')
            if not os.path.exists(ckpt_path):
                state.status = "Downloading Pow3R weights..."
                import urllib.request
                ckpt_url = "https://download.europe.naverlabs.com/ComputerVision/Pow3R/Pow3R_ViTLarge_BaseDecoder_512_linear.pth"
                urllib.request.urlretrieve(ckpt_url, ckpt_path)

            state.status = "Running Pow3R (subprocess)..."
            state.recon_frac = 0.3

            # Write a temp script that runs Pow3R in a clean Python environment
            result_path = os.path.join(tempfile.gettempdir(), 'pow3r_result.pkl')
            script = f'''
import sys, os, pickle, numpy as np, torch
os.chdir({repr(SCRIPT_DIR)})
sys.path.insert(0, {repr(POW3R_DIR)})
sys.path.insert(0, os.path.join({repr(POW3R_DIR)}, 'dust3r'))
sys.path.insert(0, os.path.join({repr(POW3R_DIR)}, 'dust3r', 'croco'))

from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from pow3r.model.model import Pow3R
import copy

# Load model
ckpt = torch.load({repr(ckpt_path)}, map_location='cpu', weights_only=False)
model = eval(ckpt['definition'])
model.load_state_dict(ckpt['weights'])
model = model.to('cuda').eval()

# Load images
imgs = load_images({repr(state.image_paths)}, size=512, verbose=True)
if len(imgs) == 1:
    imgs = [imgs[0], copy.deepcopy(imgs[0])]

# Inference
pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
output = inference(pairs, model, 'cuda', batch_size=1)

# Global alignment
mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
scene = global_aligner(output, device='cuda', mode=mode)
if mode == GlobalAlignerMode.PointCloudOptimizer:
    scene.compute_global_alignment(init='mst', niter={state.niter1}, schedule='linear', lr=0.01)

# Extract results
pts3d = [to_numpy(p) for p in scene.get_pts3d()]
confs = [to_numpy(c) for c in scene.im_conf] if hasattr(scene, 'im_conf') else [np.ones(p.shape[:2]) for p in pts3d]
c2w = scene.get_im_poses().detach().cpu().numpy()
focals = scene.get_focals().detach().cpu().numpy()
imgs_np = scene.imgs

result = dict(pts3d=pts3d, confs=confs, c2w=c2w, focals=focals, imgs=imgs_np)
with open({repr(result_path)}, 'wb') as f:
    pickle.dump(result, f)
print('SUCCESS')
'''
            script_path = os.path.join(tempfile.gettempdir(), 'pow3r_run.py')
            with open(script_path, 'w') as f:
                f.write(script)

            # Run in subprocess
            proc = subprocess.run(
                [sys.executable, script_path],
                capture_output=True, text=True, timeout=600)
            print(proc.stdout)
            if proc.stderr:
                print(proc.stderr[-2000:])  # last 2000 chars of stderr

            if proc.returncode != 0 or not os.path.exists(result_path):
                state.status = f"Pow3R failed (exit code {proc.returncode})"
            else:
                # Load results
                with open(result_path, 'rb') as f:
                    result = pickle.load(f)
                os.remove(result_path)

                # Build scene from results
                from app import VGGTScene
                pts3d = result['pts3d']
                confs = result['confs']
                c2w_all = result['c2w']
                imgs_np = result['imgs']

                extrinsic = np.zeros((len(c2w_all), 3, 4), dtype=np.float32)
                intrinsic_all = []
                for i in range(len(c2w_all)):
                    w2c = np.linalg.inv(c2w_all[i])
                    extrinsic[i] = w2c[:3, :].astype(np.float32)
                    f = float(result['focals'][i])
                    H, W = pts3d[i].shape[:2]
                    K = np.array([[f, 0, W/2], [0, f, H/2], [0, 0, 1]], dtype=np.float32)
                    intrinsic_all.append(K)

                from PIL import Image as PILImage
                orig_sizes = []
                for p in state.image_paths:
                    im = PILImage.open(p).convert('RGB')
                    orig_sizes.append(im.size)

                # Align to cached cameras if available
                cam_poses_pow3r = [c2w_all[i] for i in range(len(c2w_all))]
                if state.cached_cameras is not None and len(state.cached_cameras) == len(cam_poses_pow3r):
                    print("  Aligning Pow3R to cached cameras...")
                    pts3d, cam_poses_pow3r = _align_to_cached_cameras(pts3d, cam_poses_pow3r, state.cached_cameras)
                    # Rebuild extrinsics + intrinsics from cached cameras
                    for i in range(len(cam_poses_pow3r)):
                        extrinsic[i] = np.linalg.inv(cam_poses_pow3r[i])[:3, :].astype(np.float32)
                    intrinsic_all = [c[1].astype(np.float32) for c in state.cached_cameras]

                state.scene = VGGTScene(imgs_np, extrinsic, np.stack(intrinsic_all),
                                        pts3d, confs, orig_sizes)

                # Display
                all_pts = [p.reshape(-1, 3) for p in pts3d]
                all_cols = [(img.reshape(-1, 3) * 255).astype(np.uint8) for img in imgs_np]
                points = np.concatenate(all_pts, axis=0)
                colors = np.concatenate(all_cols, axis=0)
                if len(points) > 200000:
                    idx = np.random.choice(len(points), 200000, replace=False)
                    points, colors = points[idx], colors[idx]
                scene_gl.set_points(points, colors)
                state.has_points = True; state.needs_recenter = True

                cam_poses = [c2w_all[i] for i in range(len(c2w_all))]
                if cam_poses:
                    ext = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
                    scene_gl.set_cameras(cam_poses, scale=float(ext) * 0.05)

                state.status = f"Pow3R: {len(points):,d} points from {len(imgs_np)} images"

        elif backend == 'colmap':
            # COLMAP SfM — sparse reconstruction, accurate cameras
            state.status = "Running COLMAP SfM..."
            state.recon_frac = 0.1
            import pycolmap
            import tempfile, shutil
            from PIL import Image as PILImage

            workdir = Path(tempfile.mkdtemp(prefix="colmap_"))
            image_dir = workdir / "images"
            image_dir.mkdir()

            # Copy/link images to flat directory
            state.status = "Preparing images..."
            for p in state.image_paths:
                src = Path(p)
                dst = image_dir / src.name
                if not dst.exists():
                    shutil.copy2(str(src), str(dst))

            db_path = workdir / "database.db"
            sparse_path = workdir / "sparse"
            sparse_path.mkdir()

            state.status = "COLMAP: extracting features..."
            state.recon_frac = 0.2
            pycolmap.extract_features(db_path, image_dir)

            state.status = "COLMAP: matching features..."
            state.recon_frac = 0.4
            pycolmap.match_exhaustive(db_path)

            state.status = "COLMAP: sparse reconstruction..."
            state.recon_frac = 0.6
            maps = pycolmap.incremental_mapping(db_path, image_dir, sparse_path)

            if not maps:
                state.status = "COLMAP reconstruction failed (no valid reconstruction)"
                state.refining = False
                state.reconstructing = False
                return

            rec = maps[0]
            print(f"  COLMAP: {len(rec.images)} images, {len(rec.points3D)} points")

            # Build scene data compatible with our pipeline
            # Create a lightweight scene wrapper
            from app import VGGTScene
            import torch

            imgs_np = []
            pts3d_all = []
            conf_all = []
            extrinsics = []
            intrinsics = []

            # Map image names to our paths
            name_to_path = {}
            for p in state.image_paths:
                name_to_path[Path(p).name] = p

            registered_paths = []
            for image_id, image in sorted(rec.images.items()):
                if not image.has_pose:
                    continue
                name = image.name
                if name not in name_to_path:
                    continue

                # Load image
                img_pil = PILImage.open(name_to_path[name]).convert('RGB')
                img_np = np.array(img_pil).astype(np.float32) / 255.0
                H, W = img_np.shape[:2]

                # Camera intrinsics
                cam = rec.cameras[image.camera_id]
                K = cam.calibration_matrix()  # 3x3

                # Camera pose (w2c)
                w2c_34 = image.cam_from_world.matrix()  # 3x4
                w2c = np.eye(4, dtype=np.float32)
                w2c[:3, :] = w2c_34

                # COLMAP sparse points don't give per-pixel depth maps
                # Create a "pseudo depth map" by projecting sparse points into each image
                pts3d_img = np.zeros((H, W, 3), dtype=np.float32)
                conf_img = np.zeros((H, W), dtype=np.float32)

                R = w2c[:3, :3]; t_vec = w2c[:3, 3]
                for pt3d in rec.points3D.values():
                    # Project to this camera
                    p_cam = R @ pt3d.xyz + t_vec
                    if p_cam[2] <= 0:
                        continue
                    u = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
                    v = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
                    ui, vi = int(round(u)), int(round(v))
                    if 0 <= ui < W and 0 <= vi < H:
                        pts3d_img[vi, ui] = pt3d.xyz
                        conf_img[vi, ui] = max(pt3d.track.length(), 1.0)

                imgs_np.append(img_np)
                pts3d_all.append(pts3d_img)
                conf_all.append(conf_img)
                extrinsics.append(w2c[:3, :])
                intrinsics.append(K)
                registered_paths.append(name_to_path[name])

            if not imgs_np:
                state.status = "COLMAP: no images registered"
                state.reconstructing = False
                return

            state.image_paths = registered_paths
            orig_sizes = [(img.shape[1], img.shape[0]) for img in imgs_np]
            state.scene = VGGTScene(
                imgs_np, np.array(extrinsics), np.array(intrinsics),
                pts3d_all, conf_all, orig_sizes)

            # Also build direct point cloud for display
            pts_xyz = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float32)
            pts_rgb = np.array([p.color for p in rec.points3D.values()], dtype=np.uint8)

            scene_gl.set_points(pts_xyz, pts_rgb)
            state.has_points = True
            state.needs_recenter = True
            state.status = f"COLMAP: {len(registered_paths)} cameras, {len(pts_xyz):,d} points"

            # Set cameras for display
            cam_poses = []
            for ext in extrinsics:
                w2c = np.eye(4); w2c[:3, :] = ext
                cam_poses.append(np.linalg.inv(w2c))
            scene_gl.set_cameras(cam_poses, scale=0.1)

        else:
            # DUSt3R / MASt3R

            # Hybrid: run separate camera estimation if camera source differs from backend
            cam_source = state.camera_sources[state.camera_source_idx]
            if cam_source != 'same' and cam_source != backend and state.cached_cameras is None:
                state.status = f"Estimating cameras with {cam_source.upper()}..."
                state.recon_frac = 0.05
                try:
                    state.cached_cameras = _estimate_cameras(state, cam_source)
                    if state.cached_cameras:
                        print(f"  Camera source ({cam_source}): {len(state.cached_cameras)} cameras")
                    else:
                        print(f"  Camera estimation failed, will use {backend} cameras")
                except Exception as e:
                    print(f"  Camera estimation failed: {e}")
                    import traceback; traceback.print_exc()

            from dust3r.utils.image import load_images as dust3r_load_images
            from dust3r.utils.device import to_numpy

            if backend == 'dust3r':
                from dust3r.model import AsymmetricCroCo3DStereo
                from dust3r.inference import inference as dust3r_inference
                from dust3r.image_pairs import make_pairs
                from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

                state.status = "Loading DUSt3R model..."
                state.recon_frac = 0.1
                model = _load_pretrained_cached(
                    AsymmetricCroCo3DStereo,
                    "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt").to('cuda')
                model.eval()

                state.status = "Running DUSt3R inference..."
                state.recon_frac = 0.25
                imgs = dust3r_load_images(state.image_paths, size=512, verbose=True,
                                          patch_size=model.patch_size)
                if len(imgs) == 1:
                    imgs = [imgs[0], copy.deepcopy(imgs[0])]
                    imgs[1]['idx'] = 1

                pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
                output = dust3r_inference(pairs, model, 'cuda', batch_size=1)

                mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
                scene = global_aligner(output, device='cuda', mode=mode)
                state.recon_frac = 0.5

                # Hybrid: use external cameras if available
                print(f"  Cached cameras: {len(state.cached_cameras) if state.cached_cameras else 'None'}, imgs: {len(imgs)}, mode: {mode}")
                if state.cached_cameras is not None and len(state.cached_cameras) == len(imgs) and mode == GlobalAlignerMode.PointCloudOptimizer:
                    state.status = "Setting external cameras (fixed poses)..."
                    import torch as _torch
                    known_poses = [_torch.tensor(c[0], dtype=_torch.float32) for c in state.cached_cameras]
                    known_focals = [c[1][0, 0] for c in state.cached_cameras]
                    scene.preset_pose(_torch.stack(known_poses))
                    scene.preset_focal(_torch.tensor(known_focals))
                    state.status = "Optimizing depth with fixed cameras..."
                    state.recon_frac = 0.55
                    scene.compute_global_alignment(init='known_poses', niter=state.niter1,
                                                   schedule='linear', lr=0.01)
                elif mode == GlobalAlignerMode.PointCloudOptimizer:
                    state.status = "Global alignment..."
                    state.recon_frac = 0.55
                    scene.compute_global_alignment(init='mst', niter=state.niter1,
                                                   schedule='linear', lr=0.01)
                state.scene = scene

            else:  # mast3r
                from mast3r.model import AsymmetricMASt3R
                from mast3r.image_pairs import make_pairs
                from mast3r.cloud_opt.sparse_ga import sparse_global_alignment

                state.status = "Loading MASt3R model..."
                state.recon_frac = 0.1
                model = _load_pretrained_cached(
                    AsymmetricMASt3R,
                    "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric").to('cuda')
                model.eval()

                state.status = "Running MASt3R inference..."
                state.recon_frac = 0.25
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
                state.recon_frac = 0.5
                scene = sparse_global_alignment(
                    paths, pairs, cache_dir, model,
                    lr1=0.07, niter1=state.niter1,
                    lr2=0.014, niter2=0 if opt == 'coarse' else state.niter2,
                    device='cuda', opt_depth='depth' in opt,
                )
                state.scene = scene

            # Extract point cloud for display
            state.status = "Extracting point cloud..."
            state.recon_frac = 0.75
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
                # Build view_ids: which camera each point came from
                all_view_ids = []
                for vi in range(len(all_pts)):
                    all_view_ids.append(np.full(len(all_pts[vi]), vi, dtype=np.int32))

                points = np.concatenate(all_pts, axis=0)
                colors = np.concatenate(all_colors, axis=0)
                pt_view_ids = np.concatenate(all_view_ids, axis=0)

                if len(points) > 200000:
                    idx = np.random.choice(len(points), 200000, replace=False)
                    points, colors = points[idx], colors[idx]

                scene_gl.set_points(points, colors)
                state.has_points = True; state.needs_recenter = True

            # Show camera frustums
            try:
                cam_poses = []
                if state.scene is not None:
                    c2w_all = state.scene.get_im_poses().detach().cpu().numpy()
                    for i in range(len(c2w_all)):
                        cam_poses.append(c2w_all[i])
                if cam_poses:
                    ext = np.linalg.norm(points.max(axis=0) - points.min(axis=0)) if len(points) > 0 else 1.0
                    scene_gl.set_cameras(cam_poses, scale=float(ext) * 0.05)
            except Exception:
                pass

            del model
            torch.cuda.empty_cache()

        # Recentering handled by main loop via needs_recenter flag

        # Auto-align disabled — use Flip Up button instead (more reliable)
        if False and state.has_points and state.scene is not None:
            try:
                c2w_poses = state.scene.get_im_poses().detach().cpu().numpy()

                # Method 1: Camera up vectors (try both OpenCV and OpenGL conventions)
                avg_up_cv = np.zeros(3, dtype=np.float64)   # OpenCV: -Y is up
                avg_up_gl = np.zeros(3, dtype=np.float64)   # OpenGL: +Y is up
                avg_cam_pos = np.zeros(3, dtype=np.float64)
                for i in range(len(c2w_poses)):
                    avg_up_cv -= c2w_poses[i][:3, 1]
                    avg_up_gl += c2w_poses[i][:3, 1]
                    avg_cam_pos += c2w_poses[i][:3, 3]
                avg_up_cv /= max(np.linalg.norm(avg_up_cv), 1e-8)
                avg_up_gl /= max(np.linalg.norm(avg_up_gl), 1e-8)
                avg_cam_pos /= len(c2w_poses)

                # Method 2: Cameras should be above the scene center
                # (most photos are taken from above, looking down or level)
                pts3d_list, _ = _extract_scene_data(state)
                all_pts = []
                for p in pts3d_list:
                    if p is not None:
                        all_pts.append(p.reshape(-1, 3))
                if all_pts:
                    pts_center = np.concatenate(all_pts, axis=0).mean(axis=0)
                else:
                    pts_center = np.zeros(3)

                # Pick the convention where the up vector points from scene center toward cameras
                cam_dir = avg_cam_pos - pts_center
                cam_dir /= max(np.linalg.norm(cam_dir), 1e-8)

                # Which up convention agrees more with "cameras are above scene"?
                score_cv = np.dot(avg_up_cv, cam_dir)
                score_gl = np.dot(avg_up_gl, cam_dir)
                avg_up = avg_up_cv if score_cv > score_gl else avg_up_gl
                print(f"  Up detection: cv={score_cv:.2f}, gl={score_gl:.2f}, using {'cv' if score_cv > score_gl else 'gl'}")

                # Compute rotation to align avg_up with world Y
                target_up = np.array([0.0, 1.0, 0.0])
                dot = np.clip(np.dot(avg_up, target_up), -1.0, 1.0)

                if dot < -0.5:
                    state.scene_rot_x = 180.0
                    print(f"  Auto-flipped (upside-down)")
                elif abs(dot) < 0.99:
                    axis = np.cross(avg_up, target_up)
                    axis /= max(np.linalg.norm(axis), 1e-8)
                    angle = np.arccos(dot)
                    K_mat = np.array([[0, -axis[2], axis[1]],
                                      [axis[2], 0, -axis[0]],
                                      [-axis[1], axis[0], 0]])
                    R_align = np.eye(3) + np.sin(angle) * K_mat + (1 - np.cos(angle)) * (K_mat @ K_mat)
                    sy = np.sqrt(R_align[0, 0] ** 2 + R_align[1, 0] ** 2)
                    if sy > 1e-6:
                        rx = np.arctan2(R_align[2, 1], R_align[2, 2])
                        ry = np.arctan2(-R_align[2, 0], sy)
                        rz = np.arctan2(R_align[1, 0], R_align[0, 0])
                    else:
                        rx = np.arctan2(-R_align[1, 2], R_align[1, 1])
                        ry = np.arctan2(-R_align[2, 0], sy)
                        rz = 0
                    state.scene_rot_x = float(np.degrees(rx))
                    state.scene_rot_y = float(np.degrees(ry))
                    state.scene_rot_z = float(np.degrees(rz))
                    print(f"  Auto-aligned (rot={state.scene_rot_x:.1f}, {state.scene_rot_y:.1f}, {state.scene_rot_z:.1f})")
            except Exception as e:
                print(f"  Auto-align failed: {e}")

        state.recon_frac = 1.0
        state.status = "Reconstruction complete!"
        state.error_msg = ""

    except Exception as e:
        state.error_msg = str(e)
        state.status = f"Error: {e}"
        import traceback
        traceback.print_exc()

    state.reconstructing = False


# ── Main App ─────────────────────────────────────────────────────────────────

def run_depth_injection(state, scene_gl):
    """Inject AI depth into dust3r's scene and regenerate point cloud."""
    state.status = "Injecting AI depth..."
    try:
        from depth_inject import inject_ai_depth
        from dust3r.utils.device import to_numpy

        def progress(frac, msg):
            state.status = msg

        new_pts3d = inject_ai_depth(
            state.scene, state.scene.imgs,
            mix=state.ai_depth_mix,
            highpass_sigma=state.ai_highpass_radius,
            device='cuda', progress_fn=progress)

        # Optionally refine poses to fit the new depth
        if state.ai_refine_poses:
            from depth_inject import refine_poses_with_ai_depth
            state.status = "Refining camera poses..."
            refine_poses_with_ai_depth(
                state.scene, niter=state.ai_pose_iters, lr=0.005,
                progress_fn=progress)
            # Re-get point cloud with refined poses
            new_pts3d = to_numpy(state.scene.get_pts3d())

        # Update viewport
        imgs = state.scene.imgs
        all_pts = []
        all_colors = []
        all_view_ids = []
        for i in range(len(imgs)):
            pts = new_pts3d[i]
            flat = pts.reshape(-1, 3)
            all_pts.append(flat)
            all_colors.append((imgs[i].reshape(-1, 3) * 255).clip(0, 255).astype(np.uint8))
            all_view_ids.append(np.full(len(flat), i, dtype=np.int32))

        points = np.concatenate(all_pts, axis=0)
        colors = np.concatenate(all_colors, axis=0)
        pt_view_ids = np.concatenate(all_view_ids, axis=0)

        if len(points) > 200000:
            idx = np.random.choice(len(points), 200000, replace=False)
            points, colors = points[idx], colors[idx]

        scene_gl.set_points(points, colors)
        state.has_points = True; state.needs_recenter = True
        state.draw_mode = 0

        # Show cameras (with potentially updated poses)
        try:
            c2w_all = to_numpy(state.scene.get_im_poses().detach().cpu())
            cam_poses = [c2w_all[i] for i in range(len(imgs))]
            ext = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
            scene_gl.set_cameras(cam_poses, scale=float(ext) * 0.05)
        except Exception:
            pass

        pose_str = " + poses refined" if state.ai_refine_poses else ""
        state.status = f"AI depth injected (mix={state.ai_depth_mix:.2f}){pose_str}. {len(points):,d} points."

    except Exception as e:
        state.error_msg = str(e)
        state.status = f"Depth injection failed: {e}"
        import traceback
        traceback.print_exc()

    state.reconstructing = False


def _run_stacked_reconstruction(state, scene_gl):
    """Run multiple backends and merge their point clouds."""
    import torch

    # First: get cameras (from camera source or first backend)
    cam_source = state.camera_sources[state.camera_source_idx]
    if cam_source != 'same' and state.cached_cameras is None:
        state.status = f"Estimating cameras with {cam_source.upper()}..."
        try:
            state.cached_cameras = _estimate_cameras(state, cam_source)
        except Exception as e:
            print(f"  Camera estimation failed: {e}")

    # If no external cameras, use DUSt3R as first backend to get cameras
    if state.cached_cameras is None:
        state.status = "Running DUSt3R for cameras..."
        try:
            state.cached_cameras = _estimate_cameras(state, 'dust3r')
        except Exception as e:
            print(f"  DUSt3R camera estimation failed: {e}")

    # Run each backend that can produce dense points
    stack_backends = []
    if state.stack_dust3r: stack_backends.append('dust3r')
    if state.stack_mast3r: stack_backends.append('mast3r')
    if state.stack_vggt: stack_backends.append('vggt')
    if state.stack_pow3r: stack_backends.append('pow3r')
    if not stack_backends:
        state.status = "No backends selected for stacking"
        state.reconstructing = False
        return
    all_pts3d = []
    all_confs = []
    first_scene = None

    for bi, bname in enumerate(stack_backends):
        state.status = f"Stack: running {bname.upper()} ({bi+1}/{len(stack_backends)})..."
        state.recon_frac = bi / len(stack_backends)
        print(f"\n  === Stack: {bname} ===")

        try:
            # Temporarily set backend and run reconstruction
            old_backend_idx = state.backend_idx
            old_stack = state.stack_backends
            state.backend_idx = state.backends.index(bname)
            state.stack_backends = False  # prevent recursion
            state.pts3d_list = None  # clear cache for fresh extraction
            state.confs_list = None

            run_reconstruction(state, scene_gl)

            # Extract the points from this run
            pts3d_list, confs_list = _extract_scene_data(state)
            all_pts3d.extend(pts3d_list)
            all_confs.extend(confs_list)

            if first_scene is None:
                first_scene = state.scene

            state.backend_idx = old_backend_idx
            state.stack_backends = old_stack
            state.pts3d_list = None
            state.confs_list = None

        except Exception as e:
            print(f"  {bname} failed: {e}")
            import traceback; traceback.print_exc()
            state.backend_idx = old_backend_idx
            state.stack_backends = old_stack

        torch.cuda.empty_cache()

    if not all_pts3d:
        state.status = "All backends failed"
        state.reconstructing = False
        return

    # Merge all point clouds
    state.scene = first_scene
    state.pts3d_list = all_pts3d
    state.confs_list = all_confs

    # Display merged cloud
    all_pts, all_cols = [], []
    imgs = state.scene.imgs
    for i in range(len(all_pts3d)):
        p = all_pts3d[i]
        c = all_confs[i]
        if p.ndim == 3:
            mask = c.reshape(p.shape[:2]) > state.min_conf if c is not None else np.ones(p.shape[:2], dtype=bool)
            all_pts.append(p[mask])
            if i < len(imgs):
                all_cols.append((np.clip(imgs[i][mask], 0, 1) * 255).astype(np.uint8))
            else:
                all_cols.append(np.full((mask.sum(), 3), 180, dtype=np.uint8))
        else:
            mask = c.ravel() > state.min_conf if c is not None else np.ones(len(p), dtype=bool)
            all_pts.append(p.reshape(-1, 3)[mask.ravel()] if p.ndim > 1 else p[mask])
            all_cols.append(np.full((mask.sum(), 3), 180, dtype=np.uint8))

    if all_pts:
        points = np.concatenate(all_pts, axis=0)
        colors = np.concatenate(all_cols, axis=0)
        if len(points) > 300000:
            idx = np.random.choice(len(points), 300000, replace=False)
            points, colors = points[idx], colors[idx]
        scene_gl.set_points(points, colors)
        state.has_points = True
        state.needs_recenter = True

    state.status = f"Stacked: {len(all_pts3d)} views from {len(stack_backends)} backends, {sum(len(p.reshape(-1,3)) for p in all_pts3d):,d} total points"
    state.reconstructing = False
    state.recon_frac = 1.0


def _align_to_cached_cameras(pts3d_list, cam_poses_c2w, cached_cameras):
    """Align a reconstruction to the cached camera frame via Procrustes.
    Returns transformed pts3d_list and new c2w poses."""
    n = min(len(cam_poses_c2w), len(cached_cameras))
    if n < 2:
        return pts3d_list, cam_poses_c2w

    # Camera centers from both systems
    src_centers = np.array([cam_poses_c2w[i][:3, 3] for i in range(n)])
    tgt_centers = np.array([cached_cameras[i][0][:3, 3] for i in range(n)])

    # Procrustes: find scale, rotation, translation
    src_mean = src_centers.mean(axis=0)
    tgt_mean = tgt_centers.mean(axis=0)
    src_c = src_centers - src_mean
    tgt_c = tgt_centers - tgt_mean
    scale = np.linalg.norm(tgt_c) / max(np.linalg.norm(src_c), 1e-8)
    H_mat = src_c.T @ tgt_c
    U, _, Vt = np.linalg.svd(H_mat)
    R = (Vt.T @ U.T).astype(np.float32)
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = (Vt.T @ U.T).astype(np.float32)
    t = tgt_mean - scale * (R @ src_mean)

    print(f"  Procrustes alignment: scale={scale:.4f}")

    # Transform all 3D points
    for i in range(len(pts3d_list)):
        pts = pts3d_list[i].reshape(-1, 3).astype(np.float64)
        pts = scale * (pts @ R.T) + t
        pts3d_list[i] = pts.reshape(pts3d_list[i].shape).astype(np.float32)

    # Use cached cameras directly
    new_c2w = [cached_cameras[i][0] for i in range(len(cached_cameras))]
    return pts3d_list, new_c2w


def _extract_scene_data(state):
    """Extract pts3d and confidence from any backend into normalized numpy lists.
    Caches the result — only extracts once per reconstruction."""
    if state.pts3d_list is not None and state.confs_list is not None:
        return state.pts3d_list, state.confs_list

    from dust3r.utils.device import to_numpy
    scene = state.scene
    imgs = scene.imgs

    if hasattr(scene, '_is_vggt'):
        pts3d_list = [np.array(p) for p in scene._pts3d]
        confs_list = [np.array(c) for c in scene._depth_conf]
    elif hasattr(scene, 'canonical_paths'):
        pts3d_raw, _, confs_raw = scene.get_dense_pts3d(clean_depth=True)
        pts3d_list = [to_numpy(pts3d_raw[i]).reshape(imgs[i].shape[0], imgs[i].shape[1], 3)
                      for i in range(len(imgs))]
        confs_list = [to_numpy(confs_raw[i]) for i in range(len(imgs))]
    else:
        if hasattr(scene, 'im_conf'):
            confs_list = [to_numpy(c) for c in scene.im_conf]
        else:
            confs_list = [np.ones(imgs[i].shape[:2], dtype=np.float32) * 10 for i in range(len(imgs))]
        if hasattr(scene, 'clean_pointcloud'):
            scene = scene.clean_pointcloud()
        pts3d_list = [to_numpy(p) for p in scene.get_pts3d()]

    state.pts3d_list = pts3d_list
    state.confs_list = confs_list
    return pts3d_list, confs_list




def _run_smooth_preview(state, scene_gl):
    """Preview the smoothed point cloud without meshing."""
    state.refine_progress = "Smoothing points..."
    try:
        from mesh_export import _smooth_cloud

        pts3d_list, confs_list = _extract_scene_data(state)
        imgs = state.scene.imgs
        mesh_min_conf = state.min_conf

        all_pts, all_cols, all_vids = [], [], []
        for i in range(len(imgs)):
            p, c_arr, img = pts3d_list[i], confs_list[i], imgs[i]
            if p.ndim == 3:
                H, W = p.shape[:2]
                mask = c_arr.reshape(H, W) > mesh_min_conf if c_arr is not None else np.ones((H, W), dtype=bool)
            else:
                mask = c_arr.ravel() > mesh_min_conf if c_arr is not None else np.ones(len(p), dtype=bool)
            n = int(mask.sum())
            all_pts.append(p[mask] if p.ndim == 3 else p[mask])
            all_cols.append((np.clip((img[mask] if p.ndim == 3 else img.reshape(-1, 3)[mask]), 0, 1) * 255).astype(np.uint8))
            all_vids.append(np.full(n, i, dtype=np.int32))

        points = np.concatenate(all_pts, axis=0).astype(np.float32)
        colors = np.concatenate(all_cols, axis=0)
        view_ids = np.concatenate(all_vids, axis=0)

        pts, cols = _smooth_cloud(points, colors, radius_mult=state.smooth_radius, view_ids=view_ids)

        disp_pts, disp_cols = pts, cols
        if len(disp_pts) > 200000:
            idx = np.random.choice(len(disp_pts), 200000, replace=False)
            disp_pts, disp_cols = disp_pts[idx], disp_cols[idx]
        scene_gl.set_points(disp_pts, disp_cols)
        state.points_modified = True
        state.status = f"Smoothed: {len(points):,d} -> {len(pts):,d} points (radius={state.smooth_radius:.1f}x)"
    except Exception as e:
        state.status = f"Smooth preview failed: {e}"
        import traceback; traceback.print_exc()
    finally:
        state.refining = False
        state.refine_progress = ""


def _estimate_cameras(state, source):
    """Run a backend just for camera estimation. Returns list of (c2w, K, W, H) or None."""
    n_imgs = len(state.image_paths)

    if source == 'colmap':
        import pycolmap
        import tempfile, shutil
        workdir = Path(tempfile.mkdtemp(prefix="colmap_cams_"))
        image_dir = workdir / "images"; image_dir.mkdir()
        for p in state.image_paths:
            shutil.copy2(p, str(image_dir / Path(p).name))

        state.refine_progress = "COLMAP: features..."
        pycolmap.extract_features(workdir / "db.db", image_dir)
        state.refine_progress = "COLMAP: matching..."
        pycolmap.match_exhaustive(workdir / "db.db")
        state.refine_progress = "COLMAP: mapping..."
        sparse_dir = workdir / "sparse"; sparse_dir.mkdir()
        maps = pycolmap.incremental_mapping(workdir / "db.db", image_dir, sparse_dir)

        if not maps:
            shutil.rmtree(workdir, ignore_errors=True)
            return None

        rec = maps[0]
        name_to_idx = {Path(p).name: i for i, p in enumerate(state.image_paths)}
        cams = [None] * n_imgs
        for img_id, image in rec.images.items():
            if not image.has_pose or image.name not in name_to_idx:
                continue
            idx = name_to_idx[image.name]
            cam = rec.cameras[image.camera_id]
            K = cam.calibration_matrix().astype(np.float32)
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :] = image.cam_from_world.matrix()
            cams[idx] = (np.linalg.inv(w2c).astype(np.float32), K, cam.width, cam.height)

        shutil.rmtree(workdir, ignore_errors=True)
        return cams if all(c is not None for c in cams) else None

    elif source in ('dust3r', 'mast3r', 'vggt'):
        # Run the selected backend, extract just the cameras, discard the rest
        # For now, we run a quick low-iter reconstruction
        import torch, copy
        if source == 'dust3r':
            from dust3r.model import AsymmetricCroCo3DStereo
            from dust3r.utils.image import load_images as dust3r_load_images
            from dust3r.inference import inference as dust3r_inference
            from dust3r.image_pairs import make_pairs
            from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

            state.refine_progress = f"Loading {source} for cameras..."
            model = _load_pretrained_cached(
                AsymmetricCroCo3DStereo,
                "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt").to('cuda')
            model.eval()
            imgs = dust3r_load_images(state.image_paths, size=512, verbose=False,
                                      patch_size=model.patch_size)
            if len(imgs) == 1:
                imgs = [imgs[0], copy.deepcopy(imgs[0])]; imgs[1]['idx'] = 1
            pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
            output = dust3r_inference(pairs, model, 'cuda', batch_size=1)
            mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
            scene = global_aligner(output, device='cuda', mode=mode)
            if mode == GlobalAlignerMode.PointCloudOptimizer:
                state.refine_progress = f"{source}: aligning cameras..."
                scene.compute_global_alignment(init='mst', niter=100, schedule='linear', lr=0.01)

            c2w_all = scene.get_im_poses().detach().cpu().numpy()
            focals = scene.get_focals().detach().cpu().numpy()
            cams = []
            for i in range(len(c2w_all)):
                H, W = imgs[i]['true_shape'][0].tolist() if hasattr(imgs[i]['true_shape'], 'tolist') else imgs[i]['true_shape']
                K = np.eye(3, dtype=np.float32)
                K[0, 0] = K[1, 1] = float(focals[i])
                K[0, 2] = W / 2; K[1, 2] = H / 2
                cams.append((c2w_all[i].astype(np.float32), K, int(W), int(H)))
            del model; torch.cuda.empty_cache()
            return cams

        elif source == 'vggt':
            from vggt.models.vggt import VGGT
            from vggt.utils.load_fn import load_and_preprocess_images

            state.refine_progress = "Loading VGGT for cameras..."
            model = _load_pretrained_cached(VGGT, "facebook/VGGT-1B").to('cuda')
            model.eval()
            images = load_and_preprocess_images(state.image_paths).to('cuda')
            with torch.no_grad():
                preds = model(images)
            extrinsic = preds["extrinsic"].cpu().numpy()[0]

            cams = []
            for i in range(len(state.image_paths)):
                from PIL import Image as PILImage
                img = PILImage.open(state.image_paths[i])
                W, H = img.size
                w2c = np.eye(4, dtype=np.float32)
                w2c[:3, :] = extrinsic[i]
                c2w = np.linalg.inv(w2c).astype(np.float32)
                intrinsic = preds["intrinsic"].cpu().numpy()[0, i]
                K = intrinsic.astype(np.float32)
                cams.append((c2w, K, W, H))
            del model; torch.cuda.empty_cache()
            return cams

    return None


def run_upscale_points(state, scene_gl):
    """Upscale point cloud using monocular depth at full image resolution.

    Uses DepthAnything V2 to estimate dense depth per image, aligns it
    to the existing reconstruction's depth, then back-projects at full
    image resolution using the known cameras. This gives as many 3D points
    as there are pixels in the original images.
    """
    state.refine_progress = "Upscaling with mono depth..."
    try:
        import torch
        from PIL import Image as PILImage

        scene = state.scene
        if scene is None:
            state.status = "No reconstruction to upscale"
            state.refining = False
            return

        # Get existing scene data for depth alignment
        pts3d_list, confs_list = _extract_scene_data(state)
        c2w_all = scene.get_im_poses().detach().cpu().numpy()
        n_imgs = len(state.image_paths)

        # Load mono depth model
        state.refine_progress = "Loading depth model..."
        try:
            pipe = torch.hub.load('huggingface/pytorch-transformers', 'pipeline',
                                   'depth-estimation', model='depth-anything/Depth-Anything-V2-Small-hf',
                                   device='cuda')
        except Exception:
            # Fallback: use transformers directly
            from transformers import pipeline as hf_pipeline
            pipe = hf_pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Small-hf',
                              device='cuda')

        all_dense_pts = []
        all_dense_cols = []

        for i in range(n_imgs):
            state.refine_progress = f"Depth {i+1}/{n_imgs}..."
            img_pil = PILImage.open(state.image_paths[i]).convert('RGB')
            W_full, H_full = img_pil.size

            # Get mono depth at full resolution
            result = pipe(img_pil)
            mono_depth = np.array(result['depth']).astype(np.float32)
            if mono_depth.shape != (H_full, W_full):
                mono_depth = np.array(PILImage.fromarray(mono_depth).resize((W_full, H_full), PILImage.BILINEAR))

            # Align mono depth to existing reconstruction depth
            # Use the existing pts3d for this view to find scale + offset
            existing_pts = pts3d_list[i] if i < len(pts3d_list) else None
            if existing_pts is not None and existing_pts.ndim == 3:
                H_r, W_r = existing_pts.shape[:2]
                conf = confs_list[i] if i < len(confs_list) else None
                mask = conf.reshape(H_r, W_r) > state.min_conf if conf is not None else np.ones((H_r, W_r), dtype=bool)

                # Project existing 3D points to get their depth in camera space
                w2c = np.linalg.inv(c2w_all[i])
                R, t = w2c[:3, :3], w2c[:3, 3]
                pts_cam = (R @ existing_pts[mask].astype(np.float64).T).T + t
                existing_z = pts_cam[:, 2]

                # Sample mono depth at corresponding pixel positions
                rows, cols = np.where(mask)
                # Scale pixel coords from reconstruction res to full res
                scale_y = H_full / H_r
                scale_x = W_full / W_r
                mono_y = np.clip((rows * scale_y).astype(int), 0, H_full - 1)
                mono_x = np.clip((cols * scale_x).astype(int), 0, W_full - 1)
                mono_z = mono_depth[mono_y, mono_x]

                # Build nonlinear LUT from mono depth → metric depth
                # using matched pixel pairs from the existing reconstruction
                valid = (existing_z > 0.01) & (mono_z > 0.01)
                if valid.sum() > 50:
                    ez = existing_z[valid]
                    mz = mono_z[valid]

                    # Check if inverted (DepthAnything: larger=closer)
                    if np.corrcoef(ez, 1.0 / np.maximum(mz, 1e-6))[0, 1] > np.corrcoef(ez, mz)[0, 1]:
                        print(f"    Mono depth is inverted, converting...")
                        mono_depth = 1.0 / np.maximum(mono_depth, 1e-6)
                        mz = 1.0 / np.maximum(mz, 1e-6)

                    # Build LUT: sort mono values, map to corresponding metric values
                    # Use percentile bins for robustness
                    n_bins = 100
                    percentiles = np.linspace(0, 100, n_bins + 1)
                    mono_bins = np.percentile(mz, percentiles)
                    metric_bins = np.zeros(n_bins + 1, dtype=np.float64)
                    for bi in range(n_bins + 1):
                        lo = mono_bins[max(0, bi - 1)]
                        hi = mono_bins[min(bi + 1, n_bins)]
                        in_bin = (mz >= lo) & (mz <= hi)
                        if in_bin.any():
                            metric_bins[bi] = np.median(ez[in_bin])
                        elif bi > 0:
                            metric_bins[bi] = metric_bins[bi - 1]

                    # Smooth the LUT to avoid noise
                    from scipy.ndimage import uniform_filter1d
                    metric_bins = uniform_filter1d(metric_bins, size=5)

                    # Ensure monotonic (sort of — metric depth should increase with mono depth)
                    for bi in range(1, len(metric_bins)):
                        metric_bins[bi] = max(metric_bins[bi], metric_bins[bi - 1] + 1e-6)

                    print(f"    LUT: mono [{mono_bins[0]:.3f}-{mono_bins[-1]:.3f}] -> metric [{metric_bins[0]:.3f}-{metric_bins[-1]:.3f}]")

                    # Apply LUT to full-res mono depth via interpolation
                    aligned_depth = np.interp(mono_depth.ravel(), mono_bins, metric_bins).reshape(mono_depth.shape)
                    depth_lut = True
                else:
                    depth_lut = False
            else:
                depth_lut = False

            if not depth_lut:
                # Fallback: no alignment data, use raw mono depth
                aligned_depth = mono_depth
                print(f"    WARNING: no depth alignment data, using raw mono depth")

            aligned_depth = np.maximum(aligned_depth, 0.01).astype(np.float32)

            # Back-project full-res pixels to 3D using known camera
            # Estimate K at full resolution from existing scene
            try:
                focals = scene.get_focals().detach().cpu().numpy()
                f_val = float(focals[i].item() if hasattr(focals[i], 'item') else focals[i])
                fx = fy = f_val * (W_full / (existing_pts.shape[1] if existing_pts is not None and existing_pts.ndim == 3 else W_full))
            except Exception:
                fx = fy = W_full  # fallback
            cx, cy = W_full / 2.0, H_full / 2.0
            c2w = c2w_all[i]

            # Create pixel grid
            us, vs = np.meshgrid(np.arange(W_full), np.arange(H_full))
            us = us.astype(np.float32).ravel()
            vs = vs.astype(np.float32).ravel()
            z = aligned_depth.ravel()

            # Camera-space 3D
            x_cam = (us - cx) * z / fx
            y_cam = (vs - cy) * z / fy
            pts_cam = np.stack([x_cam, y_cam, z], axis=-1)

            # To world space
            R_c2w = c2w[:3, :3]
            t_c2w = c2w[:3, 3]
            pts_world = (R_c2w @ pts_cam.T).T + t_c2w

            # Colors from full-res image
            img_np = np.array(img_pil).reshape(-1, 3)

            # Filter: remove sky/invalid (very far or very close)
            med_z = np.median(z[z > 0.1])
            valid_mask = (z > 0.01) & (z < med_z * 5)
            pts_world = pts_world[valid_mask].astype(np.float32)
            img_np = img_np[valid_mask]

            all_dense_pts.append(pts_world)
            all_dense_cols.append(img_np)
            print(f"  Image {i+1}: {len(pts_world):,d} dense points")

        del pipe
        torch.cuda.empty_cache()

        if all_dense_pts:
            dense_pts = np.concatenate(all_dense_pts, axis=0)
            dense_cols = np.concatenate(all_dense_cols, axis=0)
            print(f"  Total dense: {len(dense_pts):,d} points")

            # Store as additional scene data
            state.pts3d_list = list(pts3d_list) + [dense_pts]
            state.confs_list = list(confs_list) + [np.ones(len(dense_pts), dtype=np.float32) * 5.0]

            # Display
            disp_pts, disp_cols = dense_pts, dense_cols
            if len(disp_pts) > 300000:
                idx = np.random.choice(len(disp_pts), 300000, replace=False)
                disp_pts, disp_cols = disp_pts[idx], disp_cols[idx]
            scene_gl.set_points(disp_pts, disp_cols)
            state.has_points = True
            state.points_modified = True
            state.status = f"Upscaled: {len(dense_pts):,d} dense points from {n_imgs} images"

    except Exception as e:
        state.status = f"Upscale failed: {e}"
        import traceback; traceback.print_exc()
    finally:
        state.refining = False
        state.refine_progress = ""


def run_dense_mesh(state, scene_gl):
    """Generate dense mesh."""
    state.refine_progress = "Generating dense mesh..."
    try:
        scene = state.scene
        if scene is None:
            state.status = "No reconstruction available"
            state.refining = False
            return

        imgs = scene.imgs
        pts3d_list, confs_list = _extract_scene_data(state)
        mesh_min_conf = state.min_conf

        from mesh_export import create_dense_mesh, _smooth_cloud, _collect_points
        from mesh_export import _mesh_local_delaunay, _mesh_ball_pivot_from_cloud, _close_holes_pymeshlab

        # Get camera poses
        cam_poses = None
        cam_center = None
        try:
            c2w = scene.get_im_poses().detach().cpu().numpy()
            cam_poses = [c2w[i] for i in range(len(imgs))]
            cam_center = np.mean([c[:3, 3] for c in cam_poses], axis=0)
        except Exception:
            pass

        mesh_mode = state.mesh_modes[state.mesh_mode_idx]

        # If smoothing enabled, collect all points, smooth, then mesh the unified cloud
        if state.use_smoothing:
            state.refine_progress = "Collecting points..."
            all_pts, all_cols, all_vids = [], [], []
            for i in range(len(imgs)):
                p, c_arr, img = pts3d_list[i], confs_list[i], imgs[i]
                if p.ndim == 3:
                    H, W = p.shape[:2]
                    mask = c_arr.reshape(H, W) > mesh_min_conf if c_arr is not None else np.ones((H, W), dtype=bool)
                    all_pts.append(p[mask]); all_cols.append((np.clip(img[mask], 0, 1) * 255).astype(np.uint8))
                    all_vids.append(np.full(mask.sum(), i, dtype=np.int32))
                else:
                    mask = c_arr.ravel() > mesh_min_conf if c_arr is not None else np.ones(len(p), dtype=bool)
                    all_pts.append(p[mask]); all_cols.append((np.clip(img.reshape(-1, 3)[mask], 0, 1) * 255).astype(np.uint8))
                    all_vids.append(np.full(mask.sum(), i, dtype=np.int32))

            points = np.concatenate(all_pts, axis=0).astype(np.float32)
            colors_raw = np.concatenate(all_cols, axis=0)
            view_ids = np.concatenate(all_vids, axis=0)

            state.refine_progress = f"Smoothing {len(points):,d} points..."
            pts, cols = _smooth_cloud(points, colors_raw,
                                       radius_mult=state.smooth_radius, view_ids=view_ids)

            state.refine_progress = f"Meshing {len(pts):,d} points ({mesh_mode})..."
            if mesh_mode == 'delaunay':
                verts, faces, colors = _mesh_local_delaunay(pts, cols, cam_center=cam_center)
            else:
                verts, faces, colors = _mesh_ball_pivot_from_cloud(pts, cols, cam_center=cam_center)

            if state.hole_cap_size > 0 and len(faces) > 0:
                verts, faces, colors = _close_holes_pymeshlab(verts, faces, colors, state.hole_cap_size)

        else:
            # No smoothing — use per-view data with selected method
            print(f"  min_conf: {mesh_min_conf:.3f}, mode: {mesh_mode}")
            state.refine_progress = f"Creating mesh ({mesh_mode})..."
            verts, faces, colors = create_dense_mesh(
                imgs, pts3d_list, confs_list,
                cam2world_list=cam_poses, min_conf=mesh_min_conf,
                mode=mesh_mode, hole_cap_size=state.hole_cap_size)

        if len(faces) > 0:
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


def run_decimate(state, scene_gl):
    """Decimate existing mesh to target face count."""
    state.refine_progress = "Decimating mesh..."
    try:
        verts, faces, colors = state.mesh_data
        target = state.target_faces
        n_before = len(faces)

        if n_before <= target:
            state.status = f"Mesh already has {n_before:,d} faces (<= target {target:,d})"
            state.refining = False
            return

        state.refine_progress = f"Decimating {n_before:,d} -> {target:,d} faces..."
        print(f"  Decimating {n_before:,d} -> {target:,d} faces...")

        # Try PyMeshLab first (best quality), fall back to Open3D
        try:
            import pymeshlab
            ms = pymeshlab.MeshSet()
            m = pymeshlab.Mesh(
                vertex_matrix=verts.astype(np.float64),
                face_matrix=faces.astype(np.int32),
                v_color_matrix=np.column_stack([
                    colors.astype(np.float64) / 255.0,
                    np.ones((len(colors), 1), dtype=np.float64)
                ])
            )
            ms.add_mesh(m)

            try:
                ms.meshing_decimation_quadric_edge_collapse(
                    targetfacenum=target,
                    qualitythr=0.5,
                    preserveboundary=True,
                    preservenormal=True,
                    preservetopology=True,
                    optimalplacement=True,
                    autoclean=True)
            except TypeError:
                # Older PyMeshLab versions may have different parameter names
                ms.simplification_quadric_edge_collapse_decimation(
                    targetfacenum=target,
                    preserveboundary=True,
                    preservenormal=True,
                    optimalplacement=True)

            mesh_out = ms.current_mesh()
            verts_out = mesh_out.vertex_matrix().astype(np.float32)
            faces_out = mesh_out.face_matrix().astype(np.int32)
            if mesh_out.has_vertex_color():
                vc = mesh_out.vertex_color_matrix()
                colors_out = (vc[:, :3] * 255).clip(0, 255).astype(np.uint8)
            else:
                colors_out = np.full((len(verts_out), 3), 128, dtype=np.uint8)
            print(f"  Decimated with PyMeshLab: {len(faces_out):,d} faces")

        except Exception as e_pml:
            print(f"  PyMeshLab decimation failed ({e_pml}), using Open3D...")
            import open3d as o3d
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
            mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target)
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_vertices()
            verts_out = np.asarray(mesh.vertices, dtype=np.float32)
            faces_out = np.asarray(mesh.triangles, dtype=np.int32)
            colors_out = (np.asarray(mesh.vertex_colors) * 255).clip(0, 255).astype(np.uint8)
            print(f"  Decimated with Open3D: {len(faces_out):,d} faces")

        state.mesh_data = (verts_out, faces_out, colors_out)
        scene_gl.set_mesh(verts_out, faces_out, colors_out)
        state.status = f"Decimated: {n_before:,d} -> {len(faces_out):,d} faces"
        print(f"  Done: {len(faces_out):,d} faces")
    except Exception as e:
        state.status = f"Decimation failed: {e}"
        print(f"  Decimation error: {e}")
        import traceback; traceback.print_exc()
    finally:
        state.refining = False


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
        state.has_points = True; state.needs_recenter = True
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


def _recolor_from_cameras(verts, faces, views):
    """Re-color vertices by projecting into cameras with incidence weighting."""
    V = len(verts)
    color_accum = np.zeros((V, 3), dtype=np.float64)
    weight_accum = np.zeros(V, dtype=np.float64)

    # Vertex normals
    v0 = verts[faces[:, 0]]; v1 = verts[faces[:, 1]]; v2 = verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn /= (np.linalg.norm(fn, axis=-1, keepdims=True) + 1e-8)
    vn = np.zeros((V, 3), dtype=np.float64)
    for ax in range(3):
        np.add.at(vn[:, ax], faces[:, 0], fn[:, ax])
        np.add.at(vn[:, ax], faces[:, 1], fn[:, ax])
        np.add.at(vn[:, ax], faces[:, 2], fn[:, ax])
    vn /= (np.linalg.norm(vn, axis=-1, keepdims=True) + 1e-8)

    for view in views:
        R = view['w2c'][:3, :3]
        t = view['w2c'][:3, 3]
        c2w = np.linalg.inv(view['w2c'])
        cam_center = c2w[:3, 3]

        pc = (R @ verts.T).T + t
        zz = np.clip(pc[:, 2], 0.01, None)
        uu = (pc[:, 0] / zz * view['K'][0, 0] + view['K'][0, 2]).astype(int)
        vv = (pc[:, 1] / zz * view['K'][1, 1] + view['K'][1, 2]).astype(int)
        ok = (zz > 0.01) & (uu >= 0) & (uu < view['W']) & (vv >= 0) & (vv < view['H'])

        if ok.any():
            view_dirs = cam_center[None, :] - verts[ok]
            view_dirs /= (np.linalg.norm(view_dirs, axis=-1, keepdims=True) + 1e-8)
            dots = (vn[ok] * view_dirs).sum(axis=-1).clip(0, 1)
            w = dots ** 2

            sampled = view['pixels'][vv[ok], uu[ok]]
            color_accum[ok] += sampled * w[:, None]
            weight_accum[ok] += w

    has_w = weight_accum > 0.001
    if not has_w.any():
        return None
    colors = np.zeros((V, 3), dtype=np.uint8)
    colors[has_w] = (color_accum[has_w] / weight_accum[has_w, None] * 255).clip(0, 255).astype(np.uint8)
    return colors


def run_recolor_mesh(state, scene_gl):
    """Recolor mesh vertices by projecting each vertex into all cameras and averaging pixel colors."""
    state.refine_progress = "Recoloring mesh from cameras..."
    try:
        from colmap_export import export_scene_to_colmap
        from refine_mesh import load_cameras
        import tempfile

        if state.mesh_data is None:
            state.status = "No mesh to recolor"
            state.refining = False
            return

        verts, faces, colors = state.mesh_data

        # Export cameras
        tmpdir = tempfile.mkdtemp()
        export_dir = os.path.join(tmpdir, 'colmap')
        state.refine_progress = "Exporting cameras..."
        export_scene_to_colmap(
            scene=state.scene, image_paths=state.image_paths,
            output_dir=export_dir, min_conf_thr=state.min_conf)
        views = load_cameras(export_dir)

        V = len(verts)
        color_sum = np.zeros((V, 3), dtype=np.float64)
        weight_sum = np.zeros(V, dtype=np.float64)

        # Compute vertex normals for incidence weighting
        v0 = verts[faces[:, 0]]; v1 = verts[faces[:, 1]]; v2 = verts[faces[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        fn /= (np.linalg.norm(fn, axis=-1, keepdims=True) + 1e-8)
        vert_normals = np.zeros((V, 3), dtype=np.float64)
        for ax in range(3):
            np.add.at(vert_normals[:, ax], faces[:, 0], fn[:, ax])
            np.add.at(vert_normals[:, ax], faces[:, 1], fn[:, ax])
            np.add.at(vert_normals[:, ax], faces[:, 2], fn[:, ax])
        vert_normals /= (np.linalg.norm(vert_normals, axis=-1, keepdims=True) + 1e-8)

        # Global color correction: compute per-camera mean color for normalization
        from texture_map import _rasterize_visibility
        cam_means = []
        cam_samples = []

        for ci, view in enumerate(views):
            state.refine_progress = f"Camera {ci+1}/{len(views)}..."
            R = view['w2c'][:3, :3]
            t_vec = view['w2c'][:3, 3]
            K = view['K']
            W_img, H_img = view['W'], view['H']
            c2w = np.linalg.inv(view['w2c'])
            cam_center = c2w[:3, 3]

            # Project all vertices
            pts_cam = (R @ verts.astype(np.float64).T).T + t_vec
            z = pts_cam[:, 2]
            valid = z > 0.01
            px = np.full(V, -1.0); py = np.full(V, -1.0)
            px[valid] = pts_cam[valid, 0] / z[valid] * K[0, 0] + K[0, 2]
            py[valid] = pts_cam[valid, 1] / z[valid] * K[1, 1] + K[1, 2]

            in_frame = valid & (px >= 0) & (px < W_img - 1) & (py >= 0) & (py < H_img - 1)
            if not in_frame.any():
                cam_means.append(np.array([0.5, 0.5, 0.5]))
                cam_samples.append(0)
                continue

            # Occlusion test
            u_vert = np.zeros(V); v_vert = np.zeros(V); z_vert = np.zeros(V)
            u_vert[valid] = px[valid]; v_vert[valid] = py[valid]; z_vert[valid] = z[valid]
            visible_faces = _rasterize_visibility(faces, u_vert, v_vert, z_vert, W_img, H_img)

            vert_visible = np.zeros(V, dtype=bool)
            if visible_faces:
                vis_face_arr = np.array(list(visible_faces), dtype=np.int64)
                vis_verts = np.unique(faces[vis_face_arr].ravel())
                vert_visible[vis_verts] = True

            in_frame &= vert_visible
            if not in_frame.any():
                cam_means.append(np.array([0.5, 0.5, 0.5]))
                cam_samples.append(0)
                continue

            # Incidence angle weight: dot(normal, view_dir)
            view_dirs = cam_center[None, :] - verts[in_frame]
            view_dirs /= (np.linalg.norm(view_dirs, axis=-1, keepdims=True) + 1e-8)
            ndot = (vert_normals[in_frame] * view_dirs).sum(axis=-1)
            ndot = np.clip(ndot, 0, 1)  # back-facing = 0 weight

            # Border falloff (smooth, using cosine)
            u_f = px[in_frame]; v_f = py[in_frame]
            norm_u = u_f / W_img  # 0 to 1
            norm_v = v_f / H_img
            border = np.minimum(
                np.minimum(norm_u, 1 - norm_u),
                np.minimum(norm_v, 1 - norm_v)
            ) * 4.0  # 0 at edge, 1 at center quarter
            border = np.clip(border, 0, 1)
            # Smooth with cosine falloff
            border = 0.5 - 0.5 * np.cos(border * np.pi)

            # Combined weight: soft incidence * smooth border
            # Use sqrt for gentler falloff — avoids sharp banding on curved surfaces
            incidence = np.clip(ndot, 0.1, 1.0)  # floor at 0.1 so grazing views still contribute
            w = incidence * border

            # Bilinear sample
            u0 = np.floor(u_f).astype(int); v0 = np.floor(v_f).astype(int)
            u1 = np.minimum(u0 + 1, W_img - 1); v1 = np.minimum(v0 + 1, H_img - 1)
            fu = u_f - u0; fv = v_f - v0
            sampled = ((1 - fu)[:, None] * (1 - fv)[:, None] * view['pixels'][v0, u0] +
                       fu[:, None] * (1 - fv)[:, None] * view['pixels'][v0, u1] +
                       (1 - fu)[:, None] * fv[:, None] * view['pixels'][v1, u0] +
                       fu[:, None] * fv[:, None] * view['pixels'][v1, u1])

            cam_means.append(sampled.mean(axis=0) if len(sampled) > 0 else np.array([0.5, 0.5, 0.5]))
            cam_samples.append((in_frame, sampled, w))

        # Global color correction: normalize all cameras to the one with most samples
        ref_idx = max(range(len(cam_samples)), key=lambda i: len(cam_samples[i][2]) if isinstance(cam_samples[i], tuple) else 0)
        ref_mean = cam_means[ref_idx]
        print(f"  Color reference: camera {ref_idx+1} (mean={ref_mean})")

        for ci in range(len(views)):
            if not isinstance(cam_samples[ci], tuple):
                continue
            in_frame, sampled, w = cam_samples[ci]
            if len(w) == 0:
                continue

            # Per-channel scale to match reference camera
            scale = np.ones(3, dtype=np.float64)
            for ch in range(3):
                if cam_means[ci][ch] > 0.01:
                    scale[ch] = np.clip(ref_mean[ch] / cam_means[ci][ch], 0.7, 1.4)

            corrected = sampled * scale[None, :]
            color_sum[in_frame] += corrected * w[:, None]
            weight_sum[in_frame] += w

        # Average
        has_color = weight_sum > 0.001
        new_colors = colors.copy()
        new_colors[has_color] = (color_sum[has_color] / weight_sum[has_color, None] * 255).clip(0, 255).astype(np.uint8)

        n_colored = has_color.sum()
        print(f"  Recolored {n_colored:,d} / {V:,d} vertices from {len(views)} cameras")

        state.mesh_data = (verts, faces, new_colors)
        scene_gl.set_mesh(verts, faces, new_colors)
        state.status = f"Recolored {n_colored:,d} vertices from {len(views)} cameras"

    except Exception as e:
        state.status = f"Recolor failed: {e}"
        import traceback; traceback.print_exc()
    finally:
        state.refining = False
        state.refine_progress = ""


def run_texture(state, scene_gl):
    """Generate UV-mapped textured mesh using PyMeshLab."""
    state.refine_progress = "Generating texture..."
    try:
        import tempfile
        from colmap_export import export_scene_to_colmap
        from refine_mesh import load_cameras

        tmpdir = tempfile.mkdtemp()
        export_dir = os.path.join(tmpdir, 'colmap')

        state.refine_progress = "Exporting cameras..."
        export_scene_to_colmap(
            scene=state.scene, image_paths=state.image_paths,
            output_dir=export_dir, min_conf_thr=state.min_conf)

        views = load_cameras(export_dir)
        verts, faces, colors = state.mesh_data

        output_dir = os.path.join(tmpdir, 'textured')
        os.makedirs(output_dir, exist_ok=True)

        state.refine_progress = f"UV unwrapping + projecting {len(faces):,d} faces..."
        from texture_map import create_textured_mesh
        obj_path, uvs, uv_faces, texture_img = create_textured_mesh(
            verts, faces, colors, views, output_dir, return_data=True)

        # Upload texture + UVs to viewport for live textured rendering
        state.refine_progress = "Uploading texture to viewport..."
        if texture_img is not None and uvs is not None:
            scene_gl.set_texture(texture_img, uvs, uv_faces, verts, faces, colors)
        else:
            # Fallback: update vertex colors
            new_colors = _recolor_from_cameras(verts, faces, views)
            if new_colors is not None:
                colors = new_colors
            state.mesh_data = (verts, faces, colors)
            scene_gl.set_mesh(verts, faces, colors)

        state.status = f"Textured mesh saved to {output_dir} ({len(faces):,d} faces)"
        state.refine_progress = ""

        import subprocess
        subprocess.Popen(['explorer', output_dir.replace('/', '\\')])

    except Exception as e:
        state.error_msg = str(e)
        state.status = f"Texture failed: {e}"
        import traceback
        traceback.print_exc()

    state.refining = False
    state.refine_progress = ""


def _sample_vertex_colors_from_obj(obj_path):
    """Load a textured OBJ, sample the texture at each vertex's UV, return vertex colors."""
    try:
        import trimesh
        mesh = trimesh.load(obj_path, process=False)
        if not hasattr(mesh, 'visual') or not hasattr(mesh.visual, 'material'):
            return None
        # Try to get per-vertex colors from the textured mesh
        # trimesh can interpolate texture to vertex colors
        vc = mesh.visual.to_color()
        if hasattr(vc, 'vertex_colors') and vc.vertex_colors is not None:
            colors = np.asarray(vc.vertex_colors[:, :3], dtype=np.uint8)
            if len(colors) > 0 and colors.sum() > 0:
                print(f"  Sampled {len(colors):,d} vertex colors from texture")
                return colors
    except Exception as e:
        print(f"  Could not sample from OBJ texture: {e}")
    return None


def _texture_pymeshlab(verts, faces, colors, views, output_dir, state):
    """Use PyMeshLab for multi-view texture projection with color correction."""
    import pymeshlab
    import trimesh
    from scipy.spatial.transform import Rotation

    state.refine_progress = "PyMeshLab: preparing mesh..."
    print(f"  PyMeshLab texturing: {len(verts):,d} verts, {len(faces):,d} faces, {len(views)} cameras")

    # Save mesh as PLY for PyMeshLab
    mesh_path = os.path.join(output_dir, 'input_mesh.ply')
    tm = trimesh.Trimesh(vertices=verts, faces=faces,
                         vertex_colors=np.column_stack([colors, np.full(len(colors), 255, dtype=np.uint8)]))
    tm.export(mesh_path)

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_path)

    # Load camera images as rasters with proper projection matrices
    state.refine_progress = "PyMeshLab: loading cameras..."
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    from PIL import Image as PILImage
    for ci, view in enumerate(views):
        # Save image
        img_uint8 = (np.clip(view['pixels'], 0, 1) * 255).astype(np.uint8)
        img_path = os.path.join(images_dir, f'cam_{ci:04d}.png')
        PILImage.fromarray(img_uint8).save(img_path)

        # Load as raster in PyMeshLab
        ms.load_new_raster(img_path)

        # Set camera parameters
        W, H = view['W'], view['H']
        K = view['K']
        w2c = view['w2c']
        c2w = np.linalg.inv(w2c)

        # PyMeshLab shot: needs rotation matrix, translation, focal, principal point
        R = w2c[:3, :3]
        t = w2c[:3, 3]

        # Create the shot
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # PyMeshLab uses a specific camera model
        # Set intrinsics via the raster's shot
        shot = ms.current_raster().shot()

        # Set extrinsics: rotation + translation
        # PyMeshLab rotation is stored as a 4x4 matrix
        rot_matrix = pymeshlab.Matrix44f()
        for r_i in range(3):
            for r_j in range(3):
                rot_matrix[r_i, r_j] = float(R[r_i, r_j])
        rot_matrix[0, 3] = float(t[0])
        rot_matrix[1, 3] = float(t[1])
        rot_matrix[2, 3] = float(t[2])
        rot_matrix[3, 3] = 1.0

        shot.set_extr_from_matrix(rot_matrix)

        # Set intrinsics
        intr = shot.intrinsics()
        intr.set_focal_mm(float(fx))
        intr.set_pixel_size_mm(pymeshlab.Point2f(1.0, 1.0))
        intr.set_center_px(pymeshlab.Point2f(float(cx), float(cy)))
        intr.set_viewport_px(pymeshlab.Point2i(int(W), int(H)))
        shot.set_intrinsics(intr)

        ms.current_raster().set_shot(shot)

    # Switch back to mesh (index 0)
    ms.set_current_mesh(0)

    # Project raster colors to vertex colors
    state.refine_progress = "PyMeshLab: projecting textures..."
    print("  Projecting raster colors...")

    try:
        ms.project_active_rasters_color_to_current_mesh(
            deptheta=0.5,
            onselectedraster=False,
            usedepth=True,
            useangle=True,
            useborders=True,
            usesil=True
        )
    except Exception as e:
        print(f"  project_active_rasters failed: {e}")
        # Try simpler version
        ms.project_active_rasters_color_to_current_mesh()

    # Try parameterization + texturing from rasters
    state.refine_progress = "PyMeshLab: generating texture atlas..."
    tex_name = "texture"
    tex_size = 4096

    try:
        ms.parameterization_and_texturing_from_registered_rasters(
            textname=tex_name,
            texsize=tex_size,
            colorcorrection=True,
            usedistanceweight=True,
            useangle=True,
            useborders=True,
        )
        obj_path = os.path.join(output_dir, 'mesh.obj')
        ms.save_current_mesh(obj_path)
        print(f"  PyMeshLab texture saved: {obj_path}")
        return obj_path

    except Exception as e:
        print(f"  parameterization_and_texturing failed: {e}")
        # Fall back: save with vertex colors at least
        obj_path = os.path.join(output_dir, 'mesh.obj')
        ms.save_current_mesh(obj_path)
        print(f"  Saved with vertex colors: {obj_path}")
        return obj_path


def main():
    _console.install()

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
        # Ignore input when window is not focused
        if not glfw.get_window_attrib(window, glfw.FOCUSED):
            return
        if imgui.get_io().want_capture_mouse:
            return
        if action == glfw.PRESS:
            mouse_down[button] = True
        elif action == glfw.RELEASE:
            mouse_down[button] = False

    def scroll_callback(window, xoffset, yoffset):
        if not glfw.get_window_attrib(window, glfw.FOCUSED):
            return
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

        # Clear mouse state when window loses focus (prevents stuck drag)
        if not glfw.get_window_attrib(window, glfw.FOCUSED):
            mouse_down[0] = mouse_down[1] = mouse_down[2] = False

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
        imgui.begin("Controls", flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_TITLE_BAR)
        imgui.begin_child("ScrollRegion", 0, 0, border=False)

        # Status + console toggle
        imgui.text_colored(state.status, 0.4, 0.8, 1.0)
        if state.error_msg:
            imgui.text_colored(state.error_msg, 1.0, 0.3, 0.3)
        _, state.show_console = imgui.checkbox("Console", state.show_console)
        imgui.separator()

        # ── Project ──
        if imgui.button("New Project"):
            # Reset everything
            if hasattr(state, '_ever_centered'):
                del state._ever_centered
            state.__init__()
            scene_gl.point_count = 0
            scene_gl.mesh_face_count = 0
            scene_gl.mesh_has_uvs = False
            scene_gl.mesh_tex_id = None
            if hasattr(scene_gl, '_mesh_uvs'):
                scene_gl._mesh_uvs = None
            camera.distance = 3.0
            camera.target = np.zeros(3, dtype=np.float32)
            camera.yaw = camera.pitch = 0.0
            state.status = "New project"

        imgui.same_line()
        if imgui.button("Save Project"):
            import tkinter as tk
            from tkinter import filedialog
            import pickle
            root = tk.Tk(); root.withdraw()
            path = filedialog.asksaveasfilename(
                title="Save Project", defaultextension=".d3d",
                filetypes=[("3D Recon Project", "*.d3d"), ("All files", "*.*")])
            root.destroy()
            if path:
                try:
                    save_data = {
                        'image_paths': state.image_paths,
                        'image_dir': state.image_dir,
                        'mesh_data': state.mesh_data,
                        'min_conf': state.min_conf,
                        'pts3d_list': state.pts3d_list,
                        'confs_list': state.confs_list,
                        'scene_rot': (state.scene_rot_x, state.scene_rot_y, state.scene_rot_z),
                    }
                    with open(path, 'wb') as f:
                        pickle.dump(save_data, f)
                    state.status = f"Project saved to {path}"
                except Exception as e:
                    state.status = f"Save failed: {e}"

        imgui.same_line()
        if imgui.button("Load Project"):
            import tkinter as tk
            from tkinter import filedialog
            import pickle
            root = tk.Tk(); root.withdraw()
            path = filedialog.askopenfilename(
                title="Load Project",
                filetypes=[("3D Recon Project", "*.d3d"), ("All files", "*.*")])
            root.destroy()
            if path:
                try:
                    with open(path, 'rb') as f:
                        save_data = pickle.load(f)
                    state.image_paths = save_data.get('image_paths', [])
                    state.image_dir = save_data.get('image_dir', '')
                    state.min_conf = save_data.get('min_conf', 2.0)
                    state.pts3d_list = save_data.get('pts3d_list')
                    state.confs_list = save_data.get('confs_list')
                    rot = save_data.get('scene_rot', (0, 0, 0))
                    state.scene_rot_x, state.scene_rot_y, state.scene_rot_z = rot

                    # Restore point cloud display
                    if state.pts3d_list is not None:
                        all_pts, all_cols = [], []
                        for i in range(len(state.pts3d_list)):
                            p = state.pts3d_list[i]
                            c = state.confs_list[i] if state.confs_list else None
                            img = None
                            # Try to load image for colors
                            if i < len(state.image_paths):
                                try:
                                    from PIL import Image as PILImage
                                    img_pil = PILImage.open(state.image_paths[i]).convert('RGB')
                                    img = np.array(img_pil).astype(np.float32) / 255.0
                                    # Resize to match pts shape if needed
                                    if p.ndim == 3 and img.shape[:2] != p.shape[:2]:
                                        img_pil = img_pil.resize((p.shape[1], p.shape[0]))
                                        img = np.array(img_pil).astype(np.float32) / 255.0
                                except Exception:
                                    pass
                            if p.ndim == 3:
                                mask = c.reshape(p.shape[:2]) > state.min_conf if c is not None else np.ones(p.shape[:2], dtype=bool)
                                all_pts.append(p[mask])
                                if img is not None:
                                    all_cols.append((np.clip(img[mask], 0, 1) * 255).astype(np.uint8))
                                else:
                                    all_cols.append(np.full((mask.sum(), 3), 180, dtype=np.uint8))
                            else:
                                mask = c.ravel() > state.min_conf if c is not None else np.ones(len(p), dtype=bool)
                                all_pts.append(p[mask])
                                all_cols.append(np.full((mask.sum(), 3), 180, dtype=np.uint8))
                        if all_pts:
                            pts = np.concatenate(all_pts, axis=0)
                            cols = np.concatenate(all_cols, axis=0)
                            if len(pts) > 200000:
                                idx = np.random.choice(len(pts), 200000, replace=False)
                                pts, cols = pts[idx], cols[idx]
                            scene_gl.set_points(pts, cols)
                            state.has_points = True
                            state.needs_recenter = True

                    # Restore mesh
                    md = save_data.get('mesh_data')
                    if md is not None:
                        state.mesh_data = md
                        scene_gl.set_mesh(md[0], md[1], md[2])
                        state.has_mesh = True

                    state.status = f"Project loaded from {path}"
                except Exception as e:
                    state.status = f"Load failed: {e}"
                    import traceback; traceback.print_exc()

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
        imgui.same_line()
        if imgui.button("Select Files..."):
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            files = filedialog.askopenfilenames(
                title="Select Images",
                filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.heic"),
                           ("All files", "*.*")])
            root.destroy()
            if files:
                state.image_paths = sorted(files)
                state.image_dir = os.path.dirname(files[0])
                state.status = f"Loaded {len(state.image_paths)} images"

        if state.image_paths:
            imgui.text(f"  {len(state.image_paths)} images")
            imgui.text(f"  {state.image_dir}")
        imgui.separator()

        # ── Backend ──
        imgui.text("Reconstruction")
        _, state.camera_source_idx = imgui.combo("Cameras",
            state.camera_source_idx, state.camera_source_labels)
        if state.cached_cameras is not None:
            imgui.same_line()
            imgui.text_colored(f"({len(state.cached_cameras)} cached)", 0.5, 1.0, 0.5)
        _, state.backend_idx = imgui.combo("Point Cloud",
            state.backend_idx, ["DUSt3R", "MASt3R", "VGGT", "COLMAP", "Pow3R"])
        _, state.stack_backends = imgui.checkbox("Stack backends", state.stack_backends)
        if state.stack_backends:
            imgui.same_line()
            _, state.stack_dust3r = imgui.checkbox("DUSt3R##stk", state.stack_dust3r)
            imgui.same_line()
            _, state.stack_mast3r = imgui.checkbox("MASt3R##stk", state.stack_mast3r)
            imgui.same_line()
            _, state.stack_vggt = imgui.checkbox("VGGT##stk", state.stack_vggt)
            imgui.same_line()
            _, state.stack_pow3r = imgui.checkbox("Pow3R##stk", state.stack_pow3r)

        if state.backends[state.backend_idx] == 'dust3r':
            _, state.niter1 = imgui.input_int("Iterations##d3r", state.niter1, 50, 100)

        if state.backends[state.backend_idx] == 'mast3r':
            _, state.optim_level = imgui.combo("Optimization##opt",
                state.optim_level, ["Coarse", "Refine", "Refine + Depth"])
            _, state.niter1 = imgui.input_int("Coarse Iters", state.niter1, 50, 100)
            _, state.niter2 = imgui.input_int("Refine Iters", state.niter2, 50, 100)

        changed_conf, state.min_conf = imgui.slider_float("Min Confidence", state.min_conf, 0.1, 20.0)

        # Live-update point cloud when confidence threshold changes
        # (only if cloud hasn't been modified by smooth/upscale)
        if changed_conf and state.scene is not None and state.has_points and not state.reconstructing and not state.points_modified:
            try:
                pts3d_list, confs_list = _extract_scene_data(state)
                all_pts, all_cols = [], []
                for i in range(len(state.scene.imgs)):
                    p = pts3d_list[i]
                    c = confs_list[i]
                    img = state.scene.imgs[i]
                    if p.ndim == 3:
                        H, W = p.shape[:2]
                        mask = c.reshape(H, W) > state.min_conf if c is not None else np.ones((H, W), dtype=bool)
                        all_pts.append(p[mask])
                        all_cols.append((np.clip(img[mask], 0, 1) * 255).astype(np.uint8))
                    else:
                        mask = c.ravel() > state.min_conf if c is not None else np.ones(len(p), dtype=bool)
                        all_pts.append(p[mask])
                        all_cols.append((np.clip(img.reshape(-1, 3)[mask], 0, 1) * 255).astype(np.uint8))
                if all_pts:
                    points = np.concatenate(all_pts, axis=0)
                    colors = np.concatenate(all_cols, axis=0)
                    if len(points) > 200000:
                        idx = np.random.choice(len(points), 200000, replace=False)
                        points, colors = points[idx], colors[idx]
                    scene_gl.set_points(points, colors)
                    state.status = f"{len(points):,d} points (conf > {state.min_conf:.1f})"
            except Exception:
                pass

        # Reconstruct button
        if state.image_paths and not state.reconstructing:
            if imgui.button("Reconstruct", width=-1):
                state.recon_thread = threading.Thread(
                    target=run_reconstruction, args=(state, scene_gl), daemon=True)
                state.recon_thread.start()

        if state.reconstructing:
            imgui.text(state.status)
            imgui.progress_bar(state.recon_frac)

        # ── AI Depth Enhancement ──
        if state.scene is not None and state.has_points and not state.reconstructing and not state.refining:
            imgui.separator()
            imgui.text("AI Depth Enhancement")
            _, state.ai_depth_mix = imgui.slider_float(
                "Detail Strength", state.ai_depth_mix, 0.0, 2.0,
                format="%.2f")
            _, state.ai_highpass_radius = imgui.slider_float(
                "Highpass Radius", state.ai_highpass_radius, 1.0, 50.0,
                format="%.1f px")

            has_depthmaps = hasattr(state.scene, 'im_depthmaps')
            if has_depthmaps:
                _, state.ai_refine_poses = imgui.checkbox("Refine poses after injection", state.ai_refine_poses)
                if state.ai_refine_poses:
                    _, state.ai_pose_iters = imgui.input_int("Pose Iters", state.ai_pose_iters, 25, 50)
                if imgui.button("Inject AI Depth (highpass blend)", width=-1):
                    state.reconstructing = True
                    state.recon_thread = threading.Thread(
                        target=run_depth_injection, args=(state, scene_gl), daemon=True)
                    state.recon_thread.start()
            else:
                imgui.text_colored("(Depth injection needs DUSt3R backend)", 0.7, 0.7, 0.3)

        imgui.separator()

        # ── Display Options ──
        imgui.text("Display")
        _, state.draw_mode = imgui.combo("Mode##draw",
            state.draw_mode, ["Points", "Mesh", "Wireframe", "Normals", "Shaded"])

        # Scene orientation
        _, state.scene_rot_x = imgui.slider_float("Rot X", state.scene_rot_x, -180, 180)
        _, state.scene_rot_y = imgui.slider_float("Rot Y", state.scene_rot_y, -180, 180)
        _, state.scene_rot_z = imgui.slider_float("Rot Z", state.scene_rot_z, -180, 180)
        if imgui.button("Flip Up"):
            state.scene_rot_x = (state.scene_rot_x + 180) % 360 - 180
        imgui.same_line()
        if imgui.button("Reset Rot"):
            state.scene_rot_x = state.scene_rot_y = state.scene_rot_z = 0.0
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

        imgui.separator()

        # ── Upscale ──
        if state.has_points and state.scene is not None and not state.refining:
            if imgui.button("Upscale Points (Mono Depth)", width=-1):
                state.refining = True
                state.refine_thread = threading.Thread(
                    target=run_upscale_points, args=(state, scene_gl), daemon=True)
                state.refine_thread.start()

        imgui.separator()

        # ── Dense Mesh ──
        imgui.text("Dense Mesh")
        _, state.mesh_mode_idx = imgui.combo("Method", state.mesh_mode_idx, state.mesh_mode_labels)
        changed_smooth, state.use_smoothing = imgui.checkbox("Smooth Points", state.use_smoothing)
        if state.use_smoothing:
            imgui.same_line()
            _, state.smooth_radius = imgui.slider_float("##smooth_r", state.smooth_radius, 0.1, 10.0, format="%.1fx")
            if state.has_points and not state.refining:
                if imgui.button("Preview Smooth", width=-1):
                    state.refining = True
                    state.refine_thread = threading.Thread(
                        target=_run_smooth_preview, args=(state, scene_gl), daemon=True)
                    state.refine_thread.start()
        # When unchecking smooth, restore original point cloud
        if changed_smooth and not state.use_smoothing and state.scene is not None and state.has_points:
            try:
                pts3d_list, confs_list = _extract_scene_data(state)
                all_pts, all_cols = [], []
                for i in range(len(state.scene.imgs)):
                    p, c_arr, img = pts3d_list[i], confs_list[i], state.scene.imgs[i]
                    if p.ndim == 3:
                        mask = c_arr.reshape(p.shape[:2]) > state.min_conf if c_arr is not None else np.ones(p.shape[:2], dtype=bool)
                    else:
                        mask = c_arr.ravel() > state.min_conf if c_arr is not None else np.ones(len(p), dtype=bool)
                    all_pts.append(p[mask] if p.ndim == 3 else p[mask])
                    all_cols.append((np.clip((img[mask] if p.ndim == 3 else img.reshape(-1, 3)[mask]), 0, 1) * 255).astype(np.uint8))
                pts = np.concatenate(all_pts, axis=0)
                cols = np.concatenate(all_cols, axis=0)
                if len(pts) > 200000:
                    idx = np.random.choice(len(pts), 200000, replace=False)
                    pts, cols = pts[idx], cols[idx]
                scene_gl.set_points(pts, cols)
                state.points_modified = False
                state.status = f"Restored {len(pts):,d} points"
            except Exception:
                pass
        _, state.hole_cap_size = imgui.slider_int("Hole Cap Size", state.hole_cap_size, 0, 500)
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

        # Recolor + Decimate buttons (post-processing)
        if state.has_mesh and state.scene is not None and not state.refining:
            if imgui.button("Recolor from Cameras", width=-1):
                state.refining = True
                state.refine_thread = threading.Thread(
                    target=run_recolor_mesh, args=(state, scene_gl), daemon=True)
                state.refine_thread.start()
        if state.has_mesh and not state.refining:
            if imgui.button("Decimate Mesh", width=-1):
                state.refining = True
                state.refine_thread = threading.Thread(
                    target=run_decimate, args=(state, scene_gl), daemon=True)
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

        imgui.end_child()
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

        # Auto-recenter camera — only on first reconstruction or project load
        # (not on subsequent reconstructions to preserve user's viewpoint)
        if state.needs_recenter and not hasattr(state, '_ever_centered'):
            state.needs_recenter = False
            state._ever_centered = True
            try:
                # Use cached pts3d_list if available (works for all backends)
                pts = None
                if state.pts3d_list is not None:
                    all_p = [p.reshape(-1, 3) for p in state.pts3d_list if p is not None]
                    if all_p:
                        pts = np.concatenate(all_p, axis=0)
                # Fallback to mesh verts
                if pts is None:
                    pts = getattr(scene_gl, '_mesh_verts', None)
                # Fallback to scene
                if pts is None and state.scene is not None:
                    try:
                        from dust3r.utils.device import to_numpy
                        all_p = to_numpy(state.scene.get_pts3d())
                        pts = np.concatenate([p.reshape(-1, 3) for p in all_p], axis=0)
                    except Exception:
                        pass
                if pts is not None and len(pts) > 0:
                    center = pts.mean(axis=0)
                    extent = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
                    camera.target = center.astype(np.float32)
                    camera.distance = float(extent * 1.5)
            except Exception:
                pass

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

        # ── Console Overlay ──
        if state.show_console:
            console_h = min(300, win_h // 3)
            imgui.set_next_window_position(400, win_h - console_h)
            imgui.set_next_window_size(win_w - 400, console_h)
            imgui.set_next_window_bg_alpha(0.85)
            imgui.begin("Console", flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE |
                        imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_FOCUS_ON_APPEARING)
            lines = _console.get_lines()
            # Show last N lines that fit
            max_lines = max(1, console_h // 16)
            for line in lines[-max_lines:]:
                imgui.text_unformatted(line)
            # Auto-scroll to bottom
            if imgui.get_scroll_y() < imgui.get_scroll_max_y():
                imgui.set_scroll_here_y(1.0)
            imgui.end()

        # Render ImGui
        gl.glViewport(0, 0, win_w, win_h)
        imgui.render()
        impl.render(imgui.get_draw_data())

        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()


if __name__ == '__main__':
    main()
