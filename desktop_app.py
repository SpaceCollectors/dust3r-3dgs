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
import tempfile
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
        near, far = 0.01, 10000.0
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
        # Use camera-local axes so pan follows screen directions at any angle
        # forward = target - eye (same direction as view matrix)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        forward = np.array([-cp * sy, -sp, -cp * cy], dtype=np.float32)
        world_up = np.array([0, 1, 0], dtype=np.float32)
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right) + 1e-8
        up = np.cross(right, forward)
        up /= np.linalg.norm(up) + 1e-8
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
out vec3 v_pos;
out vec2 v_uv;
void main() {
    gl_Position = mvp * vec4(position, 1.0);
    gl_PointSize = 3.0;
    v_color = color;
    v_pos = position;
    v_uv = uv;
}
"""

FRAG_SHADER = """
#version 330
uniform sampler2D tex;
uniform int use_texture;
uniform int shade_mode;  // 0=off, 1=shaded (v_color has normals)
uniform vec3 light_dir;
in vec3 v_color;
in vec3 v_pos;
in vec2 v_uv;
out vec4 frag_color;
void main() {
    if (use_texture == 1 && v_uv.x >= 0.0) {
        frag_color = texture(tex, v_uv);
    } else if (shade_mode == 1) {
        // v_color contains encoded normal (n*0.5+0.5), decode and shade
        vec3 normal = normalize(v_color * 2.0 - 1.0);
        float ndotl = abs(dot(normal, normalize(light_dir)));
        float ambient = 0.2;
        float shade = ambient + (1.0 - ambient) * ndotl;
        frag_color = vec4(vec3(shade * 0.85), 1.0);  // grey lambert
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
        self.shade_mode_loc = gl.glGetUniformLocation(self.program, "shade_mode")
        self.light_dir_loc = gl.glGetUniformLocation(self.program, "light_dir")
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
        self.widget_vao = None
        self.widget_vbo = None
        self.widget_line_count = 0
        self.grid_vao = None
        self.grid_vbo = None
        self.grid_line_count = 0
        # Splat display buffers (point sprite fallback)
        self.splat_vao = None
        self.splat_vbo = None
        self.splat_count = 0
        # Proper gaussian splat renderer
        self._splat_renderer = None
        self._pending_splat_params = None  # (means, quats, scales, opacities, sh0)
        # Pending uploads from background threads
        self._pending_points = None
        self._pending_mesh = None
        self._pending_cams = None
        self._pending_splats = None
        self._pending_grid = True  # build grid on first flush
        self._pending_widgets = None
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

    def set_splats(self, positions, colors_f, sizes,
                   quats=None, scales_log=None, opacities_logit=None, sh0=None):
        """Queue splat data for viewport display (thread-safe).
        If full params provided, renders as proper gaussians. Otherwise point sprites."""
        colors_u8 = (np.clip(colors_f, 0, 1) * 255).astype(np.uint8)
        with self._lock:
            self._pending_splats = (positions.copy(), colors_u8)
            if quats is not None and scales_log is not None:
                self._pending_splat_params = (
                    positions.copy(), quats.copy(), scales_log.copy(),
                    opacities_logit.copy(), sh0.copy())

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

    def _upload_splats(self, positions, colors):
        """Upload splat positions + colors as point cloud to separate buffer."""
        if len(positions) == 0:
            self.splat_count = 0
            return
        data = np.empty((len(positions), 6), dtype=np.float32)
        data[:, :3] = positions.astype(np.float32)
        data[:, 3:6] = colors.astype(np.float32) / 255.0
        if self.splat_vao is None:
            self.splat_vao = gl.glGenVertexArrays(1)
            self.splat_vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.splat_vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.splat_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, 24, None)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, False, 24, gl.ctypes.c_void_p(12))
        gl.glBindVertexArray(0)
        self.splat_count = len(positions)

    def _upload_splat_renderer(self, means, quats, scales_log, opacities_logit, sh0):
        """Initialize/update the proper gaussian splat renderer."""
        try:
            from splat_renderer import SplatRenderer, pack_splat_data_fast, sort_splats_by_depth
            if self._splat_renderer is None:
                self._splat_renderer = SplatRenderer()
            # Pack data and create initial sort (will be re-sorted each frame)
            data, sh_dim = pack_splat_data_fast(means, quats, scales_log, opacities_logit, sh0)
            self._splat_renderer.sh_dim = sh_dim
            self._splat_means = means.copy()
            # Identity sort for now (proper sort happens in draw)
            indices = np.arange(len(means), dtype=np.int32)
            self._splat_renderer.update_splats(data, indices)
            self._splat_data_packed = data
        except Exception as e:
            print(f"  Splat renderer init failed: {e}")
            import traceback; traceback.print_exc()

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

    def draw(self, mvp_grid, mvp_scene, draw_mode='points', camera_pos=None,
             view_matrix=None, proj_matrix=None, fov_y=None):
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
                # Upload normals as colors — shader computes lighting on GPU
                alt = getattr(self, '_normal_colors', None)
                if alt is not None:
                    self._upload_mesh(self._mesh_verts, self._mesh_faces, alt, stored_uvs)
                # Set light direction toward camera
                if camera_pos is not None:
                    gl.glUniform3f(self.light_dir_loc, *camera_pos.tolist())
                else:
                    gl.glUniform3f(self.light_dir_loc, 0.3, 0.7, 0.5)
                gl.glUniform1i(self.shade_mode_loc, 1)
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
            # Disable texture/shading after drawing
            gl.glUniform1i(self.use_tex_loc, 0)
            gl.glUniform1i(self.shade_mode_loc, 0)
            if self.mesh_tex_id is not None:
                gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        if draw_mode == 'wireframe' and self.mesh_face_count > 0:
            gl.glBindVertexArray(self.mesh_vao)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            gl.glDrawElements(gl.GL_TRIANGLES, self.mesh_face_count * 3,
                              gl.GL_UNSIGNED_INT, None)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

        if draw_mode == 'splats':
            if (self._splat_renderer is not None and self._splat_renderer.num_splats > 0
                    and view_matrix is not None and proj_matrix is not None):
                try:
                    import math as _math
                    from splat_renderer import sort_splats_by_depth

                    # Sort by depth using the view matrix (with scene transform)
                    view_with_scene = view_matrix @ mvp_scene
                    if hasattr(self, '_splat_means'):
                        indices = sort_splats_by_depth(self._splat_means, view_matrix)
                        self._splat_renderer.update_splats(self._splat_data_packed, indices)

                    # Compute focal from projection matrix and viewport
                    cam_p = camera_pos if camera_pos is not None else np.zeros(3)
                    fov = fov_y if fov_y else _math.radians(45)
                    # focal in pixels: proj[1,1] = 1/tan(fov/2), focal = H/2 / tan(fov/2)
                    # but we don't have H here, approximate from proj matrix
                    focal = float(proj_matrix[1, 1]) * 256  # approximate

                    self._splat_renderer.draw(
                        view_matrix.astype(np.float32),
                        proj_matrix.astype(np.float32),
                        cam_p, fov, fov, focal)
                    gl.glUseProgram(self.program)
                except Exception as e:
                    # Fall back to point sprites
                    if self.splat_count > 0:
                        gl.glBindVertexArray(self.splat_vao)
                        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
                        gl.glDrawArrays(gl.GL_POINTS, 0, self.splat_count)
            elif self.splat_count > 0:
                gl.glBindVertexArray(self.splat_vao)
                gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
                gl.glDrawArrays(gl.GL_POINTS, 0, self.splat_count)

        # Cameras: same rotation as scene
        if self.cam_line_count > 0 and getattr(self, '_show_cameras', True):
            gl.glBindVertexArray(self.cam_vao)
            gl.glDrawArrays(gl.GL_LINES, 0, self.cam_line_count)

        # Alignment widgets: discs + normals (drawn on top with depth bias)
        if self.widget_line_count > 0:
            gl.glDepthFunc(gl.GL_LEQUAL)
            gl.glLineWidth(2.0)
            gl.glBindVertexArray(self.widget_vao)
            gl.glDrawArrays(gl.GL_LINES, 0, self.widget_line_count)
            gl.glLineWidth(1.0)
            gl.glDepthFunc(gl.GL_LESS)

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

    def set_widgets(self, anchors, highlight_points=None):
        """Queue alignment widget geometry (thread-safe).
        anchors: list of {'pos': np.array(3), 'normal': np.array(3), 'radius': float, 'color': (r,g,b)}
        highlight_points: optional np.array (N,3) of points to highlight (shown as bright dots)
        """
        lines = []
        for anchor in anchors:
            pos = anchor['pos'].astype(np.float32)
            normal = anchor['normal'].astype(np.float32)
            normal = normal / max(np.linalg.norm(normal), 1e-8)
            radius = anchor['radius']
            color = np.array(anchor.get('color', (1, 1, 0)), dtype=np.float32)

            # Build tangent frame
            if abs(normal[1]) < 0.9:
                tangent = np.cross(normal, np.array([0, 1, 0], dtype=np.float32))
            else:
                tangent = np.cross(normal, np.array([1, 0, 0], dtype=np.float32))
            tangent /= max(np.linalg.norm(tangent), 1e-8)
            bitangent = np.cross(normal, tangent)
            bitangent /= max(np.linalg.norm(bitangent), 1e-8)

            # Disc outline (circle of line segments)
            n_seg = 24
            for s in range(n_seg):
                a0 = 2 * np.pi * s / n_seg
                a1 = 2 * np.pi * (s + 1) / n_seg
                p0 = pos + radius * (np.cos(a0) * tangent + np.sin(a0) * bitangent)
                p1 = pos + radius * (np.cos(a1) * tangent + np.sin(a1) * bitangent)
                lines.append(np.concatenate([p0, color]))
                lines.append(np.concatenate([p1, color]))

            # Normal arrow (from center, length = radius)
            tip = pos + normal * radius * 1.5
            lines.append(np.concatenate([pos, color]))
            lines.append(np.concatenate([tip, color * 0.8]))

            # Cross-hairs on disc for visibility
            lines.append(np.concatenate([pos - tangent * radius, color * 0.5]))
            lines.append(np.concatenate([pos + tangent * radius, color * 0.5]))
            lines.append(np.concatenate([pos - bitangent * radius, color * 0.5]))
            lines.append(np.concatenate([pos + bitangent * radius, color * 0.5]))

        # Highlighted neighborhood points — draw as small cross marks
        if highlight_points is not None and len(highlight_points) > 0:
            hi_color = np.array([1.0, 1.0, 0.0], dtype=np.float32)
            # Subsample if too many (keep it fast)
            pts = highlight_points
            if len(pts) > 200:
                pts = pts[np.random.choice(len(pts), 200, replace=False)]
            tick = float(np.linalg.norm(pts.max(0) - pts.min(0))) * 0.005
            tick = max(tick, 0.001)
            for p in pts:
                pf = p.astype(np.float32)
                for axis in range(3):
                    offset = np.zeros(3, dtype=np.float32)
                    offset[axis] = tick
                    lines.append(np.concatenate([pf - offset, hi_color]))
                    lines.append(np.concatenate([pf + offset, hi_color]))

        if not lines:
            with self._lock:
                self._pending_widgets = np.zeros((0, 6), dtype=np.float32)
            return
        with self._lock:
            self._pending_widgets = np.array(lines, dtype=np.float32)

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
            splats = self._pending_splats
            splat_params = self._pending_splat_params
            widgets = self._pending_widgets
            self._pending_points = None
            self._pending_mesh = None
            self._pending_cams = None
            self._pending_texture = None
            self._pending_splats = None
            self._pending_splat_params = None
            self._pending_widgets = None

        if pts is not None:
            self._upload_points(pts[0], pts[1])
        if msh is not None:
            self._upload_mesh(msh[0], msh[1], msh[2])
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
        if widgets is not None:
            self._upload_widgets(widgets)
        if splats is not None:
            self._upload_splats(splats[0], splats[1])
        if splat_params is not None:
            self._upload_splat_renderer(*splat_params)

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

    def _upload_widgets(self, data):
        if len(data) == 0:
            self.widget_line_count = 0
            return
        if self.widget_vao is None:
            self.widget_vao = gl.glGenVertexArrays(1)
            self.widget_vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.widget_vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.widget_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, 24, None)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, False, 24, gl.ctypes.c_void_p(12))
        gl.glBindVertexArray(0)
        self.widget_line_count = len(data)

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


# ── App Name & Temp Dirs ─────────────────────────────────────────────────────

APP_NAME = "1825_Reconstructor"

# All temp data goes under one persistent folder so it can be cleaned up
# even after a crash.  Each run creates timestamped sub-folders inside it.
_APP_TMP_ROOT = os.path.join(tempfile.gettempdir(), APP_NAME)
os.makedirs(_APP_TMP_ROOT, exist_ok=True)

def _app_mkdtemp(prefix=""):
    """Create a temp directory under the app's temp root and track it."""
    path = tempfile.mkdtemp(prefix=prefix, dir=_APP_TMP_ROOT)
    _tracked_tmpdirs.add(str(path))
    return path

def _track_tmpdir(path):
    """Register a temp directory for cleanup."""
    _tracked_tmpdirs.add(str(path))
    return path

def cleanup_temp_dirs():
    """Delete all tracked temp directories."""
    import shutil
    removed = 0
    for d in list(_tracked_tmpdirs):
        try:
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        except Exception:
            pass
    _tracked_tmpdirs.clear()
    return removed

def cleanup_all_app_temps():
    """Delete the entire app temp root (clears leftovers from crashed sessions)."""
    import shutil
    if os.path.isdir(_APP_TMP_ROOT):
        shutil.rmtree(_APP_TMP_ROOT, ignore_errors=True)
        os.makedirs(_APP_TMP_ROOT, exist_ok=True)
        print(f"Cleaned app temp root: {_APP_TMP_ROOT}")

_tracked_tmpdirs = set()


def _extract_video_frames(video_path, state):
    """Extract frames from a video file into a temp directory."""
    import cv2, tempfile

    state.video_extracting = True
    try:
        # Use os.path.shortname workaround for non-ASCII paths on Windows
        try:
            cap = cv2.VideoCapture(video_path)
        except Exception:
            cap = cv2.VideoCapture()
        if not cap.isOpened():
            # Retry with short path on Windows for non-ASCII filenames
            try:
                import ctypes
                buf = ctypes.create_unicode_buffer(512)
                ctypes.windll.kernel32.GetShortPathNameW(video_path, buf, 512)
                if buf.value:
                    cap = cv2.VideoCapture(buf.value)
            except Exception:
                pass
        if not cap.isOpened():
            state.status = f"Failed to open video: {os.path.basename(video_path)}"
            return []

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Determine frame step
        if state.video_target_fps > 0:
            step = max(1, int(round(video_fps / state.video_target_fps)))
        else:
            step = max(1, state.video_frame_interval)

        max_frames = state.video_max_frames if state.video_max_frames > 0 else 999999
        max_size = state.video_max_size if state.video_max_size > 0 else 999999
        est_total = min(max_frames, total_frames // step)

        tmpdir = _app_mkdtemp(prefix="video_frames_")

        extracted = []
        frame_idx = 0
        saved = 0

        while saved < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            # Resize if exceeds max size
            h, w = frame.shape[:2]
            longest = max(h, w)
            if longest > max_size:
                scale = max_size / longest
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            # Use ASCII-only filenames and imencode to avoid Windows Unicode path issues
            out_path = os.path.join(tmpdir, f"frame_{frame_idx:06d}.jpg")
            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ok:
                with open(out_path, 'wb') as fout:
                    fout.write(buf.tobytes())
                extracted.append(out_path)
                saved += 1

            state.status = f"Extracting frame {saved}/{est_total}..."
            frame_idx += step

        cap.release()
        return extracted
    finally:
        state.video_extracting = False


class AppState:
    def __init__(self):
        self.show_console = False  # overlay console on viewport

        # Images
        self.image_paths = []
        self.image_dir = ""

        # Video import settings
        self.video_frame_interval = 10    # extract every Nth frame
        self.video_target_fps = 0.0       # alternative: target FPS (0 = use interval mode)
        self.video_max_frames = 100       # cap total extracted frames
        self.video_max_size = 1920        # max frame size in pixels (longest edge)
        self.video_extracting = False     # True while extraction thread runs

        # Backend
        self.backend_idx = 0  # 0=dust3r, 1=mast3r, 2=vggt, 3=colmap, 4=pow3r, 5=lingbot
        self.backends = ['dust3r', 'mast3r', 'vggt', 'colmap', 'pow3r', 'lingbot']
        self.cached_cameras = None  # reference cameras from first reconstruction (for alignment)
        self.vggt_ensemble = False
        self.vggt_equirect = False  # treat single image as equirectangular panorama
        self.lingbot_keyframe_interval = 1  # process every Nth frame (1=all)
        self.lingbot_scale_frames = 8      # initial bidirectional frames for scale
        self.lingbot_kv_window = 16        # sliding window size for KV cache

        # Reconstruction
        self.scene = None
        self.reconstructing = False
        self.recon_progress = ""
        self.recon_frac = 0.0  # 0.0–1.0 progress fraction
        self.recon_thread = None

        self.mask_sky = False  # auto-detect and mask sky pixels
        self.mask_prompt = ""  # text prompt to isolate subject (empty = disabled)
        self.mask_prompt_mode = 0  # 0=keep subject, 1=remove subject
        self.mask_before_recon = True  # mask images BEFORE reconstruction (better isolation)
        self.masked_image_paths = None  # paths to pre-masked images (temp dir)

        # Bundle adjustment options
        self.ba_refine_focal = False
        self.ba_refine_pp = False
        self.ba_n_iters = 200
        self.ba_min_conf = 2.0
        self.ba_max_shift = 0.2   # max translation as fraction of scene scale
        self.ba_huber_scale = 0.1
        self.show_cameras = True

        # COLMAP PatchMatch options
        self.pm_max_image_size = -1     # -1 = full resolution
        self.pm_num_iterations = 5
        self.pm_window_radius = 5       # matching patch = (2*r+1)^2
        self.pm_min_consistent = 2      # min views that must agree
        self.pm_geom_consistency = True
        self.pm_filter_min_ncc = 0.1

        # Point cloud / Mesh
        self.has_points = False
        self.points_modified = False  # True after smooth/upscale (don't reset on conf change)
        self.has_mesh = False
        self.uv_data = None  # (uvs, uv_faces) after Create UVs
        self.draw_mode = 0  # 0=points, 1=mesh, 2=wireframe, 3=normals, 4=shaded, 5=splats
        self.draw_modes = ['points', 'mesh', 'wireframe', 'normals', 'shaded', 'splats']

        # Refinement
        self.refining = False
        self.refine_progress = ""
        self.refine_thread = None

        # Splat training
        self.splat_training = False
        self.splat_progress = ""
        self.splat_step = 0
        self.splat_total = 0
        self.splat_data = None  # trained splats ParameterDict
        self.splat_iterations = 2000
        self.splat_n_samples = 20000
        self.splat_target = 50000  # target splat count (0=no densification)
        self.splat_multi_view = True
        self.splat_multi_view_count = 2
        self.splat_strategy_idx = 4  # default to Adaptive
        self.splat_strategies = ['simple', 'mcmc', 'absgrad', 'mrnf', 'adaptive']
        self.splat_strategy_labels = ['Simple', 'MCMC', 'AbsGrad/IGS+', 'MRNF', 'Adaptive']
        self.splat_resolution = 1024  # training image resolution
        # All weights are 0-1 sliders, scaled to actual values in the training call
        self.splat_anchor = 0.0
        self.splat_flatness = 0.0
        self.splat_normal = 0.0
        self.splat_aniso = 0.0
        self.splat_depth = 0.0
        self.splat_coverage = 0.5
        self.splat_opacity_decay = 0.0
        self.splat_prune = 0.0
        self.splat_smooth = 0.0

        # Export
        self.export_path = ""
        self._deferred_action = None

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
        self.mesh_mode_idx = 0     # 0=reprojected grid, 1=ball pivot, 2=local delaunay, 3=poisson
        self.mesh_modes = ['reprojected', 'ballpivot', 'delaunay', 'poisson']
        self.mesh_mode_labels = ['Reprojected Grid', 'Ball Pivot', 'Local Delaunay', 'Poisson']
        self.hole_cap_size = 50    # max boundary edges to close (higher = close bigger holes)
        self.smooth_radius = 2.0   # neighbor merge radius multiplier
        self.poisson_depth_val = 10  # Poisson octree depth (higher = more detail)
        self.poisson_trim = 5.0      # trim percentile for low-density regions
        self.bp_radius_mult = 4.0    # ball pivot max radius multiplier (1=tight, 8=fill gaps)
        self.delaunay_edge_mult = 8.0  # radius multiplier for local Delaunay neighbor selection
        self.use_smoothing = False # whether to smooth before meshing
        self.ai_depth_mix = 0.7  # 0=pure dust3r, 1=full AI detail
        self.ai_highpass_radius = 10.0  # high-pass sigma in pixels
        self.ai_refine_poses = True  # re-optimize poses after depth injection
        self.ai_pose_iters = 100

        # Mesh generation settings (kept for compatibility)

        # Scene orientation transform (applied in shader)
        self.scene_rot_x = 180.0  # degrees (DUSt3R convention: Y-down, flip to Y-up)
        self.align_mode = None  # None, 'floor', 'wall', 'hline', 'vline'
        self.align_floor_normal = None  # stored floor normal vector
        self.align_wall_normal = None   # stored wall normal vector
        self.align_floor_anchor = None  # {'pos': np.array(3), 'normal': np.array(3), 'radius': float}
        self.align_wall_anchor = None
        self.align_preview_rot = None   # (rx, ry, rz) snapshot before preview
        self.align_line_mode = None     # None, 'hline', 'vline'
        self.align_line_start = None    # np.array(3) world-space first click
        self.align_line_start_screen = None  # (sx, sy) for rubber-band drawing
        self.scene_rot_y = 0.0
        self.scene_rot_z = 0.0
        self.scene_flip_y = False

        # Stop flag
        self.stop_requested = False
        self.needs_recenter = False

        # Camera POV cycling
        self.cam_view_idx = -1   # -1 = free orbit, 0..N-1 = scene camera index
        self.cam_view_name = ""  # displayed image name when viewing a scene camera

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
    # Keep cached_cameras — needed for Procrustes alignment across reconstructions

    try:
        backend = state.backends[state.backend_idx]

        # Pre-masking: create masked images before reconstruction
        original_paths = list(state.image_paths)
        if state.mask_before_recon and (state.mask_prompt.strip() or state.mask_sky):
            state.status = "Pre-masking images..."
            try:
                masked_paths, _ = _create_masked_images(
                    state.image_paths,
                    prompt=state.mask_prompt.strip() if state.mask_prompt.strip() else None,
                    keep=(state.mask_prompt_mode == 0),
                    mask_sky=state.mask_sky)
                state.masked_image_paths = masked_paths
                state.image_paths = masked_paths  # backends will use masked images
                print(f"  Using pre-masked images from {os.path.dirname(masked_paths[0])}")
            except Exception as e:
                print(f"  Pre-masking failed: {e}, using original images")

        if backend == 'lingbot':
            state.status = "Running LingBot-Map streaming..."
            state.recon_frac = 0.1
            from app import _reconstruct_lingbot
            def _lingbot_progress(frac, msg):
                state.recon_frac = 0.1 + frac * 0.6
                state.status = msg
            vggt_scene = _reconstruct_lingbot(
                state.image_paths,
                keyframe_interval=state.lingbot_keyframe_interval,
                num_scale_frames=state.lingbot_scale_frames,
                kv_window=state.lingbot_kv_window,
                progress_cb=_lingbot_progress)

        if backend == 'vggt':
            use_ensemble = state.vggt_ensemble and len(state.image_paths) > 4
            use_equirect = state.vggt_equirect and len(state.image_paths) == 1

            if use_equirect:
                state.status = "Running VGGT equirectangular..."
                state.recon_frac = 0.1
                from app import _reconstruct_vggt_equirect
                vggt_scene = _reconstruct_vggt_equirect(state.image_paths[0])
            elif use_ensemble:
                state.status = "Running VGGT ensemble..."
                state.recon_frac = 0.1
                from app import _reconstruct_vggt_ensemble
                vggt_scene = _reconstruct_vggt_ensemble(state.image_paths)
            else:
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

                # Register progress hooks on aggregator blocks
                n_frame = len(model.aggregator.frame_blocks)
                n_global = len(model.aggregator.global_blocks)
                total_blocks = n_frame + n_global
                block_counter = [0]
                hooks = []
                def _make_hook():
                    def _hook(module, input, output):
                        block_counter[0] += 1
                        pct = block_counter[0] / total_blocks
                        state.status = f"VGGT: block {block_counter[0]}/{total_blocks}"
                        state.recon_frac = 0.3 + 0.4 * pct
                    return _hook
                for blk in model.aggregator.frame_blocks:
                    hooks.append(blk.register_forward_hook(_make_hook()))
                for blk in model.aggregator.global_blocks:
                    hooks.append(blk.register_forward_hook(_make_hook()))

                with torch.no_grad():
                    with torch.amp.autocast('cuda', dtype=dtype):
                        predictions = model(images)

                for h in hooks:
                    h.remove()

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

                from canonical_scene import from_vggt
                from PIL import Image as PILImage
                from PIL.ImageOps import exif_transpose
                orig_sizes = []
                for p in state.image_paths:
                    im = exif_transpose(PILImage.open(p)).convert('RGB')
                    orig_sizes.append(im.size)

                imgs_list = []
                pts3d_all = []
                conf_all = []
                for i in range(len(state.image_paths)):
                    img_i = imgs_np[i]
                    conf_i = depth_conf[i].copy()
                    padding_mask = img_i.mean(axis=-1) > 0.95
                    conf_i[padding_mask] = 0
                    imgs_list.append(img_i)
                    pts3d_all.append(pts3d[i])
                    conf_all.append(conf_i)

                vggt_scene = from_vggt(imgs_list, extrinsic, intrinsic,
                                       pts3d_all, conf_all, orig_sizes)

                del model, predictions
                torch.cuda.empty_cache()

        if backend in ('vggt', 'lingbot'):
            # Extract point cloud from CanonicalScene for display
            state.status = "Extracting point cloud..."
            state.recon_frac = 0.7
            all_pts = []
            all_colors = []
            min_conf_used = state.min_conf
            for i in range(len(vggt_scene.images)):
                p = vggt_scene.pts3d[i]
                c = vggt_scene.confidence[i]
                img = vggt_scene.images[i]
                conf_mask = c > min_conf_used
                not_padding = img.mean(axis=-1) < 0.95
                valid = np.isfinite(p).all(axis=-1)
                mask = conf_mask & not_padding & valid
                n_pass = mask.sum()
                print(f"    Image {i}: conf range [{c.min():.2f}, {c.max():.2f}], "
                      f"{n_pass}/{mask.size} points pass (conf>{min_conf_used:.1f})")
                all_pts.append(p[mask])
                all_colors.append((img[mask] * 255).astype(np.uint8))

            points = np.concatenate(all_pts, axis=0) if all_pts else np.zeros((0, 3), dtype=np.float32)
            colors = np.concatenate(all_colors, axis=0) if all_pts else np.zeros((0, 3), dtype=np.uint8)

            # If min_conf filtered everything, retry with a lower threshold
            if len(points) == 0:
                print(f"  WARNING: 0 points with min_conf={min_conf_used:.1f}, retrying with min_conf=0.5")
                all_pts2 = []
                all_colors2 = []
                for i in range(len(vggt_scene.images)):
                    p = vggt_scene.pts3d[i]
                    c = vggt_scene.confidence[i]
                    img = vggt_scene.images[i]
                    mask = (c > 0.5) & (img.mean(axis=-1) < 0.95) & np.isfinite(p).all(axis=-1)
                    all_pts2.append(p[mask])
                    all_colors2.append((img[mask] * 255).astype(np.uint8))
                points = np.concatenate(all_pts2, axis=0) if all_pts2 else np.zeros((0, 3), dtype=np.float32)
                colors = np.concatenate(all_colors2, axis=0) if all_pts2 else np.zeros((0, 3), dtype=np.uint8)
                print(f"    Fallback: {len(points)} points with min_conf=0.5")

            if len(points) > 0:
                if len(points) > 200000:
                    idx = np.random.choice(len(points), 200000, replace=False)
                    points, colors = points[idx], colors[idx]
                scene_gl.set_points(points, colors)
                state.has_points = True; state.needs_recenter = True
            else:
                print("  WARNING: VGGT produced 0 valid points — check input images")

            state.status = "Building scene..."
            state.recon_frac = 0.85
            state.scene = vggt_scene

            # Set camera visualizations — c2w from CanonicalScene
            c2w_all = vggt_scene.get_im_poses().numpy()
            cam_poses = []
            for i in range(len(c2w_all)):
                c2w = c2w_all[i]
                if np.isfinite(c2w).all():
                    cam_poses.append(c2w)
                else:
                    print(f"    Camera {i}: degenerate pose (NaN/Inf), skipping")
            if cam_poses:
                if len(points) > 0:
                    ext = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
                else:
                    ext = 1.0
                scene_gl.set_cameras(cam_poses, scale=float(ext) * 0.05)

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
            result_path = os.path.join(_APP_TMP_ROOT, 'pow3r_result.pkl')
            script = f'''# -*- coding: utf-8 -*-
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

# Force landscape — Pow3R patch_embed requires W >= H
for img_dict in imgs:
    ts = img_dict['true_shape']
    if hasattr(ts, 'shape') and ts.ndim == 2:
        H, W = int(ts[0, 0]), int(ts[0, 1])
    else:
        H, W = int(ts[0]), int(ts[1])
    if H > W:
        # Rotate image tensor 90 degrees
        img_dict['img'] = img_dict['img'].rot90(1, [2, 3])
        if hasattr(ts, 'shape') and ts.ndim == 2:
            img_dict['true_shape'] = np.array([[W, H]])
        else:
            img_dict['true_shape'] = np.array([W, H])

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
            with open(script_path, 'w', encoding='utf-8') as f:
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
                from canonical_scene import from_w2c
                pts3d = result['pts3d']
                confs = result['confs']
                c2w_all = result['c2w']
                imgs_np = result['imgs']

                extrinsic = np.zeros((len(c2w_all), 3, 4), dtype=np.float32)
                intrinsic_all = []
                for i in range(len(c2w_all)):
                    w2c = np.linalg.inv(c2w_all[i])
                    extrinsic[i] = w2c[:3, :].astype(np.float32)
                    f = float(result['focals'][i].item() if hasattr(result['focals'][i], 'item') else result['focals'][i])
                    H, W = pts3d[i].shape[:2]
                    K = np.array([[f, 0, W/2], [0, f, H/2], [0, 0, 1]], dtype=np.float32)
                    intrinsic_all.append(K)

                from PIL import Image as PILImage
                orig_sizes = []
                for p in state.image_paths:
                    im = PILImage.open(p).convert('RGB')
                    orig_sizes.append(im.size)

                # Align to cached cameras if available
                state.scene = from_w2c(imgs_np, extrinsic, np.stack(intrinsic_all),
                                       pts3d, confs, orig_sizes,
                                       backend='pow3r', internal_resolution=512)

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

            workdir = Path(_app_mkdtemp(prefix="colmap_"))
            state._colmap_workdir = str(workdir)  # save for densify reuse
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
            from canonical_scene import from_w2c
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

                # Camera pose (w2c) — handle different pycolmap API versions
                w2c = np.eye(4, dtype=np.float32)
                try:
                    # pycolmap >= 3.10
                    cfw = image.cam_from_world
                    if callable(cfw):
                        w2c_34 = cfw().matrix()
                    elif hasattr(cfw, 'matrix'):
                        w2c_34 = cfw.matrix()
                    else:
                        w2c_34 = np.array(cfw)
                    w2c[:3, :] = w2c_34
                except Exception:
                    # Fallback: build from rotation + translation
                    R = image.rotmat()
                    t = image.tvec
                    w2c[:3, :3] = R
                    w2c[:3, 3] = t

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
            state.scene = from_w2c(
                imgs_np, np.array(extrinsics), np.array(intrinsics),
                pts3d_all, conf_all, orig_sizes,
                backend='colmap', internal_resolution=0)

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
                if mode == GlobalAlignerMode.PointCloudOptimizer:
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
                cache_dir = _app_mkdtemp()
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

            # Convert raw DUSt3R/MASt3R scene to CanonicalScene
            state.status = "Converting to CanonicalScene..."
            state.recon_frac = 0.7
            raw_scene = state.scene
            if backend == 'dust3r':
                from canonical_scene import from_dust3r
                state.scene = from_dust3r(raw_scene, state.image_paths)
                state._raw_dust3r_scene = raw_scene  # for AI depth injection
            else:
                from canonical_scene import from_mast3r
                state.scene = from_mast3r(raw_scene, state.image_paths)

            del model
            torch.cuda.empty_cache()

            # Extract point cloud for display
            state.status = "Extracting point cloud..."
            state.recon_frac = 0.75
            scene = state.scene
            all_pts, all_colors = [], []
            for i in range(len(scene.images)):
                p = scene.pts3d[i]
                c = scene.confidence[i]
                img = scene.images[i]
                if p.ndim == 3:
                    mask = (c > state.min_conf) & np.isfinite(p).all(axis=-1)
                    all_pts.append(p[mask])
                    all_colors.append((np.clip(img[mask], 0, 1) * 255).astype(np.uint8))
                else:
                    all_pts.append(p.reshape(-1, 3))
                    all_colors.append((np.clip(img.reshape(-1, 3), 0, 1) * 255).astype(np.uint8))

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
                    c2w_all = state.scene.get_im_poses().numpy()
                    for i in range(len(c2w_all)):
                        cam_poses.append(c2w_all[i])
                if cam_poses:
                    ext = np.linalg.norm(points.max(axis=0) - points.min(axis=0)) if len(points) > 0 else 1.0
                    scene_gl.set_cameras(cam_poses, scale=float(ext) * 0.05)
            except Exception:
                pass

        # Recentering handled by main loop via needs_recenter flag

        # Sky masking: zero confidence for sky pixels
        if state.mask_sky and state.has_points:
            try:
                state.status = "Detecting sky..."
                sky_masks = _detect_sky_masks(state.image_paths)
                if sky_masks:
                    pts3d_list, confs_list = _extract_scene_data(state)
                    for i in range(min(len(sky_masks), len(confs_list))):
                        sky = sky_masks[i]
                        c = confs_list[i]
                        if c is not None and c.ndim == 2:
                            # Resize sky mask to match confidence map if needed
                            if sky.shape != c.shape:
                                from PIL import Image as PILImage
                                sky = np.array(PILImage.fromarray(sky).resize(
                                    (c.shape[1], c.shape[0]), PILImage.NEAREST))
                            c[sky] = 0
                            confs_list[i] = c
                    state.confs_list = confs_list
                    n_masked = sum(m.sum() for m in sky_masks)
                    print(f"  Sky masked: {n_masked:,d} pixels across {len(sky_masks)} images")
            except Exception as e:
                print(f"  Sky masking failed: {e}")

        # Subject masking: keep or remove pixels matching text prompt
        if state.mask_prompt.strip() and state.has_points:
            try:
                state.status = f"Detecting '{state.mask_prompt}'..."
                keep = (state.mask_prompt_mode == 0)
                subject_masks = _detect_subject_masks(
                    state.image_paths, state.mask_prompt.strip(), keep=keep)
                if subject_masks:
                    pts3d_list, confs_list = _extract_scene_data(state)
                    for i in range(min(len(subject_masks), len(confs_list))):
                        mask = subject_masks[i]
                        c = confs_list[i]
                        if c is not None and c.ndim == 2:
                            if mask.shape != c.shape:
                                from PIL import Image as PILImage
                                mask = np.array(PILImage.fromarray(mask).resize(
                                    (c.shape[1], c.shape[0]), PILImage.NEAREST))
                            # Zero confidence for pixels NOT in the keep mask
                            c[~mask] = 0
                            confs_list[i] = c
                    state.confs_list = confs_list
                    print(f"  Subject mask applied: '{state.mask_prompt}'")
            except Exception as e:
                print(f"  Subject masking failed: {e}")

        # Rebuild point cloud after masking so sky/subject removal is visible
        if (state.mask_sky or state.mask_prompt.strip()) and state.has_points and state.scene is not None:
            try:
                state.status = "Rebuilding point cloud after masking..."
                pts3d_list, confs_list = _extract_scene_data(state)
                scene = state.scene
                all_pts, all_cols = [], []
                for i in range(len(scene.imgs)):
                    p = pts3d_list[i]
                    c = confs_list[i]
                    img = scene.imgs[i]
                    if p.ndim == 3:
                        H, W = p.shape[:2]
                        conf_2d = c.reshape(H, W) if c.ndim != 2 else c
                        mask = (conf_2d > state.min_conf) & np.isfinite(p).all(axis=-1)
                        all_pts.append(p[mask])
                        all_cols.append((np.clip(img[mask], 0, 1) * 255).astype(np.uint8))
                    else:
                        c_flat = c.ravel() if c is not None else np.ones(len(p), dtype=np.float32)
                        mask = c_flat > state.min_conf
                        all_pts.append(p[mask])
                        all_cols.append((np.clip(img.reshape(-1, 3)[mask], 0, 1) * 255).astype(np.uint8))
                if all_pts:
                    points = np.concatenate(all_pts, axis=0)
                    colors = np.concatenate(all_cols, axis=0)
                    if len(points) > 200000:
                        idx = np.random.choice(len(points), 200000, replace=False)
                        points, colors = points[idx], colors[idx]
                    scene_gl.set_points(points, colors)
                    print(f"  Rebuilt point cloud: {len(points):,d} points after masking")
            except Exception as e:
                print(f"  Point cloud rebuild after masking failed: {e}")

        # Align to reference frame: if this is the first reconstruction, cache cameras.
        # If subsequent reconstruction, align to the cached cameras via Procrustes.
        if state.has_points and state.scene is not None:
            try:
                print(f"  Alignment check: has_points={state.has_points}, scene={type(state.scene).__name__}, cached={state.cached_cameras is not None}")
                c2w_poses = state.scene.get_im_poses().numpy()
                pts3d_list, confs_list = _extract_scene_data(state)

                cam_list = []
                for i in range(len(c2w_poses)):
                    orig_w, orig_h = state.scene.original_sizes[i]
                    K = state.scene.scale_intrinsics_to(orig_w, orig_h, i).astype(np.float32)
                    cam_list.append((c2w_poses[i].astype(np.float32), K, orig_w, orig_h))

                if state.cached_cameras is not None and len(state.cached_cameras) == len(cam_list):
                    # Align this reconstruction to the reference frame
                    print("  Aligning to reference cameras via Procrustes...")
                    cam_c2w = [c2w_poses[i] for i in range(len(c2w_poses))]
                    pts3d_aligned, new_c2w = _align_to_cached_cameras(
                        list(pts3d_list), cam_c2w, state.cached_cameras)
                    # Update scene data with aligned points
                    state.pts3d_list = pts3d_aligned
                    state.confs_list = list(confs_list)
                    # Update display with confidence-filtered points
                    all_pts, all_cols = [], []
                    for i in range(len(pts3d_aligned)):
                        p = pts3d_aligned[i]
                        c = confs_list[i] if i < len(confs_list) else None
                        if p.ndim == 3:
                            H, W = p.shape[:2]
                            mask = c.reshape(H, W) > state.min_conf if c is not None else np.ones((H, W), dtype=bool)
                            all_pts.append(p[mask])
                            if i < len(state.scene.imgs):
                                all_cols.append((np.clip(state.scene.imgs[i][mask], 0, 1) * 255).astype(np.uint8))
                            else:
                                all_cols.append(np.full((mask.sum(), 3), 180, dtype=np.uint8))
                        else:
                            all_pts.append(p.reshape(-1, 3))
                            all_cols.append(np.full((len(p.reshape(-1, 3)), 3), 180, dtype=np.uint8))
                    points = np.concatenate(all_pts, axis=0)
                    colors = np.concatenate(all_cols, axis=0)
                    if len(points) > 200000:
                        idx = np.random.choice(len(points), 200000, replace=False)
                        points, colors = points[idx], colors[idx]
                    scene_gl.set_points(points, colors)
                    scene_gl.set_cameras(new_c2w, scale=float(np.linalg.norm(
                        points.max(0) - points.min(0))) * 0.05)
                else:
                    # First reconstruction — save as reference
                    state.cached_cameras = cam_list
                    print(f"  Cached {len(cam_list)} reference cameras")
            except Exception as e:
                print(f"  Camera caching/alignment failed: {e}")
                import traceback; traceback.print_exc()

        # Old auto-align disabled
        if False and state.has_points and state.scene is not None:
            try:
                c2w_poses = state.scene.get_im_poses().numpy()

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
        import traceback; traceback.print_exc()
    finally:
        # Restore original image paths (masked ones were just for reconstruction)
        try:
            if original_paths:
                state.image_paths = original_paths
        except NameError:
            pass
        state.reconstructing = False


# ── Main App ─────────────────────────────────────────────────────────────────

def run_depth_injection(state, scene_gl):
    """Inject AI depth into dust3r's scene and regenerate point cloud."""
    raw_scene = getattr(state, '_raw_dust3r_scene', None)
    if raw_scene is None:
        state.status = "AI depth injection requires DUSt3R backend"
        return
    state.status = "Injecting AI depth..."
    try:
        from depth_inject import inject_ai_depth
        from dust3r.utils.device import to_numpy

        def progress(frac, msg):
            state.status = msg

        new_pts3d = inject_ai_depth(
            raw_scene, state.scene.images,
            mix=state.ai_depth_mix,
            highpass_sigma=state.ai_highpass_radius,
            device='cuda', progress_fn=progress)

        # Optionally refine poses to fit the new depth
        if state.ai_refine_poses:
            from depth_inject import refine_poses_with_ai_depth
            state.status = "Refining camera poses..."
            refine_poses_with_ai_depth(
                raw_scene, niter=state.ai_pose_iters, lr=0.005,
                progress_fn=progress)
            new_pts3d = to_numpy(raw_scene.get_pts3d())

        # Update CanonicalScene pts3d with the new depth-injected points
        for i in range(len(new_pts3d)):
            state.scene.pts3d[i] = new_pts3d[i]
        state.pts3d_list = None  # invalidate cache

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
            c2w_all = state.scene.get_im_poses().numpy()
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


def _handle_focus_click(state, camera, mx, my, window):
    """Right-click: raycast to nearest point and set orbit target there."""
    try:
        pts3d_list = state.pts3d_list
        if pts3d_list is None:
            return
        all_pts = np.concatenate([p.reshape(-1, 3) for p in pts3d_list], axis=0).astype(np.float32)
        if len(all_pts) < 10:
            return

        # Apply scene rotation (same as rendering)
        from scipy.spatial.transform import Rotation as SciRot
        R_scene = SciRot.from_euler('xyz', [state.scene_rot_x, state.scene_rot_y, state.scene_rot_z],
                                     degrees=True).as_matrix().astype(np.float32)
        pts_display = (R_scene @ all_pts.T).T

        # Viewport
        win_w, win_h = glfw.get_window_size(window)
        vp_x = 400
        vp_w = win_w - vp_x
        vp_h = win_h
        if vp_w <= 0 or vp_h <= 0:
            return

        # Unproject click to ray
        aspect = vp_w / max(vp_h, 1)
        proj = camera.get_projection_matrix(aspect)
        view = camera.get_view_matrix()
        mvp_inv = np.linalg.inv((proj @ view).astype(np.float64))

        nx = 2.0 * (mx - vp_x) / vp_w - 1.0
        ny = 1.0 - 2.0 * my / vp_h

        near_h = mvp_inv @ np.array([nx, ny, -1, 1], dtype=np.float64)
        far_h = mvp_inv @ np.array([nx, ny, 1, 1], dtype=np.float64)
        near_pt = (near_h[:3] / near_h[3]).astype(np.float32)
        far_pt = (far_h[:3] / far_h[3]).astype(np.float32)
        ray_dir = far_pt - near_pt
        ray_dir /= max(np.linalg.norm(ray_dir), 1e-8)

        # Find nearest point to ray
        diff = pts_display - near_pt[None, :]
        cross = np.cross(diff, ray_dir[None, :])
        ray_dists = np.linalg.norm(cross, axis=-1)
        nearest_idx = np.argmin(ray_dists)
        hit_pt_display = pts_display[nearest_idx]

        # Set orbit target to hit point
        camera.target = hit_pt_display.copy()
        # Adjust distance to maintain current view feel
        camera.distance = max(0.1, np.linalg.norm(camera.get_position() - hit_pt_display))
        state.status = f"Focus: ({hit_pt_display[0]:.2f}, {hit_pt_display[1]:.2f}, {hit_pt_display[2]:.2f})"
    except Exception as e:
        print(f"Focus click error: {e}")


def _update_align_widgets(state, scene_gl, hover_anchor=None, hover_points=None):
    """Rebuild alignment widget geometry from current anchors + optional hover preview."""
    anchors = []
    if state.align_floor_anchor is not None:
        a = state.align_floor_anchor.copy()
        a['color'] = (0.3, 1.0, 0.3)  # green
        anchors.append(a)
    if state.align_wall_anchor is not None:
        a = state.align_wall_anchor.copy()
        a['color'] = (0.3, 0.7, 1.0)  # blue
        anchors.append(a)
    if hover_anchor is not None:
        anchors.append(hover_anchor)
    scene_gl.set_widgets(anchors, highlight_points=hover_points)


def _raycast_to_surface(state, camera, mx, my, window, circle_px=40, subsample=1,
                        return_neighborhood=False):
    """Raycast from screen coords into point cloud. Returns (centroid, normal, radius) or None.
    If return_neighborhood=True, returns (centroid, normal, radius, neighborhood_pts).
    Works in original (pre-rotation) space."""
    pts3d_list = state.pts3d_list
    if pts3d_list is None:
        return None
    all_pts = np.concatenate([p.reshape(-1, 3) for p in pts3d_list], axis=0).astype(np.float32)
    if len(all_pts) < 10:
        return None
    if subsample > 1:
        all_pts = all_pts[::subsample]

    from scipy.spatial.transform import Rotation as SciRot
    R_scene = SciRot.from_euler('xyz', [state.scene_rot_x, state.scene_rot_y, state.scene_rot_z],
                                 degrees=True).as_matrix().astype(np.float32)
    pts_display = (R_scene @ all_pts.T).T

    win_w, win_h = glfw.get_window_size(window)
    vp_x = 400
    vp_w = win_w - vp_x
    vp_h = win_h
    if vp_w <= 0 or vp_h <= 0:
        return None

    aspect = vp_w / max(vp_h, 1)
    proj = camera.get_projection_matrix(aspect)
    view = camera.get_view_matrix()
    mvp = proj @ view
    mvp_inv = np.linalg.inv(mvp.astype(np.float64))

    nx = 2.0 * (mx - vp_x) / vp_w - 1.0
    ny = 1.0 - 2.0 * my / vp_h
    near_h = mvp_inv @ np.array([nx, ny, -1, 1], dtype=np.float64)
    far_h = mvp_inv @ np.array([nx, ny, 1, 1], dtype=np.float64)
    near_pt = (near_h[:3] / near_h[3]).astype(np.float32)
    far_pt = (far_h[:3] / far_h[3]).astype(np.float32)
    ray_dir = far_pt - near_pt
    ray_dir /= max(np.linalg.norm(ray_dir), 1e-8)

    # Project points to screen, select within circle
    pts_h = np.column_stack([pts_display, np.ones(len(pts_display))]).astype(np.float64)
    clip = (mvp.astype(np.float64) @ pts_h.T).T
    w = clip[:, 3]
    valid_w = np.abs(w) > 1e-6
    ndc_x = np.zeros(len(all_pts)); ndc_y = np.zeros(len(all_pts))
    ndc_x[valid_w] = clip[valid_w, 0] / w[valid_w]
    ndc_y[valid_w] = clip[valid_w, 1] / w[valid_w]
    screen_x = (ndc_x + 1) * 0.5 * vp_w + vp_x
    screen_y = (1 - ndc_y) * 0.5 * vp_h
    screen_dist = np.sqrt((screen_x - mx) ** 2 + (screen_y - my) ** 2)
    mask = valid_w & (screen_dist < circle_px)
    neighborhood = all_pts[mask]

    if len(neighborhood) < 10:
        return None

    # Average normals from many random 3-point triplets.
    # Each triplet gives a candidate normal via cross product.
    # Orient all to the same hemisphere, then average — outlier
    # triplets get diluted by the large number of good ones.
    n_pts = len(neighborhood)
    n_samples = 500
    rng = np.random.default_rng()
    idx = rng.integers(0, n_pts, size=(n_samples, 3))

    p0 = neighborhood[idx[:, 0]]
    p1 = neighborhood[idx[:, 1]]
    p2 = neighborhood[idx[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)  # (n_samples, 3)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = lengths[:, 0] > 1e-10
    normals = normals[valid]
    lengths = lengths[valid]
    normals /= lengths

    if len(normals) < 10:
        return None

    # Orient all normals to same hemisphere (flip those pointing opposite to majority)
    ref = normals[0]
    dots = normals @ ref
    normals[dots < 0] *= -1

    # Average and normalize
    normal = normals.mean(axis=0).astype(np.float32)
    normal /= max(np.linalg.norm(normal), 1e-8)

    centroid = neighborhood.mean(axis=0)

    # Orient normal toward camera
    cam_pos = camera.get_position()
    R_scene_inv = R_scene.T
    cam_pos_orig = R_scene_inv @ cam_pos
    to_cam = cam_pos_orig - centroid
    if np.dot(normal, to_cam) < 0:
        normal = -normal

    radius = float(np.linalg.norm(neighborhood - centroid, axis=1).mean()) * 1.5
    radius = max(radius, 0.01)
    if return_neighborhood:
        return centroid, normal, radius, neighborhood
    return centroid, normal, radius


def _compute_align_rotation(floor_n, wall_n):
    """Compute scene rotation (Euler XYZ degrees) from floor and/or wall normals.
    Returns (rx, ry, rz) or None."""
    from scipy.spatial.transform import Rotation as SciRot

    if floor_n is not None and wall_n is not None:
        Y = floor_n / max(np.linalg.norm(floor_n), 1e-8)
        Z = wall_n - np.dot(wall_n, Y) * Y
        Z /= max(np.linalg.norm(Z), 1e-8)
        X = np.cross(Y, Z)
        X /= max(np.linalg.norm(X), 1e-8)
        R_new = np.array([X, Y, Z], dtype=np.float32)
    elif floor_n is not None:
        Y = floor_n / max(np.linalg.norm(floor_n), 1e-8)
        # Z (forward) should be orthogonal to Y, close to world-Z
        ref_fwd = np.array([0, 0, 1], dtype=np.float32)
        if abs(np.dot(Y, ref_fwd)) > 0.9:
            ref_fwd = np.array([1, 0, 0], dtype=np.float32)
        Z = ref_fwd - np.dot(ref_fwd, Y) * Y
        Z /= max(np.linalg.norm(Z), 1e-8)
        X = np.cross(Y, Z)
        X /= max(np.linalg.norm(X), 1e-8)
        R_new = np.array([X, Y, Z], dtype=np.float32)
    elif wall_n is not None:
        Z = wall_n / max(np.linalg.norm(wall_n), 1e-8)
        # Y should be as close to world-up as possible, orthogonal to Z
        ref_up = np.array([0, 1, 0], dtype=np.float32)
        if abs(np.dot(Z, ref_up)) > 0.9:
            ref_up = np.array([1, 0, 0], dtype=np.float32)
        Y = ref_up - np.dot(ref_up, Z) * Z  # project out Z component
        Y /= max(np.linalg.norm(Y), 1e-8)
        X = np.cross(Y, Z)
        X /= max(np.linalg.norm(X), 1e-8)
        R_new = np.array([X, Y, Z], dtype=np.float32)
    else:
        return None

    euler = SciRot.from_matrix(R_new.astype(np.float64)).as_euler('xyz', degrees=True)
    return (float(euler[0]), float(euler[1]), float(euler[2]))


def _handle_align_click(state, scene_gl, camera, mx, my, window):
    """Raycast from click, find surface points, fit plane, place anchor widget."""
    try:
        mode = state.align_mode
        state.align_mode = None

        result = _raycast_to_surface(state, camera, mx, my, window)
        if result is None:
            state.status = "Too few points — try a different spot"
            return
        centroid, normal, radius = result

        # Store normal and anchor (no rotation applied yet — user clicks "Align")
        anchor = {'pos': centroid.copy(), 'normal': normal.copy(), 'radius': radius}
        if mode == 'floor':
            state.align_floor_normal = normal.copy()
            state.align_floor_anchor = anchor
            state.status = "Floor anchor placed — click Align to apply"
            print(f"  Floor anchor: normal={normal}, radius={radius:.4f}")
        elif mode == 'wall':
            state.align_wall_normal = normal.copy()
            state.align_wall_anchor = anchor
            state.status = "Wall anchor placed — click Align to apply"
            print(f"  Wall anchor: normal={normal}, radius={radius:.4f}")

        _update_align_widgets(state, scene_gl)

    except Exception as e:
        state.status = f"Alignment failed: {e}"
        import traceback; traceback.print_exc()
        state.align_mode = None


def _handle_line_click(state, scene_gl, camera, mx, my, window):
    """Handle clicks for horizontal/vertical line alignment tools."""
    try:
        result = _raycast_to_surface(state, camera, mx, my, window)
        if result is None:
            state.status = "No surface found — try a different spot"
            return
        centroid, normal, radius = result

        if state.align_line_start is None:
            # First click — store start point
            state.align_line_start = centroid.copy()
            state.align_line_start_screen = (mx, my)
            state.status = f"Line start set — click end point"
            return

        # Second click — compute direction and apply alignment
        start = state.align_line_start
        end = centroid.copy()
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 1e-6:
            state.status = "Points too close — try again"
            state.align_line_start = None
            state.align_line_start_screen = None
            return
        direction /= length

        mode = state.align_line_mode

        if mode == 'vline':
            # This direction should be vertical (Y-up) — treat as floor normal constraint
            # Floor normal is perpendicular to the line direction projected to horizontal
            # Actually: vertical line means this direction IS the up direction
            state.align_floor_normal = direction.astype(np.float32)
            state.align_floor_anchor = {'pos': (start + end) / 2, 'normal': direction.astype(np.float32), 'radius': length / 2}
            print(f"  Vertical line: direction={direction}")

        elif mode == 'hline':
            # This direction should be horizontal — use it to constrain wall/yaw
            # If floor normal exists, project direction onto floor plane
            floor_n = state.align_floor_normal
            if floor_n is not None:
                direction = direction - np.dot(direction, floor_n) * floor_n
                d_len = np.linalg.norm(direction)
                if d_len < 1e-6:
                    state.status = "Line is parallel to floor normal"
                    state.align_line_start = None
                    state.align_line_start_screen = None
                    return
                direction /= d_len
            # Use projected direction as wall normal (forward direction)
            state.align_wall_normal = direction.astype(np.float32)
            state.align_wall_anchor = {'pos': (start + end) / 2, 'normal': direction.astype(np.float32), 'radius': length / 2}
            print(f"  Horizontal line: direction={direction}")

        # Update widgets (no rotation applied yet — user clicks "Align")
        _update_align_widgets(state, scene_gl)
        state.status = f"{'V-Line' if mode == 'vline' else 'H-Line'} placed — click Align to apply"

        # Reset line tool state
        state.align_line_start = None
        state.align_line_start_screen = None
        state.align_line_mode = None

    except Exception as e:
        state.status = f"Line alignment failed: {e}"
        import traceback; traceback.print_exc()
        state.align_line_start = None
        state.align_line_start_screen = None
        state.align_line_mode = None


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


def _detect_sky_masks(image_paths):
    """Detect sky pixels in images using UperNet semantic segmentation (ADE20K).
    Returns list of (H, W) boolean masks where True = sky."""
    try:
        from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor
        from PIL import Image as PILImage
        import torch

        SKY_LABEL_ID = 2  # ADE20K: index 2 = sky

        print("  Loading sky segmentation model...")
        model_name = 'openmmlab/upernet-swin-tiny'
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device).eval()

        sky_masks = []
        for i, path in enumerate(image_paths):
            img = PILImage.open(path).convert('RGB')
            inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            # Upsample logits to original image size
            logits = torch.nn.functional.interpolate(
                outputs.logits, size=(img.height, img.width),
                mode='bilinear', align_corners=False)
            seg_map = logits.argmax(dim=1)[0].cpu().numpy()
            sky_mask = seg_map == SKY_LABEL_ID

            sky_masks.append(sky_mask)
            n_sky = sky_mask.sum()
            print(f"    Image {i+1}: {n_sky:,d} sky pixels ({n_sky * 100 // max(sky_mask.size, 1)}%)")

        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return sky_masks

    except Exception as e:
        print(f"  Sky detection failed: {e}")
        return None


def _create_masked_images(image_paths, prompt, keep=True, mask_sky=False, threshold=0.5):
    """Create masked copies of images, blacking out non-subject pixels.
    Returns list of new image paths in a temp directory."""
    from PIL import Image as PILImage
    import tempfile

    tmpdir = _app_mkdtemp(prefix="masked_")
    masks = []

    # Get subject masks if prompt given
    if prompt:
        subject_masks = _detect_subject_masks(image_paths, prompt, keep=keep, threshold=threshold)
    else:
        subject_masks = None

    # Get sky masks if requested
    if mask_sky:
        sky_masks = _detect_sky_masks(image_paths)
    else:
        sky_masks = None

    new_paths = []
    for i, path in enumerate(image_paths):
        img = PILImage.open(path).convert('RGB')
        img_np = np.array(img)

        # Combine masks: start with all-keep
        keep_mask = np.ones((img.height, img.width), dtype=bool)

        if subject_masks is not None and i < len(subject_masks):
            m = subject_masks[i]
            if m.shape != keep_mask.shape:
                m = np.array(PILImage.fromarray(m).resize(
                    (img.width, img.height), PILImage.NEAREST))
            keep_mask &= m

        if sky_masks is not None and i < len(sky_masks):
            m = sky_masks[i]
            if m.shape != keep_mask.shape:
                m = np.array(PILImage.fromarray(m).resize(
                    (img.width, img.height), PILImage.NEAREST))
            keep_mask &= ~m  # sky = remove

        # Black out non-subject pixels
        img_np[~keep_mask] = 0
        masked_img = PILImage.fromarray(img_np)

        new_path = os.path.join(tmpdir, os.path.basename(path))
        masked_img.save(new_path, quality=95)
        new_paths.append(new_path)

        n_kept = keep_mask.sum()
        print(f"    Image {i+1}: {n_kept:,d}/{keep_mask.size:,d} pixels kept ({n_kept * 100 // max(keep_mask.size, 1)}%)")

    # Also save the masks for post-reconstruction confidence zeroing
    return new_paths, masks if masks else None


def _detect_subject_masks(image_paths, prompt, keep=True, threshold=0.5):
    """Detect subject pixels matching a text prompt using CLIPSeg.
    Returns list of (H, W) boolean masks where True = pixels to KEEP."""
    try:
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
        from PIL import Image as PILImage
        import torch

        print(f"  Loading CLIPSeg for prompt: '{prompt}'...")
        processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)

        masks = []
        for i, path in enumerate(image_paths):
            img = PILImage.open(path).convert('RGB')
            inputs = processor(text=[prompt], images=[img], return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            # Sigmoid to get probability map, resize to image size
            pred = torch.sigmoid(outputs.logits[0]).cpu().numpy()
            # Resize prediction to image size
            pred_resized = np.array(PILImage.fromarray(pred).resize(
                (img.width, img.height), PILImage.BILINEAR))

            if keep:
                # Keep pixels matching the prompt
                mask = pred_resized > threshold
            else:
                # Remove pixels matching the prompt
                mask = pred_resized <= threshold

            masks.append(mask)
            n_selected = mask.sum()
            print(f"    Image {i+1}: {n_selected:,d} pixels {'kept' if keep else 'removed'} ({n_selected * 100 // max(mask.size, 1)}%)")

        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return masks

    except Exception as e:
        print(f"  Subject detection failed: {e}")
        import traceback; traceback.print_exc()
        return None


def _extract_scene_data(state):
    """Extract pts3d and confidence from the CanonicalScene. Caches result."""
    if state.pts3d_list is not None and state.confs_list is not None:
        return state.pts3d_list, state.confs_list

    scene = state.scene
    state.pts3d_list = scene.pts3d
    state.confs_list = scene.confidence
    return state.pts3d_list, state.confs_list




def _run_decimate_points(state, scene_gl):
    """Merge nearby points using voxel grid, confidence-weighted position and color.

    Replaces the scene data with a single merged view so mesh generation
    and all downstream operations use the decimated cloud.
    """
    state.refine_progress = "Decimating points..."
    try:
        pts3d_list, confs_list = _extract_scene_data(state)
        scene = state.scene
        imgs = scene.imgs

        # Collect all points with confidence
        all_pts, all_cols, all_conf = [], [], []
        for i in range(len(imgs)):
            p = pts3d_list[i]
            c = confs_list[i]
            img = imgs[i]
            if p.ndim == 3:
                H, W = p.shape[:2]
                conf_2d = c.reshape(H, W) if c.ndim != 2 else c
                mask = (conf_2d > state.min_conf) & np.isfinite(p).all(axis=-1)
                all_pts.append(p[mask])
                all_cols.append((np.clip(img[mask], 0, 1) * 255).astype(np.uint8))
                all_conf.append(conf_2d[mask])
            else:
                c_flat = c.ravel()
                mask = (c_flat > state.min_conf) & np.isfinite(p).all(axis=-1)
                all_pts.append(p[mask])
                all_cols.append((np.clip(img.reshape(-1, 3)[mask], 0, 1) * 255).astype(np.uint8))
                all_conf.append(c_flat[mask])

        points = np.concatenate(all_pts, axis=0).astype(np.float32)
        colors = np.concatenate(all_cols, axis=0).astype(np.float32)
        conf = np.concatenate(all_conf, axis=0).astype(np.float32)
        n_before = len(points)

        if n_before == 0:
            state.status = "No points to decimate"
            state.refining = False
            return

        state.refine_progress = f"Voxel merging {n_before:,d} points..."

        # Use open3d for fast voxel downsampling to estimate good voxel size,
        # then do confidence-weighted merge manually.
        import open3d as o3d

        # Target ~200k output points — binary search for voxel size
        target_pts = min(200_000, n_before // 2)
        target_pts = max(target_pts, 10_000)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        bb_min = pcd.get_min_bound()
        bb_max = pcd.get_max_bound()
        extent = np.linalg.norm(bb_max - bb_min)

        # Binary search for voxel size that gives ~target_pts
        lo, hi = extent / 10000, extent / 10
        for _ in range(20):
            mid = (lo + hi) / 2
            down = pcd.voxel_down_sample(mid)
            n_down = len(down.points)
            if n_down > target_pts:
                lo = mid
            else:
                hi = mid
            if abs(n_down - target_pts) / max(target_pts, 1) < 0.1:
                break
        voxel_size = (lo + hi) / 2

        state.refine_progress = f"Confidence-weighted merge (voxel={voxel_size:.4f})..."

        # Quantize to voxel grid
        grid_coords = np.floor((points - bb_min.astype(np.float32)) / voxel_size).astype(np.int32)
        grid_max = grid_coords.max(axis=0) + 1

        # Linear voxel index
        voxel_idx = (grid_coords[:, 0].astype(np.int64) * grid_max[1] * grid_max[2] +
                     grid_coords[:, 1].astype(np.int64) * grid_max[2] +
                     grid_coords[:, 2].astype(np.int64))

        unique_voxels, inverse = np.unique(voxel_idx, return_inverse=True)
        n_voxels = len(unique_voxels)

        # Confidence-weighted accumulation
        w = conf[:, None]
        merged_pts = np.zeros((n_voxels, 3), dtype=np.float64)
        merged_cols = np.zeros((n_voxels, 3), dtype=np.float64)
        merged_w = np.zeros((n_voxels, 1), dtype=np.float64)
        merged_conf = np.zeros(n_voxels, dtype=np.float64)

        np.add.at(merged_pts, inverse, points * w)
        np.add.at(merged_cols, inverse, colors * w)
        np.add.at(merged_w, inverse, w)
        np.maximum.at(merged_conf, inverse, conf)

        valid = merged_w[:, 0] > 0
        merged_pts[valid] /= merged_w[valid]
        merged_cols[valid] /= merged_w[valid]

        merged_pts = merged_pts[valid].astype(np.float32)
        merged_cols = np.clip(merged_cols[valid], 0, 255).astype(np.uint8)
        merged_conf = merged_conf[valid].astype(np.float32)

        print(f"  Decimated: {n_before:,d} → {len(merged_pts):,d} points "
              f"(voxel={voxel_size:.4f}, extent={extent:.2f})")

        # Update display
        scene_gl.set_points(merged_pts, merged_cols)
        state.has_points = True
        state.points_modified = True

        # Sync decimated data into both state cache and CanonicalScene.
        # Keep original scene.images (needed for COLMAP image export).
        # Store decimated colors in _dense_colors for COLMAP point colors.
        state.pts3d_list = [merged_pts]
        state.confs_list = [merged_conf]
        scene.pts3d = [merged_pts]
        scene.confidence = [merged_conf]
        state._dense_colors = merged_cols
        state.status = f"Decimated: {n_before:,d} → {len(merged_pts):,d} points"

    except Exception as e:
        state.error_msg = str(e)
        state.status = f"Decimate failed: {e}"
        import traceback
        traceback.print_exc()
    finally:
        state.refining = False


def _run_smooth_preview(state, scene_gl):
    """Preview the smoothed point cloud without meshing."""
    state.refine_progress = "Smoothing points..."
    try:
        from mesh_export import _smooth_cloud

        pts3d_list, confs_list = _extract_scene_data(state)
        imgs = state.scene.imgs
        mesh_min_conf = state.min_conf

        all_pts, all_cols, all_vids = [], [], []
        for i in range(len(pts3d_list)):
            p = pts3d_list[i]
            c_arr = confs_list[i] if i < len(confs_list) else None
            if p.ndim == 3:
                H, W = p.shape[:2]
                mask = c_arr.reshape(H, W) > mesh_min_conf if c_arr is not None else np.ones((H, W), dtype=bool)
                all_pts.append(p[mask])
                if i < len(imgs):
                    all_cols.append((np.clip(imgs[i][mask], 0, 1) * 255).astype(np.uint8))
                else:
                    all_cols.append(np.full((mask.sum(), 3), 180, dtype=np.uint8))
            else:
                flat = p.reshape(-1, 3)
                mask = c_arr.ravel() > mesh_min_conf if c_arr is not None else np.ones(len(flat), dtype=bool)
                all_pts.append(flat[mask])
                all_cols.append(np.full((mask.sum(), 3), 180, dtype=np.uint8))
            all_vids.append(np.full(int(mask.sum()), i, dtype=np.int32))

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


def run_bundle_adjust(state, scene_gl):
    """Refine cameras via depth-consistency optimization on pointmaps."""
    state.refine_progress = "Starting bundle adjustment..."
    try:
        from mesh_export import bundle_adjust_depth_consistency
        import torch

        scene = state.scene

        # Extract cameras — CanonicalScene always returns a plain tensor
        c2w_all = scene.get_im_poses().numpy()
        c2w_list = [c2w_all[i].astype(np.float32) for i in range(len(c2w_all))]

        # Extract pointmaps and intrinsics
        pts3d_list, confs_list = _extract_scene_data(state)
        K_list = []
        for i in range(len(c2w_list)):
            orig_w, orig_h = scene.original_sizes[i]
            K = scene.scale_intrinsics_to(orig_w, orig_h, i).astype(np.float32)
            K_list.append(K)

        def progress(msg):
            state.refine_progress = msg

        refined = bundle_adjust_depth_consistency(
            c2w_list, K_list, pts3d_list, confs_list,
            image_paths=state.image_paths,
            refine_focal=state.ba_refine_focal,
            n_iters=state.ba_n_iters,
            min_conf=state.ba_min_conf,
            max_shift=state.ba_max_shift,
            huber_scale=state.ba_huber_scale,
            progress_fn=progress)

        if refined is None:
            state.status = "Bundle adjustment failed"
            return

        # Save old cameras before updating (needed to transform pointmaps)
        n = len(refined)
        old_c2w = [c2w_list[i].astype(np.float64) for i in range(n)]

        # Write refined cameras back into the CanonicalScene
        for i in range(n):
            c2w_new, K_new, W, H = refined[i]
            scene.c2w[i] = c2w_new.astype(np.float64)
            if state.ba_refine_focal:
                # Scale K_new (at original res) back to model's internal resolution
                if scene.internal_resolution > 0 and scene.backend in ('vggt', 'lingbot'):
                    ratio = max(W, H) / float(scene.internal_resolution)
                    scene.intrinsics[i][0, 0] = K_new[0, 0] / ratio
                    scene.intrinsics[i][1, 1] = K_new[1, 1] / ratio
                elif scene.internal_resolution > 0:
                    img = scene.images[i]
                    int_h, int_w = img.shape[:2]
                    scene.intrinsics[i][0, 0] = K_new[0, 0] * int_w / W
                    scene.intrinsics[i][1, 1] = K_new[1, 1] * int_h / H
                else:
                    scene.intrinsics[i][0, 0] = K_new[0, 0]
                    scene.intrinsics[i][1, 1] = K_new[1, 1]

        # Transform pointmaps to match refined cameras.
        # Pointmaps are in world space baked with the OLD cameras. After BA moves
        # a camera, its points must follow: new_pt = new_c2w @ old_w2c @ old_pt
        print("  Transforming pointmaps to match refined cameras...")
        for i in range(min(n, len(pts3d_list))):
            new_c2w = refined[i][0].astype(np.float64)
            old_w2c = np.linalg.inv(old_c2w[i])
            delta = new_c2w @ old_w2c  # maps old world -> new world

            # Skip if camera barely moved (identity check)
            if np.allclose(delta, np.eye(4), atol=1e-6):
                continue

            p = pts3d_list[i]
            if p.ndim == 3:
                H_m, W_m = p.shape[:2]
                flat = p.reshape(-1, 3).astype(np.float64)
                transformed = (delta[:3, :3] @ flat.T).T + delta[:3, 3]
                pts3d_list[i] = transformed.reshape(H_m, W_m, 3).astype(np.float32)
            elif p.ndim == 2:
                flat = p.astype(np.float64)
                transformed = (delta[:3, :3] @ flat.T).T + delta[:3, 3]
                pts3d_list[i] = transformed.astype(np.float32)

            # Update CanonicalScene's pointmaps directly
            scene.pts3d[i] = pts3d_list[i]

        # Cache the transformed pointmaps
        state.pts3d_list = pts3d_list
        state.confs_list = confs_list
        state.points_modified = True

        # Display points with updated cameras
        all_pts, all_cols = [], []
        for i in range(len(scene.images)):
            p = pts3d_list[i]
            c = confs_list[i]
            img = scene.images[i]
            if p.ndim == 3:
                mask = c.reshape(p.shape[:2]) > state.min_conf if c is not None else np.ones(p.shape[:2], dtype=bool)
                all_pts.append(p[mask])
                all_cols.append((np.clip(img[mask], 0, 1) * 255).astype(np.uint8))
        if all_pts:
            points = np.concatenate(all_pts, axis=0)
            colors = np.concatenate(all_cols, axis=0)
            if len(points) > 200000:
                idx = np.random.choice(len(points), 200000, replace=False)
                points, colors = points[idx], colors[idx]
            scene_gl.set_points(points, colors)

        # Update camera display
        cam_poses = scene.get_im_poses().numpy()
        ext = np.linalg.norm(points.max(axis=0) - points.min(axis=0)) if len(all_pts) > 0 and len(points) > 0 else 1.0
        scene_gl.set_cameras([cam_poses[i] for i in range(len(cam_poses))],
                             scale=float(ext) * 0.05)

        state.status = f"Bundle adjusted {n} cameras"

    except Exception as e:
        state.status = f"Bundle adjust failed: {e}"
        import traceback; traceback.print_exc()
    finally:
        state.refining = False
        state.refine_progress = ""


def run_densify_colmap(state, scene_gl):
    """Densify point cloud using COLMAP PatchMatch stereo (GPU)."""
    state.refine_progress = "Densifying with COLMAP PatchMatch..."
    try:
        from mesh_export import densify_colmap

        # Get cameras
        scene = state.scene
        c2w_all = scene.get_im_poses().numpy()
        c2w_list = [c2w_all[i].astype(np.float32) for i in range(len(c2w_all))]
        K_list = []
        pts3d_list, confs_list = _extract_scene_data(state)
        for i in range(len(c2w_all)):
            orig_w, orig_h = scene.original_sizes[i]
            K = scene.scale_intrinsics_to(orig_w, orig_h, i).astype(np.float32)
            K_list.append(K)

        def progress(msg):
            state.refine_progress = msg

        # Collect existing points for depth range estimation
        existing = None
        if pts3d_list:
            all_p = [p.reshape(-1, 3) for p in pts3d_list if p is not None]
            if all_p:
                existing = np.concatenate(all_p, axis=0).astype(np.float32)

        pm_opts = dict(
            max_image_size=state.pm_max_image_size,
            num_iterations=state.pm_num_iterations,
            window_radius=state.pm_window_radius,
            min_consistent=state.pm_min_consistent,
            geom_consistency=state.pm_geom_consistency,
            filter_min_ncc=state.pm_filter_min_ncc,
            existing_pts=existing,
            colmap_workdir=getattr(state, '_colmap_workdir', None))
        dense_pts, dense_cols, colmap_cams = densify_colmap(
            state.image_paths, c2w_list, K_list, progress_fn=progress, **pm_opts)

        if len(dense_pts) > 0:
            # Replace point data with dense cloud, keep existing scene/cameras
            state.pts3d_list = [dense_pts]
            state.confs_list = [np.ones(len(dense_pts), dtype=np.float32) * 10.0]
            state.scene.pts3d = state.pts3d_list
            state.scene.confidence = state.confs_list
            state._dense_colors = dense_cols
            state.points_modified = True

            # Display the dense cloud
            disp_pts, disp_cols = dense_pts, dense_cols
            if len(disp_pts) > 500000:
                idx = np.random.choice(len(disp_pts), 500000, replace=False)
                disp_pts, disp_cols = disp_pts[idx], disp_cols[idx]
            scene_gl.set_points(disp_pts, disp_cols)
            state.has_points = True
            state.points_modified = True
            state.status = f"COLMAP dense: {len(dense_pts):,d} points"
        else:
            state.status = "COLMAP PatchMatch produced no points (CUDA required)"

    except Exception as e:
        state.status = f"COLMAP densify failed: {e}"
        import traceback; traceback.print_exc()
    finally:
        state.refining = False
        state.refine_progress = ""


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
        c2w_all = scene.get_im_poses().numpy()
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
            try:
                K_full = scene.scale_intrinsics_to(W_full, H_full, i)
                fx, fy = K_full[0, 0], K_full[1, 1]
                cx, cy = K_full[0, 2], K_full[1, 2]
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

        # ── Equirectangular shortcut: spherical grid mesh from merged points ──
        if getattr(scene, 'equirect', None) is not None:
            state.refine_progress = "Building equirectangular mesh..."
            from equirect import equirect_mesh
            eq_pts3d = scene.equirect['merged_pts3d']
            eq_conf = scene.equirect['merged_conf']
            pano = scene.equirect['pano_img']  # uint8
            verts, faces, colors, texture, uvs = equirect_mesh(
                eq_pts3d, eq_conf, pano,
                min_conf=mesh_min_conf, depth_edge_mult=5.0, step=2)
            if len(faces) > 0:
                state.mesh_data = (verts, faces, colors)
                # Apply panorama texture with UVs
                scene_gl.set_texture(texture, uvs, faces, verts, faces, colors)
                state.uv_data = (texture, uvs, faces)
                state.has_mesh = True
                state.draw_mode = 1
                state.status = f"Equirect mesh: {len(verts):,d} verts, {len(faces):,d} faces (textured)"
            else:
                state.status = "Equirect mesh: no faces generated"
            state.refining = False
            return

        from mesh_export import create_dense_mesh, _smooth_cloud, _collect_points
        from mesh_export import _mesh_local_delaunay, _mesh_ball_pivot_from_cloud, _close_holes_pymeshlab

        # Get camera poses
        cam_poses = None
        cam_center = None
        try:
            c2w = scene.get_im_poses().numpy()
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
                verts, faces, colors = _mesh_local_delaunay(pts, cols, cam_center=cam_center,
                                                           radius_mult=state.delaunay_edge_mult)
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
                mode=mesh_mode, hole_cap_size=state.hole_cap_size,
                poisson_depth=state.poisson_depth_val,
                trim_percentile=state.poisson_trim,
                dense_colors=getattr(state, '_dense_colors', None),
                bp_radius_mult=state.bp_radius_mult,
                delaunay_edge_mult=state.delaunay_edge_mult)

        if len(faces) > 0:
            state.mesh_data = (verts, faces, colors)
            state.uv_data = None  # invalidate UVs when mesh changes
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
                    qualitythr=0.3,
                    preserveboundary=True,
                    preservenormal=True,
                    preservetopology=True,
                    optimalplacement=True,
                    autoclean=True)
            except TypeError:
                ms.simplification_quadric_edge_collapse_decimation(
                    targetfacenum=target,
                    preserveboundary=True,
                    preservenormal=True,
                    optimalplacement=True)

            # Topology cleanup: isotropic remeshing to fix star vertices
            # Compute target edge length from the decimated mesh
            state.refine_progress = "Fixing topology (isotropic remeshing)..."
            mesh_dec = ms.current_mesh()
            bbox = mesh_dec.bounding_box()
            diag = bbox.diagonal()
            n_faces_dec = mesh_dec.face_number()
            # Target edge length: approximate from face count assuming equilateral triangles
            # area ≈ n_faces * (sqrt(3)/4) * edge^2, total_area ≈ diag^2 * factor
            import math
            target_edge = diag * math.sqrt(2.0 / max(n_faces_dec, 1))
            try:
                ms.meshing_isotropic_explicit_remeshing(
                    targetlen=pymeshlab.AbsoluteValue(target_edge),
                    iterations=3,
                    adaptive=True,
                    checksurfdist=True,
                    maxsurfdist=pymeshlab.AbsoluteValue(target_edge * 0.5))
            except Exception as e_remesh:
                print(f"  Isotropic remeshing skipped: {e_remesh}")

            # Transfer vertex colors from original mesh via nearest-neighbor
            mesh_out = ms.current_mesh()
            verts_out = mesh_out.vertex_matrix().astype(np.float32)
            faces_out = mesh_out.face_matrix().astype(np.int32)

            if mesh_out.has_vertex_color() and mesh_out.vertex_number() == len(verts_out):
                vc = mesh_out.vertex_color_matrix()
                colors_out = (vc[:, :3] * 255).clip(0, 255).astype(np.uint8)
            else:
                # Remeshing added new vertices — transfer colors via nearest neighbor
                from scipy.spatial import cKDTree
                tree = cKDTree(verts)
                _, idx = tree.query(verts_out)
                colors_out = colors[idx].copy()
            print(f"  Decimated with PyMeshLab: {len(faces_out):,d} faces (remeshed)")

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
        state.uv_data = None  # invalidate UVs after decimation
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
        tmpdir = _app_mkdtemp()
        export_dir = os.path.join(tmpdir, 'colmap')
        state.refine_progress = "Exporting cameras..."
        export_scene_to_colmap(
            scene=state.scene, image_paths=state.image_paths,
            output_dir=export_dir, min_conf_thr=state.min_conf,
            dense_colors=getattr(state, '_dense_colors', None))

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
        tmpdir = _app_mkdtemp()
        export_dir = os.path.join(tmpdir, 'colmap')
        state.refine_progress = "Exporting cameras..."
        export_scene_to_colmap(
            scene=state.scene, image_paths=state.image_paths,
            output_dir=export_dir, min_conf_thr=state.min_conf,
            dense_colors=getattr(state, '_dense_colors', None))
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


def run_create_uvs(state, scene_gl):
    """Create UV unwrap using camera view directions and show debug texture."""
    state.refine_progress = "Creating UVs..."
    try:
        import tempfile
        from texture_map import create_uvs
        from colmap_export import export_scene_to_colmap
        from refine_mesh import load_cameras

        verts, faces, colors = state.mesh_data

        # Load camera views for view-based UV projection
        views = None
        if state.scene is not None:
            try:
                state.refine_progress = "Loading cameras for UV projection..."
                tmpdir = _app_mkdtemp()
                export_dir = os.path.join(tmpdir, 'colmap')
                export_scene_to_colmap(
                    scene=state.scene, image_paths=state.image_paths,
                    output_dir=export_dir, min_conf_thr=state.min_conf,
            dense_colors=getattr(state, '_dense_colors', None))
                views = load_cameras(export_dir)
                # Cache views for bake step
                state._cached_views = views
            except Exception:
                pass

        state.refine_progress = "UV unwrapping..."
        uvs, uv_faces, debug_tex = create_uvs(verts, faces, views)
        state.uv_data = (uvs, uv_faces)

        scene_gl.set_texture(debug_tex, uvs, uv_faces, verts, faces, colors)
        state.status = f"UVs created: {len(uvs):,d} coords — verify checkerboard"
    except Exception as e:
        state.error_msg = str(e)
        state.status = f"UV creation failed: {e}"
        import traceback; traceback.print_exc()
    state.refining = False
    state.refine_progress = ""


def run_texture(state, scene_gl):
    """Bake texture from camera images into UV map."""
    state.refine_progress = "Baking texture..."
    try:
        import tempfile
        from colmap_export import export_scene_to_colmap
        from refine_mesh import load_cameras

        verts, faces, colors = state.mesh_data
        uvs, uv_faces = state.uv_data

        # Reuse cached views from UV step, or load fresh
        views = getattr(state, '_cached_views', None)
        if views is None:
            state.refine_progress = "Exporting cameras..."
            tmpdir = _app_mkdtemp()
            export_dir = os.path.join(tmpdir, 'colmap')
            export_scene_to_colmap(
                scene=state.scene, image_paths=state.image_paths,
                output_dir=export_dir, min_conf_thr=state.min_conf,
            dense_colors=getattr(state, '_dense_colors', None))
            views = load_cameras(export_dir)

        # Bake
        state.refine_progress = f"Baking {len(faces):,d} faces from {len(views)} cameras..."
        from texture_map import bake_texture
        texture_img = bake_texture(verts, faces, uvs, uv_faces, views)

        # Store baked texture for export
        state._baked_texture = texture_img

        # Upload to viewport
        state.refine_progress = "Uploading texture..."
        scene_gl.set_texture(texture_img, uvs, uv_faces, verts, faces, colors)

        state.status = f"Texture baked — use Export to save"

    except Exception as e:
        state.error_msg = str(e)
        state.status = f"Texture bake failed: {e}"
        import traceback; traceback.print_exc()

    state.refining = False
    state.refine_progress = ""


def _export_textured_obj(state, path):
    """Export textured mesh to OBJ + MTL + PNG at user-chosen path."""
    try:
        from texture_map import _write_obj
        from PIL import Image

        verts, faces, colors = state.mesh_data
        uvs, uv_faces = state.uv_data
        texture_img = state._baked_texture

        out_dir = os.path.dirname(path)
        base = os.path.splitext(os.path.basename(path))[0]

        # Save texture
        tex_name = f"{base}.png"
        tex_path = os.path.join(out_dir, tex_name)
        Image.fromarray(texture_img).save(tex_path, quality=95)

        # Save MTL
        mtl_name = f"{base}.mtl"
        mtl_path = os.path.join(out_dir, mtl_name)
        with open(mtl_path, 'w') as f:
            f.write(f"newmtl material0\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\n"
                    f"d 1\nillum 1\nmap_Kd {tex_name}\n")

        # Save OBJ (flip V for OBJ convention)
        uvs_obj = uvs.copy()
        uvs_obj[:, 1] = 1.0 - uvs_obj[:, 1]

        # Write OBJ with correct MTL reference
        with open(path, 'w') as f:
            f.write(f"mtllib {mtl_name}\nusemtl material0\n")
            for i in range(len(verts)):
                f.write(f"v {verts[i,0]:.6f} {verts[i,1]:.6f} {verts[i,2]:.6f}\n")
            for i in range(len(uvs_obj)):
                f.write(f"vt {uvs_obj[i,0]:.6f} {uvs_obj[i,1]:.6f}\n")
            for i in range(len(faces)):
                a, b, c = faces[i] + 1
                ua, ub, uc = uv_faces[i] + 1
                f.write(f"f {a}/{ua} {b}/{ub} {c}/{uc}\n")

        state.status = f"Exported: {path}"
        print(f"  Exported textured mesh: {path}")
    except Exception as e:
        state.error_msg = str(e)
        state.status = f"Export failed: {e}"
        import traceback; traceback.print_exc()


def run_train_splats(state, scene_gl):
    """Background thread: train surface-constrained gaussian splats."""
    try:
        import tempfile
        from colmap_export import export_scene_to_colmap
        from surface_splats import train_surface_splats

        # Use imported COLMAP dir directly if available, otherwise re-export
        imported_dir = getattr(state, '_imported_colmap_dir', None)
        if imported_dir and os.path.isdir(os.path.join(imported_dir, 'sparse', '0')):
            export_dir = imported_dir
            state.splat_progress = "Using imported COLMAP dataset..."
            print(f"  Using imported COLMAP dataset: {export_dir}")
        else:
            state.splat_progress = "Exporting cameras..."
            tmpdir = _app_mkdtemp()
            export_dir = os.path.join(tmpdir, 'colmap')
            export_scene_to_colmap(
                scene=state.scene, image_paths=state.image_paths,
                output_dir=export_dir, min_conf_thr=state.min_conf,
                dense_colors=getattr(state, '_dense_colors', None))

        state.splat_progress = "Initializing splats..."

        # Use mesh if available, otherwise fall back to point cloud
        # Scale 0-1 sliders to actual training values
        kwargs = dict(
            colmap_dir=export_dir,
            iterations=state.splat_iterations,
            max_resolution=state.splat_resolution,
            n_samples=state.splat_n_samples,
            target_splats=state.splat_target,
            anchor_weight_start=state.splat_anchor * 0.2,       # 0-1 -> 0-0.2
            aniso_lambda=state.splat_aniso,                      # 0-1 direct (1=isotropic)
            flatness_lambda=state.splat_flatness,                # 0-1 direct (1=razor thin)
            normal_lambda=state.splat_normal,                   # 0-1 direct (1=fully aligned)
            depth_lambda=state.splat_depth * 1.0,               # 0-1 -> 0-1.0
            opacity_decay=state.splat_opacity_decay * 0.01,     # 0-1 -> 0-0.01
            prune_threshold=state.splat_prune * 0.05,           # 0-1 -> 0-0.05
            strategy_name=state.splat_strategies[state.splat_strategy_idx],
            multi_view=state.splat_multi_view,
            multi_view_count=state.splat_multi_view_count,
            smooth_strength=state.splat_smooth * 0.3,           # 0-1 -> 0-0.3
            coverage_lambda=state.splat_coverage,                # 0-1 -> 0-1
            stop_flag=lambda: state.stop_requested,
        )
        if state.mesh_data is not None:
            verts, faces, colors = state.mesh_data
            kwargs['mesh_data'] = (verts, faces, colors)
        else:
            # Collect point cloud from scene
            pts3d_list, confs_list = _extract_scene_data(state)
            dense_cols = getattr(state, '_dense_colors', None)
            all_pts, all_cols = [], []
            for i in range(len(pts3d_list)):
                p = pts3d_list[i]; c = confs_list[i]
                if p.ndim == 3:
                    # Per-image dense points (DUSt3R/MASt3R/VGGT)
                    H, W = p.shape[:2]
                    mask = c.reshape(H, W) > state.min_conf
                    all_pts.append(p[mask])
                    img = state.scene.imgs[i] if i < len(state.scene.imgs) else None
                    if img is not None and img.shape[0] == H and img.shape[1] == W:
                        all_cols.append((np.clip(img[mask], 0, 1) * 255).astype(np.uint8))
                    else:
                        all_cols.append(np.full((mask.sum(), 3), 128, dtype=np.uint8))
                else:
                    # Flat point array (COLMAP PatchMatch)
                    mask = c.ravel() > state.min_conf
                    all_pts.append(p[mask])
                    if dense_cols is not None and len(dense_cols) == len(p):
                        cols = dense_cols[mask]
                        if cols.max() <= 1.0:
                            cols = (np.clip(cols, 0, 1) * 255).astype(np.uint8)
                        all_cols.append(cols.astype(np.uint8))
                    else:
                        all_cols.append(np.full((mask.sum(), 3), 128, dtype=np.uint8))
            cloud_pts = np.concatenate(all_pts).astype(np.float32)
            cloud_cols = np.concatenate(all_cols).astype(np.uint8)
            kwargs['point_cloud'] = (cloud_pts, cloud_cols)

        gen = train_surface_splats(**kwargs)

        for progress in gen:
            state.splat_step = progress['step']
            state.splat_total = progress['total']
            state.splat_progress = (
                f"Step {progress['step']}/{progress['total']} | "
                f"loss={progress['loss']:.4f} | {progress['n_splats']//1000}K splats")

            if progress.get('means') is not None:
                scene_gl.set_splats(
                    progress['means'], progress['colors'], progress['scales'],
                    quats=progress.get('quats'),
                    scales_log=progress.get('scales_log'),
                    opacities_logit=progress.get('opacities_logit'),
                    sh0=progress.get('sh0'))
                # Auto-switch to splats view
                if 'splats' in state.draw_modes:
                    state.draw_mode = state.draw_modes.index('splats')

            if progress.get('done'):
                state.splat_data = progress.get('splats')
                state.splat_progress = f"Done: {progress['n_splats']:,d} splats"
                break

    except Exception as e:
        state.error_msg = str(e)
        state.splat_progress = f"Failed: {e}"
        import traceback; traceback.print_exc()
    finally:
        state.splat_training = False
        state.stop_requested = False


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

    # Clean stale temp data from any previous crashed sessions
    cleanup_all_app_temps()

    if not glfw.init():
        print("Could not initialize GLFW")
        return

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(1600, 900, APP_NAME, None, None)
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

    last_rclick_time = [0.0]

    def mouse_button_callback(window, button, action, mods):
        # Ignore input when window is not focused
        if not glfw.get_window_attrib(window, glfw.FOCUSED):
            return
        if imgui.get_io().want_capture_mouse:
            return
        if action == glfw.PRESS:
            # Alignment / line tool click
            if button == 0 and (state.align_mode or state.align_line_mode):
                mx, my = glfw.get_cursor_pos(window)
                if state.align_line_mode:
                    _handle_line_click(state, scene_gl, camera, mx, my, window)
                else:
                    _handle_align_click(state, scene_gl, camera, mx, my, window)
                return
            # Double right-click: focus orbit on clicked point
            if button == 1:
                now = glfw.get_time()
                if now - last_rclick_time[0] < 0.3:  # 300ms double-click
                    mx, my = glfw.get_cursor_pos(window)
                    _handle_focus_click(state, camera, mx, my, window)
                    last_rclick_time[0] = 0.0
                    return
                last_rclick_time[0] = now
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
    # Note: don't install a key callback — it would override imgui's key handler
    # and corrupt imgui state. Escape handling is done via polling in the main loop.

    while not glfw.window_should_close(window):
      try:
        glfw.poll_events()

        # Skip rendering when minimized or iconified (prevents GL crashes)
        if glfw.get_window_attrib(window, glfw.ICONIFIED):
            import time; time.sleep(0.1)
            continue

        impl.process_inputs()

        # Escape key: cancel alignment tools (polled to avoid overriding imgui's key callback)
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            if state.align_mode or state.align_line_mode:
                if state.align_preview_rot:
                    state.scene_rot_x, state.scene_rot_y, state.scene_rot_z = state.align_preview_rot
                    state.align_preview_rot = None
                state.align_mode = None
                state.align_line_mode = None
                state.align_line_start = None
                state.align_line_start_screen = None
                state.status = "Alignment cancelled"

        # Left/Right arrows: cycle through scene camera POVs
        if not imgui.get_io().want_capture_keyboard and state.cached_cameras:
            n_cams = len(state.cached_cameras)
            arrow_pressed = False
            if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS and not getattr(state, '_arrow_held', False):
                state.cam_view_idx = (state.cam_view_idx + 1) % n_cams
                arrow_pressed = True
            elif glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS and not getattr(state, '_arrow_held', False):
                state.cam_view_idx = (state.cam_view_idx - 1) % n_cams
                arrow_pressed = True
            # Track key held state for single-step per press
            state._arrow_held = (glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS or
                                 glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS)
            if arrow_pressed and 0 <= state.cam_view_idx < n_cams:
                idx = state.cam_view_idx
                if idx < len(state.image_paths):
                    state.cam_view_name = os.path.basename(state.image_paths[idx])
                else:
                    state.cam_view_name = f"Camera {idx}"

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
                state.cam_view_idx = -1; state.cam_view_name = ""
            if mouse_down[1]:  # Right = pan
                camera.pan(-dx, dy)
                state.cam_view_idx = -1; state.cam_view_name = ""
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
            # Clean all temp files
            cleanup_all_app_temps()
            # Reset everything
            if hasattr(state, '_ever_centered'):
                del state._ever_centered
            state.__init__()
            # Clear all GPU-side data
            scene_gl.point_count = 0
            scene_gl.mesh_face_count = 0
            scene_gl.mesh_has_uvs = False
            scene_gl.mesh_tex_id = None
            scene_gl.cam_line_count = 0
            if hasattr(scene_gl, '_mesh_uvs'):
                scene_gl._mesh_uvs = None
            if hasattr(scene_gl, '_splat_renderer') and scene_gl._splat_renderer is not None:
                scene_gl._splat_renderer.num_splats = 0
            if hasattr(scene_gl, '_splat_means'):
                del scene_gl._splat_means
            if hasattr(scene_gl, '_splat_data_packed'):
                del scene_gl._splat_data_packed
            # Reset camera
            camera.distance = 3.0
            camera.target = np.zeros(3, dtype=np.float32)
            camera.yaw = camera.pitch = 0.0
            n_cleaned = cleanup_temp_dirs()
            state.status = f"New project (cleaned {n_cleaned} temp folders)"

        imgui.same_line()
        if imgui.button("Clean Temp"):
            n_cleaned = cleanup_temp_dirs()
            state.status = f"Cleaned {n_cleaned} temp folders"

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
                        'cached_cameras': state.cached_cameras,
                        'dense_colors': getattr(state, '_dense_colors', None),
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
                    state.cached_cameras = save_data.get('cached_cameras')
                    state._dense_colors = save_data.get('dense_colors')
                    rot = save_data.get('scene_rot', (0, 0, 0))
                    state.scene_rot_x, state.scene_rot_y, state.scene_rot_z = rot

                    # Restore point cloud display
                    if state.pts3d_list is not None:
                        dense_cols = state._dense_colors
                        n_imgs = len(state.image_paths)
                        all_pts, all_cols = [], []
                        for i in range(len(state.pts3d_list)):
                            p = state.pts3d_list[i]
                            c = state.confs_list[i] if state.confs_list and i < len(state.confs_list) else None
                            img = None
                            # Try to load image for colors
                            if i < n_imgs:
                                try:
                                    from PIL import Image as PILImage
                                    img_pil = PILImage.open(state.image_paths[i]).convert('RGB')
                                    img = np.array(img_pil).astype(np.float32) / 255.0
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
                                flat = p.reshape(-1, 3)
                                mask = c.ravel() > state.min_conf if c is not None else np.ones(len(flat), dtype=bool)
                                all_pts.append(flat[mask])
                                # Use saved dense colors if available
                                if dense_cols is not None and i >= n_imgs and len(dense_cols) == len(flat):
                                    all_cols.append(dense_cols[mask])
                                else:
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
                folder_files = sorted([
                    os.path.join(folder, f) for f in os.listdir(folder)
                    if os.path.splitext(f)[1].lower() in exts
                ])
                # Accumulate: add new files to existing selection (deduplicated)
                existing = set(state.image_paths)
                new_files = [f for f in folder_files if f not in existing]
                state.image_paths = sorted(state.image_paths + new_files)
                state.image_dir = folder
                if new_files:
                    state.status = f"Added {len(new_files)} images from {folder} ({len(state.image_paths)} total)"
                else:
                    state.status = f"No new images from {folder}"
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
                # Accumulate: add new files to existing selection (deduplicated)
                existing = set(state.image_paths)
                new_files = [f for f in files if f not in existing]
                state.image_paths = sorted(state.image_paths + list(new_files))
                state.image_dir = os.path.dirname(files[0])
                if new_files:
                    state.status = f"Added {len(new_files)} images ({len(state.image_paths)} total)"
                else:
                    state.status = f"No new images (already have {len(state.image_paths)})"

        if imgui.button("Import Video...") and not state.video_extracting:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            video_path = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.flv *.wmv"),
                           ("All files", "*.*")])
            root.destroy()
            if video_path:
                state.status = f"Extracting frames from {os.path.basename(video_path)}..."
                def _do_extract(vp=video_path):
                    frames = _extract_video_frames(vp, state)
                    if frames:
                        existing = set(state.image_paths)
                        new_files = [f for f in frames if f not in existing]
                        state.image_paths = sorted(state.image_paths + new_files)
                        state.image_dir = os.path.dirname(frames[0])
                        state.status = f"Extracted {len(new_files)} frames from {os.path.basename(vp)} ({len(state.image_paths)} total)"
                    else:
                        state.status = "No frames extracted"
                threading.Thread(target=_do_extract, daemon=True).start()

        expanded, _ = imgui.collapsing_header("Video Settings")
        if expanded:
            mode_labels = ["Every Nth Frame", "Target FPS"]
            current_mode = 1 if state.video_target_fps > 0 else 0
            _, new_mode = imgui.combo("Extract Mode", current_mode, mode_labels)
            if new_mode == 0 and state.video_target_fps > 0:
                state.video_target_fps = 0.0
            elif new_mode == 1 and state.video_target_fps <= 0:
                state.video_target_fps = 2.0
            if state.video_target_fps > 0:
                _, state.video_target_fps = imgui.slider_float(
                    "Target FPS", state.video_target_fps, 0.5, 30.0, "%.1f")
            else:
                _, state.video_frame_interval = imgui.input_int(
                    "Frame Interval (N)", state.video_frame_interval, 1, 10)
                state.video_frame_interval = max(1, state.video_frame_interval)
            _, state.video_max_frames = imgui.input_int("Max Frames", state.video_max_frames, 10, 50)
            state.video_max_frames = max(1, state.video_max_frames)
            _, state.video_max_size = imgui.input_int("Max Size (px)", state.video_max_size, 64, 256)
            state.video_max_size = max(128, state.video_max_size)

        if state.image_paths:
            imgui.text(f"  {len(state.image_paths)} images")
            imgui.text(f"  {state.image_dir}")

        if imgui.button("Import COLMAP Dataset..."):
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            folder = filedialog.askdirectory(title="Select COLMAP Dataset Folder (contains sparse/ and images/)")
            root.destroy()
            if folder:
                try:
                    from train import load_colmap_dataset, parse_colmap_cameras, parse_colmap_images, parse_colmap_points3d
                    from canonical_scene import from_w2c
                    from PIL import Image as PILImage

                    sparse_dir = os.path.join(folder, 'sparse', '0')
                    images_dir = os.path.join(folder, 'images')

                    if not os.path.isdir(sparse_dir):
                        state.status = f"Not a COLMAP dataset: missing {sparse_dir}"
                    else:
                        cameras = parse_colmap_cameras(os.path.join(sparse_dir, 'cameras.txt'))
                        col_images = parse_colmap_images(os.path.join(sparse_dir, 'images.txt'))

                        # Prefer dense PLY if available, fall back to points3D.txt
                        dense_ply = os.path.join(folder, 'dense_point_cloud.ply')
                        if os.path.exists(dense_ply):
                            import trimesh
                            mesh = trimesh.load(dense_ply)
                            points = np.array(mesh.vertices, dtype=np.float32)
                            colors = np.array(mesh.colors[:, :3], dtype=np.uint8) if hasattr(mesh, 'colors') and mesh.colors is not None else np.full((len(points), 3), 128, dtype=np.uint8)
                            print(f"  Loaded dense PLY: {len(points):,d} points")
                        else:
                            points, colors = parse_colmap_points3d(os.path.join(sparse_dir, 'points3D.txt'))
                            print(f"  Loaded points3D: {len(points):,d} points")

                        # Load images
                        imgs_np = []
                        img_paths = []
                        for ci in col_images:
                            img_path = os.path.join(images_dir, ci['name'])
                            if os.path.exists(img_path):
                                img_paths.append(img_path)
                                pil = PILImage.open(img_path).convert('RGB')
                                imgs_np.append(np.array(pil).astype(np.float32) / 255.0)

                        if not imgs_np:
                            state.status = "No images found in COLMAP dataset"
                        else:
                            n = len(imgs_np)
                            # Build camera arrays — undo COLMAP's +0.5 principal point shift
                            extrinsic = np.zeros((n, 3, 4), dtype=np.float32)
                            intrinsic_all = []
                            colmap_cams = []
                            for i, ci in enumerate(col_images[:n]):
                                cam = cameras[ci['cam_id']]
                                W_c, H_c, fx, fy, cx, cy = cam
                                # COLMAP convention: cx,cy are +0.5 offset from OpenCV
                                cx -= 0.5; cy -= 0.5
                                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                                intrinsic_all.append(K)
                                w2c = ci['w2c'].astype(np.float32)
                                extrinsic[i] = w2c[:3, :]
                                c2w = ci['c2w'].astype(np.float32)
                                colmap_cams.append((c2w, K, W_c, H_c))

                            orig_sizes = [(img.shape[1], img.shape[0]) for img in imgs_np]
                            state.scene = from_w2c(
                                imgs_np, extrinsic, np.stack(intrinsic_all),
                                [points], [np.ones(len(points), dtype=np.float32) * 10],
                                orig_sizes, backend='colmap', internal_resolution=0)
                            state.image_paths = img_paths
                            state.image_dir = images_dir
                            state.pts3d_list = [points]
                            state.confs_list = [np.ones(len(points), dtype=np.float32) * 10]
                            state._dense_colors = colors
                            state.cached_cameras = colmap_cams
                            state.has_points = True
                            state._imported_colmap_dir = folder  # skip re-export for training

                            # Display
                            disp_pts, disp_cols = points, colors
                            if len(disp_pts) > 500000:
                                idx = np.random.choice(len(disp_pts), 500000, replace=False)
                                disp_pts, disp_cols = disp_pts[idx], disp_cols[idx]
                            scene_gl.set_points(disp_pts, disp_cols)

                            cam_poses = [c2w for c2w, K, W, H in colmap_cams]
                            ext = np.linalg.norm(points.max(0) - points.min(0))
                            scene_gl.set_cameras(cam_poses, scale=float(ext) * 0.05)

                            state.status = f"Imported COLMAP: {n} cameras, {len(points):,d} points"
                except Exception as e:
                    state.status = f"Import failed: {e}"
                    import traceback; traceback.print_exc()

        imgui.separator()

        # ── Backend ──
        imgui.text("Reconstruction")
        _, state.backend_idx = imgui.combo("Backend",
            state.backend_idx, ["DUSt3R", "MASt3R", "VGGT", "COLMAP", "Pow3R", "LingBot-Map"])

        if state.backends[state.backend_idx] == 'dust3r':
            _, state.niter1 = imgui.input_int("Iterations##d3r", state.niter1, 50, 100)

        if state.backends[state.backend_idx] == 'mast3r':
            _, state.optim_level = imgui.combo("Optimization##opt",
                state.optim_level, ["Coarse", "Refine", "Refine + Depth"])
            _, state.niter1 = imgui.input_int("Coarse Iters", state.niter1, 50, 100)
            _, state.niter2 = imgui.input_int("Refine Iters", state.niter2, 50, 100)

        if state.backends[state.backend_idx] == 'pow3r':
            _, state.niter1 = imgui.input_int("Iterations##pow3r", state.niter1, 50, 100)

        if state.backends[state.backend_idx] == 'vggt':
            _, state.vggt_ensemble = imgui.checkbox("Ensemble (bundles of 20 w/ overlap)", state.vggt_ensemble)
            _, state.vggt_equirect = imgui.checkbox("Equirectangular panorama (single 360° image)", state.vggt_equirect)

        if state.backends[state.backend_idx] == 'lingbot':
            _, state.lingbot_scale_frames = imgui.slider_int(
                "Scale Frames", state.lingbot_scale_frames, 2, 16)
            _, state.lingbot_keyframe_interval = imgui.slider_int(
                "Keyframe Interval", state.lingbot_keyframe_interval, 1, 10)
            changed_kv, state.lingbot_kv_window = imgui.slider_int(
                "KV Cache Window", state.lingbot_kv_window, 8, 64)
            if changed_kv:
                # Update KV window in-place if model is loaded (no reload needed)
                import app as app_mod
                lingbot_model = app_mod.MODELS.get('lingbot', None)
                if lingbot_model is not None:
                    app_mod._update_lingbot_kv_window(lingbot_model, state.lingbot_kv_window)
            imgui.text_colored("Streaming SLAM — handles long videos", 0.5, 0.8, 0.5, 1.0)

        changed_conf, state.min_conf = imgui.slider_float("Min Confidence", state.min_conf, 0.1, 20.0)
        _, state.mask_sky = imgui.checkbox("Mask Sky", state.mask_sky)
        _, state.mask_prompt = imgui.input_text("Isolate Subject", state.mask_prompt, 256)
        if state.mask_prompt.strip() or state.mask_sky:
            if state.mask_prompt.strip():
                imgui.same_line()
                _, state.mask_prompt_mode = imgui.combo("##mask_mode", state.mask_prompt_mode, ["Keep", "Remove"])
            _, state.mask_before_recon = imgui.checkbox("Mask before reconstruction", state.mask_before_recon)

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

        # ── Bundle Adjustment ──
        if state.scene is not None and state.has_points and not state.reconstructing and not state.refining:
            backend = state.backends[state.backend_idx]
            if backend in ('dust3r', 'mast3r', 'vggt', 'pow3r'):
                _, state.ba_refine_focal = imgui.checkbox("Refine Focal##ba", state.ba_refine_focal)
                imgui.same_line()
                _, state.ba_refine_pp = imgui.checkbox("Refine PP##ba", state.ba_refine_pp)
                _, state.ba_n_iters = imgui.slider_int("Iterations##ba", state.ba_n_iters, 50, 500)
                _, state.ba_min_conf = imgui.slider_float("Min Conf##ba", state.ba_min_conf, 0.5, 10.0, "%.1f")
                _, state.ba_max_shift = imgui.slider_float("Max Shift##ba", state.ba_max_shift, 0.05, 0.5, "%.2f")
                _, state.ba_huber_scale = imgui.slider_float("Huber Scale##ba", state.ba_huber_scale, 0.01, 1.0, "%.2f")
                if imgui.button("Bundle Adjust Cameras", width=-1):
                    state.refining = True
                    state.refine_thread = threading.Thread(
                        target=run_bundle_adjust, args=(state, scene_gl), daemon=True)
                    state.refine_thread.start()

        imgui.separator()

        # ── Display Options ──
        imgui.text("Display")
        _, state.draw_mode = imgui.combo("Mode##draw",
            state.draw_mode, ["Points", "Mesh", "Wireframe", "Normals", "Shaded", "Splats"])
        _, state.show_cameras = imgui.checkbox("Show Cameras", state.show_cameras)

        # Scene orientation
        imgui.text("Orientation")
        # Anchor indicators (clickable to remove)
        if state.align_floor_normal is not None:
            imgui.same_line()
            if imgui.small_button("F##rm_floor"):
                state.align_floor_normal = None
                state.align_floor_anchor = None
                _update_align_widgets(state, scene_gl)
        if state.align_wall_normal is not None:
            imgui.same_line()
            if imgui.small_button("W##rm_wall"):
                state.align_wall_normal = None
                state.align_wall_anchor = None
                _update_align_widgets(state, scene_gl)

        active_tool = state.align_mode or state.align_line_mode
        if active_tool:
            if state.align_line_mode:
                if state.align_line_start is not None:
                    imgui.text_colored("Click end point of line", 1.0, 1.0, 0.3)
                else:
                    label = "horizontal" if state.align_line_mode == 'hline' else "vertical"
                    imgui.text_colored(f"Click start of {label} line", 1.0, 1.0, 0.3)
            else:
                imgui.text_colored(f"Click on the {state.align_mode} to align", 1.0, 1.0, 0.3)
            if imgui.button("Cancel##align"):
                state.align_mode = None
                state.align_line_mode = None
                state.align_line_start = None
                state.align_line_start_screen = None
                if state.align_preview_rot:
                    state.scene_rot_x, state.scene_rot_y, state.scene_rot_z = state.align_preview_rot
                    state.align_preview_rot = None
        else:
            if imgui.button("Floor"):
                state.align_mode = 'floor'
            imgui.same_line()
            if imgui.button("Wall"):
                state.align_mode = 'wall'
            imgui.same_line()
            if imgui.button("H-Line"):
                state.align_line_mode = 'hline'
            imgui.same_line()
            if imgui.button("V-Line"):
                state.align_line_mode = 'vline'
            has_anchors = state.align_floor_normal is not None or state.align_wall_normal is not None
            if has_anchors:
                if imgui.button("Align", width=-1):
                    rot = _compute_align_rotation(state.align_floor_normal, state.align_wall_normal)
                    if rot:
                        state.scene_rot_x, state.scene_rot_y, state.scene_rot_z = rot
                        parts = []
                        if state.align_floor_normal is not None:
                            parts.append("floor")
                        if state.align_wall_normal is not None:
                            parts.append("wall")
                        state.status = f"Aligned: {' + '.join(parts)}"
            if imgui.button("Flip Up"):
                state.scene_rot_x = (state.scene_rot_x + 180) % 360 - 180
            imgui.same_line()
            if imgui.button("Reset"):
                state.scene_rot_x = state.scene_rot_y = state.scene_rot_z = 0.0
                state.align_floor_normal = None
                state.align_wall_normal = None
                state.align_floor_anchor = None
                state.align_wall_anchor = None
                _update_align_widgets(state, scene_gl)
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
                            output_dir=folder, min_conf_thr=state.min_conf,
                            dense_colors=getattr(state, '_dense_colors', None))
                        state.status = f"Exported to {folder}"
                    except Exception as e:
                        state.error_msg = str(e)

            if imgui.button("Export Cameras (.dae)", width=-1):
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                fpath = filedialog.asksaveasfilename(
                    title="Export Cameras as COLLADA",
                    defaultextension=".dae",
                    filetypes=[("COLLADA", "*.dae")])
                root.destroy()
                if fpath:
                    try:
                        from colmap_export import export_cameras_to_collada
                        export_cameras_to_collada(state.scene, fpath, state.image_paths)
                        state.status = f"Cameras exported to {fpath}"
                    except Exception as e:
                        state.error_msg = str(e)
            imgui.same_line()
            imgui.text_disabled("(XSI / Blender / Maya)")

        if state.has_mesh and (state.scene is not None or state.has_points):
            if imgui.button("Export Mesh (.obj, Y-up)", width=-1):
                state._deferred_action = 'export_mesh_obj_yup'
            imgui.same_line()
            imgui.text_disabled("(aligned with .dae cameras)")

        imgui.separator()

        imgui.separator()

        # ── Densify ──
        if state.has_points and state.scene is not None and not state.refining:
            if imgui.tree_node("PatchMatch Options"):
                _, state.pm_max_image_size = imgui.input_int("Max Image Size", state.pm_max_image_size, 100, 500)
                _, state.pm_num_iterations = imgui.input_int("Iterations", state.pm_num_iterations, 1, 1)
                _, state.pm_window_radius = imgui.input_int("Window Radius", state.pm_window_radius, 1, 1)
                _, state.pm_min_consistent = imgui.input_int("Min Consistent Views", state.pm_min_consistent, 1, 1)
                _, state.pm_geom_consistency = imgui.checkbox("Geometric Consistency", state.pm_geom_consistency)
                _, state.pm_filter_min_ncc = imgui.input_float("Min NCC", state.pm_filter_min_ncc, 0.01, 0.05)
                imgui.text_colored("(-1 = full resolution)", 0.5, 0.5, 0.5)
                imgui.tree_pop()
            if imgui.button("Densify (COLMAP PatchMatch)", width=-1):
                state.refining = True
                state.refine_thread = threading.Thread(
                    target=run_densify_colmap, args=(state, scene_gl), daemon=True)
                state.refine_thread.start()

        imgui.separator()

        # ── Dense Mesh ──
        imgui.text("Dense Mesh")
        _, state.mesh_mode_idx = imgui.combo("Method", state.mesh_mode_idx, state.mesh_mode_labels)
        if state.mesh_modes[state.mesh_mode_idx] == 'poisson':
            _, state.poisson_depth_val = imgui.input_int("Octree Depth", state.poisson_depth_val, 1, 1)
            _, state.poisson_trim = imgui.input_float("Trim %", state.poisson_trim, 1.0, 5.0)
        if state.mesh_modes[state.mesh_mode_idx] == 'ballpivot':
            _, state.bp_radius_mult = imgui.input_float("Radius Mult", state.bp_radius_mult, 0.5, 1.0)
            imgui.same_line()
            imgui.text_colored("(1=tight, 8=fill gaps)", 0.5, 0.5, 0.5)
        if state.mesh_modes[state.mesh_mode_idx] == 'delaunay':
            _, state.delaunay_edge_mult = imgui.slider_float("Radius##del", state.delaunay_edge_mult, 2.0, 30.0, format="%.1fx")
            imgui.same_line()
            imgui.text_colored("(higher=fill gaps)", 0.5, 0.5, 0.5)
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
        if state.has_points and not state.refining:
            if imgui.button("Decimate Points (merge neighbors)", width=-1):
                state.refining = True
                state.refine_thread = threading.Thread(
                    target=_run_decimate_points, args=(state, scene_gl), daemon=True)
                state.refine_thread.start()

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
        if state.has_mesh and not state.refining:
            if imgui.button("Create UVs", width=-1):
                state.refining = True
                state.refine_thread = threading.Thread(
                    target=run_create_uvs, args=(state, scene_gl), daemon=True)
                state.refine_thread.start()

        if (state.has_mesh and state.scene is not None and not state.refining
                and getattr(state, 'uv_data', None) is not None):
            if imgui.button("Bake Texture", width=-1):
                state.refining = True
                state.refine_thread = threading.Thread(
                    target=run_texture, args=(state, scene_gl), daemon=True)
                state.refine_thread.start()

        # Export textured OBJ (only after baking)
        if (state.has_mesh and not state.refining
                and getattr(state, 'uv_data', None) is not None
                and getattr(state, '_baked_texture', None) is not None):
            if imgui.button("Export Textured Mesh (.obj)", width=-1):
                state._deferred_action = 'export_textured_obj'

        imgui.separator()

        # ── Gaussian Splats ──
        if (state.has_mesh or state.has_points) and state.scene is not None:
            imgui.text("Gaussian Splats")
            if not state.splat_training and not state.refining:
                _, state.splat_multi_view = imgui.checkbox(
                    "Multi-View##splat", state.splat_multi_view)
                if state.splat_multi_view:
                    imgui.same_line()
                    _, state.splat_multi_view_count = imgui.input_int(
                        "##mv_count", state.splat_multi_view_count, 1, 1)
                    state.splat_multi_view_count = max(2, state.splat_multi_view_count)
                _, state.splat_strategy_idx = imgui.combo(
                    "Strategy##splat", state.splat_strategy_idx,
                    state.splat_strategy_labels)
                _, state.splat_iterations = imgui.input_int(
                    "Iterations##splat", state.splat_iterations, 500, 1000)
                _, state.splat_resolution = imgui.input_int(
                    "Resolution##splat", state.splat_resolution, 256, 512)
                _, state.splat_n_samples = imgui.input_int(
                    "Samples##splat", state.splat_n_samples, 10000, 50000)
                _, state.splat_target = imgui.input_int(
                    "Target Splats", state.splat_target, 50000, 100000)
                if state.splat_target == 0:
                    imgui.same_line()
                    imgui.text_colored("(no densify)", 0.5, 0.5, 0.5)
                _, state.splat_anchor = imgui.slider_float(
                    "Anchor##splat", state.splat_anchor, 0.0, 1.0, "%.2f")
                _, state.splat_flatness = imgui.slider_float(
                    "Flatness##splat", state.splat_flatness, 0.0, 1.0, "%.2f")
                _, state.splat_normal = imgui.slider_float(
                    "Normal Align##splat", state.splat_normal, 0.0, 1.0, "%.2f")
                _, state.splat_aniso = imgui.slider_float(
                    "Anisotropy##splat", state.splat_aniso, 0.0, 1.0, "%.2f")
                _, state.splat_coverage = imgui.slider_float(
                    "Coverage##splat", state.splat_coverage, 0.0, 1.0, "%.2f")
                _, state.splat_depth = imgui.slider_float(
                    "Depth##splat", state.splat_depth, 0.0, 1.0, "%.2f")
                _, state.splat_opacity_decay = imgui.slider_float(
                    "Opacity Decay##splat", state.splat_opacity_decay, 0.0, 1.0, "%.2f")
                _, state.splat_prune = imgui.slider_float(
                    "Prune##splat", state.splat_prune, 0.0, 1.0, "%.2f")
                _, state.splat_smooth = imgui.slider_float(
                    "Smooth##splat", state.splat_smooth, 0.0, 1.0, "%.2f")
                if imgui.button("Train Splats", width=-1):
                    state.splat_training = True
                    state.stop_requested = False
                    threading.Thread(
                        target=run_train_splats, args=(state, scene_gl),
                        daemon=True).start()

            if state.splat_training:
                if state.splat_total > 0:
                    frac = state.splat_step / state.splat_total
                    imgui.progress_bar(frac, (-1, 0), state.splat_progress)
                else:
                    imgui.text(state.splat_progress)
                if imgui.button("Stop##splat", width=-1):
                    state.stop_requested = True

            if state.splat_data is not None and not state.splat_training:
                if imgui.button("Export Splats (.ply)", width=-1):
                    state._deferred_action = 'export_splats'

        imgui.separator()

        # ── Save ──
        if state.has_mesh:
            if imgui.button("Save Mesh (.ply)", width=-1):
                state._deferred_action = 'save_mesh_ply'

        imgui.end_child()
        imgui.end()

        # ── Deferred file dialogs (run after imgui frame ends) ──
        action = getattr(state, '_deferred_action', None)
        if action:
            state._deferred_action = None
            imgui.render()
            impl.render(imgui.get_draw_data())
            glfw.swap_buffers(window)

            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()

            if action == 'export_splats':
                path = filedialog.asksaveasfilename(
                    title="Export Splats", defaultextension=".ply",
                    filetypes=[("PLY files", "*.ply")])
                root.destroy()
                if path and state.splat_data is not None:
                    import importlib.util
                    _spec = importlib.util.spec_from_file_location(
                        "_train_export", os.path.join(os.path.dirname(__file__), "train.py"))
                    _mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
                    _mod.save_ply(state.splat_data, path)
                    state.status = f"Splats exported to {path}"

            elif action == 'export_textured_obj':
                path = filedialog.asksaveasfilename(
                    title="Export Textured Mesh", defaultextension=".obj",
                    filetypes=[("OBJ files", "*.obj")])
                root.destroy()
                if path:
                    _export_textured_obj(state, path)

            elif action == 'export_mesh_obj_yup':
                path = filedialog.asksaveasfilename(
                    title="Export Mesh (Y-up, for XSI)",
                    defaultextension=".obj",
                    filetypes=[("OBJ files", "*.obj")])
                root.destroy()
                if path and state.mesh_data is not None:
                    from colmap_export import export_mesh_obj_yup
                    v, f, c = state.mesh_data
                    export_mesh_obj_yup(v, f, c, path)
                    state.status = f"Mesh exported to {path}"

            elif action == 'save_mesh_ply':
                path = filedialog.asksaveasfilename(
                    title="Save Mesh", defaultextension=".ply",
                    filetypes=[("PLY files", "*.ply")])
                root.destroy()
                if path and state.mesh_data is not None:
                    from refine_mesh import save_ply_mesh
                    v, f, c = state.mesh_data
                    save_ply_mesh(path, v, f, c)
                    state.status = f"Saved to {path}"

            continue  # skip rest of this frame since we already swapped

        # ── 3D Viewport ──
        win_w, win_h = glfw.get_window_size(window)
        if win_w <= 0 or win_h <= 0:
            imgui.render()
            glfw.swap_buffers(window)
            continue
        vp_x = 400
        vp_w = win_w - vp_x
        vp_h = win_h

        # Throttle when not focused but still render (so background tasks show results)
        is_focused = glfw.get_window_attrib(window, glfw.FOCUSED)
        if not is_focused:
            import time; time.sleep(0.05)

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
                        pts = np.concatenate([p.reshape(-1, 3) for p in state.scene.pts3d], axis=0)
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
            # Scene orientation transform
            def _rot_matrix(rx, ry, rz):
                from scipy.spatial.transform import Rotation as _R
                r = _R.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix()
                m = np.eye(4, dtype=np.float32)
                m[:3, :3] = r
                return m

            scene_tf = _rot_matrix(state.scene_rot_x, state.scene_rot_y, state.scene_rot_z)

            # Camera POV mode: use full c2w matrix (preserves roll)
            if (state.cam_view_idx >= 0 and state.cached_cameras
                    and state.cam_view_idx < len(state.cached_cameras)):
                c2w_raw = state.cached_cameras[state.cam_view_idx][0]
                K = state.cached_cameras[state.cam_view_idx][1]
                H = state.cached_cameras[state.cam_view_idx][3]
                # Convert DUSt3R convention (Y-down, Z-forward) to OpenGL (Y-up, Z-backward)
                gl_from_cv = np.diag([1, -1, -1, 1]).astype(np.float32)
                c2w_world = scene_tf @ c2w_raw.astype(np.float32)
                view = (gl_from_cv @ np.linalg.inv(c2w_world)).astype(np.float32)
                cam_fov = float(np.degrees(2 * np.arctan(H / (2 * K[1, 1]))))
                cam_pos = c2w_world[:3, 3]
                # Build projection from camera intrinsics FOV
                near, far = 0.01, 10000.0
                f = 1.0 / math.tan(math.radians(cam_fov) / 2.0)
                proj = np.zeros((4, 4), dtype=np.float32)
                proj[0, 0] = f / aspect
                proj[1, 1] = f
                proj[2, 2] = (far + near) / (near - far)
                proj[2, 3] = 2 * far * near / (near - far)
                proj[3, 2] = -1.0
                fov_y = math.radians(cam_fov)
            else:
                view = camera.get_view_matrix()
                proj = camera.get_projection_matrix(aspect)
                cam_pos = camera.get_position()
                fov_y = math.radians(camera.fov)

            mvp_base = proj @ view            # for grid + axes (fixed)
            mvp_scene = proj @ view @ scene_tf  # for point cloud / mesh / cameras

            mode = state.draw_modes[state.draw_mode]
            scene_gl._show_cameras = state.show_cameras
            scene_gl.draw(mvp_base, mvp_scene, draw_mode=mode,
                          camera_pos=cam_pos,
                          view_matrix=view @ scene_tf,
                          proj_matrix=proj,
                          fov_y=fov_y)

        gl.glDisable(gl.GL_SCISSOR_TEST)

        # ── In-app debug image viewer ──
        debug_imgs.flush()
        debug_imgs.draw_window("Debug Views")

        # ── Align cursor overlay + hover preview ──
        try:
          if state.align_mode or state.align_line_mode:
            mx_cur, my_cur = glfw.get_cursor_pos(window)
            if mx_cur > vp_x and not imgui.get_io().want_capture_mouse:
                draw_list = imgui.get_foreground_draw_list()
                if state.align_mode:
                    # Circle cursor for floor/wall click
                    circle_px = 40
                    is_floor = state.align_mode == 'floor'
                    color = imgui.get_color_u32_rgba(0.3, 1, 0.3, 0.8) if is_floor \
                            else imgui.get_color_u32_rgba(0.3, 0.7, 1, 0.8)
                    draw_list.add_circle(mx_cur, my_cur, circle_px, color, 32, 2.0)
                    draw_list.add_circle_filled(mx_cur, my_cur, 3, color, 12)

                    # Hover preview: show gizmo + highlighted points at cursor
                    result = _raycast_to_surface(state, camera, mx_cur, my_cur, window,
                                                 subsample=4, return_neighborhood=True)
                    if result is not None:
                        centroid, normal, radius, nbhood = result
                        hover_color = (0.3, 1.0, 0.3) if is_floor else (0.3, 0.7, 1.0)
                        hover_anchor = {'pos': centroid, 'normal': normal,
                                        'radius': radius, 'color': hover_color}
                        _update_align_widgets(state, scene_gl,
                                              hover_anchor=hover_anchor,
                                              hover_points=nbhood)
                        state._had_hover_widget = True
                    else:
                        # No surface under cursor — show just the placed anchors
                        _update_align_widgets(state, scene_gl)
                        state._had_hover_widget = False

                elif state.align_line_mode:
                    # Crosshair cursor for line tools
                    color = imgui.get_color_u32_rgba(0, 1, 1, 0.8) if state.align_line_mode == 'hline' \
                            else imgui.get_color_u32_rgba(1, 0, 1, 0.8)
                    size = 15
                    draw_list.add_line(mx_cur - size, my_cur, mx_cur + size, my_cur, color, 2.0)
                    draw_list.add_line(mx_cur, my_cur - size, mx_cur, my_cur + size, color, 2.0)

                    # Rubber-band line from start point
                    if state.align_line_start_screen is not None:
                        sx, sy = state.align_line_start_screen
                        draw_list.add_line(sx, sy, mx_cur, my_cur, color, 2.0)
                        draw_list.add_circle_filled(sx, sy, 4, color, 8)
          else:
            # Not in align mode — clear any leftover hover widgets
            if getattr(state, '_had_hover_widget', False):
                _update_align_widgets(state, scene_gl)
                state._had_hover_widget = False
        except Exception:
            pass  # never break the imgui frame from overlay drawing

        # ── Camera POV overlay ──
        if state.cam_view_name and state.cam_view_idx >= 0:
            try:
                draw_list = imgui.get_foreground_draw_list()
                label = f"[{state.cam_view_idx + 1}/{len(state.cached_cameras)}] {state.cam_view_name}"
                text_color = imgui.get_color_u32_rgba(1.0, 1.0, 0.3, 1.0)
                bg_color = imgui.get_color_u32_rgba(0, 0, 0, 0.6)
                tx = vp_x + 10
                ty = 10
                # Background rect
                draw_list.add_rect_filled(tx - 4, ty - 2, tx + len(label) * 8 + 4, ty + 18, bg_color, 4)
                draw_list.add_text(tx, ty, text_color, label)
            except Exception:
                pass

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
      except Exception as _frame_err:
        # Don't crash on a single bad frame — log and continue
        _err_str = str(_frame_err)
        if 'Forgot to call Render' not in _err_str:
            import traceback; traceback.print_exc()
        # Try to end the frame cleanly so next new_frame() doesn't assert
        try:
            imgui.end_frame()
        except Exception:
            pass
        try:
            glfw.swap_buffers(window)
        except Exception:
            pass

    impl.shutdown()
    glfw.terminate()


if __name__ == '__main__':
    main()
