"""
OpenGL Gaussian Splat Renderer

Based on limacv/GaussianSplattingViewer (MIT License).
Renders 3D gaussians as instanced quads with proper covariance projection
and alpha blending. Uses SSBOs (OpenGL 4.3+).
"""

import numpy as np
import OpenGL.GL as gl

# ── Shaders ──────────────────────────────────────────────────────────────────

SPLAT_VERT = """
#version 430 core

#define SH_C0 0.28209479177387814f

layout(location = 0) in vec2 position;

layout (std430, binding=0) buffer gaussian_data {
    float g_data[];
};
layout (std430, binding=1) buffer gaussian_order {
    int gi[];
};

#define POS_IDX 0
#define ROT_IDX 3
#define SCALE_IDX 7
#define OPACITY_IDX 10
#define SH_IDX 11

uniform mat4 view_matrix;
uniform mat4 projection_matrix;
uniform vec3 hfovxy_focal;
uniform vec3 cam_pos;
uniform int sh_dim;
uniform float scale_modifier;

out vec3 color;
out float alpha;
out vec3 conic;
out vec2 coordxy;
out float center_depth;

vec3 get_vec3(int offset) {
    return vec3(g_data[offset], g_data[offset + 1], g_data[offset + 2]);
}
vec4 get_vec4(int offset) {
    return vec4(g_data[offset], g_data[offset + 1], g_data[offset + 2], g_data[offset + 3]);
}

mat3 computeCov3D(vec3 scale, vec4 q) {
    mat3 S = mat3(0.f);
    S[0][0] = scale.x; S[1][1] = scale.y; S[2][2] = scale.z;
    float r = q.x, x = q.y, y = q.z, z = q.w;
    mat3 R = mat3(
        1.f - 2.f*(y*y+z*z), 2.f*(x*y-r*z), 2.f*(x*z+r*y),
        2.f*(x*y+r*z), 1.f - 2.f*(x*x+z*z), 2.f*(y*z-r*x),
        2.f*(x*z-r*y), 2.f*(y*z+r*x), 1.f - 2.f*(x*x+y*y)
    );
    mat3 M = S * R;
    return transpose(M) * M;
}

vec3 computeCov2D(vec4 mean_view, float focal_x, float focal_y,
                  float tan_fovx, float tan_fovy, mat3 cov3D, mat4 viewmatrix) {
    vec4 t = mean_view;
    float limx = 1.3f * tan_fovx;
    float limy = 1.3f * tan_fovy;
    t.x = min(limx, max(-limx, t.x / t.z)) * t.z;
    t.y = min(limy, max(-limy, t.y / t.z)) * t.z;
    mat3 J = mat3(
        focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
        0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
        0, 0, 0
    );
    mat3 W = transpose(mat3(viewmatrix));
    mat3 T = W * J;
    mat3 cov = transpose(T) * transpose(cov3D) * T;
    cov[0][0] += 0.3f;
    cov[1][1] += 0.3f;
    return vec3(cov[0][0], cov[0][1], cov[1][1]);
}

void main() {
    int boxid = gi[gl_InstanceID];
    int total_dim = 3 + 4 + 3 + 1 + sh_dim;
    int start = boxid * total_dim;
    vec4 g_pos = vec4(get_vec3(start + POS_IDX), 1.f);
    vec4 g_pos_view = view_matrix * g_pos;
    vec4 g_pos_screen = projection_matrix * g_pos_view;
    g_pos_screen.xyz = g_pos_screen.xyz / g_pos_screen.w;
    g_pos_screen.w = 1.f;

    if (any(greaterThan(abs(g_pos_screen.xyz), vec3(1.3)))) {
        gl_Position = vec4(-100, -100, -100, 1);
        return;
    }

    vec4 g_rot = get_vec4(start + ROT_IDX);
    vec3 g_scale = get_vec3(start + SCALE_IDX);
    float g_opacity = g_data[start + OPACITY_IDX];

    // Kill degenerate splats: skip if any scale axis is too large
    // or if anisotropy ratio is extreme
    float max_s = max(g_scale.x, max(g_scale.y, g_scale.z));
    float min_s = min(g_scale.x, min(g_scale.y, g_scale.z));
    if (max_s > 1.0f || max_s / (min_s + 1e-7f) > 50.f) {
        gl_Position = vec4(-100, -100, -100, 1);
        return;
    }

    mat3 cov3d = computeCov3D(g_scale * scale_modifier, g_rot);
    vec2 wh = 2 * hfovxy_focal.xy * hfovxy_focal.z;
    vec3 cov2d = computeCov2D(g_pos_view, hfovxy_focal.z, hfovxy_focal.z,
                              hfovxy_focal.x, hfovxy_focal.y, cov3d, view_matrix);

    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (det == 0.0f) { gl_Position = vec4(0,0,0,0); return; }
    float det_inv = 1.f / det;
    conic = vec3(cov2d.z * det_inv, -cov2d.y * det_inv, cov2d.x * det_inv);

    vec2 quadwh_scr = vec2(3.f * sqrt(cov2d.x), 3.f * sqrt(cov2d.z));
    // Clamp screen-space quad size to prevent giant splats covering everything
    quadwh_scr = min(quadwh_scr, wh * 0.5f);
    vec2 quadwh_ndc = quadwh_scr / wh * 2;
    g_pos_screen.xy = g_pos_screen.xy + position * quadwh_ndc;
    coordxy = position * quadwh_scr;
    gl_Position = g_pos_screen;
    center_depth = g_pos_view.z;
    alpha = g_opacity;

    // SH color (DC only for speed)
    int sh_start = start + SH_IDX;
    color = SH_C0 * get_vec3(sh_start) + 0.5f;
}
"""

SPLAT_FRAG = """
#version 430 core

in vec3 color;
in float alpha;
in vec3 conic;
in vec2 coordxy;
in float center_depth;

out vec4 FragColor;

void main() {
    float power = -0.5f * (conic.x * coordxy.x * coordxy.x +
                           conic.z * coordxy.y * coordxy.y) -
                  conic.y * coordxy.x * coordxy.y;
    if (power > 0.f) discard;
    float opacity = min(0.99f, alpha * exp(power));
    if (opacity < 1.f / 255.f) discard;

    FragColor = vec4(color, opacity);
}
"""


# ── Renderer ─────────────────────────────────────────────────────────────────

class SplatRenderer:
    """Renders gaussian splats using instanced quads + SSBOs."""

    def __init__(self):
        self.program = None
        self.vao = None
        self.gau_ssbo = None
        self.idx_ssbo = None
        self.num_splats = 0
        self.sh_dim = 3  # DC only
        self._initialized = False

    def _init_gl(self):
        """Compile shaders and set up quad geometry. Must be called from GL thread."""
        # Compile shaders
        vs = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        gl.glShaderSource(vs, SPLAT_VERT)
        gl.glCompileShader(vs)
        if not gl.glGetShaderiv(vs, gl.GL_COMPILE_STATUS):
            log = gl.glGetShaderInfoLog(vs).decode()
            raise RuntimeError(f"Vertex shader error: {log}")

        fs = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        gl.glShaderSource(fs, SPLAT_FRAG)
        gl.glCompileShader(fs)
        if not gl.glGetShaderiv(fs, gl.GL_COMPILE_STATUS):
            log = gl.glGetShaderInfoLog(fs).decode()
            raise RuntimeError(f"Fragment shader error: {log}")

        self.program = gl.glCreateProgram()
        gl.glAttachShader(self.program, vs)
        gl.glAttachShader(self.program, fs)
        gl.glLinkProgram(self.program)
        gl.glDeleteShader(vs)
        gl.glDeleteShader(fs)

        # Quad geometry: 4 vertices, 2 triangles
        quad_v = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]], dtype=np.float32)
        quad_f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)

        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

        vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, quad_v.nbytes, quad_v, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, False, 0, None)

        ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, quad_f.nbytes, quad_f, gl.GL_STATIC_DRAW)

        gl.glBindVertexArray(0)

        # SSBOs
        self.gau_ssbo = gl.glGenBuffers(1)
        self.idx_ssbo = gl.glGenBuffers(1)

        self._initialized = True

    def update_splats(self, splat_data, sort_indices):
        """Upload packed splat data and sort order to GPU.

        splat_data: (N * stride) float32 flat array
            stride = 3 (pos) + 4 (quat) + 3 (scale) + 1 (opacity) + sh_dim (SH)
        sort_indices: (N,) int32 array of back-to-front order
        """
        if not self._initialized:
            self._init_gl()

        self.num_splats = len(sort_indices)

        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, self.gau_ssbo)
        gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, splat_data.nbytes,
                        splat_data.ravel(), gl.GL_DYNAMIC_DRAW)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 0, self.gau_ssbo)

        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, self.idx_ssbo)
        gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, sort_indices.nbytes,
                        sort_indices.ravel(), gl.GL_DYNAMIC_DRAW)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 1, self.idx_ssbo)

        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)

    def draw(self, view_matrix, proj_matrix, cam_pos, fov_x, fov_y, focal):
        """Render splats. Call after glViewport/glClear."""
        if not self._initialized or self.num_splats == 0:
            return

        gl.glUseProgram(self.program)

        # Uniforms
        gl.glUniformMatrix4fv(
            gl.glGetUniformLocation(self.program, "view_matrix"),
            1, gl.GL_TRUE, view_matrix.astype(np.float32))
        gl.glUniformMatrix4fv(
            gl.glGetUniformLocation(self.program, "projection_matrix"),
            1, gl.GL_TRUE, proj_matrix.astype(np.float32))
        gl.glUniform3f(
            gl.glGetUniformLocation(self.program, "hfovxy_focal"),
            float(np.tan(fov_x / 2)), float(np.tan(fov_y / 2)), float(focal))
        gl.glUniform3f(
            gl.glGetUniformLocation(self.program, "cam_pos"),
            float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]))
        gl.glUniform1i(
            gl.glGetUniformLocation(self.program, "sh_dim"), self.sh_dim)
        gl.glUniform1f(
            gl.glGetUniformLocation(self.program, "scale_modifier"), 1.0)

        # Blending
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glDisable(gl.GL_DEPTH_TEST)  # splats handle depth via sort order

        gl.glBindVertexArray(self.vao)
        gl.glDrawElementsInstanced(
            gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT, None, self.num_splats)

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_BLEND)


def pack_splat_data(means, quats, scales_log, opacities_logit, sh0):
    """Pack splat parameters into flat SSBO format.

    Args:
        means: (N, 3) float32 - positions
        quats: (N, 4) float32 - quaternions (w,x,y,z), will be normalized
        scales_log: (N, 3) float32 - log-space scales (will be exp'd)
        opacities_logit: (N,) float32 - logit-space opacities (will be sigmoid'd)
        sh0: (N, 1, 3) float32 - DC SH coefficients

    Returns:
        data: flat float32 array
        sh_dim: int (3 for DC only)
    """
    N = len(means)
    scales = np.exp(scales_log)
    opacities = 1.0 / (1.0 + np.exp(-opacities_logit))

    # Normalize quaternions
    q_norm = np.linalg.norm(quats, axis=-1, keepdims=True)
    quats = quats / (q_norm + 1e-8)

    sh_dim = 3  # DC only
    # Pack: pos(3) + quat(4) + scale(3) + opacity(1) + sh(3) = 14 per splat
    stride = 3 + 4 + 3 + 1 + sh_dim
    data = np.zeros(N * stride, dtype=np.float32)
    for i in range(N):
        off = i * stride
        data[off:off+3] = means[i]
        data[off+3:off+7] = quats[i]
        data[off+7:off+10] = scales[i]
        data[off+10] = opacities[i]
        data[off+11:off+14] = sh0[i, 0, :]
    return data, sh_dim


def pack_splat_data_fast(means, quats, scales_log, opacities_logit, sh0):
    """Vectorized version of pack_splat_data."""
    N = len(means)
    scales = np.exp(scales_log)
    opacities = 1.0 / (1.0 + np.exp(-opacities_logit))
    q_norm = np.linalg.norm(quats, axis=-1, keepdims=True)
    quats = quats / (q_norm + 1e-8)

    sh_dim = 3
    stride = 14
    data = np.zeros((N, stride), dtype=np.float32)
    data[:, 0:3] = means
    data[:, 3:7] = quats
    data[:, 7:10] = scales
    data[:, 10] = opacities.ravel()
    data[:, 11:14] = sh0[:, 0, :]
    return data.ravel(), sh_dim


def sort_splats_by_depth(means, view_matrix):
    """Sort splat indices back-to-front for alpha blending.
    Returns int32 index array."""
    R = view_matrix[:3, :3]
    t = view_matrix[:3, 3]
    depths = (means @ R.T + t[None, :])[:, 2]
    return np.argsort(depths).astype(np.int32)
