"""
Monocular Normal Estimation — Camera-Space Normals

Uses DSINE (lightweight ViT, ~100MB) for fast camera-space normal prediction.
No diffusion models, no multi-GB downloads.

Output convention: R=X(right), G=Y(down), B=Z(into screen) — OpenCV camera space.
"""

import os
import sys
import numpy as np
from PIL import Image

_predictor = None
_predictor_type = None


def load_normal_model(device='cuda'):
    """Load DSINE normal estimation model."""
    global _predictor, _predictor_type
    if _predictor is not None:
        return _predictor, _predictor_type

    import torch

    try:
        print("  Loading DSINE normal estimator...")
        predictor = torch.hub.load("hugoycj/DSINE-hub", "DSINE", trust_repo=True)
        _predictor = predictor
        _predictor_type = 'dsine'
        print("  DSINE loaded")
        return _predictor, _predictor_type
    except Exception as e:
        print(f"  DSINE not available: {e}")

    print("  Using gradient-based fallback")
    _predictor = "gradient"
    _predictor_type = "gradient"
    return _predictor, _predictor_type


def predict_normals(image, intrinsics=None, device='cuda'):
    """
    Predict surface normals from an RGB image.

    Returns:
        normals: (H, W, 3) float numpy in camera space
                 R=X(right), G=Y(down), B=Z(into screen)
    """
    model, model_type = load_normal_model(device)

    if model_type == 'dsine':
        return _predict_dsine(model, image)
    else:
        return _predict_gradient(image)


def _predict_dsine(predictor, image):
    """Predict using DSINE — returns camera-space normals."""
    import torch

    H, W = image.shape[:2]
    img_pil = Image.fromarray((image * 255).clip(0, 255).astype(np.uint8))

    with torch.no_grad():
        result = predictor.infer_pil(img_pil)  # (1, 3, H, W) tensor, range [-1, 1]

    normals = result.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    # DSINE convention: X=right, Y=down, Z=toward camera
    # Keep as-is — we'll match the mesh renderer to this convention
    n_len = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8
    return (normals / n_len).astype(np.float32)


def _predict_gradient(image):
    """Fallback gradient-based estimation."""
    from scipy.ndimage import sobel, gaussian_filter

    H, W = image.shape[:2]
    gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]

    nx = np.zeros((H, W), dtype=np.float64)
    ny = np.zeros((H, W), dtype=np.float64)
    for sigma in [0.5, 1.0, 2.0]:
        g = gaussian_filter(gray, sigma=sigma)
        nx += sobel(g, axis=1) / (sigma + 0.5)
        ny += sobel(g, axis=0) / (sigma + 0.5)

    normal_x = -nx * 1.5
    normal_y = -ny * 1.5
    normal_z = np.ones_like(nx)
    length = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2) + 1e-8
    return np.stack([normal_x / length, normal_y / length, normal_z / length], axis=-1).astype(np.float32)


def render_mesh_normals_gl(verts, faces, w2c, K, W, H, renderer):
    """Render mesh normals in camera space via OpenGL."""
    V = len(verts)
    v0 = verts[faces[:, 0]]; v1 = verts[faces[:, 1]]; v2 = verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn /= (np.linalg.norm(fn, axis=-1, keepdims=True) + 1e-8)

    vert_normals = np.zeros((V, 3), dtype=np.float32)
    for ax in range(3):
        np.add.at(vert_normals[:, ax], faces[:, 0], fn[:, ax])
        np.add.at(vert_normals[:, ax], faces[:, 1], fn[:, ax])
        np.add.at(vert_normals[:, ax], faces[:, 2], fn[:, ax])
    vert_normals /= (np.linalg.norm(vert_normals, axis=-1, keepdims=True) + 1e-8)

    R = w2c[:3, :3]
    normals_cam = (R @ vert_normals.T).T

    # OpenCV camera space: X=right, Y=down, Z=into scene (away from camera)
    # DSINE convention: X=right, Y=up, Z=toward camera
    # Convert OpenCV -> DSINE: flip Y and flip Z
    normals_cam[:, 1] *= -1  # Y: down -> up
    normals_cam[:, 2] *= -1  # Z: into scene -> toward camera

    # Ensure camera-facing normals have positive Z
    normals_cam[normals_cam[:, 2] < 0] *= -1

    normal_colors = ((normals_cam + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)
    color_img, vert_ids = renderer.render(verts, faces, normal_colors, w2c, K, W, H)

    normal_map = color_img * 2.0 - 1.0
    n_len = np.linalg.norm(normal_map, axis=-1, keepdims=True) + 1e-8
    normal_map /= n_len
    normal_map[vert_ids < 0] = 0
    return normal_map, vert_ids


def compare_normals(predicted_normals, rendered_normals, vert_ids):
    """Compare predicted vs mesh normals. Returns (H, W) error in [0, 1]."""
    valid = vert_ids >= 0
    dot = (predicted_normals * rendered_normals).sum(axis=-1)
    error = 1.0 - np.clip(np.abs(dot), 0, 1)
    error[~valid] = 0
    return error.astype(np.float32)
