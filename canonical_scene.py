"""Canonical scene representation for all reconstruction backends.

Every backend converts its output into a CanonicalScene with consistent
conventions. All downstream code (viewport, COLMAP export, mesh, splats)
consumes only this format.

Conventions:
- c2w: always camera-to-world (4x4), OpenCV coords (x-right, y-down, z-forward)
- intrinsics: K matrices at the model's internal resolution (or original res for colmap)
- pts3d: world-space 3D points
- confidence: raw per-pixel confidence (higher = better)
- images: float [0,1] at model's internal resolution (or original res for colmap)
"""

import numpy as np
import torch


class CanonicalScene:
    """Unified scene representation for all reconstruction backends."""

    def __init__(self, images, pts3d, confidence, c2w, intrinsics, original_sizes,
                 backend="", internal_resolution=512):
        """
        Args:
            images: list of (H,W,3) float32 [0,1] numpy arrays
            pts3d: list of (H,W,3) or (N,3) float32 numpy — world-space points
            confidence: list of (H,W) or (N,) float32 numpy
            c2w: (N,4,4) float64 numpy — camera-to-world poses
            intrinsics: (N,3,3) float64 numpy — K matrices at internal resolution
            original_sizes: list of (W,H) tuples — original image sizes
            backend: str — source backend name
            internal_resolution: int — model's working resolution (512 or 518),
                                 0 means intrinsics are already at original resolution
        """
        self.images = images
        self.pts3d = pts3d
        self.confidence = confidence
        self.c2w = np.asarray(c2w, dtype=np.float64)
        self.intrinsics = np.asarray(intrinsics, dtype=np.float64)
        self.original_sizes = original_sizes
        self.backend = backend
        self.internal_resolution = internal_resolution

        # Optional equirect metadata (set by equirect backend)
        self.equirect = None

    # ── Convenience accessors ───────────────────���──────────────────────

    @property
    def imgs(self):
        """Alias for images."""
        return self.images

    @imgs.setter
    def imgs(self, value):
        self.images = value

    def get_focals(self):
        """Return focal lengths as a tensor."""
        return torch.tensor([self.intrinsics[i][0, 0] for i in range(len(self.images))])

    def get_im_poses(self):
        """Return camera-to-world poses as (N,4,4) float tensor."""
        return torch.from_numpy(self.c2w.copy()).float()

    def get_w2c(self, i):
        """Return world-to-camera (4x4) for frame i."""
        return np.linalg.inv(self.c2w[i])

    def scale_intrinsics_to(self, orig_w, orig_h, i):
        """Scale intrinsics from internal resolution to original image size.

        colmap: intrinsics already at original resolution, return as-is
        VGGT/LingBot/DUSt3R/MASt3R: scale by per-axis ratios from model
               image dimensions to original image dimensions.

        Note: VGGT/LingBot K has cx=int_w/2, cy=int_h/2 from the model,
        so we replace cx/cy with orig/2 (pinhole at image center).
        """
        K = self.intrinsics[i].copy()

        if self.internal_resolution == 0 or self.backend in ('colmap',):
            return K

        img = self.images[i]
        int_h, int_w = img.shape[:2]

        sx = orig_w / int_w
        sy = orig_h / int_h
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] = orig_w / 2.0
        K[1, 2] = orig_h / 2.0
        return K


# ── Helper ─────────────────────────────────────────────────────────────────

def _w2c_34_to_c2w_44(w2c_34):
    """Convert (N, 3, 4) w2c matrices to (N, 4, 4) c2w matrices."""
    w2c_34 = np.asarray(w2c_34)
    N = len(w2c_34)
    c2w = np.zeros((N, 4, 4), dtype=np.float64)
    for i in range(N):
        w2c_44 = np.eye(4)
        w2c_44[:3, :] = w2c_34[i]
        c2w[i] = np.linalg.inv(w2c_44)
    return c2w


# ── Backend converters ────────────────────────────────────────────────────


def from_dust3r(scene, image_paths):
    """Convert a DUSt3R GlobalAligner scene to CanonicalScene."""
    from dust3r.utils.device import to_numpy
    from PIL import Image as PILImage
    from PIL.ImageOps import exif_transpose

    imgs = scene.imgs  # list of (H,W,3) float [0,1]
    c2w = to_numpy(scene.get_im_poses().cpu())  # (N, 4, 4)

    if hasattr(scene, 'clean_pointcloud'):
        scene = scene.clean_pointcloud()

    pts3d_list = [to_numpy(p) for p in scene.get_pts3d()]

    if hasattr(scene, 'im_conf'):
        confs_list = [to_numpy(c) for c in scene.im_conf]
    else:
        confs_list = [np.ones(imgs[i].shape[:2], dtype=np.float32) * 10
                      for i in range(len(imgs))]

    intrinsics = to_numpy(scene.get_intrinsics().cpu())  # (N, 3, 3)

    original_sizes = []
    for p in image_paths:
        img = exif_transpose(PILImage.open(p)).convert('RGB')
        original_sizes.append(img.size)

    return CanonicalScene(
        images=imgs, pts3d=pts3d_list, confidence=confs_list,
        c2w=c2w, intrinsics=intrinsics, original_sizes=original_sizes,
        backend='dust3r', internal_resolution=512)


def from_mast3r(scene, image_paths):
    """Convert a MASt3R SparseGA scene to CanonicalScene."""
    from dust3r.utils.device import to_numpy
    from PIL import Image as PILImage
    from PIL.ImageOps import exif_transpose

    imgs = scene.imgs
    c2w = to_numpy(scene.get_im_poses().cpu())  # (N, 4, 4)

    pts3d_raw, _, confs_raw = scene.get_dense_pts3d(clean_depth=True)
    pts3d_list = [to_numpy(pts3d_raw[i]).reshape(imgs[i].shape[0], imgs[i].shape[1], 3)
                  for i in range(len(imgs))]
    confs_list = [to_numpy(confs_raw[i]) for i in range(len(imgs))]

    intrinsics = np.array([to_numpy(K.cpu()) for K in scene.intrinsics])

    original_sizes = []
    for p in image_paths:
        img = exif_transpose(PILImage.open(p)).convert('RGB')
        original_sizes.append(img.size)

    return CanonicalScene(
        images=imgs, pts3d=pts3d_list, confidence=confs_list,
        c2w=c2w, intrinsics=intrinsics, original_sizes=original_sizes,
        backend='mast3r', internal_resolution=512)


def from_vggt(imgs_list, extrinsic_w2c, intrinsic, pts3d_list, conf_list,
              original_sizes):
    """Convert VGGT outputs to CanonicalScene.

    Args:
        extrinsic_w2c: (N, 3, 4) w2c extrinsics from pose_encoding_to_extri_intri
    """
    c2w = _w2c_34_to_c2w_44(extrinsic_w2c)
    return CanonicalScene(
        images=imgs_list, pts3d=pts3d_list, confidence=conf_list,
        c2w=c2w, intrinsics=intrinsic, original_sizes=original_sizes,
        backend='vggt', internal_resolution=518)


def from_lingbot(imgs_list, c2w_34, intrinsic, pts3d_list, conf_list,
                 original_sizes):
    """Convert LingBot-Map outputs to CanonicalScene.

    Args:
        c2w_34: (N, 3, 4) c2w from the demo's double-inversion flow.
                The double-inverted unproject produces points in a frame where
                c2w_34 acts as w2c. Inverting it gives the pose that is
                consistent with both the points AND the COLMAP projection
                (verified: K @ inv(stored_c2w) @ pt reprojects to correct pixel).
    """
    c2w = _w2c_34_to_c2w_44(c2w_34)
    return CanonicalScene(
        images=imgs_list, pts3d=pts3d_list, confidence=conf_list,
        c2w=c2w, intrinsics=intrinsic, original_sizes=original_sizes,
        backend='lingbot', internal_resolution=518)


def from_w2c(imgs_list, w2c_34, intrinsic, pts3d_list, conf_list,
             original_sizes, backend='generic', internal_resolution=512):
    """Generic converter from w2c (N,3,4) extrinsics to CanonicalScene.

    Use this for backends like Pow3R or COLMAP SfM that produce w2c extrinsics.
    Set internal_resolution=0 when intrinsics are already at original resolution.
    """
    c2w = _w2c_34_to_c2w_44(w2c_34)
    return CanonicalScene(
        images=imgs_list, pts3d=pts3d_list, confidence=conf_list,
        c2w=c2w, intrinsics=intrinsic, original_sizes=original_sizes,
        backend=backend, internal_resolution=internal_resolution)
