"""Canonical scene representation for all reconstruction backends.

Every backend converts its output into a CanonicalScene with consistent
conventions. All downstream code (viewport, COLMAP export, mesh, splats)
consumes only this format.

Conventions:
- c2w: always camera-to-world (4x4), OpenCV coords (x-right, y-down, z-forward)
- intrinsics: K matrices at the model's internal resolution
- pts3d: world-space 3D points
- confidence: raw per-pixel confidence (higher = better)
- images: float [0,1] at model's internal resolution
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
            internal_resolution: int — model's working resolution (512 or 518)
        """
        self.images = images
        self.pts3d = pts3d
        self.confidence = confidence
        self.c2w = np.asarray(c2w, dtype=np.float64)
        self.intrinsics = np.asarray(intrinsics, dtype=np.float64)
        self.original_sizes = original_sizes
        self.backend = backend
        self.internal_resolution = internal_resolution

        # For backward compat with code that checks hasattr(scene, '_is_vggt')
        self._is_vggt = True

        # Optional equirect metadata (set by equirect backend)
        self.equirect = None

    # ── Convenience accessors ──────────────────────────────────────────

    @property
    def imgs(self):
        """Alias for images (backward compat)."""
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

        VGGT/LingBot: scale by max(W,H) / internal_res
        DUSt3R/MASt3R: scale by per-axis ratios
        """
        K = self.intrinsics[i].copy()
        img = self.images[i]
        int_h, int_w = img.shape[:2]

        if self.backend in ('vggt', 'lingbot'):
            ratio = max(orig_w, orig_h) / float(self.internal_resolution)
            K[0, 0] *= ratio
            K[1, 1] *= ratio
            K[0, 2] = orig_w / 2.0
            K[1, 2] = orig_h / 2.0
        else:
            sx = orig_w / int_w
            sy = orig_h / int_h
            K[0, 0] *= sx; K[0, 2] *= sx
            K[1, 1] *= sy; K[1, 2] *= sy
        return K

    # ── Backward compat shims ─────────────────────────────────────────

    # These properties let old code that reads scene._extrinsic, scene._pts3d,
    # scene._depth_conf, scene._intrinsic still work during migration.

    @property
    def _extrinsic(self):
        """Return w2c (3,4) matrices for backward compat."""
        w2c_list = []
        for i in range(len(self.images)):
            w2c = self.get_w2c(i)
            w2c_list.append(w2c[:3, :4])
        return np.array(w2c_list)

    @property
    def _intrinsic(self):
        return self.intrinsics

    @property
    def _pts3d(self):
        return self.pts3d

    @property
    def _depth_conf(self):
        return self.confidence


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
    # Convert w2c (3,4) → c2w (4,4)
    N = len(imgs_list)
    c2w = np.zeros((N, 4, 4), dtype=np.float64)
    for i in range(N):
        w2c_44 = np.eye(4)
        w2c_44[:3, :] = extrinsic_w2c[i]
        c2w[i] = np.linalg.inv(w2c_44)

    return CanonicalScene(
        images=imgs_list, pts3d=pts3d_list, confidence=conf_list,
        c2w=c2w, intrinsics=intrinsic, original_sizes=original_sizes,
        backend='vggt', internal_resolution=518)


def from_lingbot(imgs_list, c2w_34, intrinsic, pts3d_list, conf_list,
                 original_sizes):
    """Convert LingBot-Map outputs to CanonicalScene.

    Args:
        c2w_34: (N, 3, 4) c2w extrinsics (after the double-inversion flow,
                these are the poses that are consistent with the point cloud).
    """
    # The c2w_34 from lingbot's double-inversion flow are actually w2c
    # in the point cloud's frame. Invert to get true c2w in that frame.
    N = len(imgs_list)
    c2w = np.zeros((N, 4, 4), dtype=np.float64)
    for i in range(N):
        ext_44 = np.eye(4)
        ext_44[:3, :] = c2w_34[i]
        # c2w_34 was stored as c2w from the original convention, but in the
        # double-inverted point frame it acts as w2c. Invert to get c2w.
        c2w[i] = np.linalg.inv(ext_44)

    return CanonicalScene(
        images=imgs_list, pts3d=pts3d_list, confidence=conf_list,
        c2w=c2w, intrinsics=intrinsic, original_sizes=original_sizes,
        backend='lingbot', internal_resolution=518)
