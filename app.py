"""
DUSt3R / MASt3R / VGGT → COLMAP → Gaussian Splatting Pipeline

Gradio UI that:
1. Accepts input images
2. Runs DUSt3R, MASt3R, or VGGT 3D reconstruction
3. Previews the 3D model
4. Exports COLMAP-format dataset for Gaussian Splatting
"""

import os
import sys
import copy
import math
import shutil
import tempfile
import gc

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F

# Add repos to path
MAST3R_DIR = os.path.join(os.path.dirname(__file__), 'mast3r')
VGGT_DIR = os.path.join(os.path.dirname(__file__), 'vggt')
LINGBOT_DIR = os.path.join(os.path.dirname(__file__), 'lingbot_map')
sys.path.insert(0, MAST3R_DIR)
sys.path.insert(0, os.path.join(MAST3R_DIR, 'dust3r'))
sys.path.insert(0, VGGT_DIR)
sys.path.insert(0, LINGBOT_DIR)

from mast3r.model import AsymmetricMASt3R
from mast3r.image_pairs import make_pairs as mast3r_make_pairs
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment

from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.inference import inference as dust3r_inference
from dust3r.image_pairs import make_pairs as dust3r_make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.demo import _convert_scene_output_to_glb

from colmap_export import export_scene_to_colmap


# ── Globals ──────────────────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
MODELS = {}
TMPDIR = tempfile.mkdtemp(prefix='3dgs_pipeline_')

_original_paths = []


def load_model(backend, **kwargs):
    """Load model on demand, caching for reuse."""
    if backend in MODELS:
        return MODELS[backend]

    if backend == 'dust3r':
        from dust3r.model import AsymmetricCroCo3DStereo
        name = 'naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt'
        print(f"Loading DUSt3R ({name})...")
        model = AsymmetricCroCo3DStereo.from_pretrained(name).to(DEVICE)
        model.eval()
    elif backend == 'mast3r':
        name = 'naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric'
        print(f"Loading MASt3R ({name})...")
        model = AsymmetricMASt3R.from_pretrained(name).to(DEVICE)
        model.eval()
    elif backend == 'vggt':
        from vggt.models.vggt import VGGT
        print("Loading VGGT-1B...")
        model = VGGT.from_pretrained("facebook/VGGT-1B").to(DEVICE)
        model.eval()
    elif backend == 'lingbot':
        from lingbot_map.models.gct_stream import GCTStream
        from huggingface_hub import hf_hub_download
        print("Loading LingBot-Map (GCT-Stream)...")
        kv_window = kwargs.get('kv_window', 16)
        model = GCTStream(
            img_size=518, patch_size=14, enable_3d_rope=True,
            max_frame_num=4096, kv_cache_sliding_window=kv_window,
            kv_cache_scale_frames=8, kv_cache_cross_frame_special=True,
            kv_cache_include_scale_frames=True, use_sdpa=True)
        ckpt_path = hf_hub_download("robbyant/lingbot-map", filename="lingbot-map.pt")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        model.load_state_dict(state_dict, strict=False)
        model.to(DEVICE).eval()

    MODELS[backend] = model
    print(f"{backend} loaded.")
    return model


# ── VGGT wrapper ─────────────────────────────────────────────────────────────
# VGGT outputs raw predictions, not a scene object. This wrapper unifies the
# interface so the rest of the code can treat it like DUSt3R/MASt3R scenes.

class VGGTScene:
    """Lightweight wrapper around VGGT predictions to match scene API."""

    def __init__(self, imgs_np, extrinsic, intrinsic, pts3d, depth_conf, original_sizes):
        """
        Args:
            imgs_np: list of (H,W,3) float [0,1] numpy arrays (at VGGT's internal resolution)
            extrinsic: (N,4,4) numpy, world-to-camera (OpenCV convention)
            intrinsic: (N,3,3) numpy, K matrices at VGGT internal resolution
            pts3d: list of (H,W,3) numpy, world-frame points
            depth_conf: list of (H,W) numpy, confidence
            original_sizes: list of (W,H) original image sizes
        """
        self.imgs = imgs_np
        self._extrinsic = extrinsic  # w2c
        self._intrinsic = intrinsic  # K at internal res
        self._pts3d = pts3d
        self._depth_conf = depth_conf
        self.original_sizes = original_sizes
        self._is_vggt = True

    def get_focals(self):
        focals = torch.tensor([self._intrinsic[i][0, 0] for i in range(len(self.imgs))])
        return focals

    def get_im_poses(self):
        # extrinsic is (N, 3, 4) [R|t] w2c. Pad to 4x4, then invert to get c2w.
        c2w_list = []
        for i in range(len(self.imgs)):
            w2c_34 = self._extrinsic[i]  # (3, 4)
            w2c = np.eye(4)
            w2c[:3, :] = w2c_34
            c2w_list.append(np.linalg.inv(w2c))
        return torch.from_numpy(np.stack(c2w_list)).float()


# ── Reconstruction backends ──────────────────────────────────────────────────

def _reconstruct_dust3r(paths, imgs, scene_graph, niter1, schedule):
    model = load_model('dust3r')
    pairs = dust3r_make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    output = dust3r_inference(pairs, model, DEVICE, batch_size=1, verbose=True)
    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=DEVICE, mode=mode, verbose=True)
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        scene.compute_global_alignment(init='mst', niter=int(niter1), schedule=schedule, lr=0.01)
    return scene


def _reconstruct_mast3r(paths, imgs, scene_graph, niter1, niter2,
                        optim_level, matching_conf_thr, shared_intrinsics):
    model = load_model('mast3r')
    pairs = mast3r_make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    n2 = 0 if optim_level == 'coarse' else int(niter2)
    cache_dir = os.path.join(TMPDIR, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    scene = sparse_global_alignment(
        paths, pairs, cache_dir, model,
        lr1=0.07, niter1=int(niter1), lr2=0.014, niter2=n2,
        device=DEVICE, opt_depth='depth' in optim_level,
        shared_intrinsics=shared_intrinsics,
        matching_conf_thr=float(matching_conf_thr),
    )
    return scene


def _reconstruct_vggt(paths):
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    model = load_model('vggt')

    images = load_and_preprocess_images(paths).to(DEVICE)  # (N, 3, H, W)
    print(f"VGGT: {len(paths)} images, shape {images.shape}")

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=DTYPE):
            predictions = model(images)

    # Extract poses
    pose_enc = predictions["pose_enc"]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

    # Extract depth
    depth_map = predictions["depth"]       # (1, N, H, W, 1) or (1, N, H, W)
    depth_conf = predictions["depth_conf"]  # (1, N, H, W)

    # Squeeze batch dim
    extrinsic = extrinsic.squeeze(0).cpu().numpy()  # (N, 4, 4) w2c
    intrinsic = intrinsic.squeeze(0).cpu().numpy()   # (N, 3, 3)
    depth_map_np = depth_map.squeeze(0).cpu().numpy()  # (N, H, W) or (N, H, W, 1)
    depth_conf_np = depth_conf.squeeze(0).cpu().numpy()  # (N, H, W)

    # Ensure depth has trailing dim (N, H, W, 1) for unproject_depth_map_to_point_map
    if depth_map_np.ndim == 3:
        depth_map_np = depth_map_np[..., None]  # (N, H, W) -> (N, H, W, 1)

    # Unproject to 3D world points
    pts3d = unproject_depth_map_to_point_map(depth_map_np, extrinsic, intrinsic)
    # pts3d: (N, H, W, 3)

    # Build RGB images as numpy float [0,1]
    imgs_np = images.cpu().numpy().transpose(0, 2, 3, 1)  # (N, H, W, 3)
    imgs_list = [imgs_np[i] for i in range(len(paths))]
    pts3d_list = [pts3d[i] for i in range(len(paths))]
    conf_list = [depth_conf_np[i] for i in range(len(paths))]

    # Get original image sizes for COLMAP export
    from PIL import Image as PILImage
    from PIL.ImageOps import exif_transpose
    original_sizes = []
    for p in paths:
        img = exif_transpose(PILImage.open(p)).convert('RGB')
        original_sizes.append(img.size)  # (W, H)

    # Cleanup
    del predictions, images
    torch.cuda.empty_cache()
    gc.collect()

    return VGGTScene(imgs_list, extrinsic, intrinsic, pts3d_list, conf_list, original_sizes)


def _reconstruct_lingbot(paths, keyframe_interval=1, num_scale_frames=8,
                         kv_window=16, progress_cb=None):
    """Run LingBot-Map streaming reconstruction on a sequence of images.

    Uses causal attention with KV cache — handles long video sequences
    that would OOM with batch-only models like VGGT.

    Args:
        paths: list of image file paths
        keyframe_interval: process every Nth frame (1=all, 2=skip half, etc.)
        num_scale_frames: initial bidirectional frames for scale estimation
        kv_window: KV cache sliding window size
        progress_cb: optional callback(frac, msg) for progress updates
    """
    from lingbot_map.utils.load_fn import load_and_preprocess_images
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

    model = load_model('lingbot', kv_window=kv_window)

    images = load_and_preprocess_images(
        paths, mode="crop", image_size=518, patch_size=14).to(DEVICE)
    N = images.shape[0]
    print(f"LingBot-Map: {N} images, shape {images.shape}, "
          f"keyframe_interval={keyframe_interval}")

    if progress_cb:
        progress_cb(0.1, f"Running LingBot-Map on {N} frames...")

    # Streaming inference with KV cache
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=DTYPE):
            predictions = model.inference_streaming(
                images,
                num_scale_frames=min(num_scale_frames, N),
                keyframe_interval=keyframe_interval)

    if progress_cb:
        progress_cb(0.7, "Extracting poses and points...")

    # Decode poses
    pose_enc = predictions["pose_enc"]  # (1, N, 9)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        pose_enc, images.shape[-2:])
    extrinsic = extrinsic.squeeze(0).cpu().numpy()  # (N, 3, 4) w2c
    intrinsic = intrinsic.squeeze(0).cpu().numpy()   # (N, 3, 3)

    # Unproject depth to 3D world space
    # Use lingbot's own unproject which matches its pose convention
    from lingbot_map.utils.geometry import unproject_depth_map_to_point_map
    depth_map = predictions["depth"].squeeze(0).cpu().numpy()  # (N, H, W, 1) or (N, H, W)
    depth_conf = predictions["depth_conf"].squeeze(0).cpu().numpy()  # (N, H, W)

    if depth_map.ndim == 3:
        depth_map = depth_map[..., None]

    # lingbot demo postprocess() inverts w2c→c2w before passing to unproject.
    # unproject then inverts again internally (expects w2c, inverts to c2w).
    # So the double inversion cancels out. To match demo behavior:
    # pass c2w to unproject (it will invert to w2c, which is wrong... but matches demo).
    # Actually: let's just replicate what the demo does exactly.
    from lingbot_map.utils.geometry import closed_form_inverse_se3
    # extrinsic is (N, 3, 4) w2c. Pad to 4x4, invert to c2w, take 3x4 back.
    ext_4x4 = np.zeros((N, 4, 4), dtype=extrinsic.dtype)
    ext_4x4[:, :3, :4] = extrinsic
    ext_4x4[:, 3, 3] = 1.0
    c2w_4x4 = closed_form_inverse_se3(ext_4x4)
    c2w_34 = c2w_4x4[:, :3, :4]  # (N, 3, 4) c2w

    # Pass c2w to unproject (which will invert again → w2c, then unproject correctly)
    # This matches the demo's double-inversion flow
    pts3d = unproject_depth_map_to_point_map(depth_map, c2w_34, intrinsic)

    print(f"  depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
    print(f"  depth_conf range: [{depth_conf.min():.3f}, {depth_conf.max():.3f}]")
    print(f"  pts3d range: [{pts3d[np.isfinite(pts3d)].min():.3f}, {pts3d[np.isfinite(pts3d)].max():.3f}]")

    # Store c2w as the extrinsic (since camera viz inverts w2c→c2w, passing c2w
    # means it'll double-invert to w2c... we need to figure out what desktop_app expects)
    # VGGTScene._extrinsic should be w2c (3,4). Keep original w2c.

    # Build image list
    imgs_np = images.cpu().numpy().transpose(0, 2, 3, 1)  # (N, H, W, 3)
    imgs_list = [imgs_np[i] for i in range(N)]
    pts3d_list = [pts3d[i] for i in range(N)]
    conf_list = [depth_conf[i] for i in range(N)]

    # Original image sizes for COLMAP export
    from PIL import Image as PILImage
    from PIL.ImageOps import exif_transpose
    original_sizes = []
    for p in paths:
        img = exif_transpose(PILImage.open(p)).convert('RGB')
        original_sizes.append(img.size)

    del predictions, images
    torch.cuda.empty_cache()
    gc.collect()

    print(f"LingBot-Map complete: {N} frames, "
          f"conf range [{min(c.min() for c in conf_list):.2f}, "
          f"{max(c.max() for c in conf_list):.2f}]")
    # Points were unprojected with the double-inversion flow (c2w passed to
    # unproject which inverts again). Camera viz inverts _extrinsic (w2c→c2w).
    # Store c2w so the viz inversion produces w2c, matching the point cloud frame.
    return VGGTScene(imgs_list, c2w_34, intrinsic, pts3d_list, conf_list, original_sizes)


def _reconstruct_vggt_equirect(path):
    """Run VGGT depth estimation on a single equirectangular panorama.

    Decomposes the panorama into 12 views (4 horizontal + 4 up-diagonal +
    4 down-diagonal, 105° FOV each), runs VGGT on all 12 as a bundle,
    forces a shared camera center, then stitches depth back into a single
    equirectangular depth map. Returns a VGGTScene with one equirect view.
    """
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from equirect import equirect_to_cubemap
    from PIL import Image as PILImage
    from PIL.ImageOps import exif_transpose

    print("VGGT equirectangular: decomposing panorama into 12 views...")
    pano_img = exif_transpose(PILImage.open(path)).convert('RGB')
    eq_w, eq_h = pano_img.size
    print(f"  Panorama size: {eq_w}x{eq_h}")

    # Cap output resolution to avoid excessive memory for stitching
    max_eq_w = 2048
    if eq_w > max_eq_w:
        scale = max_eq_w / eq_w
        eq_w = max_eq_w
        eq_h = int(eq_h * scale)
        pano_img = pano_img.resize((eq_w, eq_h), PILImage.BICUBIC)
        print(f"  Downscaled to {eq_w}x{eq_h} for stitching")

    # Decompose into cube faces at VGGT's preferred resolution
    # FOV > 90° gives overlap between adjacent faces for better VGGT matching
    face_size = 518  # VGGT internal resolution (divisible by 14)
    face_fov = 105.0  # degrees — 15° overlap on each edge

    # Create a textured version of the panorama for VGGT matching.
    # Featureless surfaces (white walls, ceilings) confuse VGGT because
    # there's nothing to match between faces. We overlay a faint
    # thresholded perlin-like noise to create sharp, non-repeating blobs.
    # Used ONLY for VGGT inference — original colors in the output.
    pano_arr = np.array(pano_img).astype(np.float32)
    rng = np.random.RandomState(42)
    # Multi-octave smooth noise → threshold → sharp non-repeating blobs.
    # Target ~20-40px features at face_size=518, which means ~40-80px at equirect.
    pattern = np.zeros((eq_h, eq_w), dtype=np.float32)
    for octave_scale in [20, 10, 5]:  # feature sizes in pixels at equirect res
        gh = max(2, eq_h // octave_scale)
        gw = max(2, eq_w // octave_scale)
        small = rng.randn(gh, gw).astype(np.float32)
        small_up = np.array(PILImage.fromarray(small, mode='F').resize(
            (eq_w, eq_h), PILImage.BILINEAR))
        pattern += small_up * (octave_scale / 40)  # weight larger features more
    # Threshold to create sharp edges
    pattern = np.sign(pattern) * 15  # ±15 intensity (~6% of 255)
    pano_textured = np.clip(pano_arr + pattern[:, :, None], 0, 255).astype(np.uint8)
    pano_textured_img = PILImage.fromarray(pano_textured)
    # Save textured panorama + pattern for inspection
    debug_pattern = ((pattern + 15) / 30 * 255).clip(0, 255).astype(np.uint8)
    PILImage.fromarray(debug_pattern).save(os.path.join(TMPDIR, 'debug_pattern.png'))
    pano_textured_img.save(os.path.join(TMPDIR, 'debug_pano_textured.png'))
    print(f"  Added noise pattern for VGGT matching (saved debug_pattern.png and debug_pano_textured.png to {TMPDIR})")

    # Decompose TEXTURED panorama for VGGT inference
    faces_textured, face_names = equirect_to_cubemap(pano_textured_img, face_size=face_size, fov_deg=face_fov)

    print(f"  {len(face_names)} faces extracted ({face_fov:.0f}° FOV)")

    # Build VGGT input tensor directly — skip disk save/reload entirely.
    # Faces are already 518×518 RGB, just convert to (N, 3, H, W) float [0,1].
    from torchvision import transforms as TF
    to_tensor = TF.ToTensor()
    face_tensors = [to_tensor(f) for f in faces_textured]
    images = torch.stack(face_tensors).to(DEVICE)

    model = load_model('vggt')
    print(f"  VGGT input: {images.shape}")

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=DTYPE):
            predictions = model(images)

    pose_enc = predictions["pose_enc"]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

    depth_map = predictions["depth"].squeeze(0).cpu().numpy()
    depth_conf = predictions["depth_conf"].squeeze(0).cpu().numpy()
    n_faces = len(face_names)
    extrinsic = extrinsic.squeeze(0).cpu().numpy()  # (N, 3, 4)
    intrinsic = intrinsic.squeeze(0).cpu().numpy()   # (N, 3, 3)

    if depth_map.ndim == 4:
        depth_map = depth_map[..., 0]  # (N, H, W)

    # Unproject each face to 3D (in VGGT's coordinate frame)
    depth_4d = depth_map[..., None]  # (N, H, W, 1)
    pts3d = unproject_depth_map_to_point_map(depth_4d, extrinsic, intrinsic)

    # Keep cameras exactly where VGGT placed them — moving them creates
    # seams in the point cloud at face boundaries.

    # Extract CLEAN face crops from the original panorama at full resolution.
    # Determine a face size that preserves panorama detail:
    # each face covers ~(fov/360)*eq_w pixels horizontally.
    clean_face_size = max(face_size, int(eq_w * face_fov / 360))
    # Round to nearest multiple of 14 for consistency
    clean_face_size = ((clean_face_size + 13) // 14) * 14
    print(f"  Clean face crops: {clean_face_size}x{clean_face_size} "
          f"(from {eq_w}x{eq_h} panorama)")
    faces_clean, _ = equirect_to_cubemap(pano_img, face_size=clean_face_size, fov_deg=face_fov)

    # Build N-view VGGTScene (one per face), mesh builder handles per-face grid.
    # scene.imgs = textured faces (for point cloud display, matches VGGT input)
    # scene._equirect_clean_imgs = clean faces at full res (for mesh coloring)
    imgs_list = []
    pts3d_list = []
    conf_list = []
    clean_imgs_list = []
    for i in range(n_faces):
        face_np = np.array(faces_textured[i]).astype(np.float32) / 255.0
        imgs_list.append(face_np)
        clean_np = np.array(faces_clean[i]).astype(np.float32) / 255.0
        clean_imgs_list.append(clean_np)
        pts3d_list.append(pts3d[i])
        conf_list.append(depth_conf[i])

    original_sizes = [(face_size, face_size)] * n_faces

    del predictions, images
    torch.cuda.empty_cache()
    gc.collect()

    # ── Merge overlapping faces into a single equirect point grid ───────
    from equirect import merge_faces_to_equirect
    eq_pts3d, eq_conf, eq_color = merge_faces_to_equirect(
        pts3d_list, conf_list, clean_imgs_list, eq_h, eq_w, fov_deg=face_fov)

    # Build a single-view VGGTScene from the merged equirect grid
    eq_img = np.array(pano_img).astype(np.float32) / 255.0
    eq_extrinsic = np.zeros((1, 3, 4), dtype=np.float64)
    eq_extrinsic[0, :3, :3] = np.eye(3)
    eq_intrinsic = np.zeros((1, 3, 3), dtype=np.float64)
    eq_intrinsic[0] = np.eye(3)

    scene = VGGTScene(
        [eq_img], eq_extrinsic, eq_intrinsic,
        [eq_pts3d], [eq_conf], [(eq_w, eq_h)])
    scene._equirect = True
    scene._equirect_merged_pts3d = eq_pts3d
    scene._equirect_merged_conf = eq_conf
    scene._equirect_merged_color = eq_color
    scene._equirect_pano_img = np.array(pano_img)  # uint8 for texture
    scene._equirect_pano_size = (eq_w, eq_h)

    print(f"VGGT equirectangular complete: {n_faces} faces → {eq_w}x{eq_h} merged")
    return scene


def _reconstruct_vggt_ensemble(paths, bundle_size=20, n_anchors=16):
    """Run VGGT on scattered bundles with shared anchors for global alignment.

    Strategy:
    1. Pick `n_anchors` images spread evenly across the full sequence — these
       go in EVERY bundle and serve as Procrustes anchors.
    2. Remaining images are grouped into consecutive pairs.
    3. Pairs are distributed round-robin across bundles so each bundle's pairs
       are scattered across the full sequence (wide baselines).
    4. Bundle 0 defines the reference frame; others align via Procrustes on
       the shared anchors.
    5. Each image gets depth/extrinsic from exactly ONE bundle (no mixing).
    """
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    N = len(paths)
    if N <= bundle_size:
        print(f"VGGT ensemble: only {N} images, falling back to single pass")
        return _reconstruct_vggt(paths)

    model = load_model('vggt')
    all_images = load_and_preprocess_images(paths).to(DEVICE)  # (N, 3, H, W)
    H_int, W_int = all_images.shape[2], all_images.shape[3]

    # ── 1. Pick anchor images spread across the sequence ──────────────────
    anchor_indices = np.round(np.linspace(0, N - 1, n_anchors)).astype(int).tolist()
    anchor_set = set(anchor_indices)

    # ── 2. Group remaining images into consecutive pairs ──────────────────
    remaining = [i for i in range(N) if i not in anchor_set]
    pairs = []
    i = 0
    while i < len(remaining):
        if i + 1 < len(remaining) and remaining[i + 1] == remaining[i] + 1:
            pairs.append((remaining[i], remaining[i + 1]))
            i += 2
        else:
            pairs.append((remaining[i],))
            i += 1

    # ── 3. Distribute pairs round-robin across bundles ────────────────────
    max_pairs_per_bundle = (bundle_size - n_anchors) // 2
    n_bundles = max(1, -(-len(pairs) // max_pairs_per_bundle))  # ceil div

    bundle_pairs = [[] for _ in range(n_bundles)]
    for pi, pair in enumerate(pairs):
        bundle_pairs[pi % n_bundles].append(pair)

    # Build final bundle index lists (anchors + scattered pairs)
    bundles = []
    for bi in range(n_bundles):
        imgs = set(anchor_indices)
        for pair in bundle_pairs[bi]:
            imgs.update(pair)
        bundles.append(sorted(imgs))

    # Which bundle owns each image (for depth assignment)
    best_bundle = np.full(N, 0, dtype=int)
    for bi in range(n_bundles):
        for pair in bundle_pairs[bi]:
            for img_idx in pair:
                best_bundle[img_idx] = bi
    # Anchors: assign to bundle 0
    for ai in anchor_indices:
        best_bundle[ai] = 0

    print(f"VGGT ensemble: {N} images, {n_bundles} scattered bundles "
          f"(max {bundle_size}/bundle, {n_anchors} shared anchors)")
    for bi, b in enumerate(bundles):
        print(f"  Bundle {bi}: {len(b)} images, "
              f"indices [{b[0]}..{b[-1]}], "
              f"{len(bundle_pairs[bi])} pairs")

    # ── Helper functions ──────────────────────────────────────────────────

    def cam_frames(extrinsics):
        """Extract camera centers + axis endpoints from w2c matrices (3,4)."""
        centers, rights, ups, fwds = [], [], [], []
        for w2c in extrinsics:
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            C = -R.T @ t
            right = R.T @ np.array([1, 0, 0])
            up    = R.T @ np.array([0, 1, 0])
            fwd   = R.T @ np.array([0, 0, 1])
            centers.append(C)
            rights.append(C + right)
            ups.append(C + up)
            fwds.append(C + fwd)
        return (np.array(centers), np.array(rights),
                np.array(ups), np.array(fwds))

    def rigid_align(src, dst):
        """Rigid alignment (rotation + translation only, no scale).
        Finds R, t minimizing ||dst - (R@src + t)||^2.
        VGGT predicts metric depth so scale should be ~1.0 between bundles."""
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)
        src_c = src - src_mean
        dst_c = dst - dst_mean
        H_mat = src_c.T @ dst_c
        U, _, Vt = np.linalg.svd(H_mat)
        d = np.linalg.det(Vt.T @ U.T)
        R = Vt.T @ np.diag([1, 1, d]) @ U.T
        t = dst_mean - R @ src_mean
        return R, t

    def procrustes_rigid(bun_w2c, ref_w2c):
        """Align bundle cameras to reference using positions + orientations.
        Rigid (no scale) — uses 4 points per camera for robustness."""
        bun_c, bun_r, bun_u, bun_f = cam_frames(bun_w2c)
        ref_c, ref_r, ref_u, ref_f = cam_frames(ref_w2c)
        src = np.concatenate([bun_c, bun_r, bun_u, bun_f], axis=0)
        dst = np.concatenate([ref_c, ref_r, ref_u, ref_f], axis=0)
        return rigid_align(src, dst)

    def transform_w2c(w2c, R_align, t_align):
        """Transform a w2c (3,4) matrix from bundle frame to reference frame.
        Rigid transform (no scale): P_ref = R_align @ P_bun + t_align."""
        R_w = w2c[:3, :3]
        t_w = w2c[:3, 3]
        C_bun = -R_w.T @ t_w
        C_ref = R_align @ C_bun + t_align
        R_ref = R_w @ R_align.T
        t_ref = -R_ref @ C_ref
        w2c_new = np.zeros((3, 4), dtype=np.float64)
        w2c_new[:3, :3] = R_ref
        w2c_new[:3, 3] = t_ref
        return w2c_new

    # ── Per-image storage ─────────────────────────────────────────────────
    global_extrinsic = np.zeros((N, 3, 4), dtype=np.float64)
    global_intrinsic = np.zeros((N, 3, 3), dtype=np.float64)
    global_depth = np.zeros((N, H_int, W_int), dtype=np.float32)
    global_conf = np.zeros((N, H_int, W_int), dtype=np.float32)
    assigned_depth = np.zeros(N, dtype=bool)

    # ── Process each bundle ───────────────────────────────────────────────
    ref_anchor_w2c = None  # anchor w2c from bundle 0 (reference frame)

    for bi, bundle_idx in enumerate(bundles):
        B = len(bundle_idx)
        print(f"  Running bundle {bi+1}/{n_bundles}: {B} images")

        # Map from global image index → local index in this bundle
        global_to_local = {gi: li for li, gi in enumerate(bundle_idx)}

        bundle_images = all_images[bundle_idx]
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=DTYPE):
                pred = model(bundle_images)

        b_pose_enc = pred["pose_enc"]
        b_extri, b_intri = pose_encoding_to_extri_intri(b_pose_enc, (H_int, W_int))
        b_extri = b_extri.squeeze(0).cpu().numpy()  # (B, 3, 4)
        b_intri = b_intri.squeeze(0).cpu().numpy()   # (B, 3, 3)

        b_depth = pred["depth"].squeeze(0).cpu().numpy()
        b_conf = pred["depth_conf"].squeeze(0).cpu().numpy()
        if b_depth.ndim == 4:
            b_depth = b_depth[..., 0]

        del pred
        torch.cuda.empty_cache()

        if bi == 0:
            # First bundle defines the reference frame
            ref_anchor_w2c = {}
            for ai in anchor_indices:
                li = global_to_local[ai]
                ref_anchor_w2c[ai] = b_extri[li].copy()

            # Store all images owned by bundle 0
            for li, gi in enumerate(bundle_idx):
                if best_bundle[gi] == 0:
                    global_extrinsic[gi] = b_extri[li]
                    global_intrinsic[gi] = b_intri[li]
                    global_depth[gi] = b_depth[li]
                    global_conf[gi] = b_conf[li]
                    assigned_depth[gi] = True

            print(f"    Reference frame set ({len(ref_anchor_w2c)} anchors)")
        else:
            # Align to reference frame via shared anchors
            anchor_local_indices = [global_to_local[ai] for ai in anchor_indices]
            bun_anchor_w2c = b_extri[anchor_local_indices]
            ref_anchor_arr = np.array([ref_anchor_w2c[ai] for ai in anchor_indices])

            R_align, t_align = procrustes_rigid(bun_anchor_w2c, ref_anchor_arr)

            # Alignment error
            bun_centers = cam_frames(bun_anchor_w2c)[0]
            ref_centers = cam_frames(ref_anchor_arr)[0]
            aligned_pos = (R_align @ bun_centers.T).T + t_align
            per_cam_err = np.sqrt(((aligned_pos - ref_centers) ** 2).sum(axis=1))
            print(f"    Alignment: mean cam error={per_cam_err.mean():.6f}, "
                  f"max={per_cam_err.max():.6f} ({n_anchors} anchors)")

            # Store images owned by this bundle (rigid-aligned, no depth scaling)
            for li, gi in enumerate(bundle_idx):
                if best_bundle[gi] == bi and not assigned_depth[gi]:
                    aligned_w2c = transform_w2c(b_extri[li], R_align, t_align)
                    global_extrinsic[gi] = aligned_w2c
                    global_intrinsic[gi] = b_intri[li]
                    global_depth[gi] = b_depth[li]
                    global_conf[gi] = b_conf[li]
                    assigned_depth[gi] = True

    # Check coverage
    missing = np.where(~assigned_depth)[0]
    if len(missing) > 0:
        print(f"  WARNING: {len(missing)} images missing depth!")
        for img_idx in missing:
            print(f"    Image {img_idx} (best_bundle={best_bundle[img_idx]})")

    # Unproject depth to 3D
    depth_4d = global_depth[..., None]  # (N, H, W, 1)
    pts3d = unproject_depth_map_to_point_map(depth_4d, global_extrinsic, global_intrinsic)

    # Build output
    imgs_np = all_images.cpu().numpy().transpose(0, 2, 3, 1)
    imgs_list = [imgs_np[i] for i in range(N)]
    pts3d_list = [pts3d[i] for i in range(N)]
    conf_list = [global_conf[i] for i in range(N)]

    from PIL import Image as PILImage
    from PIL.ImageOps import exif_transpose
    original_sizes = []
    for p in paths:
        img = exif_transpose(PILImage.open(p)).convert('RGB')
        original_sizes.append(img.size)

    del all_images
    torch.cuda.empty_cache()
    gc.collect()

    print(f"VGGT ensemble complete: {N} views, {n_bundles} bundles")
    return VGGTScene(imgs_list, global_extrinsic, global_intrinsic, pts3d_list, conf_list, original_sizes)


# ── Scene data extraction (handles all backends) ────────────────────────────

def _is_mast3r_scene(scene):
    return hasattr(scene, 'canonical_paths')


def _is_vggt_scene(scene):
    return hasattr(scene, '_is_vggt')


def _extract_dense_pts3d(scene, min_conf_thr, clean_depth):
    """Extract pts3d and masks from any scene type, returning (H,W,3) arrays."""
    rgbimg = scene.imgs

    if _is_vggt_scene(scene):
        pts3d = scene._pts3d
        # VGGT conf is percentile-based, threshold differently
        msk = [scene._depth_conf[i] > min_conf_thr for i in range(len(rgbimg))]
        return pts3d, msk
    elif _is_mast3r_scene(scene):
        pts3d_raw, _, confs_raw = scene.get_dense_pts3d(clean_depth=clean_depth)
        pts3d = []
        msk = []
        for i in range(len(rgbimg)):
            H, W = rgbimg[i].shape[:2]
            pts3d.append(to_numpy(pts3d_raw[i]).reshape(H, W, 3))
            msk.append(to_numpy(confs_raw[i]) > min_conf_thr)
        return pts3d, msk
    else:
        # DUSt3R
        if clean_depth:
            scene = scene.clean_pointcloud()
        pts3d = to_numpy(scene.get_pts3d())
        scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
        msk = to_numpy(scene.get_masks())
        return pts3d, msk


# ── Core Pipeline ────────────────────────────────────────────────────────────

def reconstruct(filelist, backend, optim_level, schedule, niter1, niter2, min_conf_thr,
                matching_conf_thr,
                as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                scenegraph_type, winsize, refid, shared_intrinsics,
                vggt_ensemble=False, vggt_equirect=False):
    global _original_paths

    if not filelist or len(filelist) == 0:
        raise gr.Error("Please upload at least 2 images.")

    if isinstance(filelist[0], str):
        paths = filelist
    else:
        paths = [f.name if hasattr(f, 'name') else f for f in filelist]

    _original_paths = list(paths)

    if backend == 'vggt':
        if vggt_equirect and len(paths) == 1:
            scene = _reconstruct_vggt_equirect(paths[0])
        elif vggt_ensemble and len(paths) > 4:
            scene = _reconstruct_vggt_ensemble(paths)
        else:
            scene = _reconstruct_vggt(paths)
    else:
        # DUSt3R / MASt3R need dust3r-style image loading
        model = load_model(backend)
        imgs = load_images(paths, size=512, verbose=True, patch_size=model.patch_size)
        if len(imgs) == 1:
            imgs = [imgs[0], copy.deepcopy(imgs[0])]
            imgs[1]['idx'] = 1
            paths = [paths[0], paths[0] + '_2']

        sg_parts = [scenegraph_type]
        if scenegraph_type in ["swin", "logwin"]:
            sg_parts.append(str(int(winsize)))
        elif scenegraph_type == "oneref":
            sg_parts.append(str(int(refid)))
        scene_graph = '-'.join(sg_parts)

        if backend == 'dust3r':
            scene = _reconstruct_dust3r(paths, imgs, scene_graph, niter1, schedule)
        else:
            scene = _reconstruct_mast3r(paths, imgs, scene_graph, niter1, niter2,
                                        optim_level, matching_conf_thr, shared_intrinsics)

    outfile = get_3d_preview(scene, min_conf_thr, as_pointcloud,
                             mask_sky, clean_depth, transparent_cams, cam_size)
    return scene, outfile


def get_3d_preview(scene, min_conf_thr=2, as_pointcloud=False, mask_sky=False,
                   clean_depth=True, transparent_cams=False, cam_size=0.05):
    if scene is None:
        return None

    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    pts3d, msk = _extract_dense_pts3d(scene, min_conf_thr, clean_depth)

    return _convert_scene_output_to_glb(TMPDIR, rgbimg, pts3d, msk, focals, cams2world,
                                        as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams,
                                        cam_size=cam_size)


def update_preview(scene, min_conf_thr, as_pointcloud, mask_sky,
                   clean_depth, transparent_cams, cam_size):
    return get_3d_preview(scene, min_conf_thr, as_pointcloud,
                          mask_sky, clean_depth, transparent_cams, cam_size)


def refine_mesh_fn(scene, min_conf_thr, clean_depth, refine_iters, compare_mode,
                   depth_reg, smooth_reg, progress=gr.Progress(track_tqdm=False)):
    """Export mesh, then refine with multi-view photoconsistency."""
    if scene is None:
        raise gr.Error("No reconstruction available. Run reconstruction first!")

    # Step 1: Export COLMAP (needed for camera data)
    progress(0, desc="Exporting COLMAP dataset...")
    export_dir = os.path.join(TMPDIR, 'colmap_export')
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    export_scene_to_colmap(
        scene=scene, image_paths=_original_paths,
        output_dir=export_dir, min_conf_thr=float(min_conf_thr),
        clean_depth=clean_depth,
    )

    # Step 2: Check if dense mesh exists, if not create it
    mesh_path = os.path.join(export_dir, 'dense_mesh.ply')
    if not os.path.exists(mesh_path):
        progress(0.1, desc="Creating dense mesh...")
        from mesh_export import tsdf_fusion, save_mesh_ply, save_dense_ply
        pts3d, msk = _extract_dense_pts3d(scene, min_conf_thr, clean_depth)
        imgs = scene.imgs
        confs = [np.ones(imgs[i].shape[:2], dtype=np.float32) for i in range(len(imgs))]
        for i in range(len(imgs)):
            confs[i][~msk[i]] = 0.0
        try:
            verts, faces, colors = tsdf_fusion(imgs, pts3d, confs, min_conf=0.5)
            if len(faces) > 0:
                save_mesh_ply(mesh_path, verts, faces, colors)
            else:
                raise ValueError("No faces generated")
        except Exception as e:
            raise gr.Error(f"Mesh generation failed: {e}. Try 'Export Dense Mesh' first.")

    # Step 3: Refine
    progress(0.2, desc="Refining mesh...")
    output_path = os.path.join(TMPDIR, 'refined_mesh.ply')

    import subprocess, re
    cmd = [
        sys.executable, '-u',
        os.path.join(os.path.dirname(__file__), 'refine_mesh.py'),
        '--data_dir', export_dir,
        '--mesh_path', mesh_path,
        '--output_path', output_path,
        '--iterations', str(int(refine_iters)),
        '--compare_mode', str(compare_mode),
        '--depth_reg', str(float(depth_reg)),
        '--smooth_reg', str(float(smooth_reg)),
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1)
    step_re = re.compile(r'\[\s*(\d+)/\s*(\d+)\]')
    for line in proc.stdout:
        line = line.rstrip()
        print(line)
        match = step_re.search(line)
        if match:
            s = int(match.group(1))
            t = int(match.group(2))
            frac = 0.2 + 0.8 * (s / max(t, 1))
            progress(frac, desc=f"Refining: {s}/{t}")

    proc.wait()
    if proc.returncode != 0:
        err = proc.stderr.read()
        raise gr.Error(f"Refinement failed: {err[-500:]}")

    progress(1.0, desc="Refinement complete!")
    return output_path


def export_dense_mesh(scene, min_conf_thr, clean_depth):
    """Export dense mesh/point cloud from reconstruction depth maps."""
    if scene is None:
        raise gr.Error("No reconstruction available. Run reconstruction first!")

    from mesh_export import tsdf_fusion, save_mesh_ply, save_dense_ply

    pts3d, msk = _extract_dense_pts3d(scene, min_conf_thr, clean_depth)
    imgs = scene.imgs

    # Build confidence maps (use mask as binary conf)
    confs = []
    for i in range(len(imgs)):
        c = np.ones(imgs[i].shape[:2], dtype=np.float32)
        c[~msk[i]] = 0.0
        confs.append(c)

    # Try TSDF mesh first, fallback to dense point cloud
    output_path = os.path.join(TMPDIR, 'dense_mesh.ply')
    try:
        verts, faces, colors = tsdf_fusion(imgs, pts3d, confs, min_conf=0.5)
        if len(faces) > 0:
            save_mesh_ply(output_path, verts, faces, colors)
        else:
            # No mesh, save as dense point cloud
            all_pts = np.concatenate([pts3d[i][msk[i]] for i in range(len(imgs))], axis=0)
            all_cols = np.concatenate(
                [(np.clip(imgs[i][msk[i]], 0, 1) * 255).astype(np.uint8) for i in range(len(imgs))], axis=0)
            save_dense_ply(output_path, all_pts, all_cols)
    except Exception as e:
        print(f"TSDF failed ({e}), exporting dense point cloud instead")
        all_pts = np.concatenate([pts3d[i][msk[i]] for i in range(len(imgs))], axis=0)
        all_cols = np.concatenate(
            [(np.clip(imgs[i][msk[i]], 0, 1) * 255).astype(np.uint8) for i in range(len(imgs))], axis=0)
        save_dense_ply(output_path, all_pts, all_cols)

    return output_path


def export_colmap(scene, min_conf_thr, clean_depth):
    if scene is None:
        raise gr.Error("No reconstruction available. Run reconstruction first!")

    export_dir = os.path.join(TMPDIR, 'colmap_export')
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)

    export_scene_to_colmap(
        scene=scene,
        image_paths=_original_paths,
        output_dir=export_dir,
        min_conf_thr=float(min_conf_thr),
        clean_depth=clean_depth,
    )

    zip_path = os.path.join(TMPDIR, 'colmap_dataset')
    shutil.make_archive(zip_path, 'zip', export_dir)
    return zip_path + '.zip'


def export_and_train(scene, min_conf_thr, clean_depth, train_iters, target_splats,
                     depth_lambda, aniso_lambda, progress=gr.Progress(track_tqdm=False)):
    """Export COLMAP, then train depth-regularized gaussians."""
    if scene is None:
        raise gr.Error("No reconstruction available. Run reconstruction first!")

    total_iters = int(train_iters)

    # Step 1: Export COLMAP
    progress(0, desc="Exporting COLMAP dataset...")
    export_dir = os.path.join(TMPDIR, 'colmap_export')
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    export_scene_to_colmap(
        scene=scene, image_paths=_original_paths,
        output_dir=export_dir, min_conf_thr=float(min_conf_thr),
        clean_depth=clean_depth,
    )

    # Step 2: Train with live progress
    output_dir = os.path.join(TMPDIR, 'trained_gaussians')
    os.makedirs(output_dir, exist_ok=True)

    import subprocess, re
    cmd = [
        sys.executable, '-u',  # unbuffered output
        os.path.join(os.path.dirname(__file__), 'train.py'),
        '--data_dir', export_dir,
        '--output_dir', output_dir,
        '--iterations', str(total_iters),
        '--target_splats', str(int(target_splats)),
        '--depth_lambda', str(float(depth_lambda)),
        '--aniso_lambda', str(float(aniso_lambda)),
    ]
    print(f"Running: {' '.join(cmd)}")

    progress(0, desc="Starting training...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1)

    # Parse stdout for progress updates
    stderr_lines = []
    step_re = re.compile(r'\[\s*(\d+)/\s*(\d+)\].*loss=([\d.]+)')

    for line in proc.stdout:
        line = line.rstrip()
        print(line)
        match = step_re.search(line)
        if match:
            step = int(match.group(1))
            total = int(match.group(2))
            loss = match.group(3)
            frac = step / max(total, 1)
            progress(frac, desc=f"Training: step {step}/{total}, loss={loss}")

    proc.wait()
    stderr_out = proc.stderr.read()
    if stderr_out.strip():
        real_errors = [l for l in stderr_out.splitlines()
                       if 'FutureWarning' not in l and 'UserWarning' not in l]
        if real_errors:
            stderr_lines = real_errors

    if proc.returncode != 0:
        err = '\n'.join(stderr_lines[-10:]) if stderr_lines else stderr_out[-500:]
        print(err)
        raise gr.Error(f"Training failed:\n{err}")

    progress(1.0, desc="Training complete!")

    ply_path = os.path.join(output_dir, 'point_cloud.ply')
    if not os.path.exists(ply_path):
        raise gr.Error("Training completed but no output PLY found")

    preview_path = os.path.join(output_dir, 'preview', 'latest.png')
    preview = preview_path if os.path.exists(preview_path) else None
    return ply_path, preview


# ── UI Helpers ───────────────────────────────────────────────────────────────

def set_scenegraph_options(inputfiles, winsize, refid, scenegraph_type):
    num_files = len(inputfiles) if inputfiles is not None else 1
    max_winsize = max(1, math.ceil((num_files - 1) / 2))
    if scenegraph_type in ["swin", "logwin"]:
        winsize = gr.Slider(label="Window Size", value=max_winsize,
                            minimum=1, maximum=max_winsize, step=1, visible=True)
        refid = gr.Slider(label="Reference Id", value=0, minimum=0,
                          maximum=num_files - 1, step=1, visible=False)
    elif scenegraph_type == "oneref":
        winsize = gr.Slider(label="Window Size", value=max_winsize,
                            minimum=1, maximum=max_winsize, step=1, visible=False)
        refid = gr.Slider(label="Reference Id", value=0, minimum=0,
                          maximum=num_files - 1, step=1, visible=True)
    else:
        winsize = gr.Slider(label="Window Size", value=max_winsize,
                            minimum=1, maximum=max_winsize, step=1, visible=False)
        refid = gr.Slider(label="Reference Id", value=0, minimum=0,
                          maximum=num_files - 1, step=1, visible=False)
    return winsize, refid


def toggle_backend_options(backend):
    is_mast3r = backend == 'mast3r'
    is_dust3r = backend == 'dust3r'
    is_vggt = backend == 'vggt'
    show_pairing = is_dust3r or is_mast3r  # VGGT doesn't use scene graphs
    return (
        gr.update(visible=is_mast3r),       # optim_level
        gr.update(visible=is_mast3r),       # niter2
        gr.update(visible=is_mast3r),       # matching_conf_thr
        gr.update(visible=is_mast3r),       # shared_intrinsics
        gr.update(visible=is_dust3r),       # schedule
        gr.update(visible=show_pairing),    # niter1
        gr.update(visible=show_pairing),    # scenegraph_type
        gr.update(visible=show_pairing),    # winsize
        gr.update(visible=show_pairing),    # refid
        gr.update(visible=is_vggt),         # vggt_ensemble
        gr.update(visible=is_vggt),         # vggt_equirect
    )


# ── Gradio UI ────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(
        css=".gradio-container {margin: 0 !important; min-width: 100%};",
        title="3D Reconstruction → COLMAP → 3DGS"
    ) as demo:
        scene_state = gr.State(None)

        gr.HTML('<h2 style="text-align: center;">3D Reconstruction → COLMAP → Gaussian Splatting</h2>')
        gr.HTML('<p style="text-align: center; color: #666;">'
                'Upload images → Reconstruct with DUSt3R / MASt3R / VGGT → Export COLMAP → Train 3DGS</p>')

        with gr.Column():
            inputfiles = gr.File(file_count="multiple", label="Upload Images")

            backend = gr.Radio(
                [("DUSt3R (dense, simple)", "dust3r"),
                 ("MASt3R (descriptors, metric)", "mast3r"),
                 ("VGGT (feed-forward, fast)", "vggt")],
                value="mast3r", label="Reconstruction Backend")

            with gr.Row():
                scenegraph_type = gr.Dropdown(
                    [("Complete (all pairs)", "complete"),
                     ("Sliding Window", "swin"),
                     ("Log Window", "logwin"),
                     ("One Reference", "oneref")],
                    value='complete', label="Scene Graph",
                    info="How to pair images", interactive=True)
                winsize = gr.Slider(label="Window Size", value=1,
                                    minimum=1, maximum=1, step=1, visible=False)
                refid = gr.Slider(label="Reference Id", value=0,
                                  minimum=0, maximum=0, step=1, visible=False)

            with gr.Row():
                schedule = gr.Dropdown(["linear", "cosine"], value='linear',
                                       label="Schedule", info="Global alignment schedule",
                                       visible=False)
                optim_level = gr.Dropdown(
                    [("Coarse (fast)", "coarse"),
                     ("Refine (recommended)", "refine"),
                     ("Refine + Depth", "refine+depth")],
                    value='refine+depth', label="Optimization",
                    info="Coarse=3D only, Refine=+2D reproj, Depth=+depth optim",
                    visible=True)
                niter1 = gr.Number(value=300, precision=0, minimum=0, maximum=5000,
                                   label="Iterations (coarse)")
                niter2 = gr.Number(value=300, precision=0, minimum=0, maximum=5000,
                                   label="Iterations (refine)", visible=True)
                matching_conf_thr = gr.Slider(label="Matching Confidence",
                                              value=5.0, minimum=0.0, maximum=30.0, step=0.5,
                                              visible=True)
                shared_intrinsics = gr.Checkbox(value=False, label="Shared Intrinsics",
                                                visible=True)

            vggt_ensemble = gr.Checkbox(value=False, label="VGGT Ensemble (bundles of 20 with overlap — scales to many images)",
                                        visible=False)
            vggt_equirect = gr.Checkbox(value=False, label="Equirectangular panorama (single 360° image → cubemap depth)",
                                        visible=False)

            run_btn = gr.Button("Reconstruct", variant="primary", size="lg")

            with gr.Row():
                min_conf_thr = gr.Slider(label="Min Confidence Threshold",
                                         value=2.0, minimum=0.1, maximum=20, step=0.1)
                cam_size = gr.Slider(label="Camera Size (preview)",
                                     value=0.05, minimum=0.001, maximum=0.1, step=0.001)
            with gr.Row():
                as_pointcloud = gr.Checkbox(value=True, label="As Pointcloud")
                mask_sky = gr.Checkbox(value=False, label="Mask Sky")
                clean_depth = gr.Checkbox(value=True, label="Clean Depth")
                transparent_cams = gr.Checkbox(value=False, label="Transparent Cameras")

            outmodel = gr.Model3D(label="3D Preview")

            gr.HTML('<h3>Export</h3>')
            with gr.Row():
                export_btn = gr.Button("Export COLMAP Dataset (.zip)", variant="secondary")
                export_mesh_btn = gr.Button("Export Dense Mesh (.ply)", variant="secondary")
            with gr.Row():
                export_file = gr.File(label="Download COLMAP Dataset")
                export_mesh_file = gr.File(label="Download Dense Mesh")

            # ── Refine Mesh ──
            gr.HTML('<h3>Refine Mesh (Photoconsistency)</h3>')
            gr.HTML('<p style="color: #666;">Moves vertices to minimize color disagreement between cameras. Export Dense Mesh first.</p>')
            with gr.Row():
                refine_iters = gr.Number(value=500, precision=0, label="Iterations")
                refine_compare = gr.Dropdown(
                    [("Edges (geometry-focused)", "edges"),
                     ("High Frequency (edges + texture)", "highfreq"),
                     ("Color (raw RGB)", "color"),
                     ("Both (edges + color)", "both")],
                    value="edges", label="Compare Mode",
                    info="Edges = best for geometry, ignores flat color differences")
                refine_depth_reg = gr.Slider(value=0.1, minimum=0.0, maximum=1.0, step=0.01,
                                              label="Depth Reg", info="How much to anchor to original position")
                refine_smooth_reg = gr.Slider(value=0.01, minimum=0.0, maximum=0.1, step=0.005,
                                               label="Smooth Reg", info="Laplacian smoothing weight")
            refine_btn = gr.Button("Refine Mesh", variant="secondary", size="lg")
            refine_output = gr.File(label="Download Refined Mesh (.ply)")

            # ── Train Gaussians ──
            gr.HTML('<h3>Train Depth-Regularized Gaussians</h3>')
            gr.HTML('<p style="color: #666;">Custom few-view training with depth supervision and anisotropy regularization.</p>')
            with gr.Row():
                train_iters = gr.Number(value=7000, precision=0, label="Iterations")
                train_target_splats = gr.Number(value=0, precision=0, label="Target Splats",
                                                 info="0 = keep initial count from point cloud. Set e.g. 50000 to grow.")
                train_depth_lambda = gr.Slider(value=0.5, minimum=0.0, maximum=2.0, step=0.05,
                                                label="Depth Weight", info="How much to trust depth priors")
                train_aniso_lambda = gr.Slider(value=0.01, minimum=0.0, maximum=0.1, step=0.005,
                                                label="Anisotropy Reg", info="Penalize elongated gaussians")
            train_btn = gr.Button("Export + Train Gaussians", variant="primary", size="lg")
            with gr.Row():
                train_output = gr.File(label="Download Trained Model (.ply)")
                train_preview = gr.Image(label="Training Preview")

        # ── Events ──
        backend.change(toggle_backend_options, inputs=[backend],
                       outputs=[optim_level, niter2, matching_conf_thr,
                                shared_intrinsics, schedule, niter1,
                                scenegraph_type, winsize, refid, vggt_ensemble,
                                vggt_equirect])

        scenegraph_type.change(set_scenegraph_options,
                               inputs=[inputfiles, winsize, refid, scenegraph_type],
                               outputs=[winsize, refid])
        inputfiles.change(set_scenegraph_options,
                          inputs=[inputfiles, winsize, refid, scenegraph_type],
                          outputs=[winsize, refid])

        run_btn.click(
            fn=reconstruct,
            inputs=[inputfiles, backend, optim_level, schedule,
                    niter1, niter2, min_conf_thr, matching_conf_thr,
                    as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                    scenegraph_type, winsize, refid, shared_intrinsics,
                    vggt_ensemble, vggt_equirect],
            outputs=[scene_state, outmodel]
        )

        for ctrl in [min_conf_thr, cam_size, as_pointcloud, mask_sky, clean_depth, transparent_cams]:
            trigger = ctrl.release if hasattr(ctrl, 'release') and isinstance(ctrl, gr.Slider) else ctrl.change
            trigger(fn=update_preview,
                    inputs=[scene_state, min_conf_thr, as_pointcloud, mask_sky,
                            clean_depth, transparent_cams, cam_size],
                    outputs=outmodel)

        export_btn.click(fn=export_colmap,
                         inputs=[scene_state, min_conf_thr, clean_depth],
                         outputs=export_file)

        export_mesh_btn.click(fn=export_dense_mesh,
                              inputs=[scene_state, min_conf_thr, clean_depth],
                              outputs=export_mesh_file)

        refine_btn.click(fn=refine_mesh_fn,
                         inputs=[scene_state, min_conf_thr, clean_depth,
                                 refine_iters, refine_compare,
                                 refine_depth_reg, refine_smooth_reg],
                         outputs=refine_output)

        train_btn.click(fn=export_and_train,
                        inputs=[scene_state, min_conf_thr, clean_depth,
                                train_iters, train_target_splats,
                                train_depth_lambda, train_aniso_lambda],
                        outputs=[train_output, train_preview])

    return demo


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="3D Reconstruction → COLMAP → 3DGS")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--server_name", type=str, default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    if args.device:
        DEVICE = args.device

    # Models load on demand when first selected
    demo = build_ui()
    demo.launch(server_name=args.server_name, server_port=args.server_port,
                share=args.share)
