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
sys.path.insert(0, MAST3R_DIR)
sys.path.insert(0, os.path.join(MAST3R_DIR, 'dust3r'))
sys.path.insert(0, VGGT_DIR)

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


def load_model(backend):
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
                scenegraph_type, winsize, refid, shared_intrinsics):
    global _original_paths

    if not filelist or len(filelist) == 0:
        raise gr.Error("Please upload at least 2 images.")

    if isinstance(filelist[0], str):
        paths = filelist
    else:
        paths = [f.name if hasattr(f, 'name') else f for f in filelist]

    _original_paths = list(paths)

    if backend == 'vggt':
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
                                scenegraph_type, winsize, refid])

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
                    scenegraph_type, winsize, refid, shared_intrinsics],
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
