"""
Export a CanonicalScene to COLMAP format for Gaussian Splatting.

All reconstruction backends produce a CanonicalScene, which this module
consumes uniformly — no backend-specific branching.

Output structure:
  <output_dir>/
    images/           # original full-resolution images
    sparse/0/
      cameras.txt     # intrinsics scaled to match original resolution
      images.txt
      points3D.txt
"""

import os
import numpy as np
from PIL import Image
from PIL.ImageOps import exif_transpose
from scipy.spatial.transform import Rotation

import sys
MAST3R_DIR = os.path.join(os.path.dirname(__file__), 'mast3r')
sys.path.insert(0, MAST3R_DIR)
sys.path.insert(0, os.path.join(MAST3R_DIR, 'dust3r'))

from dust3r.utils.geometry import opencv_to_colmap_intrinsics


def rotmat_to_qvec(R):
    """Convert 3x3 rotation matrix to COLMAP quaternion (qw, qx, qy, qz)."""
    rot = Rotation.from_matrix(R)
    quat = rot.as_quat()  # (qx, qy, qz, qw)
    return np.array([quat[3], quat[0], quat[1], quat[2]])


def export_scene_to_colmap(scene, image_paths, output_dir, min_conf_thr=2.0,
                           clean_depth=True, dense_colors=None):
    """
    Export a CanonicalScene to COLMAP text format.

    Saves original full-resolution images and scales intrinsics to match.
    """
    if scene is None:
        raise ValueError("Scene is None, run reconstruction first.")

    imgs = scene.images
    intrinsics = [scene.intrinsics[i] for i in range(len(imgs))]
    cam2world = scene.c2w
    pts3d_list = scene.pts3d
    confs_list = scene.confidence

    n_images = len(imgs)

    # Create output directories
    images_dir = os.path.join(output_dir, 'images')
    sparse_dir = os.path.join(output_dir, 'sparse', '0')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    # --- Copy original images and compute scale factors ---
    image_names = []
    scale_factors = []
    original_sizes = []

    is_equirect = getattr(scene, 'equirect', None) is not None

    for i in range(n_images):
        name = f'frame_{i:05d}.jpg'
        image_names.append(name)

        if is_equirect or i >= len(image_paths):
            # Equirect scene: face images are in scene.imgs, no original files
            dust3r_H, dust3r_W = imgs[i].shape[:2]
            original_sizes.append((dust3r_W, dust3r_H))
            scale_factors.append((1.0, 1.0))
            face_rgb = (np.clip(imgs[i], 0, 1) * 255).astype(np.uint8)
            Image.fromarray(face_rgb).save(os.path.join(images_dir, name), quality=95)
        else:
            orig_img = exif_transpose(Image.open(image_paths[i])).convert('RGB')
            orig_W, orig_H = orig_img.size
            original_sizes.append((orig_W, orig_H))

            dust3r_H, dust3r_W = imgs[i].shape[:2]
            sx = orig_W / dust3r_W
            sy = orig_H / dust3r_H
            scale_factors.append((sx, sy))

            orig_img.save(os.path.join(images_dir, name), quality=95)

    # --- Compute scaled intrinsics for each image ---
    cam_params = []  # list of (orig_W, orig_H, fx, fy, cx, cy)
    for i in range(n_images):
        K = intrinsics[i].copy()
        orig_W, orig_H = original_sizes[i]

        K = scene.scale_intrinsics_to(orig_W, orig_H, i)

        K = opencv_to_colmap_intrinsics(K)
        cam_params.append((orig_W, orig_H, K[0, 0], K[1, 1], K[0, 2], K[1, 2]))

    # --- Compute poses ---
    # cam2world is always c2w (CanonicalScene convention). Invert for COLMAP.
    pose_data = []  # list of (qvec, tvec)
    for i in range(n_images):
        w2c = np.linalg.inv(cam2world[i])
        pose_data.append((rotmat_to_qvec(w2c[:3, :3]), w2c[:3, 3]))

    # --- Write TEXT format (for 3DGS tools) ---
    _write_cameras_txt(os.path.join(sparse_dir, 'cameras.txt'), cam_params, n_images)
    _write_images_txt(os.path.join(sparse_dir, 'images.txt'), pose_data, image_names, n_images)

    # --- Write BINARY format (for RealityCapture and other tools) ---
    _write_cameras_bin(os.path.join(sparse_dir, 'cameras.bin'), cam_params, n_images)
    _write_images_bin(os.path.join(sparse_dir, 'images.bin'), pose_data, image_names, n_images)

    # --- Write points3D.txt ---
    all_pts = []
    all_colors = []
    all_confs_flat = []

    # Collect all flat points first, then match colors
    dense_colors_offset = 0  # track offset into dense_colors for flat arrays

    for i in range(min(n_images, len(pts3d_list))):
        pts = pts3d_list[i]
        conf = confs_list[i] if i < len(confs_list) else None

        if conf is None:
            continue

        if pts.ndim == 3 and i < len(imgs) and pts.shape[:2] == imgs[i].shape[:2]:
            # Dense per-view pointmap — extract colors from corresponding image
            mask = conf > min_conf_thr
            all_pts.append(pts[mask])
            all_colors.append((np.clip(imgs[i][mask], 0, 1) * 255).astype(np.uint8))
            all_confs_flat.append(conf[mask])
        elif pts.ndim == 2:
            # Flat array (decimated/densified cloud)
            mask = conf.ravel() > min_conf_thr if conf is not None else np.ones(len(pts), dtype=bool)
            all_pts.append(pts[mask])
            # Use dense_colors if available
            if dense_colors is not None and len(dense_colors) >= dense_colors_offset + len(pts):
                cols = dense_colors[dense_colors_offset:dense_colors_offset + len(pts)][mask]
                if cols.dtype != np.uint8:
                    if cols.max() <= 1.0:
                        cols = (np.clip(cols, 0, 1) * 255).astype(np.uint8)
                    else:
                        cols = np.clip(cols, 0, 255).astype(np.uint8)
                all_colors.append(cols)
            elif dense_colors is not None and len(dense_colors) == len(pts):
                cols = dense_colors[mask]
                if cols.dtype != np.uint8:
                    if cols.max() <= 1.0:
                        cols = (np.clip(cols, 0, 1) * 255).astype(np.uint8)
                    else:
                        cols = np.clip(cols, 0, 255).astype(np.uint8)
                all_colors.append(cols)
            else:
                all_colors.append(np.full((mask.sum(), 3), 128, dtype=np.uint8))
            all_confs_flat.append(conf.ravel()[mask] if conf is not None else np.ones(mask.sum()))
            dense_colors_offset += len(pts)
        elif pts.ndim == 3:
            # Dense pointmap but no matching image — extract colors if possible
            mask = conf.reshape(pts.shape[:2]) > min_conf_thr
            all_pts.append(pts[mask])
            all_colors.append(np.full((mask.sum(), 3), 180, dtype=np.uint8))
            all_confs_flat.append(conf.reshape(pts.shape[:2])[mask])
        else:
            continue

    if len(all_pts) > 0:
        all_pts = np.concatenate(all_pts, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)
        all_confs_flat = np.concatenate(all_confs_flat, axis=0)
    else:
        all_pts = np.zeros((0, 3))
        all_colors = np.zeros((0, 3), dtype=np.uint8)
        all_confs_flat = np.zeros((0,))

    # Keep the full dense cloud for PLY export
    if len(all_pts) > 0:
        dense_pts = np.concatenate([all_pts] if isinstance(all_pts, np.ndarray) else all_pts, axis=0) if not isinstance(all_pts, np.ndarray) else all_pts
        dense_colors = np.concatenate([all_colors] if isinstance(all_colors, np.ndarray) else all_colors, axis=0) if not isinstance(all_colors, np.ndarray) else all_colors
    else:
        dense_pts = np.zeros((0, 3))
        dense_colors = np.zeros((0, 3), dtype=np.uint8)

    all_pts = np.concatenate(all_pts, axis=0) if not isinstance(all_pts, np.ndarray) else all_pts
    all_colors = np.concatenate(all_colors, axis=0) if not isinstance(all_colors, np.ndarray) else all_colors
    all_confs_flat = np.concatenate(all_confs_flat, axis=0) if not isinstance(all_confs_flat, np.ndarray) else all_confs_flat

    # Sparse version for COLMAP points3D (downsampled for compatibility)
    sparse_pts, sparse_colors = all_pts, all_colors
    if len(sparse_pts) > 0:
        sparse_pts, sparse_colors = _voxel_downsample(sparse_pts.copy(), sparse_colors.copy(), all_confs_flat)
    max_points = 500_000
    if len(sparse_pts) > max_points:
        indices = np.random.choice(len(sparse_pts), max_points, replace=False)
        sparse_pts = sparse_pts[indices]
        sparse_colors = sparse_colors[indices]

    # Write sparse points3D (for COLMAP/3DGS)
    _write_points3d_txt(os.path.join(sparse_dir, 'points3D.txt'), sparse_pts, sparse_colors)
    _write_points3d_bin(os.path.join(sparse_dir, 'points3D.bin'), sparse_pts, sparse_colors)

    # --- Write Bundler format (for RealityCapture) ---
    _write_bundler(
        os.path.join(output_dir, 'bundle.out'),
        os.path.join(output_dir, 'image_list.txt'),
        cam_params, pose_data, image_names, sparse_pts, sparse_colors
    )

    # --- Write XMP sidecar files (for RealityCapture auto-detection) ---
    _write_xmp_sidecars(images_dir, image_names, cam_params, cam2world)

    # --- Write dense PLY (full reconstruction, for RC import) ---
    _write_ply(os.path.join(output_dir, 'dense_point_cloud.ply'), dense_pts, dense_colors)

    # --- Write sparse PLY (downsampled, for quick preview) ---
    _write_ply(os.path.join(output_dir, 'point_cloud.ply'), sparse_pts, sparse_colors)

    print(f"COLMAP dataset exported to {output_dir}")
    print(f"  Formats: COLMAP txt+bin, Bundler, XMP, PLY")
    print(f"  - {n_images} cameras/images (original resolution)")
    for i in range(n_images):
        dust3r_H, dust3r_W = imgs[i].shape[:2]
        orig_W, orig_H = original_sizes[i]
        orient = "landscape" if orig_W >= orig_H else "portrait"
        W_c, H_c, fx, fy, cx, cy = cam_params[i]
        print(f"    image {i}: {dust3r_W}x{dust3r_H} -> {orig_W}x{orig_H} [{orient}] "
              f"(fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f})")
    print(f"  - {len(all_pts)} 3D points")
    return output_dir


import struct


# ── PLY point cloud writer ────────────────────────────────────────────────────

def _write_ply(path, pts, colors):
    """Write a simple colored PLY point cloud."""
    n = len(pts)
    header = f"""ply
format binary_little_endian 1.0
element vertex {n}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        for j in range(n):
            f.write(struct.pack('<3f', *pts[j].astype(np.float32)))
            f.write(struct.pack('<3B', *colors[j].astype(np.uint8)))
    print(f"  PLY point cloud: {n:,d} points -> {path}")


# ── XMP sidecar writer (for RealityCapture) ──────────────────────────────────

def _write_xmp_sidecars(images_dir, image_names, cam_params, cam2world):
    """
    Write XMP sidecar files next to each image.
    RealityCapture automatically reads these when images are added.

    xcr:Rotation = world-to-camera rotation matrix (row-major, 9 values)
    xcr:Position = camera center in world coordinates
    """
    for i, name in enumerate(image_names):
        W, H, fx, fy, cx, cy = cam_params[i]
        c2w = cam2world[i]

        # Camera center in world space
        pos = c2w[:3, 3]

        # RC expects world-to-camera rotation (transpose of c2w rotation)
        w2c_R = c2w[:3, :3].T

        # Focal length in 35mm equivalent
        focal_35mm = float(fx) * 36.0 / float(W)

        # Principal point as offset from center, normalized by image size
        pp_u = float(cx) / float(W) - 0.5
        pp_v = float(cy) / float(H) - 0.5

        r = w2c_R
        rot_str = f"{r[0,0]:.10f} {r[0,1]:.10f} {r[0,2]:.10f} {r[1,0]:.10f} {r[1,1]:.10f} {r[1,2]:.10f} {r[2,0]:.10f} {r[2,1]:.10f} {r[2,2]:.10f}"
        pos_str = f"{pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}"

        xmp_content = (
            '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
            '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            '<rdf:Description '
            'xmlns:xcr="http://www.capturingreality.com/ns/xcr/1.1#" '
            f'xcr:Version="3" '
            f'xcr:PosePrior="locked" '
            f'xcr:Coordinates="local" '
            f'xcr:DistortionModel="brown3" '
            f'xcr:FocalLength35mm="{focal_35mm:.4f}" '
            f'xcr:Skew="0" '
            f'xcr:AspectRatio="1" '
            f'xcr:PrincipalPointU="{pp_u:.6f}" '
            f'xcr:PrincipalPointV="{pp_v:.6f}" '
            f'xcr:CalibrationPrior="locked">'
            f'<xcr:Rotation>{rot_str}</xcr:Rotation>'
            f'<xcr:Position>{pos_str}</xcr:Position>'
            '</rdf:Description>'
            '</rdf:RDF>'
            '</x:xmpmeta>'
        )

        xmp_path = os.path.join(images_dir, os.path.splitext(name)[0] + '.xmp')
        with open(xmp_path, 'wb') as f:
            f.write(xmp_content.encode('ascii'))

    print(f"  XMP sidecars written for {len(image_names)} images")


# ── Bundler format writer (for RealityCapture) ──────────────────────────────

def _write_bundler(bundle_path, imglist_path, cam_params, pose_data, image_names,
                   pts3d, colors):
    """
    Write Bundler .out format + image_list.txt.

    RealityCapture import workflow:
      1. Add images to project
      2. Import Metadata > select bundle.out
      3. When prompted, select image_list.txt

    Bundler format:
      - Uses a single focal length (average of fx/fy)
      - Rotation is world-to-camera (3x3 matrix, row-major)
      - Translation is -R * camera_center
      - Y and Z axes are negated vs. COLMAP convention
    """
    n_cams = len(cam_params)
    n_pts = len(pts3d)

    # image_list.txt — one image path per line
    with open(imglist_path, 'w') as f:
        for name in image_names:
            f.write(f"images/{name}\n")

    with open(bundle_path, 'w') as f:
        # Header
        f.write("# Bundle file v0.3\n")
        f.write(f"{n_cams} {n_pts}\n")

        # Cameras
        for i in range(n_cams):
            W, H, fx, fy, cx, cy = cam_params[i]
            qvec, tvec = pose_data[i]

            # Bundler uses a single focal length
            focal = (fx + fy) / 2.0

            # Radial distortion (k1, k2) — we have none
            k1, k2 = 0.0, 0.0

            # Rotation: Bundler uses w2c rotation but with Y,Z negated
            # COLMAP qvec -> rotation matrix
            R = Rotation.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()

            # Bundler convention: negate Y and Z rows
            R[1, :] *= -1
            R[2, :] *= -1

            # Translation: Bundler t = -R * C, but also negate Y,Z
            t = tvec.copy()
            t[1] *= -1
            t[2] *= -1

            f.write(f"{focal:.6f} {k1:.6f} {k2:.6f}\n")
            f.write(f"{R[0,0]:.6f} {R[0,1]:.6f} {R[0,2]:.6f}\n")
            f.write(f"{R[1,0]:.6f} {R[1,1]:.6f} {R[1,2]:.6f}\n")
            f.write(f"{R[2,0]:.6f} {R[2,1]:.6f} {R[2,2]:.6f}\n")
            f.write(f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f}\n")

        # Points
        for j in range(n_pts):
            x, y, z = pts3d[j].astype(float)
            r, g, b = colors[j].astype(int)

            # Position (negate Y,Z for Bundler convention)
            f.write(f"{x:.6f} {-y:.6f} {-z:.6f}\n")
            # Color
            f.write(f"{r} {g} {b}\n")
            # View list (empty — no tracks)
            f.write("0\n")

    print(f"  Bundler export: {bundle_path}")


# ── Text format writers ──────────────────────────────────────────────────────

def _write_cameras_txt(path, cam_params, n):
    with open(path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {n}\n")
        for i, (W, H, fx, fy, cx, cy) in enumerate(cam_params):
            f.write(f"{i + 1} PINHOLE {W} {H} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")


def _write_images_txt(path, pose_data, image_names, n):
    with open(path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {n}\n")
        for i, (qvec, tvec) in enumerate(pose_data):
            f.write(f"{i + 1} {qvec[0]:.10f} {qvec[1]:.10f} {qvec[2]:.10f} {qvec[3]:.10f} "
                    f"{tvec[0]:.10f} {tvec[1]:.10f} {tvec[2]:.10f} {i + 1} {image_names[i]}\n")
            f.write("\n")


def _write_points3d_txt(path, pts, colors):
    with open(path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(pts)}\n")
        for j in range(len(pts)):
            x, y, z = pts[j]
            r, g, b = colors[j]
            f.write(f"{j + 1} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 0.0\n")


# ── Binary format writers (COLMAP binary convention) ─────────────────────────
# See: https://colmap.github.io/format.html#binary-file-format

# Camera model IDs: PINHOLE = 1
COLMAP_PINHOLE_ID = 1

def _write_cameras_bin(path, cam_params, n):
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', n))  # num cameras
        for i, (W, H, fx, fy, cx, cy) in enumerate(cam_params):
            cam_id = i + 1
            f.write(struct.pack('<i', cam_id))          # camera_id
            f.write(struct.pack('<i', COLMAP_PINHOLE_ID))  # model_id (PINHOLE=1)
            f.write(struct.pack('<Q', W))               # width
            f.write(struct.pack('<Q', H))               # height
            # PINHOLE params: fx, fy, cx, cy
            f.write(struct.pack('<4d', fx, fy, cx, cy))


def _write_images_bin(path, pose_data, image_names, n):
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', n))  # num images
        for i, (qvec, tvec) in enumerate(pose_data):
            img_id = i + 1
            cam_id = i + 1
            f.write(struct.pack('<i', img_id))                  # image_id
            f.write(struct.pack('<4d', *qvec.tolist()))         # qw, qx, qy, qz
            f.write(struct.pack('<3d', *tvec.tolist()))         # tx, ty, tz
            f.write(struct.pack('<i', cam_id))                  # camera_id
            # Image name as null-terminated string
            name_bytes = image_names[i].encode('utf-8') + b'\x00'
            f.write(name_bytes)
            # Number of 2D points (0 — we don't have tracks)
            f.write(struct.pack('<Q', 0))


def _write_points3d_bin(path, pts, colors):
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(pts)))  # num points
        for j in range(len(pts)):
            x, y, z = pts[j].astype(np.float64)
            r, g, b = colors[j].astype(np.uint8)
            pt_id = j + 1
            f.write(struct.pack('<Q', pt_id))           # point3d_id
            f.write(struct.pack('<3d', x, y, z))        # xyz
            f.write(struct.pack('<3B', r, g, b))        # rgb
            f.write(struct.pack('<d', 0.0))             # error
            f.write(struct.pack('<Q', 0))               # track length (0)


def _voxel_downsample(pts, colors, confs, voxel_size=None):
    """Voxel-grid downsample: keep highest-confidence point per voxel."""
    if len(pts) == 0:
        return pts, colors

    if voxel_size is None:
        extent = pts.max(axis=0) - pts.min(axis=0)
        volume = np.prod(extent + 1e-8)
        target_points = 150_000
        voxel_size = (volume / target_points) ** (1.0 / 3.0)
        scene_scale = np.linalg.norm(extent)
        voxel_size = np.clip(voxel_size, scene_scale * 1e-4, scene_scale * 0.01)

    mins = pts.min(axis=0)
    voxel_indices = ((pts - mins) / voxel_size).astype(np.int32)

    keys = (voxel_indices[:, 0].astype(np.int64) * 73856093 ^
            voxel_indices[:, 1].astype(np.int64) * 19349669 ^
            voxel_indices[:, 2].astype(np.int64) * 83492791)

    unique_keys = np.unique(keys)
    selected = np.empty(len(unique_keys), dtype=np.int64)
    for idx, key in enumerate(unique_keys):
        voxel_mask = keys == key
        voxel_indices_arr = np.where(voxel_mask)[0]
        best = voxel_indices_arr[np.argmax(confs[voxel_mask])]
        selected[idx] = best

    print(f"  Voxel downsample: {len(pts)} -> {len(selected)} points (voxel_size={voxel_size:.6f})")
    return pts[selected], colors[selected]


# ── COLLADA (.dae) camera exporter (for Softimage XSI / Crosswalk) ───────────

def export_cameras_to_collada(scene, output_path, image_paths=None):
    """
    Export cameras from a CanonicalScene to COLLADA (.dae) format.

    Works with Softimage XSI (via Crosswalk), Blender, Maya, etc.
    Converts from OpenCV convention (y-down, z-forward) to COLLADA Y-up.
    Camera names are derived from source image filenames when available.
    """
    import math
    import re

    if scene is None:
        raise ValueError("Scene is None, run reconstruction first.")

    n = len(scene.images)
    cam2world = scene.c2w
    intrinsics = scene.intrinsics

    # Derive camera names from image filenames
    cam_names = []
    for i in range(n):
        if image_paths and i < len(image_paths):
            stem = os.path.splitext(os.path.basename(image_paths[i]))[0]
            # Sanitize for XML id: replace non-alphanumeric with underscore
            safe = re.sub(r'[^A-Za-z0-9_.-]', '_', stem)
            cam_names.append(safe)
        else:
            cam_names.append(f"camera_{i:03d}")

    # OpenCV -> Y-up flip: diag(1, -1, -1, 1)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])

    lines = []
    lines.append('<?xml version="1.0" encoding="utf-8"?>')
    lines.append('<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">')
    lines.append('  <asset>')
    lines.append('    <up_axis>Y_UP</up_axis>')
    lines.append('  </asset>')

    # Library: camera optics
    lines.append('  <library_cameras>')
    for i in range(n):
        K = intrinsics[i]
        fx, fy = K[0, 0], K[1, 1]
        H, W = scene.images[i].shape[:2]
        yfov_rad = 2.0 * math.atan2(H, 2.0 * fy)
        yfov_deg = math.degrees(yfov_rad)
        aspect = W / H
        name = cam_names[i]

        lines.append(f'    <camera id="{name}_cam" name="{name}">')
        lines.append(f'      <optics>')
        lines.append(f'        <technique_common>')
        lines.append(f'          <perspective>')
        lines.append(f'            <yfov>{yfov_deg:.6f}</yfov>')
        lines.append(f'            <aspect_ratio>{aspect:.6f}</aspect_ratio>')
        lines.append(f'            <znear>0.01</znear>')
        lines.append(f'            <zfar>10000</zfar>')
        lines.append(f'          </perspective>')
        lines.append(f'        </technique_common>')
        lines.append(f'      </optics>')
        lines.append(f'    </camera>')
    lines.append('  </library_cameras>')

    # Visual scene: camera nodes with transforms
    lines.append('  <library_visual_scenes>')
    lines.append('    <visual_scene id="Scene" name="Scene">')
    for i in range(n):
        c2w_dae = flip @ cam2world[i] @ flip
        # COLLADA matrix is row-major in the XML
        m = c2w_dae
        mat_str = ' '.join(f'{m[r, c]:.10f}' for r in range(4) for c in range(4))
        name = cam_names[i]

        lines.append(f'      <node id="{name}_node" name="{name}" type="NODE">')
        lines.append(f'        <matrix sid="transform">{mat_str}</matrix>')
        lines.append(f'        <instance_camera url="#{name}_cam"/>')
        lines.append(f'      </node>')
    lines.append('    </visual_scene>')
    lines.append('  </library_visual_scenes>')

    lines.append('  <scene>')
    lines.append('    <instance_visual_scene url="#Scene"/>')
    lines.append('  </scene>')
    lines.append('</COLLADA>')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"COLLADA cameras exported: {n} cameras -> {output_path}")


def export_mesh_obj_yup(verts, faces, colors, output_path):
    """
    Export a vertex-colored mesh to OBJ with the same Y-up coordinate
    transform used by export_cameras_to_collada, so they align in XSI.

    Applies flip: (x, y, z) -> (x, -y, -z)
    """
    n_v = len(verts)
    n_f = len(faces)
    has_colors = colors is not None and len(colors) == n_v

    with open(output_path, 'w') as f:
        f.write(f"# OBJ export (Y-up, aligned with COLLADA cameras)\n")
        f.write(f"# {n_v} vertices, {n_f} faces\n")
        for i in range(n_v):
            x, y, z = float(verts[i, 0]), float(-verts[i, 1]), float(-verts[i, 2])
            if has_colors:
                r, g, b = colors[i]
                if not isinstance(r, float) or r > 1.0:
                    r, g, b = r / 255.0, g / 255.0, b / 255.0
                f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.4f} {g:.4f} {b:.4f}\n")
            else:
                f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for i in range(n_f):
            a, b, c = faces[i] + 1  # OBJ is 1-indexed
            f.write(f"f {a} {b} {c}\n")

    print(f"OBJ mesh exported (Y-up): {n_v} verts, {n_f} faces -> {output_path}")
