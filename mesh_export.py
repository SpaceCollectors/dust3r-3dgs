"""
TSDF Fusion: Create a dense mesh from depth maps + camera poses.
No external tools needed — uses the depth maps from dust3r/mast3r/vggt directly.
"""

import numpy as np
import torch

import sys, os
MAST3R_DIR = os.path.join(os.path.dirname(__file__), 'mast3r')
sys.path.insert(0, MAST3R_DIR)
sys.path.insert(0, os.path.join(MAST3R_DIR, 'dust3r'))
from dust3r.utils.device import to_numpy


def tsdf_fusion(imgs, pts3d_list, confs_list, min_conf=2.0,
                voxel_size=None, trunc_factor=5.0):
    """
    TSDF volumetric fusion from dense point maps.

    Args:
        imgs: list of (H,W,3) float [0,1]
        pts3d_list: list of (H,W,3) numpy — world-frame points per view
        confs_list: list of (H,W) numpy — confidence per pixel
        min_conf: minimum confidence threshold
        voxel_size: voxel size (auto-computed if None)
        trunc_factor: truncation distance = trunc_factor * voxel_size

    Returns:
        vertices: (M, 3) numpy
        faces: (F, 3) numpy
        colors: (M, 3) numpy uint8
    """
    # Gather all valid points to determine volume bounds
    all_pts = []
    for i in range(len(imgs)):
        mask = confs_list[i] > min_conf
        all_pts.append(pts3d_list[i][mask])

    all_pts = np.concatenate(all_pts, axis=0)
    if len(all_pts) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int), np.zeros((0, 3), dtype=np.uint8)

    # Volume bounds
    pts_min = all_pts.min(axis=0)
    pts_max = all_pts.max(axis=0)
    extent = pts_max - pts_min

    # Auto voxel size: aim for ~256^3 volume max
    if voxel_size is None:
        voxel_size = float(max(extent)) / 256.0
        voxel_size = max(voxel_size, 1e-6)

    trunc_dist = trunc_factor * voxel_size

    # Pad bounds
    pad = trunc_dist * 2
    vol_min = pts_min - pad
    vol_max = pts_max + pad

    # Grid dimensions
    dims = np.ceil((vol_max - vol_min) / voxel_size).astype(int)
    dims = np.clip(dims, 1, 512)  # cap at 512^3

    print(f"  TSDF volume: {dims[0]}x{dims[1]}x{dims[2]}, voxel_size={voxel_size:.6f}")

    # Initialize TSDF volume
    tsdf = np.ones(dims, dtype=np.float32)  # 1.0 = empty
    weights = np.zeros(dims, dtype=np.float32)
    color_vol = np.zeros((*dims, 3), dtype=np.float32)

    # Integrate each view
    for v in range(len(imgs)):
        print(f"  Integrating view {v+1}/{len(imgs)}...")
        pts = pts3d_list[v]  # (H, W, 3)
        conf = confs_list[v]  # (H, W)
        img = imgs[v]  # (H, W, 3)
        H, W = pts.shape[:2]

        mask = conf > min_conf

        # For each valid pixel, update the TSDF
        valid_pts = pts[mask]  # (N, 3)
        valid_colors = img[mask]  # (N, 3)
        valid_confs = conf[mask]  # (N,)

        # Voxel indices for each point
        voxel_idx = ((valid_pts - vol_min) / voxel_size).astype(int)

        # Clamp to volume
        for d in range(3):
            voxel_idx[:, d] = np.clip(voxel_idx[:, d], 0, dims[d] - 1)

        # Simple nearest-voxel integration (fast approximation of TSDF)
        # For each point, set the corresponding voxel to "surface"
        ix, iy, iz = voxel_idx[:, 0], voxel_idx[:, 1], voxel_idx[:, 2]
        w = valid_confs

        # Update with weighted average
        old_w = weights[ix, iy, iz]
        new_w = old_w + w
        safe_w = np.where(new_w > 0, new_w, 1.0)

        tsdf[ix, iy, iz] = (tsdf[ix, iy, iz] * old_w + 0.0 * w) / safe_w
        color_vol[ix, iy, iz] = (color_vol[ix, iy, iz] * old_w[:, None] +
                                  valid_colors * w[:, None]) / safe_w[:, None]
        weights[ix, iy, iz] = new_w

    # Extract surface: voxels that received points
    occupied = weights > 0

    # Convert to mesh using marching cubes on the weight field
    try:
        from skimage.measure import marching_cubes
        # Use weights as the volume — surface is where weights > threshold
        volume = weights.copy()
        volume[volume == 0] = -1  # empty space
        threshold = np.median(volume[volume > 0]) * 0.5 if (volume > 0).any() else 0

        verts, faces, _, _ = marching_cubes(volume, level=threshold)
        verts = verts * voxel_size + vol_min

        # Color vertices by nearest voxel
        vi = ((verts - vol_min) / voxel_size).astype(int)
        for d in range(3):
            vi[:, d] = np.clip(vi[:, d], 0, dims[d] - 1)
        vert_colors = (color_vol[vi[:, 0], vi[:, 1], vi[:, 2]] * 255).astype(np.uint8)

        print(f"  Mesh: {len(verts):,d} vertices, {len(faces):,d} faces")
        return verts, faces, vert_colors

    except ImportError:
        print("  scikit-image not available, exporting as point cloud only")
        # Fallback: return point cloud (no mesh)
        occ_idx = np.where(occupied)
        verts = np.stack(occ_idx, axis=-1).astype(np.float32) * voxel_size + vol_min
        vert_colors = (color_vol[occ_idx] * 255).astype(np.uint8)
        return verts, np.zeros((0, 3), dtype=int), vert_colors


def save_mesh_ply(path, vertices, faces, colors):
    """Save mesh as PLY."""
    import struct
    n_verts = len(vertices)
    n_faces = len(faces)

    header = f"""ply
format binary_little_endian 1.0
element vertex {n_verts}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face {n_faces}
property list uchar int vertex_indices
end_header
"""
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        for i in range(n_verts):
            f.write(struct.pack('<3f', *vertices[i].astype(np.float32)))
            f.write(struct.pack('<3B', *colors[i].astype(np.uint8)))
        for i in range(n_faces):
            f.write(struct.pack('<B3i', 3, *faces[i].astype(np.int32)))

    print(f"  Saved mesh: {n_verts:,d} verts, {n_faces:,d} faces -> {path}")


def save_dense_ply(path, pts, colors):
    """Save dense point cloud as PLY."""
    import struct
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
        # Write all at once for speed
        verts = pts.astype(np.float32)
        cols = colors.astype(np.uint8)
        for i in range(n):
            f.write(struct.pack('<3f', *verts[i]))
            f.write(struct.pack('<3B', *cols[i]))

    print(f"  Saved dense PLY: {n:,d} points -> {path}")
