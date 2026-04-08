"""
Multi-View Photoconsistency Mesh Refinement via Differentiable Rendering

For each iteration:
  1. Render the mesh from each camera using differentiable rasterization
  2. Compare rendered image vs actual photo → error map
  3. Backprop pixel errors through the renderer to vertex positions
  4. Vertices that cause visual artifacts get moved to reduce the error

This is fundamentally different from per-vertex projection:
  - Error is computed in IMAGE SPACE (where we can see exactly what's wrong)
  - Gradients flow through the rasterizer to vertex positions
  - The error heatmap tells us WHERE the geometry is incorrect
"""

import os
import sys
import time
import argparse
import struct
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation


# ── Mesh I/O ─────────────────────────────────────────────────────────────────

def load_ply_mesh(path):
    import re
    with open(path, 'rb') as f:
        header = b''
        while True:
            line = f.readline()
            header += line
            if b'end_header' in line:
                break
        header_str = header.decode('ascii')
        n_verts = int(re.search(r'element vertex (\d+)', header_str).group(1))
        face_match = re.search(r'element face (\d+)', header_str)
        n_faces = int(face_match.group(1)) if face_match else 0

        verts = np.zeros((n_verts, 3), dtype=np.float32)
        colors = np.zeros((n_verts, 3), dtype=np.uint8)
        for i in range(n_verts):
            data = struct.unpack('<3f3B', f.read(15))
            verts[i] = data[:3]
            colors[i] = data[3:6]

        faces = np.zeros((n_faces, 3), dtype=np.int32)
        for i in range(n_faces):
            n = struct.unpack('<B', f.read(1))[0]
            idx = struct.unpack(f'<{n}i', f.read(4 * n))
            if n >= 3:
                faces[i] = idx[:3]
    return verts, faces, colors


def save_ply_mesh(path, verts, faces, colors):
    n_verts, n_faces = len(verts), len(faces)
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
            f.write(struct.pack('<3f', *verts[i].astype(np.float32)))
            f.write(struct.pack('<3B', *colors[i].astype(np.uint8)))
        for i in range(n_faces):
            f.write(struct.pack('<B3i', 3, *faces[i].astype(np.int32)))


# ── COLMAP Loading ───────────────────────────────────────────────────────────

def load_cameras(data_dir):
    sparse_dir = os.path.join(data_dir, 'sparse', '0')
    images_dir = os.path.join(data_dir, 'images')

    cameras = {}
    with open(os.path.join(sparse_dir, 'cameras.txt')) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            p = line.strip().split()
            cameras[int(p[0])] = (int(p[2]), int(p[3]),
                                   float(p[4]), float(p[5]), float(p[6]), float(p[7]))

    views = []
    with open(os.path.join(sparse_dir, 'images.txt')) as f:
        lines = [l.strip() for l in f if not l.startswith('#')]
    for line in [l for l in lines if l and len(l.split()) >= 9]:
        p = line.split()
        qw, qx, qy, qz = float(p[1]), float(p[2]), float(p[3]), float(p[4])
        tx, ty, tz = float(p[5]), float(p[6]), float(p[7])
        cam = cameras[int(p[8])]
        W, H, fx, fy, cx, cy = cam

        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R
        w2c[:3, 3] = [tx, ty, tz]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        img = np.array(Image.open(os.path.join(images_dir, p[9])).convert('RGB'),
                       dtype=np.float32) / 255.0
        views.append({'w2c': w2c, 'K': K, 'W': W, 'H': H, 'pixels': img})
    return views


# ── Differentiable Mesh Renderer ─────────────────────────────────────────────
# Renders a textured mesh by:
#   1. Project all face vertices to 2D
#   2. For each pixel, find which face covers it (z-buffer via splatting)
#   3. Compute barycentric coords → interpolate vertex colors
#   4. All differentiable w.r.t vertex positions

def render_mesh(verts, faces, vert_colors, w2c, K, W, H, gt_image):
    """
    Render mesh and compute per-pixel error against ground truth.
    Uses a soft rasterization approach: each face contributes to nearby pixels
    weighted by proximity, making gradients smooth.

    Returns:
        rendered: (H, W, 3) rendered image
        error_map: (H, W) per-pixel L1 error
        loss: scalar photometric loss
    """
    device = verts.device

    # Project vertices
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    verts_cam = verts @ R.T + t[None, :]
    z = verts_cam[:, 2].clamp(min=0.001)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    verts_2d = torch.stack([
        verts_cam[:, 0] / z * fx + cx,
        verts_cam[:, 1] / z * fy + cy,
    ], dim=-1)  # (V, 2)

    # Get face vertex positions and colors
    v0 = verts_2d[faces[:, 0]]  # (F, 2)
    v1 = verts_2d[faces[:, 1]]
    v2 = verts_2d[faces[:, 2]]
    z0 = z[faces[:, 0]]  # (F,)
    z1 = z[faces[:, 1]]
    z2 = z[faces[:, 2]]
    c0 = vert_colors[faces[:, 0]]  # (F, 3)
    c1 = vert_colors[faces[:, 1]]
    c2 = vert_colors[faces[:, 2]]

    # Downscale for speed
    scale = max(1, max(W, H) // 256)
    rW, rH = W // scale, H // scale

    # Pixel grid
    py = torch.arange(rH, device=device, dtype=torch.float32) * scale + scale / 2
    px = torch.arange(rW, device=device, dtype=torch.float32) * scale + scale / 2
    gy, gx = torch.meshgrid(py, px, indexing='ij')
    pixels = torch.stack([gx, gy], dim=-1)  # (rH, rW, 2)

    # Initialize buffers
    rendered = torch.zeros(rH, rW, 3, device=device)
    zbuf = torch.full((rH, rW), float('inf'), device=device)

    # Process faces in batches for memory
    BATCH = 2048
    n_faces = faces.shape[0]

    for fi in range(0, n_faces, BATCH):
        fe = min(fi + BATCH, n_faces)
        bv0, bv1, bv2 = v0[fi:fe], v1[fi:fe], v2[fi:fe]
        bz0, bz1, bz2 = z0[fi:fe], z1[fi:fe], z2[fi:fe]
        bc0, bc1, bc2 = c0[fi:fe], c1[fi:fe], c2[fi:fe]
        B = fe - fi

        # Bounding box per face
        face_min_x = torch.stack([bv0[:, 0], bv1[:, 0], bv2[:, 0]], dim=-1).min(dim=-1).values
        face_max_x = torch.stack([bv0[:, 0], bv1[:, 0], bv2[:, 0]], dim=-1).max(dim=-1).values
        face_min_y = torch.stack([bv0[:, 1], bv1[:, 1], bv2[:, 1]], dim=-1).min(dim=-1).values
        face_max_y = torch.stack([bv0[:, 1], bv1[:, 1], bv2[:, 1]], dim=-1).max(dim=-1).values

        # For each face, find pixels in its bbox and test barycentric coords
        for b in range(B):
            x_lo = max(0, int((face_min_x[b].item() - scale) / scale))
            x_hi = min(rW, int((face_max_x[b].item() + scale) / scale) + 1)
            y_lo = max(0, int((face_min_y[b].item() - scale) / scale))
            y_hi = min(rH, int((face_max_y[b].item() + scale) / scale) + 1)

            if x_lo >= x_hi or y_lo >= y_hi:
                continue

            # Pixels in bbox
            tile_px = pixels[y_lo:y_hi, x_lo:x_hi]  # (th, tw, 2)
            th, tw = tile_px.shape[:2]

            # Barycentric coordinates
            p = tile_px.reshape(-1, 2)  # (N, 2)
            a, bb, c = bv0[b], bv1[b], bv2[b]  # (2,) each

            v0v1 = bb - a
            v0v2 = c - a
            v0p = p - a[None, :]

            d00 = (v0v1 * v0v1).sum()
            d01 = (v0v1 * v0v2).sum()
            d11 = (v0v2 * v0v2).sum()
            d20 = (v0p * v0v1[None, :]).sum(dim=-1)
            d21 = (v0p * v0v2[None, :]).sum(dim=-1)

            denom = d00 * d11 - d01 * d01
            if abs(denom.item()) < 1e-10:
                continue
            inv_denom = 1.0 / denom
            bary_v = (d11 * d20 - d01 * d21) * inv_denom
            bary_w = (d00 * d21 - d01 * d20) * inv_denom
            bary_u = 1.0 - bary_v - bary_w

            # Inside triangle
            inside = (bary_u >= -0.001) & (bary_v >= -0.001) & (bary_w >= -0.001)

            if not inside.any():
                continue

            # Interpolate depth and color
            face_z = bary_u * bz0[b] + bary_v * bz1[b] + bary_w * bz2[b]  # (N,)
            face_color = (bary_u[:, None] * bc0[b][None, :] +
                          bary_v[:, None] * bc1[b][None, :] +
                          bary_w[:, None] * bc2[b][None, :])  # (N, 3)

            # Z-buffer test: update pixels where this face is closer
            face_z_2d = face_z.reshape(th, tw)
            inside_2d = inside.reshape(th, tw)
            face_color_2d = face_color.reshape(th, tw, 3)

            tile_z = zbuf[y_lo:y_hi, x_lo:x_hi]
            closer = inside_2d & (face_z_2d < tile_z)

            if closer.any():
                zbuf[y_lo:y_hi, x_lo:x_hi] = torch.where(closer, face_z_2d, tile_z)
                rendered[y_lo:y_hi, x_lo:x_hi] = torch.where(
                    closer.unsqueeze(-1), face_color_2d, rendered[y_lo:y_hi, x_lo:x_hi])

    # Downscale GT to match
    gt_small = F.interpolate(gt_image.permute(2, 0, 1).unsqueeze(0),
                             size=(rH, rW), mode='bilinear', align_corners=False)
    gt_small = gt_small.squeeze(0).permute(1, 2, 0)  # (rH, rW, 3)

    # Error map: only where mesh was rendered (zbuf != inf)
    has_render = zbuf < float('inf')
    error_map = torch.zeros(rH, rW, device=device)
    if has_render.any():
        pixel_error = (rendered - gt_small).abs().mean(dim=-1)  # (rH, rW)
        error_map = pixel_error * has_render.float()
        loss = pixel_error[has_render].mean()
    else:
        loss = torch.tensor(0.0, device=device)

    return rendered, error_map, loss


# ── Laplacian ────────────────────────────────────────────────────────────────

def build_laplacian(faces, n_verts, device):
    adj = [set() for _ in range(n_verts)]
    for f in faces:
        adj[f[0]].update([f[1], f[2]])
        adj[f[1]].update([f[0], f[2]])
        adj[f[2]].update([f[0], f[1]])

    max_nbrs = max((len(a) for a in adj), default=0)
    if max_nbrs == 0:
        return None, None

    nbr_idx = torch.zeros(n_verts, max_nbrs, dtype=torch.long, device=device)
    nbr_mask = torch.zeros(n_verts, max_nbrs, dtype=torch.bool, device=device)
    for vi in range(n_verts):
        nbrs = list(adj[vi])[:max_nbrs]
        if nbrs:
            nbr_idx[vi, :len(nbrs)] = torch.tensor(nbrs, device=device)
            nbr_mask[vi, :len(nbrs)] = True
    return nbr_idx, nbr_mask


def laplacian_loss(verts, nbr_idx, nbr_mask):
    nbr_pos = verts[nbr_idx] * nbr_mask.unsqueeze(-1)
    counts = nbr_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
    nbr_mean = nbr_pos.sum(dim=1) / counts
    has = nbr_mask.any(dim=1)
    return ((verts - nbr_mean)[has] ** 2).mean()


# ── Mesh Decimation ──────────────────────────────────────────────────────────

def decimate_mesh(verts, faces, colors, target_faces):
    """
    Decimate mesh to target face count using quadric error metrics.
    Falls back to uniform subsampling if open3d/pyfqmr not available.
    """
    if len(faces) <= target_faces:
        return verts, faces, colors

    try:
        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        mesh = mesh.simplify_quadric_decimation(target_faces)
        out_v = np.asarray(mesh.vertices, dtype=np.float32)
        out_f = np.asarray(mesh.triangles, dtype=np.int32)
        out_c = (np.asarray(mesh.vertex_colors) * 255).clip(0, 255).astype(np.uint8)
        print(f"  Decimated with Open3D: {len(verts):,d} -> {len(out_v):,d} verts, "
              f"{len(faces):,d} -> {len(out_f):,d} faces")
        return out_v, out_f, out_c
    except ImportError:
        pass

    # Fallback: edge collapse by removing shortest edges
    # Simple but effective — collapse edges shorter than a threshold
    print(f"  Decimating with edge collapse (no Open3D)...")
    current_faces = faces.copy()
    current_verts = verts.copy()
    current_colors = colors.copy()

    while len(current_faces) > target_faces:
        # Find shortest edge
        edges = {}
        for f in current_faces:
            for a, b in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]:
                key = (min(a, b), max(a, b))
                if key not in edges:
                    d = np.linalg.norm(current_verts[a] - current_verts[b])
                    edges[key] = d

        if not edges:
            break

        # Sort edges by length, collapse shortest ones
        sorted_edges = sorted(edges.items(), key=lambda x: x[1])
        n_collapse = max(1, (len(current_faces) - target_faces) // 2)

        collapsed = set()
        remap = np.arange(len(current_verts))

        for (vi, vj), _ in sorted_edges[:n_collapse]:
            if vi in collapsed or vj in collapsed:
                continue
            # Merge vj into vi
            current_verts[vi] = (current_verts[vi] + current_verts[vj]) / 2
            current_colors[vi] = ((current_colors[vi].astype(np.int32) +
                                    current_colors[vj].astype(np.int32)) // 2).astype(np.uint8)
            remap[vj] = vi
            collapsed.add(vj)

        # Remap faces
        new_faces = []
        for f in current_faces:
            rf = [remap[f[0]], remap[f[1]], remap[f[2]]]
            if rf[0] != rf[1] and rf[1] != rf[2] and rf[0] != rf[2]:
                new_faces.append(rf)
        current_faces = np.array(new_faces, dtype=np.int32) if new_faces else np.zeros((0, 3), dtype=np.int32)

        if len(current_faces) == 0:
            break

    # Compact: remove unused vertices
    used = np.unique(current_faces.ravel())
    idx_map = np.full(len(current_verts), -1, dtype=np.int32)
    idx_map[used] = np.arange(len(used), dtype=np.int32)
    out_v = current_verts[used]
    out_c = current_colors[used]
    out_f = idx_map[current_faces]

    print(f"  Decimated: {len(verts):,d} -> {len(out_v):,d} verts, "
          f"{len(faces):,d} -> {len(out_f):,d} faces")
    return out_v, out_f, out_c


# ── Adaptive Edge Subdivision ─────────────────────────────────────────────────

def subdivide_high_error_edges(verts_np, faces_np, colors_np, per_vertex_error, threshold=None):
    """
    Split edges where the error difference between endpoints is high.
    Inserts a new vertex at the midpoint of each high-error edge.

    Returns new verts, faces, colors arrays.
    """
    if threshold is None:
        threshold = np.percentile(per_vertex_error, 75)

    # Collect unique edges and their error
    edges = {}  # (vi, vj) -> max_error
    for f in faces_np:
        for a, b in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]:
            key = (min(a, b), max(a, b))
            err = max(per_vertex_error[a], per_vertex_error[b])
            if key not in edges or err > edges[key]:
                edges[key] = err

    # Find edges to split
    split_edges = {k: v for k, v in edges.items() if v > threshold}
    if not split_edges:
        return verts_np, faces_np, colors_np, 0

    # Create midpoint vertices
    new_verts = list(verts_np)
    new_colors = list(colors_np)
    edge_to_mid = {}  # (vi, vj) -> new_vertex_index

    for (vi, vj) in split_edges:
        mid_pos = (verts_np[vi] + verts_np[vj]) / 2.0
        mid_col = ((colors_np[vi].astype(np.float32) + colors_np[vj].astype(np.float32)) / 2.0).astype(np.uint8)
        mid_idx = len(new_verts)
        new_verts.append(mid_pos)
        new_colors.append(mid_col)
        edge_to_mid[(vi, vj)] = mid_idx

    # Rebuild faces: for each face, check which edges are split
    new_faces = []
    for f in faces_np:
        a, b, c = f[0], f[1], f[2]
        e_ab = edge_to_mid.get((min(a, b), max(a, b)))
        e_bc = edge_to_mid.get((min(b, c), max(b, c)))
        e_ca = edge_to_mid.get((min(c, a), max(c, a)))

        splits = (e_ab is not None) + (e_bc is not None) + (e_ca is not None)

        if splits == 0:
            new_faces.append([a, b, c])
        elif splits == 1:
            # Split one edge: 1 triangle -> 2 triangles
            if e_ab is not None:
                new_faces.append([a, e_ab, c])
                new_faces.append([e_ab, b, c])
            elif e_bc is not None:
                new_faces.append([a, b, e_bc])
                new_faces.append([a, e_bc, c])
            else:
                new_faces.append([a, b, e_ca])
                new_faces.append([b, c, e_ca])
        elif splits == 2:
            # Split two edges: 1 -> 3 triangles
            if e_ab is not None and e_bc is not None:
                new_faces.append([a, e_ab, e_bc])
                new_faces.append([a, e_bc, c])
                new_faces.append([e_ab, b, e_bc])
            elif e_bc is not None and e_ca is not None:
                new_faces.append([a, b, e_bc])
                new_faces.append([a, e_bc, e_ca])
                new_faces.append([e_bc, c, e_ca])
            else:
                new_faces.append([e_ab, b, e_ca])
                new_faces.append([a, e_ab, e_ca])
                new_faces.append([e_ab, c, e_ca])  # fix: use b->c
        else:
            # All 3 edges split: 1 -> 4 triangles
            new_faces.append([a, e_ab, e_ca])
            new_faces.append([e_ab, b, e_bc])
            new_faces.append([e_ca, e_bc, c])
            new_faces.append([e_ab, e_bc, e_ca])

    n_added = len(new_verts) - len(verts_np)
    return (np.array(new_verts, dtype=np.float32),
            np.array(new_faces, dtype=np.int32),
            np.array(new_colors, dtype=np.uint8),
            n_added)


def compute_per_vertex_error(verts, faces_t, vert_colors, views, dev):
    """Render from each camera, accumulate per-vertex error via projection."""
    V = verts.shape[0]
    error_accum = torch.zeros(V, device=dev)
    counts = torch.zeros(V, device=dev)

    with torch.no_grad():
        for view in views:
            R = view['w2c_t'][:3, :3]
            t_vec = view['w2c_t'][:3, 3]
            pts_cam = verts @ R.T + t_vec[None, :]
            z_v = pts_cam[:, 2]
            u = (pts_cam[:, 0] / z_v.clamp(min=0.01) * view['K_t'][0, 0] + view['K_t'][0, 2]).long()
            v_px = (pts_cam[:, 1] / z_v.clamp(min=0.01) * view['K_t'][1, 1] + view['K_t'][1, 2]).long()
            valid = (z_v > 0.01) & (u >= 0) & (u < view['W']) & (v_px >= 0) & (v_px < view['H'])

            if valid.any():
                sampled = view['pixels_t'][v_px[valid], u[valid]]  # (N, 3)
                predicted = vert_colors[valid]
                err = (sampled - predicted).abs().mean(dim=-1)  # (N,)
                error_accum[valid] += err
                counts[valid] += 1

    counts = counts.clamp(min=1)
    return (error_accum / counts).cpu().numpy()


# ── Main Refinement: Multi-Pass with Adaptive Subdivision ────────────────────

def refine_mesh(data_dir, mesh_path, output_path, iterations=300,
                lr=0.0005, depth_reg=0.1, smooth_reg=0.01, device='cuda'):

    dev = torch.device(device)
    print(f"Loading cameras from {data_dir}")
    views = load_cameras(data_dir)
    C = len(views)
    print(f"  {C} cameras")

    print(f"Loading mesh from {mesh_path}")
    verts_np, faces_np, colors_np = load_ply_mesh(mesh_path)
    print(f"  Original: {len(verts_np):,d} vertices, {len(faces_np):,d} faces")

    if len(verts_np) == 0 or len(faces_np) == 0:
        print("No mesh to refine")
        save_ply_mesh(output_path, verts_np, faces_np, colors_np)
        return

    # Decimate to ~5k faces for fast first pass
    # Each pass will subdivide high-error areas back up
    start_faces = 5000
    if len(faces_np) > start_faces * 1.5:
        print(f"  Decimating to ~{start_faces:,d} faces for coarse pass...")
        verts_np, faces_np, colors_np = decimate_mesh(verts_np, faces_np, colors_np, start_faces)

    for v in views:
        v['w2c_t'] = torch.from_numpy(v['w2c']).float().to(dev)
        v['K_t'] = torch.from_numpy(v['K']).float().to(dev)
        v['pixels_t'] = torch.from_numpy(v['pixels']).float().to(dev)

    preview_dir = os.path.join(os.path.dirname(output_path), 'refine_preview')
    os.makedirs(preview_dir, exist_ok=True)

    t0 = time.time()
    n_passes = 3  # coarse → medium → fine
    iters_per_pass = iterations // n_passes

    for pass_idx in range(n_passes):
        V, n_F = len(verts_np), len(faces_np)
        print(f"\n=== Pass {pass_idx + 1}/{n_passes}: {V:,d} verts, {n_F:,d} faces, {iters_per_pass} iters ===")

        # To GPU
        verts = torch.from_numpy(verts_np).float().to(dev).requires_grad_(True)
        verts_init = verts.detach().clone()
        faces_t = torch.from_numpy(faces_np).long().to(dev)
        vert_colors = torch.from_numpy(colors_np / 255.0).float().to(dev)

        # Laplacian
        nbr_idx, nbr_mask = build_laplacian(faces_np, V, dev)
        optimizer = torch.optim.Adam([verts], lr=lr)

        for step in range(iters_per_pass):
            global_step = pass_idx * iters_per_pass + step
            optimizer.zero_grad()

            ci = step % C
            view = views[ci]

            rendered, error_map, photo_loss = render_mesh(
                verts, faces_t, vert_colors,
                view['w2c_t'], view['K_t'],
                view['W'], view['H'], view['pixels_t']
            )

            depth_loss = F.mse_loss(verts, verts_init)
            smooth_loss = laplacian_loss(verts, nbr_idx, nbr_mask) if nbr_idx is not None else torch.tensor(0.0, device=dev)

            loss = photo_loss + depth_reg * depth_loss + smooth_reg * smooth_loss
            loss.backward()
            optimizer.step()

            # Update vertex colors
            if step % 50 == 0:
                with torch.no_grad():
                    for vi_cam in range(C):
                        v_data = views[vi_cam]
                        R = v_data['w2c_t'][:3, :3]
                        t_vec = v_data['w2c_t'][:3, 3]
                        pts_cam = verts.detach() @ R.T + t_vec[None, :]
                        z_v = pts_cam[:, 2]
                        u = (pts_cam[:, 0] / z_v.clamp(min=0.01) * v_data['K_t'][0, 0] + v_data['K_t'][0, 2]).long()
                        v_px = (pts_cam[:, 1] / z_v.clamp(min=0.01) * v_data['K_t'][1, 1] + v_data['K_t'][1, 2]).long()
                        valid = (z_v > 0.01) & (u >= 0) & (u < v_data['W']) & (v_px >= 0) & (v_px < v_data['H'])
                        if valid.any():
                            vert_colors[valid] = v_data['pixels_t'][v_px[valid], u[valid]]

            if step % 25 == 0 or step == iters_per_pass - 1:
                print(f"[{global_step:4d}/{iterations}] photo={photo_loss.item():.6f} "
                      f"depth={depth_loss.item():.6f} smooth={smooth_loss.item():.6f} "
                      f"verts={V:,d} cam={ci} time={time.time() - t0:.1f}s")

            if step % 50 == 0:
                with torch.no_grad():
                    emap = error_map.detach().cpu().numpy()
                    emap = (emap / (emap.max() + 1e-8) * 255).astype(np.uint8)
                    import matplotlib.cm as cm
                    heatmap = (cm.jet(emap)[:, :, :3] * 255).astype(np.uint8)
                    Image.fromarray(heatmap).save(os.path.join(preview_dir, 'error_heatmap.png'))
                    rend = rendered.detach().clamp(0, 1).cpu().numpy()
                    Image.fromarray((rend * 255).astype(np.uint8)).save(
                        os.path.join(preview_dir, 'render.png'))

        # After optimization, compute per-vertex error and subdivide high-error edges
        verts_np = verts.detach().cpu().numpy()
        colors_np = (vert_colors.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        if pass_idx < n_passes - 1:
            print("  Computing per-vertex error for adaptive subdivision...")
            per_vert_error = compute_per_vertex_error(verts.detach(), faces_t, vert_colors, views, dev)

            verts_np, faces_np, colors_np, n_added = subdivide_high_error_edges(
                verts_np, faces_np, colors_np, per_vert_error
            )
            print(f"  Subdivided: +{n_added:,d} vertices -> {len(verts_np):,d} total, {len(faces_np):,d} faces")
        else:
            faces_np = faces_np  # keep as-is on final pass

    # Final save
    save_ply_mesh(output_path, verts_np, faces_np, colors_np)
    print(f"\nDone. {time.time() - t0:.1f}s. Saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--mesh_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--depth_reg', type=float, default=0.1)
    parser.add_argument('--smooth_reg', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    refine_mesh(**vars(args))
