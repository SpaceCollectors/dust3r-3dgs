"""Equirectangular panorama ↔ cubemap conversion utilities.

Used to decompose a 360° equirectangular panorama into 6 perspective cube faces
for VGGT depth estimation, then stitch the per-face depth back into equirectangular.
"""

import numpy as np
from PIL import Image


def _make_camera_layout():
    """Build camera layout: 4 horizontal + 4 up-diagonal + 4 down-diagonal.

    Horizontal cameras at azimuth 0°, 90°, 180°, 270° (elevation 0°).
    Up-diagonals at azimuth 45°, 135°, 225°, 315° (elevation +45°).
    Down-diagonals at same azimuths (elevation -45°).

    With ~105° FOV, up/down diagonals just reach the poles, giving full
    sphere coverage without degenerate straight-up/down views.
    """
    cameras = []

    def _fwd_up(az_deg, el_deg):
        az = np.radians(az_deg)
        el = np.radians(el_deg)
        fwd = np.array([np.sin(az) * np.cos(el),
                         np.sin(el),
                         -np.cos(az) * np.cos(el)], dtype=np.float64)
        # Up = world-up projected perpendicular to forward
        world_up = np.array([0, 1, 0], dtype=np.float64)
        right = np.cross(fwd, world_up)
        if np.linalg.norm(right) < 1e-6:
            # Nearly vertical — use world-Z as fallback
            world_up = np.array([0, 0, 1], dtype=np.float64)
            right = np.cross(fwd, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, fwd)
        up = up / np.linalg.norm(up)
        return fwd, up

    # 4 horizontal cameras
    for az, name in [(0, "front"), (90, "right"), (180, "back"), (270, "left")]:
        fwd, up = _fwd_up(az, 0)
        cameras.append((name, fwd, up))

    # 4 up-diagonal cameras (45° elevation, rotated 45° from horizontal)
    for az, name in [(45, "up_FR"), (135, "up_BR"), (225, "up_BL"), (315, "up_FL")]:
        fwd, up = _fwd_up(az, 45)
        cameras.append((name, fwd, up))

    # 4 down-diagonal cameras (-45° elevation, same azimuths)
    for az, name in [(45, "dn_FR"), (135, "dn_BR"), (225, "dn_BL"), (315, "dn_FL")]:
        fwd, up = _fwd_up(az, -45)
        cameras.append((name, fwd, up))

    return cameras


CUBE_FACES = _make_camera_layout()


def _make_face_rays(face_size, fwd, up, fov_deg=90.0):
    """Generate unit ray directions for a perspective face with given FOV.

    Args:
        face_size: output resolution (pixels)
        fwd: (3,) forward direction unit vector
        up: (3,) up direction unit vector
        fov_deg: field of view in degrees (default 90°, use >90 for overlap)

    Returns:
        rays: (face_size, face_size, 3) unit vectors in world space
    """
    right = np.cross(fwd, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, fwd)  # ensure orthonormal
    up = up / np.linalg.norm(up)

    # FOV → half-extent in NDC
    half_extent = np.tan(np.radians(fov_deg / 2))

    u = np.linspace(-half_extent, half_extent, face_size, dtype=np.float64)
    v = np.linspace(-half_extent, half_extent, face_size, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)

    rays = fwd[None, None, :] + uu[:, :, None] * right[None, None, :] + vv[:, :, None] * (-up[None, None, :])
    rays = rays / np.linalg.norm(rays, axis=2, keepdims=True)
    return rays


def _rays_to_equirect_coords(rays, eq_h, eq_w):
    """Convert unit ray directions to equirectangular pixel coordinates.

    Args:
        rays: (..., 3) unit vectors
        eq_h, eq_w: equirectangular image dimensions

    Returns:
        u, v: (...) pixel coordinates in the equirectangular image (float)
    """
    x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
    # Longitude: atan2(x, -z) → [-π, π] → [0, W]
    lon = np.arctan2(x, -z)
    # Latitude: asin(y) → [-π/2, π/2]
    # In equirect images: top row = +90° (up), bottom row = -90° (down)
    # y-up convention: positive y = up
    lat = np.arcsin(np.clip(y, -1, 1))

    u = (lon / (2 * np.pi) + 0.5) * eq_w
    v = (0.5 - lat / np.pi) * eq_h  # +90° → row 0, -90° → row H
    return u, v


def _equirect_coords_to_rays(eq_h, eq_w):
    """Generate ray directions for every pixel in an equirectangular image.

    Returns:
        rays: (eq_h, eq_w, 3) unit vectors
    """
    u = np.arange(eq_w, dtype=np.float64) + 0.5
    v = np.arange(eq_h, dtype=np.float64) + 0.5

    lon = (u / eq_w - 0.5) * 2 * np.pi   # [-π, π]
    lat = (0.5 - v / eq_h) * np.pi        # top=+π/2, bottom=-π/2

    lon, lat = np.meshgrid(lon, lat)

    x = np.sin(lon) * np.cos(lat)
    y = np.sin(lat)                        # y-up
    z = -np.cos(lon) * np.cos(lat)

    return np.stack([x, y, z], axis=-1)


def equirect_to_cubemap(equirect_img, face_size=518, fov_deg=95.0):
    """Decompose an equirectangular panorama into 6 perspective face images.

    Args:
        equirect_img: PIL Image or numpy array (H, W, 3), 2:1 aspect ratio
        face_size: output face resolution (default 518 to match VGGT input)
        fov_deg: field of view per face in degrees (>90 gives overlap between faces)

    Returns:
        faces: list of 6 PIL Images, one per cube face
        face_names: list of face name strings
    """
    if isinstance(equirect_img, Image.Image):
        equirect_img = np.array(equirect_img.convert('RGB'))

    eq_h, eq_w = equirect_img.shape[:2]

    faces = []
    face_names = []

    for name, fwd, up in CUBE_FACES:
        rays = _make_face_rays(face_size, fwd, up, fov_deg=fov_deg)
        u, v = _rays_to_equirect_coords(rays, eq_h, eq_w)

        # Bilinear sampling from equirectangular
        u = u % eq_w  # wrap longitude
        v = np.clip(v, 0, eq_h - 1)

        u0 = np.floor(u).astype(int) % eq_w
        v0 = np.clip(np.floor(v).astype(int), 0, eq_h - 2)
        u1 = (u0 + 1) % eq_w
        v1 = np.clip(v0 + 1, 0, eq_h - 1)

        du = (u - np.floor(u))[:, :, None]
        dv = (v - np.floor(v))[:, :, None]

        face = (
            equirect_img[v0, u0] * (1 - du) * (1 - dv) +
            equirect_img[v0, u1] * du * (1 - dv) +
            equirect_img[v1, u0] * (1 - du) * dv +
            equirect_img[v1, u1] * du * dv
        )

        faces.append(Image.fromarray(face.astype(np.uint8)))
        face_names.append(name)

    return faces, face_names


def _stitch_faces_to_equirect(face_data_list, eq_h, eq_w, fov_deg=95.0):
    """Generic stitching: project equirect rays onto each face, pick best face per pixel.

    Args:
        face_data_list: list of 6 numpy arrays, each (face_size, face_size, ...).
                        Can be depth (H,W), confidence (H,W), or pts3d (H,W,3).
        eq_h, eq_w: output equirectangular dimensions
        fov_deg: FOV used when generating the faces

    Returns:
        eq_data: (eq_h, eq_w, ...) stitched result
    """
    eq_rays = _equirect_coords_to_rays(eq_h, eq_w)  # (H, W, 3)
    sample_shape = face_data_list[0].shape[2:]  # () for depth, (3,) for pts3d
    eq_data = np.zeros((eq_h, eq_w) + sample_shape, dtype=np.float32)
    best_dot = np.full((eq_h, eq_w), -1.0, dtype=np.float64)

    half_extent = np.tan(np.radians(fov_deg / 2))

    for fi, (name, fwd, up) in enumerate(CUBE_FACES):
        right = np.cross(fwd, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, fwd)
        up = up / np.linalg.norm(up)

        face_size = face_data_list[fi].shape[0]

        dot_fwd = np.sum(eq_rays * fwd[None, None, :], axis=2)
        facing = dot_fwd > 0

        with np.errstate(divide='ignore', invalid='ignore'):
            proj = eq_rays / dot_fwd[:, :, None]

        # Face-local coordinates in [-half_extent, half_extent]
        face_u = np.sum(proj * right[None, None, :], axis=2)
        face_v = np.sum(proj * (-up[None, None, :]), axis=2)

        in_face = facing & (np.abs(face_u) <= half_extent) & (np.abs(face_v) <= half_extent)

        # Convert to pixel coordinates
        px = ((face_u / half_extent + 1) / 2 * (face_size - 1)).astype(int)
        py = ((face_v / half_extent + 1) / 2 * (face_size - 1)).astype(int)
        px = np.clip(px, 0, face_size - 1)
        py = np.clip(py, 0, face_size - 1)

        better = in_face & (dot_fwd > best_dot)
        eq_data[better] = face_data_list[fi][py[better], px[better]]
        best_dot[better] = dot_fwd[better]

    return eq_data


def cubemap_depth_to_equirect(face_depths, face_confs, eq_h, eq_w, fov_deg=95.0):
    """Stitch 6 cube face depth/confidence maps back into equirectangular."""
    # Add trailing dim so stitch function handles them uniformly
    depths_3d = [d[:, :, None] for d in face_depths]
    confs_3d = [c[:, :, None] for c in face_confs]
    eq_depth = _stitch_faces_to_equirect(depths_3d, eq_h, eq_w, fov_deg)[:, :, 0]
    eq_conf = _stitch_faces_to_equirect(confs_3d, eq_h, eq_w, fov_deg)[:, :, 0]
    return eq_depth, eq_conf


def cubemap_pts3d_to_equirect(face_pts3d, eq_h, eq_w, fov_deg=95.0):
    """Stitch 6 cube face 3D point maps back into equirectangular."""
    return _stitch_faces_to_equirect(face_pts3d, eq_h, eq_w, fov_deg)


def merge_faces_to_equirect(pts3d_list, conf_list, imgs_list, eq_h, eq_w, fov_deg,
                            min_conf=0.0):
    """Merge overlapping face point clouds into a single equirectangular grid.

    Each face pixel is mapped to its equirect coordinate via the known camera
    layout.  Where faces overlap, 3D positions and colors are averaged weighted
    by confidence.  Result is a seamless (eq_h, eq_w) point map.

    Args:
        pts3d_list: list of N (H, W, 3) world-space point maps
        conf_list:  list of N (H, W) confidence maps
        imgs_list:  list of N (Hi, Wi, 3) float [0,1] face images
        eq_h, eq_w: output equirectangular resolution
        fov_deg:    FOV used when extracting the faces
        min_conf:   ignore face pixels below this confidence

    Returns:
        eq_pts3d: (eq_h, eq_w, 3) float32 — merged 3D points
        eq_conf:  (eq_h, eq_w) float32 — max confidence at each pixel
        eq_color: (eq_h, eq_w, 3) float32 [0,1] — merged colors
    """
    # Accumulators
    acc_pts = np.zeros((eq_h, eq_w, 3), dtype=np.float64)
    acc_color = np.zeros((eq_h, eq_w, 3), dtype=np.float64)
    acc_weight = np.zeros((eq_h, eq_w), dtype=np.float64)
    max_conf = np.zeros((eq_h, eq_w), dtype=np.float32)

    for fi, (name, fwd, up) in enumerate(CUBE_FACES):
        pts = pts3d_list[fi]    # (H, W, 3)
        conf = conf_list[fi]    # (H, W)
        img = imgs_list[fi]     # (Hi, Wi, 3)
        H, W = pts.shape[:2]
        Hi, Wi = img.shape[:2]

        # Ray directions for this face's pixels
        rays = _make_face_rays(H, fwd, up, fov_deg=fov_deg)  # (H, W, 3)
        eu, ev = _rays_to_equirect_coords(rays, eq_h, eq_w)  # (H, W) each

        # Valid mask
        valid = (conf > min_conf) & np.isfinite(pts).all(axis=-1)

        # Equirect pixel indices (nearest neighbor scatter)
        eu_i = np.clip(np.round(eu).astype(int), 0, eq_w - 1)
        ev_i = np.clip(np.round(ev).astype(int), 0, eq_h - 1)

        # Sample colors from (possibly higher-res) image
        img_rr = np.clip((np.arange(H)[:, None] * Hi / H).astype(int), 0, Hi - 1)
        img_cc = np.clip((np.arange(W)[None, :] * Wi / W).astype(int), 0, Wi - 1)
        img_rr = np.broadcast_to(img_rr, (H, W))
        img_cc = np.broadcast_to(img_cc, (H, W))
        face_colors = img[img_rr, img_cc]  # (H, W, 3)

        # Scatter with confidence weighting
        w = conf.astype(np.float64) * valid  # zero weight for invalid
        rows_valid = ev_i[valid]
        cols_valid = eu_i[valid]
        w_valid = w[valid]
        pts_valid = pts[valid]
        col_valid = face_colors[valid]

        np.add.at(acc_pts, (rows_valid, cols_valid), pts_valid * w_valid[:, None])
        np.add.at(acc_color, (rows_valid, cols_valid), col_valid * w_valid[:, None])
        np.add.at(acc_weight, (rows_valid, cols_valid), w_valid)
        np.maximum.at(max_conf, (rows_valid, cols_valid), conf[valid])

    # Normalize
    has_data = acc_weight > 0
    eq_pts3d = np.zeros((eq_h, eq_w, 3), dtype=np.float32)
    eq_color = np.zeros((eq_h, eq_w, 3), dtype=np.float32)
    eq_pts3d[has_data] = (acc_pts[has_data] / acc_weight[has_data, None]).astype(np.float32)
    eq_color[has_data] = (acc_color[has_data] / acc_weight[has_data, None]).astype(np.float32)

    n_filled = has_data.sum()
    print(f"  Merged {len(pts3d_list)} faces → {eq_w}x{eq_h} equirect: "
          f"{n_filled:,d}/{eq_h * eq_w:,d} pixels filled ({100*n_filled/(eq_h*eq_w):.1f}%)")
    return eq_pts3d, max_conf, eq_color


def equirect_mesh(eq_pts3d, eq_conf, pano_img, min_conf=0.5,
                  depth_edge_mult=5.0, step=2):
    """Build a spherical grid mesh from a merged equirectangular point map.

    The mesh is a lat/lon grid on the equirect pixel lattice.  Vertex positions
    come from the merged 3D points, vertex colors from the panorama, and UVs
    map directly into the panorama texture.  Wraps horizontally at the 360° seam.
    Triangles spanning depth discontinuities or with >85° incidence are removed.

    Args:
        eq_pts3d: (eq_h, eq_w, 3) merged world-space points
        eq_conf:  (eq_h, eq_w) confidence
        pano_img: (eq_h, eq_w, 3) uint8 panorama (used as texture)
        min_conf: minimum confidence threshold
        depth_edge_mult: max depth ratio for triangle culling
        step: pixel step for grid subsampling (1=full, 2=half, etc.)

    Returns:
        verts:  (V, 3) float32
        faces:  (F, 3) int32
        colors: (V, 3) uint8
        texture: (eq_h, eq_w, 3) uint8 — panorama texture
        uvs:    (V, 2) float32 — per-vertex UVs into the panorama
    """
    eq_h, eq_w = eq_pts3d.shape[:2]

    # Sub-sample grid
    rows = np.arange(0, eq_h, step)
    cols = np.arange(0, eq_w, step)
    gh, gw = len(rows), len(cols)
    rr, cc = np.meshgrid(rows, cols, indexing='ij')

    g_pts = eq_pts3d[rr, cc]       # (gh, gw, 3)
    g_conf = eq_conf[rr, cc]       # (gh, gw)
    g_color = pano_img[rr, cc]     # (gh, gw, 3) uint8

    valid = (g_conf > min_conf) & np.isfinite(g_pts).all(axis=-1)
    g_depth = np.linalg.norm(g_pts, axis=-1)
    valid &= g_depth > 0

    # UVs: map grid pixel → panorama texture coordinates
    # U = col / W, V = 1 - row / H (flip for OpenGL)
    uv_u = (cc.astype(np.float32) + 0.5) / eq_w
    uv_v = 1.0 - (rr.astype(np.float32) + 0.5) / eq_h
    g_uvs = np.stack([uv_u, uv_v], axis=-1)  # (gh, gw, 2)

    # Flatten
    verts = g_pts.reshape(-1, 3).astype(np.float32)
    colors = g_color.reshape(-1, 3).copy()
    uvs = g_uvs.reshape(-1, 2).astype(np.float32)

    # Vertex index grid
    vidx = np.arange(gh * gw, dtype=np.int32).reshape(gh, gw)

    # Grid triangulation — wrap horizontally for 360° seam
    i00 = vidx[:-1, :]
    i10 = vidx[1:, :]
    i01 = np.roll(vidx, -1, axis=1)[:-1, :]  # col+1, wraps
    i11 = np.roll(vidx, -1, axis=1)[1:, :]

    v00 = valid[:-1, :]
    v10 = valid[1:, :]
    v01 = np.roll(valid, -1, axis=1)[:-1, :]
    v11 = np.roll(valid, -1, axis=1)[1:, :]

    d00 = g_depth[:-1, :]
    d10 = g_depth[1:, :]
    d01 = np.roll(g_depth, -1, axis=1)[:-1, :]
    d11 = np.roll(g_depth, -1, axis=1)[1:, :]

    def _depth_ok(da, db):
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(da > db, da / db, db / da)
        return ratio < depth_edge_mult

    # Triangle 1: i00-i10-i01
    t1_valid = v00 & v10 & v01
    t1_valid &= _depth_ok(d00, d10) & _depth_ok(d00, d01) & _depth_ok(d10, d01)
    tri1 = np.stack([i00[t1_valid], i10[t1_valid], i01[t1_valid]], axis=-1)

    # Triangle 2: i10-i11-i01
    t2_valid = v10 & v11 & v01
    t2_valid &= _depth_ok(d10, d11) & _depth_ok(d01, d11) & _depth_ok(d10, d01)
    tri2 = np.stack([i10[t2_valid], i11[t2_valid], i01[t2_valid]], axis=-1)

    faces = np.concatenate([tri1, tri2], axis=0).astype(np.int32)

    # ── Remove grazing faces (incidence angle > 85°) ───────────────────
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / (norm_len + 1e-12)
    centroids = (v0 + v1 + v2) / 3.0
    view_dir = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    cos_angle = np.abs(np.sum(normals * view_dir, axis=1))
    cos_85 = np.cos(np.radians(85.0))
    keep = cos_angle >= cos_85
    n_removed = (~keep).sum()
    if n_removed > 0:
        faces = faces[keep]
        print(f"  Removed {n_removed:,d} grazing faces (>85°)")

    # Clean up unreferenced vertices
    used = np.unique(faces.ravel())
    remap = np.full(len(verts), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    verts = verts[used]
    colors = colors[used]
    uvs = uvs[used]
    faces = remap[faces]

    print(f"  Equirect mesh: {len(verts):,d} verts, {len(faces):,d} faces "
          f"(grid {gw}x{gh}, step={step})")
    return verts, faces, colors, pano_img, uvs
