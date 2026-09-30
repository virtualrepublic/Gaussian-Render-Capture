"""Camera sphere helpers: framing, radii, model points."""

import bpy
import math
import numpy as np
from mathutils import Vector


# ----------------------------------------------------------------------
# Camera sphere generator: helpers
# ----------------------------------------------------------------------
def _sph_apply_render_settings(scene):
    """Sets the render settings for Gaussian splatting captures once.
    Each setting defensively (try/except), so that a property path that
    differs in this Blender version does not abort the whole operator --
    the others are set anyway. Returns a list of the items that could not
    be set (for a notice to the user)."""
    skipped = []

    def tryset(setter, label):
        try:
            setter()
        except Exception:
            skipped.append(label)

    # Engine to Cycles + GPU.
    tryset(lambda: setattr(scene.render, "engine", 'CYCLES'),
           "Cycles engine")
    tryset(lambda: setattr(scene.cycles, "device", 'GPU'),
           "Cycles GPU device")

    # Denoiser: enable Render + Viewport.
    tryset(lambda: setattr(scene.cycles, "use_denoising", True),
           "Render denoiser")
    tryset(lambda: setattr(scene.cycles, "use_preview_denoising", True),
           "Viewport denoiser")
    # Denoiser TYPE to OpenImageDenoise (OIDN) -- Render + Viewport.
    tryset(lambda: setattr(scene.cycles, "denoiser", 'OPENIMAGEDENOISE'),
           "Render denoiser = OpenImageDenoise")
    tryset(lambda: setattr(scene.cycles, "preview_denoiser", 'OPENIMAGEDENOISE'),
           "Viewport denoiser = OpenImageDenoise")
    # Denoising device on GPU (Blender 4.2+: the denoiser can use the GPU,
    # even when rendering). The property name varies by version, so
    # try several variants defensively.
    tryset(lambda: setattr(scene.cycles, "denoising_use_gpu", True),
           "Render denoiser GPU")
    tryset(lambda: setattr(scene.cycles, "preview_denoising_use_gpu", True),
           "Viewport denoiser GPU")

    # Samples (Render). Viewport samples are left untouched.
    tryset(lambda: setattr(scene.cycles, "samples", 512),
           "Render samples 512")

    # Film: Transparent. Transparent Glass stays as the scene has it (1.3.1):
    # the user decides with the option next to Prepare Scene.
    tryset(lambda: setattr(scene.render, "film_transparent", True),
           "Film transparent")

    # World background: Color Value to 1 (white).
    def set_world_bg():
        world = scene.world
        if world is None:
            world = bpy.data.worlds.new("World")
            scene.world = world
        world.use_nodes = True
        bg = world.node_tree.nodes.get("Background")
        if bg is None:
            for node in world.node_tree.nodes:
                if node.type == 'BACKGROUND':
                    bg = node
                    break
        if bg is not None:
            bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        else:
            world.color = (1.0, 1.0, 1.0)
    tryset(set_world_bg, "World background color = 1")

    return skipped


def _sph_apply_camera_clip(s):
    """Sets clip start/end of the configured camera (for splatting:
    start small, end very large)."""
    cam = bpy.data.objects.get(s.camera_name)
    if cam is None or cam.type != 'CAMERA':
        cam = bpy.context.scene.camera
    if cam is not None and cam.type == 'CAMERA':
        cam.data.clip_start = 0.01
        cam.data.clip_end = 10000.0
        return True
    return False


def _sph_collect_objects(s, exclude=None):
    """All VISIBLE mesh objects from the selected target collections
    (recursively incl. child collections). 'exclude' (object) is skipped
    -- important so that the camera sphere itself does NOT enter the size
    calculation. Only visible objects count (visible_get): this way hidden
    helper objects (ground planes, rigs, helper meshes) do not inflate the
    bounding box -- that was the reason why the sphere became far too
    large for models made of many parts."""
    objs = []
    seen = set()
    excl_name = exclude.name if exclude is not None else None

    def walk(coll):
        for o in coll.objects:
            if (o.type == 'MESH' and o.name not in seen
                    and o.name != excl_name and o.visible_get()):
                seen.add(o.name)
                objs.append(o)
        for child in coll.children:
            walk(child)

    for item in s.sph_target_colls:
        if item.coll is not None:
            walk(item.coll)
    return objs


def _sph_world_bounds(objs):
    """(min_vec, max_vec, center, max_dim, biggest) of the world-space bounding
    box over all objects. max_dim is the largest axis extent.
    'biggest' = (name, dim) of the object with the largest single diagonal
    (for diagnosing which part inflates the bounds). Reads the bound_box
    from the EVALUATED object (incl. modifiers)."""
    import math as _m
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mn = Vector((_m.inf, _m.inf, _m.inf))
    mx = Vector((-_m.inf, -_m.inf, -_m.inf))
    found = False
    biggest_name = None
    biggest_dim = -1.0
    for o in objs:
        o_eval = o.evaluated_get(depsgraph)
        mw = o_eval.matrix_world
        omn = Vector((_m.inf, _m.inf, _m.inf))
        omx = Vector((-_m.inf, -_m.inf, -_m.inf))
        for corner in o_eval.bound_box:
            wc = mw @ Vector(corner)
            mn.x = min(mn.x, wc.x); mn.y = min(mn.y, wc.y); mn.z = min(mn.z, wc.z)
            mx.x = max(mx.x, wc.x); mx.y = max(mx.y, wc.y); mx.z = max(mx.z, wc.z)
            omn.x = min(omn.x, wc.x); omn.y = min(omn.y, wc.y); omn.z = min(omn.z, wc.z)
            omx.x = max(omx.x, wc.x); omx.y = max(omx.y, wc.y); omx.z = max(omx.z, wc.z)
            found = True
        # Individual extent of this object.
        odim = max(omx.x - omn.x, omx.y - omn.y, omx.z - omn.z)
        if odim > biggest_dim:
            biggest_dim = odim
            biggest_name = o.name
    if not found:
        return None
    center = (mn + mx) * 0.5
    max_dim = max(mx.x - mn.x, mx.y - mn.y, mx.z - mn.z)
    # Space diagonal of the overall bounding box: the largest dimension a
    # camera can see from ANY direction. As the reference for the radius it
    # makes the framing shape-independent -- compact (cube) and elongated
    # (car) objects are framed equally well with the same margin.
    diag = math.sqrt((mx.x - mn.x) ** 2 + (mx.y - mn.y) ** 2 +
                     (mx.z - mn.z) ** 2)
    return (mn, mx, center, max_dim, diag, (biggest_name, biggest_dim))


def _sph_camera_min_fov(s, scene):
    """Smaller of the two FOV axes (horizontal/vertical) in radians --
    the limiting axis for 'object fully in frame'. Based on the focal
    length set in the add-on. Sensor width 36mm.

    IMPORTANT: a SQUARE aspect ratio (1:1) is DELIBERATELY assumed, NOT
    the currently set render aspect ratio. Reason: the build renders
    square for splatting (set_resolution sets res_x = res_y). If the
    generator used the current aspect ratio, the radius would change as
    soon as the build switches the resolution to square -- the sphere
    would then shrink/grow once on the next creation. With a fixed 1:1
    the generator is consistent with the render result from the start
    and stays stable."""
    lens = max(s.focal_length, 1e-3)
    sensor_w = 36.0
    # Square: fov_h == fov_v, so one axis is enough.
    fov = 2.0 * math.atan((sensor_w / 2.0) / lens)
    return fov


# Mesh attribute of the Camera Sphere (face domain): factor by which the
# camera moves toward the center per face (Fill Each View, v96).
_SPH_FIT_ATTR = "gcapture_fit"
_SPH_FIT_MIN = 0.6   # no camera closer than 60 % of the sphere distance
_SPH_SHIFT_ATTR = "gcapture_shift"   # lateral offset per camera (v135)


def _sph_model_points(objs):
    """All vertices of the objects in world coordinates as an (N,3) array."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    chunks = []
    for o in objs:
        oe = o.evaluated_get(depsgraph)
        me = oe.to_mesh()
        if me is None:
            continue
        n = len(me.vertices)
        if n:
            co = np.empty(n * 3, dtype=np.float32)
            me.vertices.foreach_get("co", co)
            co = co.reshape(n, 3)
            mw = np.array(oe.matrix_world, dtype=np.float64)
            chunks.append(co @ mw[:3, :3].T + mw[:3, 3])
        oe.to_mesh_clear()
    if not chunks:
        return None
    return np.vstack(chunks)


def _sph_required_distances(points, center, dirs, fov, margin):
    """Per view direction, the distance from the center at which ALL points
    fit into the (square) image with margin. dirs: unit vectors from the
    center to the camera; the camera looks at the center (-dir). For a
    point p with rel = p - center, the camera image requires |rel.right| <=
    t * depth and |rel.up| <= t * depth with depth = d - rel.u and
    t = tan(fov/2) / margin -> d >= rel.u + max(|rel.right|, |rel.up|) / t."""
    tt = math.tan(fov / 2.0) / max(margin, 1e-6)
    points, safety = _sph_reduce_points(points)
    safety *= (1.0 + 1.0 / tt)
    rel = points - np.array(center, dtype=np.float64)
    out = []
    for u in dirs:
        q = (-u).to_track_quat('-Z', 'Y')
        r = q @ Vector((1.0, 0.0, 0.0))
        up = q @ Vector((0.0, 1.0, 0.0))
        best = -np.inf
        for k in range(0, len(rel), 500000):   # memory-saving
            blk = rel[k:k + 500000]
            du = blk @ np.array(u)
            lat = np.maximum(np.abs(blk @ np.array(r)),
                             np.abs(blk @ np.array(up)))
            best = max(best, float((du + lat / tt).max()))
        out.append(best + safety)
    return out


def _sph_framed_positions(points, center, dirs, fov, margin):
    """Per view direction (unit vector center -> camera) the camera position
    at which all points fit centered into the square image with margin,
    without changing the view direction (v135). Returns per direction
    (distance along dir, lateral offset as Vector, relative to the center).
    Per image axis: a = rel.axis, z = rel.forward, M = max(a - t z),
    m = min(a + t z); camera p_a = (m + M) / 2, p_f <= (m - M) / (2 t)."""
    tt = math.tan(fov / 2.0) / max(margin, 1e-6)
    points, safety = _sph_reduce_points(points)
    safety *= (1.0 + 1.0 / tt)
    rel = points - np.array(center, dtype=np.float64)
    out = []
    for u in dirs:
        f = -u
        q = f.to_track_quat('-Z', 'Y')
        r = q @ Vector((1.0, 0.0, 0.0))
        up = q @ Vector((0.0, 1.0, 0.0))
        M1 = M2 = -np.inf
        m1 = m2 = np.inf
        for k in range(0, len(rel), 500000):   # memory-saving
            blk = rel[k:k + 500000]
            z = blk @ np.array(f)
            a = blk @ np.array(r)
            b = blk @ np.array(up)
            M1 = max(M1, float((a - tt * z).max()))
            m1 = min(m1, float((a + tt * z).min()))
            M2 = max(M2, float((b - tt * z).max()))
            m2 = min(m2, float((b + tt * z).min()))
        p_f = min((m1 - M1) / (2.0 * tt), (m2 - M2) / (2.0 * tt))
        shift = r * ((m1 + M1) / 2.0) + up * ((m2 + M2) / 2.0)
        out.append((-p_f + safety, shift))
    return out


def _sph_center_shifts(points, center, dirs, fov, dists, shifts, iters=8):
    """Adjust the offset so that the model sits centered in both image axes
    at the final distance (after the 60 % limit) (v135). The closed-form
    solution centers only the tight axis exactly; due to perspective the
    other axis needs a short Newton iteration: center of the image span ->
    move the camera by center * t * depth."""
    tt = math.tan(fov / 2.0)
    points, _ = _sph_reduce_points(points)
    P = np.asarray(points, dtype=np.float64)
    c0 = np.array(center, dtype=np.float64)
    out = []
    for u, d, sv in zip(dirs, dists, shifts):
        f = -u
        q = f.to_track_quat('-Z', 'Y')
        axes = (np.array(q @ Vector((1.0, 0.0, 0.0))),
                np.array(q @ Vector((0.0, 1.0, 0.0))))
        fa = np.array(f)
        pos = c0 + np.array(u) * d + np.array(sv)
        for ax in axes:
            for _ in range(iters):
                rel = P - pos
                z = rel @ fa
                v = (rel @ ax) / (tt * z)
                i_max, i_min = int(v.argmax()), int(v.argmin())
                off = 0.5 * (v[i_max] + v[i_min])
                if abs(off) < 1e-5:
                    break
                pos = pos + ax * (off * tt * 0.5 * (z[i_max] + z[i_min]))
        out.append(Vector(pos - c0 - np.array(u) * d))
    return out


def _sph_reduce_points(points, grid=384):
    """Reduces large point sets for the distance calculation without missing
    a result: per grid column (x/y cell) only the lowest and the highest
    point count -- every point in between lies on their connecting segment
    and can never lie further out for linear constraints. x/y are set to
    the cell center; the maximum resulting offset (half the cell diagonal)
    is returned as a safety margin. Small sets stay unchanged (returns 0.0)."""
    if len(points) <= 200000:
        return points, 0.0
    lo = points.min(axis=0)
    ext = float((points.max(axis=0) - lo)[:2].max())
    if ext <= 0.0:
        return points, 0.0
    cell = ext / grid
    ij = np.minimum(((points[:, :2] - lo[:2]) / cell).astype(np.int64),
                    grid - 1)
    key = ij[:, 0] * grid + ij[:, 1]
    zmin = np.full(grid * grid, np.inf)
    zmax = np.full(grid * grid, -np.inf)
    np.minimum.at(zmin, key, points[:, 2])
    np.maximum.at(zmax, key, points[:, 2])
    used = np.nonzero(np.isfinite(zmin))[0]
    cx = lo[0] + (used // grid + 0.5) * cell
    cy = lo[1] + (used % grid + 0.5) * cell
    red = np.vstack([np.column_stack([cx, cy, zmin[used]]),
                     np.column_stack([cx, cy, zmax[used]])])
    return red, cell * 0.7072


def _sph_required_radius(max_dim, fov, margin):
    """Sphere radius at which an object with the largest extent max_dim
    (with margin) fits completely into the image: d = (max_dim*margin/2)/tan(fov/2).
    This way the object is fully captured from EVERY direction, because
    the sizing is based on the longest extent."""
    half = (max_dim * margin) / 2.0
    t = math.tan(fov / 2.0)
    if t < 1e-6:
        return half  # guard against division by ~0
    return half / t
