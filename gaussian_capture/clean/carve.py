"""Clean Splat: dataset cameras, model depth, free-space rule, job and result text."""

import bpy
import os
import math
import numpy as np
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

from ..core.gpu_depth import _CLN_TOL_REL, _ExpGpuDepth
from ..core.scene import _GCAPTURE_RIG_COLL
from ..core.versions import _exp_resolve_output_dir


def _cln_result_text(removed, total, out_name, mode, warn=False):
    """Result line of Clean Splat: numbers - file (path). The panel shows it
    in two lines, split at the first " - "."""
    text = "Removed {:,} of {:,} splats ({:.1f} %) - {}".format(
        removed, total, 100.0 * removed / max(total, 1), out_name)
    if warn:
        text += " - more than half: splat from another scene version?"
    return "%s (%s)" % (text, mode)


def _cln_check(context, s):
    """What step 8 is missing (red lines) and the paths."""
    errs = []
    splat = bpy.path.abspath(s.clean_splat_file or "").strip()
    if not splat:
        errs.append("Pick the trained splat (.ply)")
    elif not os.path.isfile(splat):
        errs.append("Splat file not found: %s" % os.path.basename(splat))
    elif not splat.lower().endswith(".ply"):
        errs.append("Only .ply splats can be cleaned")
    ds = _exp_resolve_output_dir(s)
    js = _cln_export_json_path(ds)
    if not os.path.isfile(js):
        errs.append("No %s - export the dataset first (step 7)"
                    % os.path.basename(js))
    elif not os.path.isfile(os.path.join(ds, "sparse", "0", "images.txt")):
        errs.append("No cameras in %s - export the dataset first (step 7)"
                    % os.path.basename(os.path.normpath(ds)))
    stem = os.path.splitext(splat)[0]
    return errs, {"splat": splat, "dataset": ds, "json": js,
                  "out": stem + "_clean.ply"}


def _cln_export_json_path(ds_dir):
    """<vNNN>_gcapture_export.json sits NEXT TO the dataset folder (v145).
    Datasets from Gaussian Render Scan 1.0.x have the same file under its old
    names: <vNNN>_gscan_export.json next to the dataset (from 0.145) or
    gscan_export.json inside it (1.126.0-0.144). The first one that exists
    wins; otherwise the current name (for the message)."""
    ds = os.path.normpath(ds_dir)
    current = os.path.join(os.path.dirname(ds),
                           os.path.basename(ds) + "_gcapture_export.json")
    for path in (current,
                 os.path.join(os.path.dirname(ds),
                              os.path.basename(ds) + "_gscan_export.json"),
                 os.path.join(ds, "gscan_export.json")):
        if os.path.isfile(path):
            return path
    return current


def _cln_load_transform(json_path):
    """(blender_from_dataset as a 4x4 Matrix, scale) from the export JSON."""
    import json
    with open(json_path, encoding="utf-8") as f:
        info = json.load(f)
    mat, scale = info.get("blender_from_dataset"), info.get("scale")
    if not mat or not scale:
        raise ValueError("%s has no blender_from_dataset / scale"
                         % os.path.basename(json_path))
    return Matrix(mat), float(scale)


def _cln_dataset_views(ds_dir, bfd):
    """Cameras of the dataset in Blender world space: [(pos, fwd, fov)]. Exactly
    the poses the trainer used (including Live Camera Adjust, without
    rejected interior cameras). fov covers the whole image (larger axis,
    shift included) -- the depth map is square."""
    from mathutils import Quaternion
    sp = os.path.join(ds_dir, "sparse", "0")
    fovs = {}
    with open(os.path.join(sp, "cameras.txt"), encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) < 8 or p[0].startswith("#"):
                continue
            w, h = float(p[2]), float(p[3])
            fx, fy, cx, cy = map(float, p[4:8])
            half = max(max(cx, w - cx) / fx, max(cy, h - cy) / fy)
            fovs[p[0]] = 2.0 * math.atan(half)
    rot = bfd.to_3x3()
    views = []
    with open(os.path.join(sp, "images.txt"), encoding="utf-8") as f:
        for line in f:
            p = line.split()
            # Image line: ID QW QX QY QZ TX TY TZ CAMERA_ID NAME (the lines of the
            # 2D points have a multiple of 3 fields or are empty).
            if len(p) != 10 or p[0].startswith("#"):
                continue
            try:
                int(p[0])
                q = Quaternion(tuple(float(v) for v in p[1:5]))
                t = Vector(tuple(float(v) for v in p[5:8]))
            except ValueError:
                continue
            fov = fovs.get(p[8])
            if fov is None:
                continue
            r_w2c = q.to_matrix()
            c_ds = -(r_w2c.transposed() @ t)
            fwd_ds = r_w2c.transposed() @ Vector((0.0, 0.0, 1.0))
            pos = (bfd @ c_ds.to_4d()).to_3d()
            fwd = (rot @ fwd_ds).normalized()
            views.append((pos, fwd, fov))
    return views


_CLN_GEOM_TYPES = {'MESH', 'CURVE', 'SURFACE', 'FONT', 'META'}


def _cln_render_geometry(context):
    """Everything that is rendered, as triangles in world coordinates -- not
    only the look target: a rendered floor is not empty space. Evaluated as
    for the render: objects hidden in the viewport (eye or monitor icon) are
    shown, modifiers that are on for the render only are switched on, and
    subdivision uses its render levels -- all restored afterwards. Includes
    instances (collection instances, Geometry Nodes). Without the camera
    sphere, guides, crop object, Capture_Rig and hide_render objects."""
    scene = context.scene
    s = scene.gcapture_settings
    excl = set()
    if s.sph_object is not None:
        excl.add(s.sph_object.name)
    for item in s.guides:
        for ref in item.objects:
            if ref.obj is not None:
                excl.add(ref.obj.name)
    if s.exp_crop_object is not None:
        excl.add(s.exp_crop_object.name)
    rig = bpy.data.collections.get(_GCAPTURE_RIG_COLL)
    if rig is not None:
        excl.update(o.name for o in rig.all_objects)
    view_layer = context.view_layer
    undo = []                     # (restore function) in reverse order
    try:
        for o in scene.objects:
            if o.hide_render or o.name in excl:
                continue
            if o.hide_viewport:
                o.hide_viewport = False
                undo.append(lambda o=o: setattr(o, "hide_viewport", True))
            try:
                if o.hide_get(view_layer=view_layer):
                    o.hide_set(False, view_layer=view_layer)
                    undo.append(lambda o=o: o.hide_set(True, view_layer=view_layer))
            except RuntimeError:  # not in this view layer
                pass
            for md in o.modifiers:
                if md.show_render and not md.show_viewport:
                    md.show_viewport = True
                    undo.append(lambda md=md: setattr(md, "show_viewport", False))
                if md.type in {'SUBSURF', 'MULTIRES'} and md.levels != md.render_levels:
                    old = md.levels
                    md.levels = md.render_levels
                    undo.append(lambda md=md, old=old: setattr(md, "levels", old))
        if undo:
            view_layer.update()
        dg = context.evaluated_depsgraph_get()
        cache, verts, tris = {}, [], []
        off = 0
        for inst in dg.object_instances:
            ob = inst.object
            orig = ob.original
            if ob.type not in _CLN_GEOM_TYPES or orig.hide_render or orig.name in excl:
                continue
            # Key by the evaluated data: Geometry Nodes instances of different
            # generated meshes all carry the same temporary name.
            key = (ob.data.as_pointer() if ob.data is not None
                   else hash(orig.name))
            if key not in cache:
                co = tri = None
                me = ob.to_mesh()
                if me is not None:
                    try:
                        me.calc_loop_triangles()
                        if len(me.vertices) and len(me.loop_triangles):
                            co = np.empty(len(me.vertices) * 3, dtype=np.float32)
                            me.vertices.foreach_get("co", co)
                            co = co.reshape(-1, 3)
                            tri = np.empty(len(me.loop_triangles) * 3, dtype=np.int32)
                            me.loop_triangles.foreach_get("vertices", tri)
                            tri = tri.reshape(-1, 3)
                    finally:
                        ob.to_mesh_clear()
                cache[key] = (co, tri)
            co, tri = cache[key]
            if co is None:
                continue
            mwn = np.array(inst.matrix_world, dtype=np.float32)
            verts.append(co @ mwn[:3, :3].T + mwn[:3, 3])
            tris.append(tri + off)
            off += len(co)
    finally:
        for fn in reversed(undo):
            fn()
        if undo:
            view_layer.update()
    if not verts:
        raise ValueError("No rendered geometry in the scene")
    return (np.vstack(verts).astype(np.float32),
            np.vstack(tris).astype(np.int32))


def _cln_model_size(context, verts):
    """Bounding-box diagonal of the look target (step 2) -- the scale of the
    absolute tolerance. A big rendered floor must not widen it. Without a
    look target: all rendered geometry."""
    pts = []
    for it in context.scene.gcapture_settings.sph_target_colls:
        if it.coll is None:
            continue
        for o in it.coll.all_objects:
            if o.type in _CLN_GEOM_TYPES and not o.hide_render:
                mw = o.matrix_world
                pts.extend(tuple(mw @ Vector(c)) for c in o.bound_box)
    arr = np.array(pts) if pts else verts
    return float(np.linalg.norm(arr.max(axis=0) - arr.min(axis=0))) or 1.0
_CLN_TOL_SIZE = 2e-3     # absolute tolerance: share of the model size
# Depth map resolution for GPU and ray casting alike. Coarse on purpose: the
# 3x3 neighbourhood then reaches about 1 % of the image sideways, so splats
# that hug an edge or a silhouette stay. Beetle, 26.09.2026 (1.92 M splats,
# 40 test views): 2048 px removed 12 % and cost 1 dB PSNR, 256 px removes
# 0.45 % at -0.1 dB and still 62 % of the floaters farther than 10 mm.
_CLN_DEPTH_RES = 256


def _cln_splat_points(ply, bfd, scale):
    """Splat centres in Blender world space and extent sigma (largest axis,
    Blender units). Without scale_* sigma is 0."""
    names, rows = ply["names"], ply["rows"]
    xyz = rows[:, [names.index(c) for c in ("x", "y", "z")]].astype(np.float64)
    mat = np.array(bfd, dtype=np.float64)
    points = xyz @ mat[:3, :3].T + mat[:3, 3]
    idx = [names.index("scale_%d" % i) for i in range(3)
           if "scale_%d" % i in names]
    if len(idx) == 3:
        sigma = np.exp(rows[:, idx].astype(np.float64)).max(axis=1) / scale
    else:
        sigma = np.zeros(len(points))
    return points, sigma


def _cln_camera_frame(pos, fwd, fov):
    """Camera axes as in _ExpGpuDepth._render (roll does not matter, only consistency)."""
    q = fwd.to_track_quat('-Z', 'Y')
    return (np.array(pos, dtype=np.float64),
            np.array(q @ Vector((1.0, 0.0, 0.0)), dtype=np.float64),
            np.array(q @ Vector((0.0, 1.0, 0.0)), dtype=np.float64),
            np.array(fwd.normalized(), dtype=np.float64),
            1.0 / math.tan(fov / 2.0))


def _cln_depth_cpu(bvh, frame, res):
    """Linear depth per pixel by ray casting (row 0 = bottom, inf = nothing)."""
    pos, right, up, fwd, f = frame
    origin = Vector(pos)
    c = (np.arange(res) + 0.5) / res * 2.0 - 1.0
    lin = np.full((res, res), np.inf)
    for py in range(res):
        row_dir = fwd + c[py] * up / f
        for px in range(res):
            d = row_dir + c[px] * right / f
            hit = bvh.ray_cast(origin, Vector(d))
            if hit[0] is not None:
                lin[py, px] = float(np.dot(np.array(hit[0]) - pos, fwd))
    return lin


def _cln_judge(points, sigma, lin, frame, near, tol_abs):
    """The same rule as _CLN_CARVE_COMPUTE_SRC: (in front of the surface, in the
    image) per splat. Nearest depth of the 3x3 neighbourhood, background = infinite."""
    pos, right, up, fwd, f = frame
    res = lin.shape[0]
    rel = points - pos
    d = rel @ fwd
    safe = np.where(d > near, d, 1.0)
    xn = (rel @ right) / safe * f
    yn = (rel @ up) / safe * f
    seen = (d > near) & (np.abs(xn) <= 1.0) & (np.abs(yn) <= 1.0)
    px = np.clip(((xn + 1.0) * 0.5 * res).astype(np.int64), 0, res - 1)
    py = np.clip(((yn + 1.0) * 0.5 * res).astype(np.int64), 0, res - 1)
    fin = np.where(np.isfinite(lin), lin, 1e30)
    pad = np.pad(fin, 1, mode='edge')
    zmin = fin.copy()
    for dy in range(3):
        for dx in range(3):
            zmin = np.minimum(zmin, pad[dy:dy + res, dx:dx + res])
    z = zmin[py, px]
    front = seen & (d + 3.0 * sigma + _CLN_TOL_REL * d + tol_abs < z)
    return front, seen


class _ClnJob:
    """One cleaning run: draw the depth per camera and count per splat how often
    it lies in front of the surface and how often in the image. GPU if
    possible; otherwise ray casting (blender --background, render nodes)."""

    def __init__(self, verts, tris, points, sigma, views, use_gpu, size=None):
        self.points, self.sigma, self.views = points, sigma, views
        self.total, self.done = len(views), 0
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        self.size = float(np.linalg.norm(hi - lo)) or 1.0
        self.center = (lo + hi) / 2.0
        self.radius = float(np.linalg.norm(verts - self.center, axis=1).max()) or 1.0
        # tolerance from the model (look target), the rest from all geometry
        self.tol_abs = _CLN_TOL_SIZE * (size or self.size)
        self.viol = np.zeros(len(points), dtype=np.int32)
        self.seen = np.zeros(len(points), dtype=np.int32)
        self.gpu = None
        if use_gpu:
            try:
                self.gpu = _ExpGpuDepth(None, res=_CLN_DEPTH_RES,
                                        geometry=(verts, tris))
                self.gpu.carve_begin(points, sigma)
            except Exception as exc:
                print("[Gaussian Render Capture] GPU carving unavailable, "
                      "using ray casting:", exc)
                self.gpu = None
        self.bvh = None
        if self.gpu is None:
            self.bvh = BVHTree.FromPolygons(verts.tolist(), tris.tolist(),
                                            all_triangles=True)

    @property
    def mode(self):
        return "GPU" if self.gpu is not None else "ray casting"

    def _near(self, pos):
        dist = float(np.linalg.norm(np.array(pos) - self.center))
        return max(self.size * 1e-4, dist - self.radius * 1.01)

    def step(self):
        pos, fwd, fov = self.views[self.done]
        if self.gpu is not None:
            self.gpu.carve_camera(pos, fwd, fov, self.tol_abs)
        else:
            frame = _cln_camera_frame(pos, fwd, fov)
            lin = _cln_depth_cpu(self.bvh, frame, _CLN_DEPTH_RES)
            front, seen = _cln_judge(self.points, self.sigma, lin, frame,
                                     self._near(pos), self.tol_abs)
            self.viol += front
            self.seen += seen
        self.done += 1

    def result(self, min_views):
        """bool per splat: remove."""
        if self.gpu is not None:
            self.viol, self.seen = self.gpu.carve_result()
        return (self.viol >= min_views) | (self.seen == 0)
