"""Start points: vertices, face points, point-cloud signature for reuse."""

import bpy
import numpy as np
from mathutils import Vector

from ..core.glass import _gcapture_see_through_slots
from ..core.gpu_depth import _EXP_GPU_RES
from ..export.colmap import (
    _exp_crop_bounds_world, _exp_object_material_colors, _exp_vertex_color_lookup)


def _exp_sample_face_points(objs, count, scale, W, world_out=None,
                            crop_bounds=None, colors_out=None, glass_out=None):
    """Distributes 'count' points area-weighted over the surfaces of the
    objects (uniform initial density regardless of the vertex layout).
    Returns: list (x,y,z) in Y-up + scale; world_out receives the world
    coordinates in parallel (for the visibility filter). Barycentric
    sqrt sampling -> uniformly distributed in the triangle.

    Since v110: triangles via foreach_get/numpy (previously bmesh + Python
    loop per triangle) and crop box as for the vertex points -- points
    outside are discarded and replenished until 'count'
    is reached (at most 6 rounds).

    Since v1.1.5: colors_out receives per point the base color of the
    face's material (sRGB bytes), like the vertex points in Material mode."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    tri_chunks = []
    col_chunks = []
    glass_chunks = []
    glass_cache = {}
    for obj in objs:
        if obj.type != 'MESH':
            continue
        obj_eval = obj.evaluated_get(depsgraph)
        mesh = obj_eval.to_mesh()
        if mesh is None:
            continue
        try:
            mesh.calc_loop_triangles()
            n = len(mesh.vertices)
            nt = len(mesh.loop_triangles)
            if not n or not nt:
                continue
            co = np.empty(n * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", co)
            mw = np.array(obj_eval.matrix_world, dtype=np.float64)
            world = co.reshape(n, 3) @ mw[:3, :3].T + mw[:3, 3]
            idx = np.empty(nt * 3, dtype=np.int32)
            mesh.loop_triangles.foreach_get("vertices", idx)
            tri_chunks.append(world[idx.reshape(nt, 3).astype(np.int64)])
            mi = None
            if colors_out is not None or glass_out is not None:
                mi = np.empty(nt, dtype=np.int32)
                mesh.loop_triangles.foreach_get("material_index", mi)
            if colors_out is not None:
                mc = np.array(_exp_object_material_colors(obj), dtype=np.float64)
                col_chunks.append(mc[np.clip(mi, 0, len(mc) - 1)])
            if glass_out is not None:
                slots = _gcapture_see_through_slots(obj, glass_cache)
                glass_chunks.append(
                    np.array(slots)[np.clip(mi, 0, len(slots) - 1)] if slots
                    else np.zeros(nt, dtype=bool))
        finally:
            obj_eval.to_mesh_clear()
    if not tri_chunks or count <= 0:
        return []
    A = np.vstack(tri_chunks)                                   # (T,3,3)
    C = np.vstack(col_chunks) if col_chunks else None           # (T,3)
    G = np.concatenate(glass_chunks) if glass_chunks else None  # (T,)
    area = 0.5 * np.linalg.norm(np.cross(A[:, 1] - A[:, 0],
                                         A[:, 2] - A[:, 0]), axis=1)
    good = area > 0.0
    A, area = A[good], area[good]
    if C is not None:
        C = C[good]
    if G is not None:
        G = G[good]
    if not len(A):
        return []
    prob = area / area.sum()
    rng = np.random.default_rng()
    picks = []

    def draw(k):
        pick = rng.choice(len(A), size=int(k), p=prob)
        r1 = np.sqrt(rng.random(len(pick)))[:, None]
        r2 = rng.random(len(pick))[:, None]
        tri = A[pick]
        picks.append(pick)
        return (tri[:, 0] * (1.0 - r1) + tri[:, 1] * (r1 * (1.0 - r2))
                + tri[:, 2] * (r1 * r2))

    count = int(count)
    if crop_bounds is None:
        world = draw(count)
        pick_all = picks[0]
    else:
        c_mn = np.array(tuple(crop_bounds[0]), dtype=np.float64)
        c_mx = np.array(tuple(crop_bounds[1]), dtype=np.float64)
        got, have, frac = [], 0, None
        for _ in range(6):
            need = count - have
            if need <= 0:
                break
            k = need if frac is None else int(need / max(frac, 1e-3) * 1.1) + 1
            k = min(k, count * 50)
            smp = draw(k)
            keep = np.all((smp >= c_mn) & (smp <= c_mx), axis=1)
            inside = smp[keep]
            picks[-1] = picks[-1][keep]
            frac = len(inside) / float(len(smp))
            got.append(inside)
            have += len(inside)
            if frac == 0.0:
                break
        world = np.vstack(got)[:count] if got else np.empty((0, 3))
        pick_all = (np.concatenate(picks)[:count] if picks
                    else np.empty(0, dtype=np.int64))
    Wm = np.array(W, dtype=np.float64)
    out = (world @ Wm.T) * scale
    if world_out is not None:
        world_out.extend(map(Vector, world.tolist()))
    if colors_out is not None and C is not None:
        colors_out.extend(map(tuple, _exp_srgb_bytes_np(C[pick_all]).tolist()))
    if glass_out is not None and G is not None:
        glass_out.extend(G[pick_all].tolist())
    # tolist() -> real Python floats for points3D.txt (v109 fix).
    return out.tolist()


def _exp_point_source_objects(scene, s, extra_exclude=()):
    """Mesh objects the point cloud is built from. Explicitly named
    objects (exp_point_objects) apply unchanged. Otherwise: all visible
    meshes that are also RENDERED -- without helper objects. Excluded
    are objects with hide_render (e.g. the camera sphere), the sphere and
    the guide meshes of the guide list, the crop object and extra_exclude.
    Up to v81 the vertices of the Camera_Sphere thus ended up as points in
    mid-air in points3D.txt (v82 fix)."""
    names = [x.strip() for x in s.exp_point_objects.split(",") if x.strip()]
    mode = s.exp_point_source
    # Scenes from before v129: name list set, Point Source never chosen.
    if names and not s.is_property_set("exp_point_source"):
        mode = 'OBJECTS'
    if mode == 'OBJECTS' and names:
        objs = [bpy.data.objects.get(n) for n in names]
        return [o for o in objs if o and o.type == 'MESH']
    excl = set(o for o in extra_exclude if o is not None)
    if s.sph_object is not None:
        excl.add(s.sph_object)
    if s.use_guide_list:
        for item in s.guides:
            for ref in item.objects:
                if ref.obj is not None:
                    excl.add(ref.obj)
    if s.exp_crop_object is not None:
        excl.add(s.exp_crop_object)
    objs = [o for o in scene.objects
            if o.type == 'MESH' and o.visible_get() and not o.hide_render
            and o not in excl]
    if mode == 'TARGETS':
        colls = [it.coll for it in s.sph_target_colls if it.coll]
        if colls:
            members = {o for c in colls for o in c.all_objects}
            objs = [o for o in objs if o in members]
    return objs


def _exp_vertex_counts(scene, s):
    """(vertices per object, vertices per unique mesh, objects, meshes) of the
    point cloud objects, evaluated incl. modifiers -- the same numbers the
    export uses. Reads the already evaluated meshes (no to_mesh),
    hence cheap even in the panel draw. The point cloud counts each object
    separately, Blender's statistics each mesh only once (linked copies)."""
    objs = _exp_point_source_objects(scene, s)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    per_obj = 0
    unique = {}
    for o in objs:
        oe = o.evaluated_get(depsgraph)
        n = len(oe.data.vertices) if oe.data is not None else 0
        per_obj += n
        unique[o.data.name] = n
    return per_obj, sum(unique.values()), len(objs), len(unique)


def _exp_max_points_effective(s):
    """Upper limit of vertex points: none with Use Vertex Count (all)."""
    return None if s.exp_use_vertex_count else s.exp_max_points


def _exp_srgb_bytes_np(rgb):
    """Vectorized version of _exp_srgb_byte for an (N,3) array
    of linear colors (same threshold, same coefficients, same
    rounding). Returns (N,3) int."""
    c = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    srgb = np.where(c <= 0.0031308, 12.92 * c,
                    1.055 * np.power(c, 1.0 / 2.4) - 0.055)
    return np.round(np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.int64)


def _exp_vertex_first_material(mesh, n):
    """Material index per vertex: that of the first polygon containing the
    vertex (polygon order), otherwise 0 -- like the earlier loop."""
    vm = np.zeros(n, dtype=np.int64)
    npoly = len(mesh.polygons)
    if not npoly:
        return vm
    mi = np.empty(npoly, dtype=np.int32)
    mesh.polygons.foreach_get("material_index", mi)
    total = np.empty(npoly, dtype=np.int32)
    mesh.polygons.foreach_get("loop_total", total)
    nloop = len(mesh.loops)
    lv = np.empty(nloop, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", lv)
    # Blender stores the loops polygon by polygon in sequence (ascending
    # loop_start), loop order = polygon order.
    poly_of_loop = np.repeat(np.arange(npoly), total)
    uniq, first = np.unique(lv, return_index=True)
    vm[uniq] = mi[poly_of_loop[first]]
    return vm


def _exp_gather_points(scene, scale, W, s, world_out=None, glass_out=None):
    """Main cloud from the vertices of the point cloud objects, with crop,
    color and upper limit. Since v110 with numpy instead of a Python loop
    per vertex (Vespa: 5.4 million vertices); same result as before."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    objs = _exp_point_source_objects(scene, s)
    if not objs:
        raise RuntimeError("No mesh objects found for the point cloud.")

    mode = s.exp_point_color
    crop_bounds = _exp_crop_bounds_world(s)
    if crop_bounds is not None:
        c_mn = np.array(tuple(crop_bounds[0]), dtype=np.float64)
        c_mx = np.array(tuple(crop_bounds[1]), dtype=np.float64)
    worlds, colors, glasses = [], [], []
    glass_cache = {}

    for obj in objs:
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        if mesh is None:
            continue
        try:
            n = len(mesh.vertices)
            if not n:
                continue
            co = np.empty(n * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", co)
            mw = np.array(eval_obj.matrix_world, dtype=np.float64)
            world = co.reshape(n, 3) @ mw[:3, :3].T + mw[:3, 3]

            # Color per vertex (linear 0..1); NaN = no color -> gray.
            rgb = None
            if mode != 'NONE':
                rgb = np.full((n, 3), np.nan)
                lookup = (_exp_vertex_color_lookup(mesh)
                          if mode == 'VERTEX' else None)
                if lookup is not None:
                    for vi, c in lookup.items():
                        rgb[vi] = c
                else:
                    mat_colors = np.array(_exp_object_material_colors(obj),
                                          dtype=np.float64)
                    vm = _exp_vertex_first_material(mesh, n)
                    ok = (vm >= 0) & (vm < len(mat_colors))
                    rgb[ok] = mat_colors[vm[ok]]

            # Point on glass (1.3.0): its first material is see-through.
            glass = np.zeros(n, dtype=bool)
            if glass_out is not None:
                slots = _gcapture_see_through_slots(obj, glass_cache)
                if slots:
                    vm = _exp_vertex_first_material(mesh, n)
                    ok = vm >= 0
                    glass[ok] = np.array(slots)[np.clip(vm[ok], 0, len(slots) - 1)]
            if crop_bounds is not None:
                keep = np.all((world >= c_mn) & (world <= c_mx), axis=1)
                world = world[keep]
                glass = glass[keep]
                if rgb is not None:
                    rgb = rgb[keep]
            worlds.append(world)
            glasses.append(glass)
            if rgb is None:
                colors.append(np.full((len(world), 3), 200, dtype=np.int64))
            else:
                grey = np.isnan(rgb).any(axis=1)
                cb = _exp_srgb_bytes_np(np.nan_to_num(rgb))
                cb[grey] = 200
                colors.append(cb)
        finally:
            eval_obj.to_mesh_clear()

    if not worlds:
        return [], []
    world_all = np.vstack(worlds)
    col_all = np.vstack(colors)
    glass_all = np.concatenate(glasses)

    # Downsample together (positions and colors in sync).
    cap = _exp_max_points_effective(s)
    if cap is not None and len(world_all) > cap:
        step = len(world_all) / float(cap)
        idx = np.array([int(i * step) for i in range(cap)], dtype=np.int64)
        world_all = world_all[idx]
        col_all = col_all[idx]
        glass_all = glass_all[idx]

    Wm = np.array(W, dtype=np.float64)
    pts = (world_all @ Wm.T) * scale
    if world_out is not None:
        world_out.extend(map(Vector, world_all.tolist()))
    if glass_out is not None:
        glass_out.extend(glass_all.tolist())
    # tolist(): real Python numbers for points3D.txt (see v109). Rows
    # stay lists -- building tuples from them cost several seconds with
    # 5 million points; the writing code unpacks both the same way.
    return pts.tolist(), col_all.tolist()


_EXP_SIGNATURE_VERSION = 4


def _exp_face_count(s, n_vertex_points):
    """Number of face points as the export uses them (also for the
    preview in the panel): with Use Vertex Count a share of the vertex
    points, at least 100; otherwise the entered value."""
    if s.exp_use_vertex_count:
        return max(100, int(round(n_vertex_points * s.exp_face_points_percent
                                  / 100.0)))
    return s.exp_face_points_count


def _exp_point_signature(scene, s, cam_views, scale, W):
    """Fingerprint of everything that determines the point cloud. Same
    fingerprint -> same point cloud; the export then reuses the most recently
    written points3D.txt (v106). Paths are NOT included.

    v110: instead of only the vertex count, the evaluated vertex positions
    (modifiers, Edit Mode), for color modes the color sources, the face
    points share, view directions and focal length of the cameras, and the
    method of the visibility filter (GPU or ray cast yield slightly
    different sets)."""
    import hashlib
    h = hashlib.sha1()

    def add(*vals):
        h.update(repr(vals).encode("utf-8"))

    add("signature", _EXP_SIGNATURE_VERSION)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    glass_cache = {}
    for o in _exp_point_source_objects(scene, s):
        oe = o.evaluated_get(depsgraph)
        add(o.name, [round(v, 6) for row in oe.matrix_world for v in row])
        # glass lets the visibility filter look through (1.3.0)
        add("see_through", _gcapture_see_through_slots(o, glass_cache))
        me = oe.to_mesh()
        if me is None:
            add("no mesh")
            continue
        try:
            n = len(me.vertices)
            co = np.empty(n * 3, dtype=np.float32)
            me.vertices.foreach_get("co", co)
            add(n)
            h.update(co.tobytes())
            if s.exp_point_color != 'NONE':
                add(_exp_object_material_colors(o))
                npoly = len(me.polygons)
                mi = np.empty(npoly, dtype=np.int32)
                me.polygons.foreach_get("material_index", mi)
                h.update(mi.tobytes())
                ca = getattr(me, "color_attributes", None)
                if s.exp_point_color == 'VERTEX' and ca and len(ca):
                    layer = ca.active_color or ca[0]
                    col = np.empty(len(layer.data) * 4, dtype=np.float32)
                    layer.data.foreach_get("color", col)
                    add(layer.name, layer.domain)
                    h.update(col.tobytes())
        finally:
            oe.to_mesh_clear()
    add([[round(c, 5) for c in pos] + [round(c, 5) for c in look]
         for pos, look in cam_views])
    add(s.exp_use_vertex_count, s.exp_max_points, s.exp_face_points,
        s.exp_face_points_count, round(s.exp_face_points_percent, 6),
        s.exp_face_points_vischeck, s.exp_point_color, s.exp_point_objects,
        s.exp_crop_enable,
        s.exp_crop_object.name if s.exp_crop_object else "",
        round(s.exp_crop_margin, 6), s.exp_zup_to_yup, round(scale, 9),
        [round(v, 6) for row in W for v in row],
        round(s.focal_length, 6), s.look_mode)
    if s.exp_crop_enable and s.exp_crop_object is not None:
        add([round(v, 6) for row in s.exp_crop_object.matrix_world
             for v in row], [tuple(round(c, 6) for c in v)
                             for v in s.exp_crop_object.bound_box])
    if s.exp_face_points_vischeck == 'RAYCAST':
        add("GPU" if not bpy.app.background else "RAY", _EXP_GPU_RES)
    return h.hexdigest()
