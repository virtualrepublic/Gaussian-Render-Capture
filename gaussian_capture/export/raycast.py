"""Ray-casting helpers adapted from "Gauss Cannon" by Arash Keshmirian
(Warpgate Labs), utils/ray_casting.py - https://github.com/warpgatelabs/gauss-cannon

Copyright (C) 2025 Arash Keshmirian / Warpgate Labs
Adaptation Copyright (C) 2026 Prof. Michael Klein
License: GPL-3.0-or-later (see LICENSE and THIRD_PARTY.md).
"""

import bpy
import math
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from ..core.glass import _gcapture_see_through_slots


# ----------------------------------------------------------------------
# Ray-casting helpers (interior camera detection)
# Derived from "Gauss Cannon" by Arash Keshmirian (Warpgate Labs),
# GPL-3.0-or-later.
# Source: https://github.com/warpgatelabs/gauss-cannon
# Adapted to our data structures; see the license section in the header.
# ----------------------------------------------------------------------
def _rc_is_camera_inside_mesh(context, camera_pos, guide_objects):
    """True if camera_pos lies inside a visible non-guide
    mesh. Odd-even rule: odd number of intersections along
    a ray -> inside."""
    for obj in context.view_layer.objects:
        if (obj.type != 'MESH' or not obj.visible_get()
                or obj.hide_render or obj in guide_objects):
            continue
        mat_inv = obj.matrix_world.inverted_safe()
        pos_local = mat_inv @ camera_pos
        ray_dir = Vector((1.0, 0.0, 0.0))
        count = 0
        cur = pos_local.copy()
        # Limit the iterations to reliably avoid infinite loops.
        for _ in range(1000):
            result, location, normal, index = obj.ray_cast(cur, ray_dir)
            if not result:
                break
            count += 1
            cur = location + ray_dir * 0.0001
        if count % 2 == 1:
            return True
    return False


def _rc_build_visible_mesh_bvh_cache(context, guide_objects):
    """List of world-space BVHTrees of all visible non-guide
    meshes (depsgraph-evaluated, modifiers taken into account)."""
    depsgraph = context.evaluated_depsgraph_get()
    cache = []
    for obj in context.view_layer.objects:
        if (obj.type != 'MESH' or not obj.visible_get()
                or obj.hide_render or obj in guide_objects):
            continue
        obj_eval = obj.evaluated_get(depsgraph)
        mesh = obj_eval.to_mesh()
        if mesh is None:
            continue
        try:
            mw = obj.matrix_world
            world_verts = [mw @ v.co for v in mesh.vertices]
            polys = [list(p.vertices) for p in mesh.polygons]
            if world_verts and polys:
                cache.append(BVHTree.FromPolygons(world_verts, polys))
        finally:
            obj_eval.to_mesh_clear()
    return cache


def _rc_build_near_frustum_bvh(cam_data, cam_pos, look_dir, right, up, aspect):
    """World-space BVH of the volume between camera and near plane.
    PERSP -> 5-vertex pyramid, ORTHO -> 8-vertex box."""
    near_clip = cam_data.clip_start
    if cam_data.type == 'PERSP':
        tan_half = math.tan(cam_data.angle / 2.0)
        half_w = near_clip * tan_half * aspect
        half_h = near_clip * tan_half
        near_center = cam_pos + look_dir * near_clip
        verts = [
            cam_pos.copy(),
            near_center - right * half_w + up * half_h,
            near_center + right * half_w + up * half_h,
            near_center + right * half_w - up * half_h,
            near_center - right * half_w - up * half_h,
        ]
        tris = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1),
                (1, 2, 3), (1, 3, 4)]
    else:
        half_w = cam_data.ortho_scale * aspect / 2.0
        half_h = cam_data.ortho_scale / 2.0
        forward = look_dir * near_clip
        verts = [
            cam_pos - right * half_w - up * half_h,
            cam_pos + right * half_w - up * half_h,
            cam_pos + right * half_w + up * half_h,
            cam_pos - right * half_w + up * half_h,
            cam_pos - right * half_w - up * half_h + forward,
            cam_pos + right * half_w - up * half_h + forward,
            cam_pos + right * half_w + up * half_h + forward,
            cam_pos - right * half_w + up * half_h + forward,
        ]
        tris = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
                (0, 3, 7), (0, 7, 4), (1, 5, 6), (1, 6, 2),
                (0, 4, 5), (0, 5, 1), (2, 6, 7), (2, 7, 3)]
    return BVHTree.FromPolygons(verts, tris)


def _rc_geometry_within_near_clip(cam_data, cam_pos, look_dir, right, up,
                                  mesh_bvh_cache, aspect):
    """True if visible geometry reaches into the camera's near-clip volume
    (two tests: nearest point within near radius + BVH overlap)."""
    if not mesh_bvh_cache:
        return False
    near_clip = cam_data.clip_start
    if cam_data.type == 'PERSP':
        tan_half = math.tan(cam_data.angle / 2.0)
        tan_half_x = tan_half * aspect
        tan_half_y = tan_half
        ortho_half_w = ortho_half_h = 0.0
    else:
        tan_half_x = tan_half_y = 0.0
        ortho_half_w = cam_data.ortho_scale * aspect / 2.0
        ortho_half_h = cam_data.ortho_scale / 2.0

    def in_frustum(rel, forward):
        x_off = abs(rel.dot(right))
        y_off = abs(rel.dot(up))
        if cam_data.type == 'PERSP':
            return x_off <= forward * tan_half_x and y_off <= forward * tan_half_y
        return x_off <= ortho_half_w and y_off <= ortho_half_h

    for mesh_bvh in mesh_bvh_cache:
        location, normal, index, dist = mesh_bvh.find_nearest(cam_pos, near_clip)
        if location is None:
            continue
        rel = location - cam_pos
        forward = rel.dot(look_dir)
        if forward <= 0.0:
            continue
        if in_frustum(rel, forward):
            return True

    frustum_bvh = _rc_build_near_frustum_bvh(cam_data, cam_pos, look_dir,
                                             right, up, aspect)
    for mesh_bvh in mesh_bvh_cache:
        if frustum_bvh.overlap(mesh_bvh):
            return True
    return False


def _exp_build_scene_bvh(objs, see_through=False):
    """Single world-space BVHTree over all given meshes.
    Derived from "Gauss Cannon" by Arash Keshmirian (Warpgate Labs),
    GPL-3.0-or-later;
    see the license section in the header."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    vert_arrays = []
    all_polys = []
    offset = 0
    cache = {}
    for obj in objs:
        if obj.type != 'MESH':
            continue
        # see_through (1.3.0): glass does not block the view
        skip = _gcapture_see_through_slots(obj, cache) if see_through else None
        obj_eval = obj.evaluated_get(depsgraph)
        mesh = obj_eval.to_mesh()
        if mesh is None:
            continue
        try:
            mesh.transform(obj.matrix_world)
            n_verts = len(mesh.vertices)
            flat = np.empty(n_verts * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", flat)
            vert_arrays.append(flat.reshape(-1, 3))
            last = len(skip) - 1 if skip else 0
            all_polys.extend([i + offset for i in p.vertices]
                             for p in mesh.polygons
                             if not (skip and skip[min(p.material_index, last)]))
            offset += n_verts
        finally:
            obj_eval.to_mesh_clear()
    if not vert_arrays or not all_polys:
        return None
    all_verts = np.vstack(vert_arrays)
    return BVHTree.FromPolygons(all_verts.tolist(), all_polys)
