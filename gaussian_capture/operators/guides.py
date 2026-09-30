"""Guide and look-target lists, Group & Align."""

import bpy
import numpy as np
from bpy.types import Operator


# ----------------------------------------------------------------------
# Operator
# ----------------------------------------------------------------------
class GCAPTURE_OT_guide_add(Operator):
    bl_idname = "gcapture.guide_add"
    bl_label = "Add Guide"
    bl_description = "Add the selected mesh objects to the guide list"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.gcapture_settings
        mesh_objs = [o for o in context.selected_objects if o.type == 'MESH']
        if not mesh_objs:
            self.report({'WARNING'}, "No mesh objects selected.")
            return {'CANCELLED'}

        if len(mesh_objs) == 1:
            # Single object -> entry with the object name as label.
            item = s.guides.add()
            item.label = mesh_objs[0].name
            ref = item.objects.add()
            ref.obj = mesh_objs[0]
            msg = "Added guide '%s'." % mesh_objs[0].name
        else:
            # Multiple objects -> one collective entry "Guide" (numbered consecutively).
            existing_labels = {it.label for it in s.guides}
            base = "Guide"
            label = base
            n = 1
            while label in existing_labels:
                n += 1
                label = "%s.%03d" % (base, n)
            item = s.guides.add()
            item.label = label
            for obj in mesh_objs:
                ref = item.objects.add()
                ref.obj = obj
            msg = "Added group '%s' with %d objects." % (label, len(mesh_objs))

        s.guide_index = len(s.guides) - 1
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class GCAPTURE_OT_guide_remove(Operator):
    bl_idname = "gcapture.guide_remove"
    bl_label = "Remove Guide"
    bl_description = "Remove the selected guide from the list"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.gcapture_settings
        if 0 <= s.guide_index < len(s.guides):
            s.guides.remove(s.guide_index)
            s.guide_index = min(s.guide_index, len(s.guides) - 1)
        return {'FINISHED'}


class GCAPTURE_OT_guide_clear(Operator):
    bl_idname = "gcapture.guide_clear"
    bl_label = "Clear Guides"
    bl_description = "Remove all guides from the list"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        context.scene.gcapture_settings.guides.clear()
        context.scene.gcapture_settings.guide_index = 0
        return {'FINISHED'}


class GCAPTURE_OT_coll_add(Operator):
    bl_idname = "gcapture.coll_add"
    bl_label = "Add Collection"
    bl_description = ("Add the active object's collection (or the scene's "
                      "active collection) to the target list")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.gcapture_settings
        # Prefers the active collection of the view layer.
        coll = context.view_layer.active_layer_collection.collection
        if coll is None:
            self.report({'WARNING'}, "No active collection.")
            return {'CANCELLED'}
        existing = {it.coll for it in s.sph_target_colls if it.coll}
        if coll in existing:
            self.report({'INFO'}, "Collection already in list.")
            return {'CANCELLED'}
        item = s.sph_target_colls.add()
        item.coll = coll
        s.sph_coll_index = len(s.sph_target_colls) - 1
        self.report({'INFO'}, "Added collection '%s'." % coll.name)
        return {'FINISHED'}


class GCAPTURE_OT_coll_remove(Operator):
    bl_idname = "gcapture.coll_remove"
    bl_label = "Remove Collection"
    bl_description = "Remove the selected collection from the target list"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.gcapture_settings
        if 0 <= s.sph_coll_index < len(s.sph_target_colls):
            s.sph_target_colls.remove(s.sph_coll_index)
            s.sph_coll_index = min(s.sph_coll_index,
                                   len(s.sph_target_colls) - 1)
        return {'FINISHED'}


class GCAPTURE_OT_coll_clear(Operator):
    bl_idname = "gcapture.coll_clear"
    bl_label = "Clear Collections"
    bl_description = "Remove all collections from the target list"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        context.scene.gcapture_settings.sph_target_colls.clear()
        context.scene.gcapture_settings.sph_coll_index = 0
        return {'FINISHED'}


def _ga_collect_top_parents(colls):
    """Collects the topmost parent objects from the given collections.
    'Topmost' = object has no parent OR its parent is NOT in the
    collections (then this object is the top element of its group).
    Existing groups are preserved this way -- their children are not
    collected individually, only the group's top element. Returns: list
    of unique objects."""
    all_objs = set()
    for coll in colls:
        if coll is None:
            continue
        for o in coll.all_objects:
            all_objs.add(o)
    tops = []
    seen = set()
    for o in all_objs:
        p = o.parent
        # Topmost if there is no parent or the parent is not in the selection.
        if p is None or p not in all_objs:
            if o.name not in seen:
                seen.add(o.name)
                tops.append(o)
    return tops


def _ga_world_bounds(objs, depsgraph, fast_bbox=False):
    """World bounding box over all objs AND their descendants.

    fast_bbox=False (default, precise): numpy-vectorized over the real
    mesh vertices -- exact, even for rotated objects, but much faster
    than a Python vertex loop (foreach_get + matrix multiplication
    in batch).

    fast_bbox=True (fast, slightly less accurate): uses only the 8 corners
    of the object's own bounding box (obj.bound_box) per object. For rotated
    objects the resulting world box is slightly too large.

    Returns (min_v, max_v) as mathutils.Vector or (None, None)."""
    import mathutils
    mn = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    mx = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    found = False

    def _walk(o):
        yield o
        for c in o.children:
            yield from _walk(c)

    for top in objs:
        for o in _walk(top):
            if o.type != 'MESH':
                continue
            ob_eval = o.evaluated_get(depsgraph)
            mw = np.array(ob_eval.matrix_world, dtype=np.float64)  # (4,4)

            if fast_bbox:
                # 8 corners of the local bounding box.
                corners = np.array([list(c) for c in ob_eval.bound_box],
                                   dtype=np.float64)  # (8,3)
                co = corners
            else:
                me = ob_eval.to_mesh()
                if me is None:
                    continue
                n = len(me.vertices)
                if n == 0:
                    ob_eval.to_mesh_clear()
                    continue
                flat = np.empty(n * 3, dtype=np.float32)
                me.vertices.foreach_get("co", flat)
                co = flat.reshape(n, 3)

            # Homogeneous transform: (N,4) @ (4,4)^T -> (N,4).
            homog = np.column_stack([co, np.ones(len(co))])
            world = homog @ mw.T
            wxyz = world[:, :3]
            mn = np.minimum(mn, wxyz.min(axis=0))
            mx = np.maximum(mx, wxyz.max(axis=0))
            found = True

            if not fast_bbox:
                ob_eval.to_mesh_clear()

    if not found:
        return None, None
    return (mathutils.Vector(mn.tolist()), mathutils.Vector(mx.tolist()))


def _gcapture_frame_collections(context, colls):
    """Frames the visible geometry (mesh) of the collections in the 3D viewport
    (View Selected), in the viewport of the button, otherwise in the first
    of the windows. The selection is restored afterwards (v117)."""
    if bpy.app.background:
        return
    win = context.window
    area = context.area if context.area and context.area.type == 'VIEW_3D' else None
    if area is None and win is not None:
        area = next((a for a in win.screen.areas if a.type == 'VIEW_3D'), None)
    if area is None:
        return
    region = next((r for r in area.regions if r.type == 'WINDOW'), None)
    vl = context.view_layer
    objs = {o for c in colls if c for o in c.all_objects
            if o.type == 'MESH' and o.name in vl.objects and o.visible_get()}
    if region is None or not objs:
        return
    prev_sel = [o for o in vl.objects if o.select_get()]
    prev_act = vl.objects.active
    try:
        for o in prev_sel:
            o.select_set(False)
        for o in objs:
            o.select_set(True)
        with context.temp_override(window=win, area=area, region=region):
            bpy.ops.view3d.view_selected()
    except Exception as exc:
        print("[Gaussian Render Capture] framing skipped: %s" % exc)
    finally:
        for o in objs:
            try:
                o.select_set(False)
            except RuntimeError:
                pass
        for o in prev_sel:
            try:
                o.select_set(True)
            except RuntimeError:
                pass
        vl.objects.active = prev_act


class GCAPTURE_OT_group_align(Operator):
    bl_idname = "gcapture.group_align"
    bl_label = "Group & Align to Ground"
    bl_description = ("Group the top-level objects of the target collection(s) "
                     "under an Empty 'GCapture_Group'. The Empty sits centered in "
                     "X/Y and at the lowest point (ground) in Z. The whole "
                     "group is then moved so the ground sits on Z=0. Existing "
                     "sub-groups are kept intact; child transforms are "
                     "preserved")
    bl_options = {'REGISTER', 'UNDO'}

    EMPTY_NAME = "GCapture_Group"

    @classmethod
    def poll(cls, context):
        s = context.scene.gcapture_settings
        return len(s.sph_target_colls) > 0

    def execute(self, context):
        import mathutils
        s = context.scene.gcapture_settings
        colls = [it.coll for it in s.sph_target_colls if it.coll]
        if not colls:
            self.report({'ERROR'}, "No target collection set.")
            return {'CANCELLED'}

        tops = _ga_collect_top_parents(colls)
        if not tops:
            self.report({'ERROR'}, "No top-level objects found.")
            return {'CANCELLED'}

        depsgraph = context.evaluated_depsgraph_get()
        mn, mx = _ga_world_bounds(tops, depsgraph, fast_bbox=s.ga_fast_bbox)
        if mn is None:
            self.report({'ERROR'}, "No mesh geometry found for bounds.")
            return {'CANCELLED'}

        # Empty position (reference): X/Y center of the bbox, Z at floor (min Z).
        cx = (mn.x + mx.x) * 0.5
        cy = (mn.y + mx.y) * 0.5
        cz = mn.z
        ref = mathutils.Vector((cx, cy, cz))

        # Translation vector so that this reference point (and thus the
        # empty) ends up at the world origin 0/0/0. ALL top objects are
        # moved by the same vector -> relative arrangement stays,
        # car stands centered above the origin, wheels at Z=0.
        d = -ref

        # Reuse the existing GCapture_Group empty or create a new one.
        empty = bpy.data.objects.get(self.EMPTY_NAME)
        if empty is None or empty.type != 'EMPTY':
            empty = bpy.data.objects.new(self.EMPTY_NAME, None)
            empty.empty_display_type = 'PLAIN_AXES'
            empty.empty_display_size = max((mx - mn).length * 0.1, 0.1)
            colls[0].objects.link(empty)

        # First move all top objects by d (world translation),
        # while they are NOT yet parented.
        for o in tops:
            if o == empty:
                continue
            o.location = o.location + d

        # Empty to the world origin.
        empty.location = mathutils.Vector((0.0, 0.0, 0.0))
        context.view_layer.update()

        # Parent with the (already moved) world transform preserved.
        for o in tops:
            if o == empty:
                continue
            if o.parent == empty:
                continue
            mw = o.matrix_world.copy()
            o.parent = empty
            o.matrix_parent_inverse = empty.matrix_world.inverted()
            o.matrix_world = mw

        # Sphere target info: geometric center (after the move) -- the
        # sphere is created separately and sits volume-centered.
        sph_center_z = (mn.z + mx.z) * 0.5 + d.z
        self.report(
            {'INFO'},
            "Grouped %d top object(s) under '%s' at world origin. Ground on "
            "Z=0. Sphere center will be at Z=%.3f (volume center)."
            % (len(tops), self.EMPTY_NAME, sph_center_z))
        _gcapture_frame_collections(context, colls)
        return {'FINISHED'}
