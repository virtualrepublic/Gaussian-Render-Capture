"""Rig collection, camera, guide objects, render setup and per-view adjustments."""

import bpy
import bmesh
from mathutils import Vector, Matrix

from ..core.sphere import _SPH_FIT_ATTR, _SPH_SHIFT_ATTR, _sph_collect_objects, _sph_world_bounds


# ----------------------------------------------------------------------
# Core logic
# ----------------------------------------------------------------------
def face_centers_and_normals(obj):
    """Per face (world center, world normal), evaluated incl. modifiers."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)

    bm = bmesh.new()
    bm.from_mesh(eval_obj.to_mesh())
    bm.faces.ensure_lookup_table()

    mw = obj.matrix_world
    normal_matrix = mw.to_3x3().inverted_safe().transposed()

    data = []
    for f in bm.faces:
        center_world = mw @ f.calc_center_median()
        normal_world = (normal_matrix @ f.normal).normalized()
        data.append((center_world, normal_world))

    bm.free()
    eval_obj.to_mesh_clear()
    return data


_GCAPTURE_RIG_COLL = "Capture_Rig"


def _gcapture_rig_collection(scene):
    """Dedicated collection for the camera and Camera Sphere (v136)."""
    coll = bpy.data.collections.get(_GCAPTURE_RIG_COLL)
    if coll is None or coll.library is not None:
        coll = bpy.data.collections.new(_GCAPTURE_RIG_COLL)
    if coll not in scene.collection.children_recursive:
        scene.collection.children.link(coll)
    return coll


def _gcapture_move_to_rig(scene, obj):
    """Keep the object only in the rig collection (v136)."""
    if obj is None:
        return
    coll = _gcapture_rig_collection(scene)
    if coll not in obj.users_collection:
        coll.objects.link(obj)
    for c in list(obj.users_collection):
        if c != coll:
            c.objects.unlink(obj)


def get_or_create_camera(name):
    cam = bpy.data.objects.get(name)
    if cam and cam.type == 'CAMERA':
        _gcapture_move_to_rig(bpy.context.scene, cam)
        return cam
    cam_data = bpy.data.cameras.new(name)
    cam = bpy.data.objects.new(name, cam_data)
    _gcapture_rig_collection(bpy.context.scene).objects.link(cam)
    return cam


def _gcapture_apply_render_setup(scene, s):
    """Apply the render settings from step 5 to the scene (v127) --
    the same as Build does at the end, as far as camera/keyframes exist."""
    cam = bpy.data.objects.get(s.camera_name)
    if cam is not None and cam.type != 'CAMERA':
        cam = None
    fcs = iter_action_fcurves(cam) if cam is not None else []
    if fcs:
        interp = ('CONSTANT' if s.constant_interp else
                  bpy.context.preferences.edit.keyframe_new_interpolation_type)
        for fc in fcs:
            for kp in fc.keyframe_points:
                kp.interpolation = interp
    if s.set_active_camera and cam is not None:
        scene.camera = cam
    if s.set_frame_range and fcs:
        frames = [kp.co[0] for fc in fcs for kp in fc.keyframe_points]
        scene.frame_start = int(round(min(frames)))
        scene.frame_end = int(round(max(frames)))
    if s.set_resolution:
        rnd = scene.render
        rnd.resolution_x = s.resolution
        rnd.resolution_y = s.resolution
        rnd.resolution_percentage = 100


def iter_action_fcurves(obj):
    """Returns all F-curves of the action of 'obj' -- version-safe.

    Blender 4.4+ introduced 'Slotted Actions': action.fcurves was moved
    into a channelbag per slot and is removed entirely in Blender 5.0.
    This function covers both worlds:
      1. new: via action.layers[].strips[].channelbag(slot).fcurves
      2. old: via action.fcurves (legacy, < 4.4)
    """
    ad = obj.animation_data
    if not ad or not ad.action:
        return []
    action = ad.action

    # --- New way (Slotted Actions, 4.4+/5.x) ---
    # Prefers the official helper function if available.
    try:
        from bpy_extras import anim_utils
        slot = getattr(ad, "action_slot", None)
        getter = getattr(anim_utils, "action_get_channelbag_for_slot", None)
        if slot is not None and getter is not None:
            cbag = getter(action, slot)
            if cbag is not None:
                return list(cbag.fcurves)
    except Exception:
        pass

    # Manual way via the layer/strip/channelbag hierarchy.
    try:
        if len(action.layers) > 0:
            strip = action.layers[0].strips[0]
            slot = getattr(ad, "action_slot", None)
            if slot is not None:
                cbag = strip.channelbag(slot)
                if cbag is not None:
                    return list(cbag.fcurves)
            # If no slot is available: take all channelbags of the strip.
            bags = getattr(strip, "channelbags", None)
            if bags:
                fcurves = []
                for cb in bags:
                    fcurves.extend(list(cb.fcurves))
                return fcurves
    except Exception:
        pass

    # --- Legacy way (< 4.4) ---
    if hasattr(action, "fcurves"):
        try:
            return list(action.fcurves)
        except Exception:
            pass

    return []


def setup_guide_object(obj, settings):
    """Shows the guide mesh as wireframe and removes it from the render.
    Always since v93 (previously optional) and without renaming --
    rename_guide, guide_target_name and setup_guide_display remain only as
    properties for old .blend files. Returns the object name."""
    obj.display_type = 'WIRE'      # Viewport: wireframe only
    obj.hide_render = True          # exclude from the final render
    obj.show_in_front = False
    # Additionally remove it from all ray visibilities so that it does
    # not contribute indirectly either (reflections/shadows).
    try:
        obj.visible_camera = False
        obj.visible_diffuse = False
        obj.visible_glossy = False
        obj.visible_transmission = False
        obj.visible_volume_scatter = False
        obj.visible_shadow = False
    except AttributeError:
        # These properties exist only with the Cycles engine active.
        pass

    return obj.name


def _gcapture_has_custom_guides(s):
    """True if the guide list contains its own guide meshes (other than the
    sphere created by 'Camera Sphere')."""
    if not s.use_guide_list:
        return False
    for item in s.guides:
        for ref in item.objects:
            if (ref.obj is not None and ref.obj != s.sph_object
                    and ref.obj.users_scene):
                return True
    return False


def _collect_guide_objects(context, active_obj):
    """Returns the list of guide mesh objects to process.
    With the guide list enabled the collection, otherwise the active object."""
    s = context.scene.gcapture_settings
    if s.use_guide_list and len(s.guides) > 0:
        objs = []
        for item in s.guides:
            for ref in item.objects:
                o = ref.obj
                # Only objects of the current scene (guide meshes deleted in the
                # viewport can live on as orphaned data, v94).
                if (o and o.type == 'MESH' and o not in objs
                        and o.name in context.scene.objects):
                    objs.append(o)
        return objs
    if active_obj and active_obj.type == 'MESH':
        return [active_obj]
    return []


# Custom property on the camera sphere: its center in object
# coordinates (= center of the target bounding box at creation).
_SPH_CENTER_PROP = "gcapture_center"


def _gcapture_sphere_origin_to_center(obj):
    """Moves the origin of a Camera Sphere to its center without moving
    it (spheres before v95 had their origin at the world origin). Uses the
    stored center; without it, everything stays as it is.
    Returns: True if it was changed."""
    stored = obj.get(_SPH_CENTER_PROP) if obj is not None else None
    if stored is None or len(stored) != 3 or obj.type != 'MESH':
        return False
    c = Vector(stored)
    if c.length < 1e-9:
        return False
    obj.data.transform(Matrix.Translation(-c))
    obj.matrix_world = obj.matrix_world @ Matrix.Translation(c)
    obj[_SPH_CENTER_PROP] = (0.0, 0.0, 0.0)
    obj.data.update()
    return True


def _guide_faces_with_targets(obj):
    """Returns per face (center, normal, geo_center, origin) in world coords.
    geo_center is the stored sphere center if the guide mesh is a sphere
    created with 'Camera Sphere', otherwise the bbox center of this object's
    face centers. For the upper hemisphere the bbox of the face centers is
    NOT usable as a target point: it lies far above the model there, and
    the cameras aimed past it (v81 fix)."""
    faces = face_centers_and_normals(obj)
    if not faces:
        return []
    origin = obj.matrix_world.translation.copy()
    stored = obj.get(_SPH_CENTER_PROP)
    if stored is not None and len(stored) == 3:
        # Stored in object coordinates -> follows moving/scaling
        # of the sphere (e.g. with Live Camera Lock).
        geo_center = obj.matrix_world @ Vector(stored)
    else:
        xs = [c.x for (c, _) in faces]
        ys = [c.y for (c, _) in faces]
        zs = [c.z for (c, _) in faces]
        geo_center = Vector((0.5 * (max(xs) + min(xs)),
                             0.5 * (max(ys) + min(ys)),
                             0.5 * (max(zs) + min(zs))))
    # Fill Each View (v96): move the camera per face toward the center by
    # the stored factor. Follows moving/scaling of the sphere (Live Lock).
    fit = None
    if obj.type == 'MESH':
        a = obj.data.attributes.get(_SPH_FIT_ATTR)
        if (a is not None and a.domain == 'FACE'
                and len(a.data) == len(faces)):
            fit = [d.value for d in a.data]
    if fit is not None:
        out = [(geo_center + (c - geo_center) * q, nrm, geo_center, origin)
               for (c, nrm), q in zip(faces, fit)]
    else:
        out = [(c, nrm, geo_center, origin) for (c, nrm) in faces]
    # Center in Each View (v135): offset camera and look target sideways
    # together -- the view direction stays, the model sits centered.
    if obj.type == 'MESH':
        sa = obj.data.attributes.get(_SPH_SHIFT_ATTR)
        if (sa is not None and sa.domain == 'FACE'
                and len(sa.data) == len(faces)):
            rot = obj.matrix_world.to_3x3()
            for i, d in enumerate(sa.data):
                sv = rot @ Vector(d.vector)
                c, nrm, g, o = out[i]
                out[i] = (c + sv, nrm, g + sv, o)
    # Individually adjusted cameras (v123, Finish Live Adjust > Only this
    # camera): position and look target per face in object coordinates.
    for i, pos, tgt in _sph_view_adjustments(obj, len(faces)):
        mw = obj.matrix_world
        out[i] = (mw @ pos, out[i][1], mw @ tgt, origin)
    return out


_SPH_ADJ_FLAG = "gcapture_adj"
_SPH_ADJ_POS = "gcapture_adj_pos"
_SPH_ADJ_TGT = "gcapture_adj_tgt"


def _sph_view_adjustments(obj, n_faces):
    """[(face index, position, look target)] of the individually adjusted
    cameras, both in object coordinates (v123)."""
    if obj.type != 'MESH':
        return []
    attrs = obj.data.attributes
    flag, pos, tgt = (attrs.get(_SPH_ADJ_FLAG), attrs.get(_SPH_ADJ_POS),
                      attrs.get(_SPH_ADJ_TGT))
    if (flag is None or pos is None or tgt is None
            or not (len(flag.data) == len(pos.data) == len(tgt.data) == n_faces)):
        return []
    return [(i, Vector(pos.data[i].vector), Vector(tgt.data[i].vector))
            for i in range(n_faces) if flag.data[i].value]


def _sph_base_matrix(obj):
    """Placement of the sphere after Create / Update (v133)."""
    base = obj.get("gcapture_base_loc")
    if base is not None and len(base) == 3:
        return Matrix.Translation(Vector(base))
    return Matrix.Translation(obj.matrix_world.translation)


def _sph_ensure_base(obj, s):
    """Spheres from before v133 do not know their placement after Create /
    Update: fill it in from the look target collections, the way Create /
    Update computes it (v134)."""
    if obj is None or obj.get("gcapture_base_loc") is not None:
        return
    try:
        bounds = _sph_world_bounds(_sph_collect_objects(s, exclude=obj))
    except Exception:
        bounds = None
    centre = bounds[2] if bounds else obj.matrix_world.translation
    obj["gcapture_base_loc"] = tuple(centre)


def _sph_has_adjustments(obj):
    """True if Live Camera Adjust changed the sphere or adjusted views
    individually (v133)."""
    if obj is None or obj.type != 'MESH':
        return False
    flag = obj.data.attributes.get(_SPH_ADJ_FLAG)
    if flag is not None and any(d.value for d in flag.data):
        return True
    base = _sph_base_matrix(obj)
    return any(abs(a - b) > 1e-5 for ra, rb in zip(obj.matrix_world, base)
               for a, b in zip(ra, rb))


def _sph_reset_adjustments(obj):
    """Reset the sphere to its placement after Create / Update and delete
    individually adjusted views (v133). Returns: whether anything changed."""
    if not _sph_has_adjustments(obj):
        return False
    attrs = obj.data.attributes
    for name in (_SPH_ADJ_FLAG, _SPH_ADJ_POS, _SPH_ADJ_TGT):
        a = attrs.get(name)
        if a is not None:
            attrs.remove(a)
    obj.matrix_world = _sph_base_matrix(obj)
    obj.data.update()
    return True


def _sph_store_view_adjustment(obj, face_idx, pos_local, tgt_local):
    """Stores the pose of an individually adjusted camera on its face."""
    attrs = obj.data.attributes
    n = len(obj.data.polygons)
    for name, typ in ((_SPH_ADJ_FLAG, 'BOOLEAN'), (_SPH_ADJ_POS, 'FLOAT_VECTOR'),
                      (_SPH_ADJ_TGT, 'FLOAT_VECTOR')):
        a = attrs.get(name)
        if a is not None and (a.domain != 'FACE' or a.data_type != typ
                              or len(a.data) != n):
            attrs.remove(a)
            a = None
        if a is None:
            attrs.new(name, typ, 'FACE')
    attrs[_SPH_ADJ_FLAG].data[face_idx].value = True
    attrs[_SPH_ADJ_POS].data[face_idx].vector = tuple(pos_local)
    attrs[_SPH_ADJ_TGT].data[face_idx].vector = tuple(tgt_local)
    obj.data.update()


def _gcapture_look_dir_for(s, center, normal, geo_center, origin):
    """View direction of a camera at the face, consistent with build_animation."""
    if s.look_mode == 'CENTER':
        look_dir = (geo_center - center)
    elif s.look_mode == 'ORIGIN':
        look_dir = (origin - center)
    else:
        look_dir = -normal
    if look_dir.length < 1e-9:
        look_dir = -normal if normal.length > 1e-9 else Vector((0, 0, -1))
    return look_dir.normalized()


def _gcapture_all_guide_faces(context, active_obj=None, guide_objs=None):
    """All faces of all current guide meshes as a flat list, in exactly
    the order that build_animation also uses. If guide_objs is passed
    explicitly (e.g. the objects remembered at lock time), this list is
    used instead of collecting via active_object."""
    if guide_objs is None:
        guide_objs = _collect_guide_objects(context, active_obj)
    faces = []
    for gobj in guide_objs:
        if gobj is not None:
            faces.extend(_guide_faces_with_targets(gobj))
    return faces


def _gcapture_wt_status_build_poses(context, s):
    cam = bpy.data.objects.get(s.camera_name)
    fcs = iter_action_fcurves(cam) if cam is not None else []
    frames = set()
    for fc in fcs:
        for kp in fc.keyframe_points:
            frames.add(int(round(kp.co[0])))
    if frames:
        return 'DONE', "%d camera poses (frames %d-%d)" % (
            len(frames), min(frames), max(frames))
    return 'TODO', "No camera animation yet"
