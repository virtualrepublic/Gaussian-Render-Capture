"""Live Camera Lock: binds one camera live to the guide mesh (state, handler)."""

import bpy
import bpy.app.handlers as _gcapture_handlers

from ..core.scene import (
    _collect_guide_objects, _gcapture_all_guide_faces, _gcapture_look_dir_for,
    iter_action_fcurves)


# ----------------------------------------------------------------------
# Live Camera Lock: binds ONE camera (active frame -> face index) live
# to the guide mesh. Follows scaling/movement immediately without baking
# the whole animation. Lightweight handler (one pose instead of many).
# ----------------------------------------------------------------------

_GCAPTURE_LOCK_BUSY = False
_GCAPTURE_LOCK_START_MW = {}    # guide mesh name -> matrix_world at switch-on (v123)
# Face index frozen at switch-on (variant 1). The handler always keeps
# the camera at this index, regardless of the current frame -- this
# avoids the fragile frame reading in the handler context. None = nothing
# frozen.
_GCAPTURE_LOCK_FROZEN_IDX = None
# Diagnostics: writes to the Blender system console what the lock does.
GCAPTURE_LOCK_DEBUG = False
# Guide mesh objects remembered at switch-on. The handler must NOT rely
# on bpy.context.active_object -- it is unreliable in the handler
# context (the scaled object is not necessarily "active" there).
# Instead we remember the guides explicitly at switch-on.
_GCAPTURE_LOCK_GUIDES = []


def _gcapture_redraw_view3d():
    """Triggers a redraw of all 3D viewports. Uses tag_redraw() -- this
    only redraws and does NOT trigger a depsgraph update (unlike
    cam.update_tag(), which would create an update loop in the handler).
    Needed because setting the camera by script does not otherwise redraw
    the viewport in the handler path -- the change would only become
    visible on the next UI event."""
    wm = bpy.context.window_manager
    if not wm:
        return
    for win in wm.windows:
        scr = win.screen
        if not scr:
            continue
        for area in scr.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _gcapture_lock_apply(context, scene=None, active_obj=None, tag_update=False):
    """Places the active camera for the current frame at the face belonging
    to the frame (frame - frame_start -> face index). 'scene' is passed
    explicitly (in the handler from the scene argument), because
    bpy.context can return stale values in the handler context -- among
    others an old frame_current.

    tag_update: True only from the toggle callback (for the redraw). In the
    handler path it MUST stay False, otherwise cam.update_tag() triggers a
    new depsgraph update -> the handler fires again, this time with the
    camera (instead of the sphere) in the updates, which leads to a frame/
    face offset when scaling."""
    if scene is None:
        scene = context.scene
    s = scene.gcapture_settings
    cam = bpy.data.objects.get(s.camera_name)
    if GCAPTURE_LOCK_DEBUG:
        print("[GCAPTURE Lock] apply: camera_name='%s' -> %s | scene.camera=%s"
              % (s.camera_name,
                 cam.name if cam else "NOT FOUND",
                 scene.camera.name if scene.camera else "None"))
    if cam is None or cam.type != 'CAMERA':
        # If no named camera exists, take the scene camera.
        cam = scene.camera
        if GCAPTURE_LOCK_DEBUG:
            print("[GCAPTURE Lock] apply: falling back to scene.camera=%s"
                  % (cam.name if cam else "None"))
    if cam is None or cam.type != 'CAMERA':
        if GCAPTURE_LOCK_DEBUG:
            print("[GCAPTURE Lock] apply: NO usable camera -> abort")
        return False

    # Faces from the guides remembered at lock (robust against active_object).
    lock_guides = [o for o in _GCAPTURE_LOCK_GUIDES if o is not None]
    faces = _gcapture_all_guide_faces(context, active_obj,
                                 guide_objs=lock_guides if lock_guides else None)
    if not faces:
        return False

    # Variant 1: if an index is frozen (lock active), use it -- do NOT
    # read the current frame. This makes the lock robust against the fragile
    # frame context in the handler. Only as a fallback (e.g. a direct call
    # without freezing) is the frame used.
    if _GCAPTURE_LOCK_FROZEN_IDX is not None:
        idx = _GCAPTURE_LOCK_FROZEN_IDX
    else:
        idx = scene.frame_current - s.frame_start
    if idx < 0 or idx >= len(faces):
        return False  # Index outside the valid face range

    center, normal, geo_center, origin = faces[idx]
    look_dir = _gcapture_look_dir_for(s, center, normal, geo_center, origin)
    cam.data.lens_unit = 'MILLIMETERS'
    cam.data.lens = s.focal_length
    cam.rotation_mode = 'QUATERNION'
    cam.location = center
    cam.rotation_quaternion = look_dir.to_track_quat('-Z', 'Y')

    # UPDATE the keyframe belonging to the index (keyframe_insert) instead
    # of only setting it directly. Reason: if the camera already has keyframes
    # (after a build), the animation overwrites any direct cam.location on
    # the next depsgraph eval -- the camera "does not move". keyframe_insert
    # writes the live value INTO the animation, so it wins.
    # Frame = frame_start + idx (the frame belonging to the face).
    target_frame = s.frame_start + idx
    if cam.animation_data and cam.animation_data.action:
        cam.keyframe_insert(data_path="location", frame=target_frame)
        cam.keyframe_insert(data_path="rotation_quaternion", frame=target_frame)
        # Keep constant interpolation if built that way.
        if s.constant_interp:
            for fcurve in iter_action_fcurves(cam):
                for kp in fcurve.keyframe_points:
                    if abs(kp.co[0] - target_frame) < 0.5:
                        kp.interpolation = 'CONSTANT'
    if GCAPTURE_LOCK_DEBUG:
        print("[GCAPTURE Lock] apply: set '%s' idx %d @frame %d, loc=(%.3f,%.3f,%.3f)"
              % (cam.name, idx, target_frame, center.x, center.y, center.z))
    # Only tag a depsgraph update when explicitly requested (toggle
    # switch-on, for the redraw). NOT in the handler path -- otherwise
    # an update loop with frame/face offset (see docstring).
    if tag_update:
        cam.update_tag()
    return True


def _gcapture_lock_handler(scene, depsgraph):
    global _GCAPTURE_LOCK_BUSY
    if GCAPTURE_LOCK_DEBUG:
        print("[GCAPTURE Lock] handler FIRED. busy=%s" % _GCAPTURE_LOCK_BUSY)
    if _GCAPTURE_LOCK_BUSY:
        return
    s = getattr(scene, "gcapture_settings", None)
    if not s or not s.live_lock:
        if GCAPTURE_LOCK_DEBUG:
            print("[GCAPTURE Lock] handler: lock off or no settings -> skip "
                  "(live_lock=%s)" % (getattr(s, "live_lock", "n/a")))
        return

    # Watched guide meshes: the objects REMEMBERED at switch-on, NOT via
    # active_object (that is unreliable in the handler -- exactly that was
    # the bug: the scaled Camera_Array was not "active", so it did not
    # match the watch set and the update was considered irrelevant).
    guide_objs = [o for o in _GCAPTURE_LOCK_GUIDES if o is not None]
    if not guide_objs:
        # Fallback: collect from the current settings.
        guide_objs = _collect_guide_objects(bpy.context,
                                            bpy.context.active_object)
    if not guide_objs:
        if GCAPTURE_LOCK_DEBUG:
            print("[GCAPTURE Lock] handler: no guide objects -> skip")
        return
    watch = set(guide_objs)
    if GCAPTURE_LOCK_DEBUG:
        names = [o.name for o in guide_objs]
        upd_names = [getattr(getattr(u.id, "original", u.id), "name", "?")
                     for u in depsgraph.updates]
        print("[GCAPTURE Lock] handler: watching %s | updates this tick: %s"
              % (names, upd_names))

    # Variant 1: the face index is frozen. We react ONLY to
    # transform/geometry changes of the guide meshes (scaling/moving) --
    # frame-change handling is no longer needed since the index is fixed.
    # This eliminates the former source of errors (frame reading in handler).
    relevant = False
    for upd in depsgraph.updates:
        orig = getattr(upd.id, "original", upd.id)
        if orig in watch:
            if GCAPTURE_LOCK_DEBUG:
                print("[GCAPTURE Lock] handler: update on guide '%s' "
                      "transform=%s geometry=%s"
                      % (orig.name,
                         getattr(upd, "is_updated_transform", "n/a"),
                         getattr(upd, "is_updated_geometry", "n/a")))
            if (getattr(upd, "is_updated_transform", False) or
                    getattr(upd, "is_updated_geometry", False)):
                relevant = True
                break
    if not relevant:
        return

    if GCAPTURE_LOCK_DEBUG:
        print("[GCAPTURE Lock] handler: relevant update -> applying")
    _GCAPTURE_LOCK_BUSY = True
    try:
        _gcapture_lock_apply(bpy.context, scene=scene)
        # Redraw the viewport so that the camera movement is visible
        # immediately (tag_redraw does NOT trigger a depsgraph update -> no loop).
        _gcapture_redraw_view3d()
    except Exception as exc:
        print("[Gaussian Render Capture] Live Camera Lock failed:", exc)
    finally:
        _GCAPTURE_LOCK_BUSY = False


def _gcapture_lock_installed():
    return _gcapture_lock_handler in _gcapture_handlers.depsgraph_update_post


def _gcapture_lock_install():
    if not _gcapture_lock_installed():
        _gcapture_handlers.depsgraph_update_post.append(_gcapture_lock_handler)


def _gcapture_lock_remove():
    if _gcapture_lock_installed():
        _gcapture_handlers.depsgraph_update_post.remove(_gcapture_lock_handler)


def _gcapture_on_lock_toggle(settings, context):
    """Switches the Live Camera Lock handler on/off. At switch-on the face
    index of the CURRENT frame is frozen (variant 1); the lock then keeps
    the camera at this index, even when scaling/moving the guide mesh.
    For a different face: switch the lock off and on again (it then
    freezes the new frame)."""
    global _GCAPTURE_LOCK_BUSY, _GCAPTURE_LOCK_FROZEN_IDX, _GCAPTURE_LOCK_GUIDES
    if settings.live_lock:
        scene = context.scene
        # Freeze the current frame index.
        _GCAPTURE_LOCK_FROZEN_IDX = scene.frame_current - settings.frame_start
        # Remember the guide meshes NOW (active_object is still reliable here,
        # in the toggle context -- later in the handler it no longer is).
        _GCAPTURE_LOCK_GUIDES = _collect_guide_objects(context,
                                                  context.active_object)
        # Placement of the guide meshes at switch-on (v123): "Only this camera"
        # resets them on finishing.
        _GCAPTURE_LOCK_START_MW.clear()
        for g in _GCAPTURE_LOCK_GUIDES:
            if g is not None:
                _GCAPTURE_LOCK_START_MW[g.name] = g.matrix_world.copy()
        _gcapture_lock_install()
        if GCAPTURE_LOCK_DEBUG:
            print("[GCAPTURE Lock] toggle ON: frozen idx=%d, guides=%s, "
                  "handler installed=%s"
                  % (_GCAPTURE_LOCK_FROZEN_IDX,
                     [o.name for o in _GCAPTURE_LOCK_GUIDES if o],
                     _gcapture_lock_installed()))
        _GCAPTURE_LOCK_BUSY = True
        try:
            _gcapture_lock_apply(context, scene=scene,
                            active_obj=context.active_object,
                            tag_update=True)
        except Exception as exc:
            print("[Gaussian Render Capture] Live Camera Lock init failed:", exc)
        finally:
            _GCAPTURE_LOCK_BUSY = False
        # Force a viewport redraw -- in the property update context there is
        # otherwise no redraw (Blender T74000); the camera would be placed
        # correctly, but the viewport would still show the old state.
        _gcapture_redraw_view3d()
    else:
        _GCAPTURE_LOCK_FROZEN_IDX = None
        _GCAPTURE_LOCK_GUIDES = []
        _gcapture_lock_remove()
