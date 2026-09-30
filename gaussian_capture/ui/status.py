"""Step status of the guide (done / open / hint) per step."""

import bpy
import os

from ..core.render import _gcapture_format_ext, _gcapture_scene_prepared
from ..core.scene import _gcapture_wt_status_build_poses, iter_action_fcurves
from ..core.versions import _exp_output_has_export, _exp_resolve_output_dir
from ..export.colmap import _exp_detect_frames, _exp_resolve_image_dir
from ..operators.guides import GCAPTURE_OT_group_align


def _gcapture_wt_status_camera(context, s):
    if _gcapture_scene_prepared(context.scene):
        return 'DONE', "Scene prepared (Cycles on GPU)"
    return 'TODO', "Scene not prepared yet"


def _gcapture_wt_status_target(context, s):
    names = [it.coll.name for it in s.sph_target_colls if it.coll]
    if names:
        return 'DONE', "Target: %s" % ", ".join(names)
    return 'TODO', "No collection added yet"


def _gcapture_wt_status_group(context, s):
    if bpy.data.objects.get(GCAPTURE_OT_group_align.EMPTY_NAME) is not None:
        return 'DONE', "Grouped under '%s'" % GCAPTURE_OT_group_align.EMPTY_NAME
    return 'OPTIONAL', "Not grouped yet (optional)"


def _gcapture_wt_status_sphere(context, s):
    sph = s.sph_object
    if sph is not None and sph.name in context.scene.objects:
        return 'DONE', "%s: %d cameras" % (sph.name, len(sph.data.polygons))
    return 'TODO', "No camera sphere yet"


def _gcapture_wt_status_build(context, s):
    if not bpy.data.filepath:
        cam = bpy.data.objects.get(s.camera_name)
        if cam is not None and iter_action_fcurves(cam):
            return 'TODO', "Built - now save the scene (Save Scene Version)"
    return _gcapture_wt_status_build_poses(context, s)


def _gcapture_wt_status_render(context, s):
    scene = context.scene
    expected = scene.frame_end - scene.frame_start + 1
    folder = _exp_resolve_image_dir(s)
    _, _, frames = _exp_detect_frames(folder,
                                      prefer=_gcapture_format_ext(scene))
    have = len([f for f in frames
                if scene.frame_start <= f <= scene.frame_end])
    short = os.path.basename(os.path.normpath(folder)) if folder else "?"
    if have >= expected > 0:
        return 'DONE', "All %d images found in '%s'" % (expected, short)
    if have:
        return 'MISSING', "%d of %d images found in '%s'" % (have, expected, short)
    return 'MISSING', "No rendered images found in '%s'" % short


def _gcapture_wt_status_export(context, s):
    out_dir = os.path.normpath(_exp_resolve_output_dir(s))
    tail = os.path.join(os.path.basename(os.path.dirname(out_dir)),
                        os.path.basename(out_dir))
    if _exp_output_has_export(out_dir):
        return 'DONE', "Dataset written: %s" % tail
    return 'TODO', "Not exported yet (%s)" % os.path.basename(out_dir)


def _gcapture_wt_status_clean(context, s):
    if s.clean_last_result:
        # numbers only: the file is named in the field above, and a longer
        # line would be cut off in the guide
        return 'DONE', s.clean_last_result.partition(" - ")[0]
    return 'OPTIONAL', "Optional, after training: pick the splat and clean it"
