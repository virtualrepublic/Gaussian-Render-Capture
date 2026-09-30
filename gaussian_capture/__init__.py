"""
Gaussian Render Capture - Blender add-on
=========================================================================
Turns a 3D model into a ready-to-train COLMAP dataset for Gaussian
Splatting (Postshot, LichtFeld Studio): cameras on a sphere around the
model, one rendered image per camera, exact camera poses and a start point
cloud. Panel: 3D viewport sidebar (N), tab "Gaussian Render Capture";
beginners press "Start Guide". Documentation: README.md and docs/.

Copyright (C) 2026 Prof. Michael Klein
Portions (ray-casting helpers, see below) Copyright (C) 2025 Arash
Keshmirian / Warpgate Labs.
License: GPL-3.0-or-later (see LICENSE).

Third-party code: the add-on started from "Gauss Cannon" by Arash
Keshmirian (Warpgate Labs; GPL-3.0-or-later,
https://github.com/warpgatelabs/gauss-cannon). Adapted
from its utils/ray_casting.py, now in export/raycast.py:
    _rc_is_camera_inside_mesh, _rc_build_visible_mesh_bvh_cache,
    _rc_build_near_frustum_bvh, _rc_geometry_within_near_clip
        (Skip Interior Cameras, Advanced)
    _exp_build_scene_bvh
        (ray-cast fallback of the visibility filter)
and the idea of a list of guide meshes. Everything else is original work.
See THIRD_PARTY.md.

Maintainer: Prof. Michael Klein
    Digital Film Design – Animation/VFX · Mediadesign University of
    Applied Sciences
    https://www.mediadesign.de  https://www.virtualrepublic.org
    https://www.renderbricks.com
    https://www.linkedin.com/in/virtualrepublic/

Transparency note on the use of AI: the author is not a programmer but
has worked in CGI since 1987. The add-on was developed entirely
through vibe coding with Anthropic Claude (Claude Code), which wrote the
code and the tests. Concept, design decisions, tests and the acceptance
of every version: the author.

Code comments are in English.
=========================================================================
"""

bl_info = {
    "name": "Gaussian Render Capture",
    "author": "Prof. Michael Klein - Mediadesign University of Applied Sciences",
    "version": (1, 3, 1),
    "blender": (4, 1, 0),
    "location": "View3D > Sidebar (N) > Gaussian Render Capture",
    "description": "Synthetic COLMAP datasets for Gaussian Splatting: camera "
                   "sphere, render, export for Postshot / LichtFeld. "
                   "Vibe-coded with Anthropic Claude (Claude Code)",
    "doc_url": "https://github.com/virtualrepublic/Gaussian-Render-Capture",
    "tracker_url": "https://github.com/virtualrepublic/Gaussian-Render-Capture/issues",
    "license": ["SPDX:GPL-3.0-or-later"],
    "category": "Camera",
}

import bpy
from bpy.props import PointerProperty
import bpy.app.handlers as _gcapture_handlers

from .core.live_lock import _gcapture_lock_remove
from .core.versions import (
    _GCAPTURE_MSGBUS_OWNER, _gcapture_migrate_timer, _gcapture_msgbus_subscribe,
    _gcapture_on_load_post, _gcapture_on_save_pre)
from .operators.clean import GCAPTURE_OT_clean_pick, GCAPTURE_OT_clean_splat
from .operators.export import GCAPTURE_OT_export_colmap, GCAPTURE_OT_path_relative
from .operators.guides import (
    GCAPTURE_OT_coll_add, GCAPTURE_OT_coll_clear, GCAPTURE_OT_coll_remove,
    GCAPTURE_OT_group_align, GCAPTURE_OT_guide_add, GCAPTURE_OT_guide_clear,
    GCAPTURE_OT_guide_remove)
from .operators.render import (
    GCAPTURE_OT_confirm_resolution, GCAPTURE_OT_prepare_scene, GCAPTURE_OT_render_images,
    GCAPTURE_OT_save_version, GCAPTURE_OT_selection_to_collection)
from .operators.sphere import (
    GCAPTURE_OT_build, GCAPTURE_OT_lock_finish, GCAPTURE_OT_lock_start, GCAPTURE_OT_make_sphere)
from .settings import (
    GCAPTURE_CollItem, GCAPTURE_GuideItem, GCAPTURE_GuideObjRef, GCAPTURE_Settings)
from .ui.icons import _gcapture_icons_load, _gcapture_icons_unload
from .ui.lists import GCAPTURE_UL_colls, GCAPTURE_UL_guides
from .ui.panels import (
    GCAPTURE_PT_advanced, GCAPTURE_PT_build, GCAPTURE_PT_camera, GCAPTURE_PT_clean,
    GCAPTURE_PT_export, GCAPTURE_PT_group, GCAPTURE_PT_panel, GCAPTURE_PT_render,
    GCAPTURE_PT_sphere, GCAPTURE_PT_target)
from .ui.walkthrough import (
    GCAPTURE_OT_walkthrough_exit, GCAPTURE_OT_walkthrough_nav, GCAPTURE_OT_walkthrough_start)


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------
classes = (
    GCAPTURE_GuideObjRef,
    GCAPTURE_GuideItem,
    GCAPTURE_CollItem,
    GCAPTURE_Settings,
    GCAPTURE_OT_guide_add,
    GCAPTURE_OT_guide_remove,
    GCAPTURE_OT_guide_clear,
    GCAPTURE_OT_coll_add,
    GCAPTURE_OT_coll_remove,
    GCAPTURE_OT_coll_clear,
    GCAPTURE_OT_group_align,
    GCAPTURE_OT_make_sphere,
    GCAPTURE_OT_lock_start,
    GCAPTURE_OT_lock_finish,
    GCAPTURE_OT_build,
    GCAPTURE_OT_path_relative,
    GCAPTURE_OT_export_colmap,
    GCAPTURE_OT_walkthrough_start,
    GCAPTURE_OT_prepare_scene,
    GCAPTURE_OT_render_images,
    GCAPTURE_OT_confirm_resolution,
    GCAPTURE_OT_selection_to_collection,
    GCAPTURE_OT_save_version,
    GCAPTURE_OT_walkthrough_nav,
    GCAPTURE_OT_walkthrough_exit,
    GCAPTURE_OT_clean_pick,
    GCAPTURE_OT_clean_splat,
    GCAPTURE_UL_guides,
    GCAPTURE_UL_colls,
    GCAPTURE_PT_panel,
    GCAPTURE_PT_camera,
    GCAPTURE_PT_target,
    GCAPTURE_PT_group,
    GCAPTURE_PT_sphere,
    GCAPTURE_PT_build,
    GCAPTURE_PT_render,
    GCAPTURE_PT_export,
    GCAPTURE_PT_clean,
    GCAPTURE_PT_advanced,
)


def _gcapture_check_single_install():
    """Refuse to register next to another copy (1.3.1): the old single file
    gaussian_capture.py enabled beside the package would register the same
    classes twice. Raises before anything is registered, so the other copy
    keeps working."""
    other = getattr(bpy.types, "GCAPTURE_PT_panel", None)
    if other is not None and not other.__module__.startswith(__name__ + "."):
        raise RuntimeError(
            "Remove the old Gaussian Render Capture (gaussian_capture.py) in "
            "Preferences → Add-ons first.")


def register():
    _gcapture_check_single_install()
    _gcapture_icons_load()
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.gcapture_settings = PointerProperty(type=GCAPTURE_Settings)
    if _gcapture_on_load_post not in _gcapture_handlers.load_post:
        _gcapture_handlers.load_post.append(_gcapture_on_load_post)
    if _gcapture_on_save_pre not in _gcapture_handlers.save_pre:
        _gcapture_handlers.save_pre.append(_gcapture_on_save_pre)
    try:
        _gcapture_msgbus_subscribe()
    except Exception as exc:
        print("[Gaussian Render Capture] msgbus:", exc)
    # bpy.data is locked on enable -> apply shortly after.
    bpy.app.timers.register(_gcapture_migrate_timer, first_interval=0.1)


def unregister():
    # Remove the live-lock handler safely.
    _gcapture_lock_remove()
    for lst, fn in ((_gcapture_handlers.load_post, _gcapture_on_load_post),
                    (_gcapture_handlers.save_pre, _gcapture_on_save_pre)):
        if fn in lst:
            lst.remove(fn)
    bpy.msgbus.clear_by_owner(_GCAPTURE_MSGBUS_OWNER)
    del bpy.types.Scene.gcapture_settings
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    _gcapture_icons_unload()
