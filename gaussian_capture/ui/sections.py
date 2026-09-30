"""Step contents shared by the subpanels and the guide card."""

import bpy
import os
import sys
import textwrap as _gcapture_textwrap

from ..core import live_lock
from ..core.scene import (
    _collect_guide_objects, _gcapture_wt_status_build_poses, _guide_faces_with_targets,
    _sph_has_adjustments)
from ..core.versions import _exp_output_has_export, _exp_resolve_output_dir, _exp_same_dir
from ..export.colmap import _exp_resolve_image_dir
from ..export.points import _exp_face_count, _exp_vertex_counts
from ..ui.status import _gcapture_wt_status_render
from ..ui.widgets import _gcapture_action, _gcapture_lock, _gcapture_progress_draw


# ----------------------------------------------------------------------
# Section contents: used by subpanels AND the guide card (v89)
# ----------------------------------------------------------------------
def _gcapture_draw_camera(layout, context):
    s = context.scene.gcapture_settings
    obj = context.active_object
    sph = s.sph_object
    if s.use_guide_list and sph is not None and sph.name in context.scene.objects:
        n = len(sph.data.polygons)
        layout.label(text="Guide: %s (%d faces -> %d frames)"
                     % (sph.name, n, n), icon='MESH_ICOSPHERE')
    elif obj and obj.type == 'MESH':
        face_count = len(obj.data.polygons)
        layout.label(text="Active: %s" % obj.name, icon='MESH_ICOSPHERE')
        layout.label(text="Faces: %d  ->  %d frames"
                     % (face_count, face_count))
    else:
        layout.label(text="Cameras come from the Camera Sphere (step 4)",
                     icon='INFO')
    ccol = layout.column(align=True)
    ccol.prop(s, "camera_name")
    ccol.prop(s, "frame_start")
    ccol.prop(s, "focal_length")
    ccol.prop(s, "look_mode")
    # Warning: non-uniform scale + normal mode do not go together.
    if obj and obj.type == 'MESH' and s.look_mode == 'NORMAL':
        sc = obj.scale
        non_uniform = (abs(sc.x - sc.y) > 1e-4 or abs(sc.y - sc.z) > 1e-4)
        if non_uniform:
            wbox = layout.box()
            wbox.alert = True
            wbox.label(text="Non-uniform scale detected!", icon='ERROR')
            wbox.label(text="Normals do not point to the center.")
            wbox.label(text="-> Choose 'Geometry Center'.")


def _gcapture_draw_target(layout, context):
    s = context.scene.gcapture_settings
    row = layout.row()
    row.enabled = bool(context.selected_objects)
    _gcapture_action(row, "gcapture.selection_to_collection",
                not any(it.coll for it in s.sph_target_colls), 'COLLECTION_NEW')
    if not s.wt_active:
        layout.label(text="Which collection the cameras look at:")
    srow = layout.row()
    srow.template_list("GCAPTURE_UL_colls", "gcapture_colls", s, "sph_target_colls",
                       s, "sph_coll_index", rows=3)
    scol = srow.column(align=True)
    scol.operator("gcapture.coll_add", text="", icon='ADD')
    scol.operator("gcapture.coll_remove", text="", icon='REMOVE')
    scol.operator("gcapture.coll_clear", text="", icon='TRASH')


def _gcapture_draw_group(layout, context):
    s = context.scene.gcapture_settings
    if not s.wt_active:
        layout.label(text="Group models under an Empty, put them on Z=0:")
    layout.prop(s, "ga_fast_bbox")
    garow = layout.row()
    garow.scale_y = 1.2
    garow.operator("gcapture.group_align", icon='EMPTY_AXIS')


def _gcapture_draw_sphere(layout, context):
    s = context.scene.gcapture_settings
    if not s.wt_active:
        layout.label(text="Where the cameras sit (auto-fit):")
    layout.prop(s, "sph_subdivisions")
    layout.prop(s, "sph_fit_each")
    crow = layout.row()
    crow.enabled = s.sph_fit_each
    crow.prop(s, "sph_center_view")
    layout.prop(s, "sph_margin")
    layout.prop(s, "sph_upper_only")
    layout.prop(s, "sph_set_as_guide")
    _gcapture_action(layout, "gcapture.make_sphere",
                not (s.sph_object is not None
                     and s.sph_object.name in context.scene.objects),
                'MESH_ICOSPHERE')


def _gcapture_draw_build(layout, context):
    s = context.scene.gcapture_settings
    outer = layout
    if not s.wt_active:
        _gcapture_lock(outer).label(text="One keyframe per sphere face:")
    if not _gcapture_progress_draw(outer, "gcapture.build_animation"):
        brow = _gcapture_lock(outer).row()
        brow.scale_y = 1.4
        # While Live Camera Lock is active: Build blocked (Finish first).
        brow.enabled = not s.live_lock
        _gcapture_action(brow, "gcapture.build_animation",
                    _gcapture_wt_status_build_poses(context, s)[0] != 'DONE',
                    'CON_CAMERASOLVER')
    layout = _gcapture_lock(outer)
    if not s.live_lock and _sph_has_adjustments(s.sph_object):
        wrow = layout.row()
        wrow.alert = True
        wrow.label(text="Adjusted views - Build discards them", icon='ERROR')
    # Takes effect on the next Build -> belongs here, not to step 5 (v126).
    layout.prop(s, "save_after_build")
    if s.live_lock:
        layout.label(text="Finish Live Camera Adjust before rebuilding",
                     icon='INFO')

    # Optional: Live Camera Lock (fine adjustment). Enabling it immediately
    # starts scaling the sphere (v95).
    lbox = layout.box()
    if not s.live_lock:
        lbox.operator("gcapture.lock_start", icon='CON_CAMERASOLVER')
    else:
        lbox.label(text="Live Camera Adjust active", icon='REC')
        hint = lbox.column(align=True)
        hint.label(text="Press S in the viewport to scale the sphere",
                   icon='EVENT_S')
        hint.label(text="(G moves it) - the camera follows live.")
    if s.live_lock:
        faces_total = sum(len(_guide_faces_with_targets(o))
                          for o in _collect_guide_objects(
                              context, context.active_object))
        frozen = live_lock._GCAPTURE_LOCK_FROZEN_IDX
        if frozen is not None and 0 <= frozen < faces_total:
            lbox.label(text="Adjusting face %d / %d"
                       % (frozen, faces_total), icon='TRACKER')
            cur_idx = context.scene.frame_current - s.frame_start
            if cur_idx != frozen:
                lbox.label(text="Finish, then start again for this frame",
                           icon='INFO')
        else:
            lbox.label(text="Frame out of face range (0..%d)"
                       % (max(faces_total - 1, 0)), icon='INFO')
        lbox.label(text="Finish - apply the adjustment to:")
        frow = lbox.row(align=True)
        frow.scale_y = 1.3
        op = frow.operator("gcapture.lock_finish", text="This Camera",
                           icon='CAMERA_DATA')
        op.apply_to = 'THIS'
        op = frow.operator("gcapture.lock_finish", text="All Cameras",
                           icon='MESH_ICOSPHERE')
        op.apply_to = 'ALL'
        crow = lbox.row()
        op = crow.operator("gcapture.lock_finish", text="Cancel", icon='X')
        op.apply_to = 'CANCEL'


def _gcapture_draw_render(layout, context):
    s = context.scene.gcapture_settings
    # Only the resolution; the always-needed switches are under
    # Advanced > Render Setup (v129).
    rrow = layout.row(align=True)
    rrow.enabled = s.set_resolution
    # Blue once at the start: deliberately confirm or change (v140).
    if s.set_resolution and not context.scene.get("gcapture_res_ok"):
        split = rrow.split(factor=0.8, align=True)
        split.prop(s, "resolution")
        split.operator("gcapture.confirm_resolution", text="OK", depress=True)
    else:
        rrow.prop(s, "resolution")

    # Save scene + render target (v90).
    box = layout.box()
    if bpy.data.filepath:
        box.label(text="Scene: %s" % os.path.basename(bpy.data.filepath),
                  icon='FILE_BLEND')
    else:
        row = box.row()
        row.alert = True
        row.label(text="Scene not saved yet", icon='FILE_BLEND')
    out_images = os.path.join(_exp_resolve_output_dir(s), "images")
    if _exp_same_dir(_exp_resolve_image_dir(s), out_images):
        tail = os.path.relpath(out_images, os.path.dirname(
            os.path.dirname(os.path.dirname(out_images))))
        box.label(text="Renders go to %s" % tail, icon='CHECKMARK')
    else:
        box.label(text="Render output: %s" % context.scene.render.filepath,
                  icon='OUTPUT')
    _gcapture_action(box, "gcapture.save_version", not bpy.data.filepath, 'FILE_TICK')

    # Render directly from the add-on (v1.1.5). Blue: the open step, namely
    # the engine the scene is currently set to.
    pending = _gcapture_wt_status_render(context, s)[0] != 'DONE'
    eevee = context.scene.render.engine != 'CYCLES'
    rcol = layout.column(align=True)
    rcol.enabled = bool(bpy.data.filepath)
    rcol.row(align=True).prop(s, "render_format", expand=True)
    row = rcol.row(align=True)
    row.scale_y = 1.4
    # The label says what the click does: render or write a script.
    if s.render_headless:
        ext = (".cmd" if sys.platform.startswith("win") else
               ".command" if sys.platform == "darwin" else ".sh")
        t_cyc, t_eev = "Write Cycles " + ext, "Write EEVEE " + ext
    else:
        t_cyc, t_eev = "Render Cycles", "Render EEVEE"
    op = row.operator("gcapture.render_images", text=t_eev,
                      icon='CONSOLE' if s.render_headless else 'SHADING_TEXTURE',
                      depress=pending and eevee)
    op.engine = 'EEVEE'
    op = row.operator("gcapture.render_images", text=t_cyc,
                      icon='CONSOLE' if s.render_headless else 'SHADING_RENDERED',
                      depress=pending and not eevee)
    op.engine = 'CYCLES'
    rcol.prop(s, "render_headless")
    if s.render_headless and s.render_script:
        rcol.label(text="Double-click: %s" % os.path.basename(s.render_script),
                   icon='CONSOLE')
    if not s.wt_active:
        layout.label(text="Render farm: render the saved scene there.",
                     icon='INFO')


def _gcapture_section(layout, title, icon):
    """Box with a heading, structures long steps (v1.1.5)."""
    box = layout.box()
    box.label(text=title, icon=icon)
    return box.column()


def _gcapture_draw_export(layout, context):
    s = context.scene.gcapture_settings
    outer = layout
    layout = _gcapture_lock(outer)

    # --- Dataset: where to, and where the images come from ---------------
    ds = _gcapture_section(layout, "Dataset", 'FILE_FOLDER')
    ecol = ds.column(align=True)
    # Label above the field: in the narrow sidebar it was cut off
    # otherwise, and the path gets the full width (v129).
    ecol.label(text="Output Folder:")
    orow = ecol.row(align=True)
    orow.prop(s, "exp_output_dir", text="")
    if (s.exp_output_dir or "").strip() and not s.exp_output_dir.startswith("//"):
        op = orow.operator("gcapture.path_relative", text="", icon='DOT')
        op.prop = "exp_output_dir"
    if not (s.exp_output_dir or "").strip():
        try:
            auto = _exp_resolve_output_dir(s)
            tail = os.path.join(os.path.basename(os.path.dirname(auto)),
                                os.path.basename(auto))
            ecol.label(text="Auto: %s" % tail, icon='FILE_FOLDER')
        except Exception:
            pass
    ecol.label(text="Image Folder (rendered images):")
    irow = ecol.row(align=True)
    irow.prop(s, "exp_image_dir", text="")
    if (s.exp_image_dir or "").strip() and not s.exp_image_dir.startswith("//"):
        op = irow.operator("gcapture.path_relative", text="", icon='DOT')
        op.prop = "exp_image_dir"
    # If the scene already renders into the dataset, Copy/Move are unnecessary.
    inplace = _exp_same_dir(_exp_resolve_image_dir(s),
                            os.path.join(_exp_resolve_output_dir(s), "images"))
    if inplace:
        ds.label(text="Images render into the dataset", icon='CHECKMARK')
    else:
        ds.prop(s, "exp_image_mode")
        if s.exp_image_mode == 'MOVE':
            wcol = ds.column(align=True)
            wrow = wcol.row()
            wrow.alert = True
            wrow.label(text="Move removes images from the render folder!",
                       icon='ERROR')
            wcol.label(text="Only runs if all images are present", icon='INFO')
        elif s.exp_image_mode == 'NONE':
            # LichtFeld/COLMAP read images/ next to sparse/ -- no images, no
            # training. With a separate image folder this is the most common mistake.
            wrow = ds.row()
            wrow.alert = True       # always red: no images, no training (v125)
            wrow.label(text="Images are not in the dataset - choose Copy or Move",
                       icon='ERROR')

    # --- Start points: source, amount, filter, reuse -----------------------
    pts = _gcapture_section(layout, "Start Points", 'OUTLINER_DATA_POINTCLOUD')
    pcol = pts.column(align=True)
    pcol.label(text="Point Source:")
    pcol.prop(s, "exp_point_source", text="")
    if s.exp_point_source == 'OBJECTS':
        pcol.prop(s, "exp_point_objects")
    elif s.exp_point_source == 'TARGETS':
        names = [it.coll.name for it in s.sph_target_colls if it.coll]
        pcol.label(text="From: %s" % (", ".join(names) if names
                                       else "no collection - all visible meshes"),
                   icon='OUTLINER_COLLECTION')
    vcol = pts.column(align=True)
    vcol.prop(s, "exp_use_vertex_count")
    # Determine vertex counts only once per redraw (v110).
    try:
        counts = _exp_vertex_counts(context.scene, s)
    except Exception:
        counts = None
    if s.exp_use_vertex_count:
        try:
            n_obj, n_unique, k_obj, k_mesh = counts
            vcol.label(text="{:,} vertices".format(n_obj), icon='VERTEXSEL')
            if n_unique != n_obj:
                vcol.label(text="{:,} objects, {:,} unique meshes".format(
                    k_obj, k_mesh))
            if n_obj > 5000000:
                vcol.label(text="More than LichtFeld's Max Cap", icon='INFO')
                vcol.label(text="(5,000,000) - raise it there")
        except Exception:
            pass
    else:
        vcol.prop(s, "exp_max_points")
    fcol = pts.column(align=True)
    fcol.prop(s, "exp_face_points")
    if s.exp_face_points:
        if s.exp_use_vertex_count:
            fcol.prop(s, "exp_face_points_percent")
            if counts is not None:
                if s.exp_crop_enable:
                    fcol.label(text="Share of the vertices inside the crop box")
                else:
                    fcol.label(text="~{:,} points".format(
                        _exp_face_count(s, counts[0])))
        else:
            fcol.prop(s, "exp_face_points_count")
    ccol = pts.column(align=True)
    ccol.label(text="Visibility Filter:")
    ccol.prop(s, "exp_face_points_vischeck", text="")
    if s.exp_face_points_vischeck == 'RAYCAST':
        ccol.label(text="GPU check per camera, Esc cancels", icon='INFO')
    ccol.label(text="Point Color:")
    ccol.prop(s, "exp_point_color", text="")
    cbox2 = pts.column(align=True)
    cbox2.prop(s, "exp_crop_enable")
    if s.exp_crop_enable:
        cbox2.prop(s, "exp_crop_object")
        cbox2.prop(s, "exp_crop_margin")
        if s.exp_crop_object is None:
            cbox2.label(text="Pick a bounds object", icon='INFO')
    rrow = pts.column(align=True)
    rrow.prop(s, "exp_reuse_points")
    if s.exp_reuse_points:
        stored = bpy.path.abspath(s.exp_last_points_file or "")
        if stored and os.path.isfile(stored):
            rrow.label(text="Stored: %s - reused if unchanged"
                       % os.path.basename(os.path.normpath(os.path.dirname(
                           os.path.dirname(os.path.dirname(stored))))),
                       icon='FILE_TICK')
        else:
            rrow.label(text="No stored point cloud yet", icon='INFO')

    # --- Scale and axes --------------------------------------------------
    sc = _gcapture_section(layout, "Scale & Axes", 'ORIENTATION_GLOBAL')
    scol = sc.column(align=True)
    scol.prop(s, "exp_auto_scale")
    sub = scol.row(align=True)
    sub.enabled = s.exp_auto_scale
    sub.prop(s, "exp_target_radius")
    sc.prop(s, "exp_zup_to_yup")

    # Warning if the target folder already contains an export -- the export
    # overwrites sparse/0 (and images/) without asking (v84).
    try:
        out_dir = _exp_resolve_output_dir(s)
        if _exp_output_has_export(out_dir):
            wrow = layout.row()
            wrow.alert = True
            wrow.label(text="%s exists - export will overwrite it"
                       % os.path.basename(os.path.normpath(out_dir)),
                       icon='ERROR')
    except Exception:
        pass
    if not _gcapture_progress_draw(outer, "gcapture.export_colmap"):
        erow2 = _gcapture_lock(outer).row()
        erow2.scale_y = 1.3
        _gcapture_action(erow2, "gcapture.export_colmap",
                    not _exp_output_has_export(_exp_resolve_output_dir(s)),
                    'EXPORT')
    layout = _gcapture_lock(outer)
    if s.exp_last_points:
        rcol = layout.column(align=True)
        rcol.label(text="Last export: {:,} points".format(s.exp_last_points),
                   icon='CHECKMARK')
        if s.exp_last_detail:
            # Wrap instead of truncating (v1.1.5).
            for line in _gcapture_textwrap.wrap(s.exp_last_detail, 42):
                rcol.label(text=line)
