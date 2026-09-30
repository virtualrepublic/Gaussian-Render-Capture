"""Guide texts, step status, progress and lock drawing."""

import textwrap as _gcapture_textwrap

from ..clean.carve import _cln_check
from ..core.progress import _gcapture_busy
from ..core.render import _gcapture_scene_prepared
from ..ui.sections import (
    _gcapture_draw_build, _gcapture_draw_camera, _gcapture_draw_export, _gcapture_draw_group,
    _gcapture_draw_render, _gcapture_draw_sphere, _gcapture_draw_target)
from ..ui.status import (
    _gcapture_wt_status_build, _gcapture_wt_status_camera, _gcapture_wt_status_clean,
    _gcapture_wt_status_export, _gcapture_wt_status_group, _gcapture_wt_status_render,
    _gcapture_wt_status_sphere, _gcapture_wt_status_target)
from ..ui.widgets import (
    _gcapture_action, _gcapture_badge_label, _gcapture_lock, _gcapture_progress_draw)


# ----------------------------------------------------------------------
# Guide: walks beginners through the workflow step by step (v89)
# ----------------------------------------------------------------------
# Blender does not let add-ons expand/collapse panels. So the guide
# hides the subpanels (poll) and shows a card in the main panel with an
# explanation, the controls of the step (the same _gcapture_draw_*
# functions as the subpanels) and a status display.


def _gcapture_draw_intro(layout, context):
    row = layout.row()
    row.scale_y = 1.2
    _gcapture_action(row, "gcapture.prepare_scene",
                not _gcapture_scene_prepared(context.scene), 'SCENE_DATA')
    layout.prop(context.scene.gcapture_settings, "prep_see_through_glass")
    layout.separator(factor=0.5)
    _gcapture_draw_camera(layout, context)


def _gcapture_draw_clean(layout, context):
    s = context.scene.gcapture_settings
    _gcapture_progress_draw(layout, "gcapture.clean_splat")
    layout = _gcapture_lock(layout)
    row = layout.row(align=True)
    row.prop(s, "clean_splat_file", text="")
    row.operator("gcapture.clean_pick", text="", icon='FILEBROWSER')
    errs, _ = _cln_check(context, s)
    for e in errs:
        r = layout.row()
        r.alert = True
        r.label(text=e, icon='ERROR')
    col = layout.column()
    col.scale_y = 1.4
    col.enabled = not errs
    col.operator("gcapture.clean_splat", icon='OUTLINER_OB_POINTCLOUD',
                 depress=not errs and not s.clean_last_result)
    # In the guide the status line shows the result; the panel shows it here,
    # in two lines so the numbers are not cut off.
    if s.clean_last_result and not s.wt_active:
        head, _, tail = s.clean_last_result.partition(" - ")
        icon = ('ERROR' if "another scene version" in s.clean_last_result
                else 'CHECKMARK')
        layout.label(text=head, icon=icon)
        if tail:
            layout.label(text=tail, icon='BLANK1')


# Steps whose draw function shows a progress bar; the guide draws them
# outside its locked column (1.2.0).
_GCAPTURE_WT_PROGRESS_DRAWS = (_gcapture_draw_build, _gcapture_draw_export,
                               _gcapture_draw_clean)


_GCAPTURE_WT_STEPS = [
    dict(title="1. Scene & Camera", badge='gcapture_cam', icon='CAMERA_DATA',
         draw=_gcapture_draw_intro, status=_gcapture_wt_status_camera,
         goal="This guide turns your model into a Gaussian Splatting "
              "dataset: cameras on a sphere around the model, one rendered "
              "image per camera, then a COLMAP export for Postshot or "
              "LichtFeld Studio.",
         steps=["Press Prepare Scene: Cycles renders on your GPU with the "
                "capture render settings; Blender's start cube, camera and "
                "light are removed. Leave Transparent Glass off - glass then "
                "shows what lies behind it and stays clean in the splat.",
                "Import your model and put it into its own collection.",
                "Set the Focal Length of the capture camera - 50 mm is a good "
                "default; the sphere adapts its size to the lens.",
                "Keep Look Target at Geometry Center."],
         check="",
         note=["The camera itself is created in step 5 (Build)."]),
    dict(title="2. Look Target (Collection)", badge='gcapture_1', icon='HIDE_OFF',
         draw=_gcapture_draw_target, status=_gcapture_wt_status_target,
         goal="Tells the add-on which collection holds your model: the "
              "cameras look at it, and the sphere is sized to fit it.",
         steps=["Model in its own collection: select it in the Outliner and "
                "press + below the list.",
                "No own collection yet: select all parts of the model and "
                "press Put Selection into Collection - this also adds it to "
                "the list."],
         check="the collection appears in the list.",
         note=[]),
    dict(title="3. Group & Align to Ground", badge='gcapture_2', icon='EMPTY_AXIS',
         draw=_gcapture_draw_group, status=_gcapture_wt_status_group,
         goal="Only for objects that stand on a ground: places the model "
              "at the world origin, standing on Z = 0, so the sphere is "
              "centred and the dataset upright.",
         steps=["Leave Fast (bounding box) off for an exact result.",
                "Press Group & Align to Ground."],
         check="the model stands on the grid, centred on the origin.",
         note=["Floating object without a ground, or model already in "
               "place: skip with Next.",
               "A floating object gets the full sphere in the next step."]),
    dict(title="4. Camera Sphere", badge='gcapture_3', icon='MESH_ICOSPHERE',
         draw=_gcapture_draw_sphere, status=_gcapture_wt_status_sphere,
         goal="The cameras sit on the faces of an ico-sphere around the "
              "model - one camera per face.",
         steps=["Choose Subdivisions: 1 = 20, 2 = 80, 3 = 320, 4 = 1280 "
                "cameras. More cameras give better splats but take longer "
                "to render.",
                "Keep Fill Each View and Center in Each View on: every "
                "camera moves as close as the model allows and centres it "
                "in its image.",
                "Object on a ground: turn on Upper Hemisphere Only. "
                "Floating object: leave it off - the full sphere also sees "
                "it from below.",
                "Keep Build Cameras on This Sphere on and press Create / "
                "Update Camera Sphere."],
         check="the sphere surrounds the model; the status line shows the "
               "number of cameras.",
         note=["Framing Margin adds room around the model; 1.0 fills the "
               "image.",
               "Create / Update fits the sphere anew and discards Live "
               "Camera Adjust changes."]),
    dict(title="5. Build Camera Animation", badge='gcapture_4', icon='KEYFRAME_HLT',
         draw=_gcapture_draw_build, status=_gcapture_wt_status_build,
         goal="Puts one camera keyframe on every sphere face - one frame "
              "per viewpoint.",
         steps=["Press Build Camera Animation.",
                "When asked, save the scene: the add-on proposes "
                "<Collection>_v001.blend. After a rebuild it keeps the "
                "loaded version (Overwrite); choose New Version only when "
                "you want one.",
                "Scrub the timeline to check the views."],
         check="the status line shows the number of camera poses.",
         note=["Fine-tuning: go to a frame, press Live Camera Adjust, then "
               "press S (or G) in the viewport - the camera of this frame "
               "follows.",
               "Finish with This Camera (only this view changes) or All "
               "Cameras (the whole sphere, all views rebuilt); Cancel "
               "undoes it.",
               "Build Camera Animation starts again from step 4 and discards "
               "all adjustments (it warns first)."]),
    dict(title="6. Render Settings & Output", badge='gcapture_5',
         icon='RENDER_ANIMATION',
         draw=_gcapture_draw_render, status=_gcapture_wt_status_render,
         props_tab='OUTPUT',
         goal="Build has set the frame range, resolution and active camera; "
              "saving the version has pointed the render output into the "
              "dataset folder (<Name>_COLMAP, subfolder <vNNN> / images) - "
              "see the Output tab.",
         steps=["Check the Resolution and confirm it with OK - or change "
                "it.",
                "Check the render output in the box below.",
                "Choose PNG or TIFF (both RGBA 8 bit, read by Postshot and "
                "LichtFeld).",
                "Press Render EEVEE (much faster, lighting approximated) or "
                "Render Cycles (exact path tracing, slower): the scene is "
                "saved, then rendering starts; Esc cancels."],
         check="the status line shows all images found.",
         note=["Command Line Render: the buttons save the scene and write a "
               "script next to it - double-click it to render headless; "
               "Blender stays free.",
               "Render farm: render the saved scene there.",
               "A change of the resolution applies at once."]),
    dict(title="7. COLMAP Export", badge='gcapture_6', icon='EXPORT',
         draw=_gcapture_draw_export, status=_gcapture_wt_status_export,
         goal="Writes the cameras, their poses and a start point cloud into "
              "the dataset folder next to the images.",
         steps=["Keep Use Vertex Count on: every vertex of the model becomes "
                "a start point.",
                "Keep Visibility Filter at Visible Only: points no camera "
                "can see (hidden inner parts, e.g. of brick models) are dropped.",
                "Press Export COLMAP (Postshot / LichtFeld).",
                "Open <Name>_COLMAP/<vNNN> in Postshot or LichtFeld Studio "
                "and start training."],
         check="the status line shows \"Dataset written\".",
         note=["The settings are grouped into Dataset, Start Points and "
               "Scale & Axes; the defaults fit most models.",
               "Copy/Move only appear if you render into your own folder "
               "structure."]),
    dict(title="8. Clean Splat", badge='gcapture_7', icon='OUTLINER_OB_POINTCLOUD',
         draw=_gcapture_draw_clean, status=_gcapture_wt_status_clean,
         goal="After training: removes the splats in empty space (floaters) "
              "from the trained splat.",
         steps=["Export a .ply from LichtFeld Studio or Postshot.",
                "Pick it with the folder button (opens in the dataset "
                "folder).",
                "Press Clean Splat."],
         check="the result line shows how many splats were removed.",
         note=["Writes <name>_clean.ply next to the splat - the original "
               "stays untouched.",
               "Use the scene version the dataset was exported from."]),
]

_GCAPTURE_WT_STATUS_ICON = {'DONE': 'CHECKMARK', 'TODO': 'ERROR',
                       'MISSING': 'ERROR',
                       'OPTIONAL': 'RADIOBUT_OFF', 'INFO': 'INFO'}


def _gcapture_wrap(layout, context, text, indent_px=0):
    """Blender does not wrap labels: wrap text to the panel width.
    indent_px: width already occupied on the left (instruction number)."""
    width = context.region.width if context.region else 300
    try:
        scale = context.preferences.system.ui_scale
    except Exception:
        scale = 1.0
    scale = scale if scale and scale > 0.1 else 1.0   # background: 0
    chars = max(20, int((width - 40 - indent_px * scale) / (6.2 * scale)))
    col = layout.column(align=True)
    col.scale_y = 0.85
    for line in _gcapture_textwrap.wrap(text, chars):
        col.label(text=line)


def _gcapture_wt_draw_text(layout, context, step):
    """Guide text (v112): purpose, numbered steps with hanging
    indent, check and hint."""
    _gcapture_wrap(layout, context, step['goal'])
    if step['steps']:
        layout.separator(factor=0.5)
        scol = layout.column(align=True)
        for i, line in enumerate(step['steps'], 1):
            row = scol.row(align=True)
            num = row.column(align=True)
            num.ui_units_x = 1.0
            num.scale_y = 0.85
            num.label(text="%d." % i)
            _gcapture_wrap(row.column(align=True), context, line, indent_px=34)
    # Check and hints as a bulleted list (v131).
    note = step['note']
    bullets = ((["Check: " + step['check']] if step['check'] else [])
               + ([note] if isinstance(note, str) and note else list(note or [])))
    if bullets:
        layout.separator(factor=0.5)
        bcol = layout.column(align=True)
        for line in bullets:
            row = bcol.row(align=True)
            dot = row.column(align=True)
            dot.ui_units_x = 1.0
            dot.scale_y = 0.85
            dot.label(text="•")
            _gcapture_wrap(row.column(align=True), context, line, indent_px=34)


def _gcapture_draw_walkthrough(layout, context):
    s = context.scene.gcapture_settings
    n = len(_GCAPTURE_WT_STEPS)
    idx = max(0, min(s.wt_step, n - 1))
    step = _GCAPTURE_WT_STEPS[idx]

    card = layout.box()
    head = card.row(align=True)
    _gcapture_badge_label(head, step['badge'], None)
    head.label(text=step['title'], icon=step['icon'])
    right = head.row()
    right.alignment = 'RIGHT'
    right.label(text="Step %d of %d" % (idx + 1, n))

    _gcapture_wt_draw_text(card, context, step)
    card.separator()
    if step['draw'] in _GCAPTURE_WT_PROGRESS_DRAWS:
        step['draw'](card.column(), context)   # lock itself, bar stays free
    else:
        step['draw'](_gcapture_lock(card), context)
    card.separator()

    try:
        state, msg = step['status'](context, s)
    except Exception as exc:
        state, msg = 'INFO', "Status unavailable (%s)" % exc
    srow = card.row()
    srow.alert = state in {'TODO', 'MISSING'}   # open one red (v124/v125)
    srow.label(text=msg, icon=_GCAPTURE_WT_STATUS_ICON.get(state, 'INFO'))

    nav = card.row(align=True)
    nav.scale_y = 1.3
    nav.enabled = not _gcapture_busy()
    back = nav.row(align=True)
    back.enabled = idx > 0
    op = back.operator("gcapture.walkthrough_nav", text="Back", icon='TRIA_LEFT')
    op.delta = -1
    nav.operator("gcapture.walkthrough_exit", text="Exit", icon='X')
    # Prerequisites met -> Next blue (v139).
    ready = state in {'DONE', 'OPTIONAL'}
    if idx < n - 1:
        op = nav.operator("gcapture.walkthrough_nav", text="Next", icon='TRIA_RIGHT',
                          depress=ready)
        op.delta = 1
    else:
        nav.operator("gcapture.walkthrough_exit", text="Finish", icon='CHECKMARK',
                     depress=ready)
