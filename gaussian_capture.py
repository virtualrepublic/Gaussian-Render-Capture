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
from its utils/ray_casting.py and still part of this file:
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

Code comments are in German.
=========================================================================
"""

bl_info = {
    "name": "Gaussian Render Capture",
    "author": "Prof. Michael Klein - Mediadesign University of Applied Sciences",
    "version": (1, 1, 5),
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
import bmesh
import os
import shutil
import sys
import re
import math
import time
import numpy as np
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from bpy.props import (StringProperty, IntProperty, EnumProperty,
                       FloatProperty, CollectionProperty,
                       BoolProperty, PointerProperty)
from bpy.types import Operator, Panel, PropertyGroup


def _gcapture_on_param_change(settings, context):
    """Wird bei Aenderung interaktiver Parameter (z. B. Focal Length)
    aufgerufen. Setzt die Brennweite sofort auf die (bereits existierende)
    Kamera -- unabhaengig vom Live-Modus, da die Brennweite nur den
    Kamera-Datenblock betrifft und kein Neu-Backen der Posen erfordert."""
    cam = bpy.data.objects.get(settings.camera_name)
    if cam and cam.type == 'CAMERA':
        cam.data.lens_unit = 'MILLIMETERS'
        cam.data.lens = settings.focal_length


# ----------------------------------------------------------------------
# Einstellungen (im N-Panel editierbar, in der .blend gespeichert)
# ----------------------------------------------------------------------
def _gcapture_on_guide_index_change(self, context):
    """Wird aufgerufen, wenn in der Guide-Liste ein Eintrag ausgewaehlt
    wird. Selektiert die zum Eintrag gehoerenden Objekte im Outliner/
    Viewport und macht das erste zum aktiven Objekt."""
    s = context.scene.gcapture_settings
    if not (0 <= s.guide_index < len(s.guides)):
        return
    item = s.guides[s.guide_index]
    objs = [r.obj for r in item.objects if r.obj]
    if not objs:
        return
    try:
        # Bestehende Auswahl aufheben, dann die Eintragsobjekte selektieren.
        for o in context.view_layer.objects:
            o.select_set(False)
        for o in objs:
            try:
                o.select_set(True)
            except RuntimeError:
                pass  # Objekt evtl. in ausgeblendeter Collection
        context.view_layer.objects.active = objs[0]
    except Exception:
        pass


class GCAPTURE_GuideObjRef(PropertyGroup):
    """Verweis auf ein einzelnes Mesh-Objekt innerhalb eines Guide-Eintrags."""
    obj: PointerProperty(
        name="Object",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'MESH',
    )


class GCAPTURE_GuideItem(PropertyGroup):
    """Ein Leitgitter-Eintrag: editierbares Label + eine oder mehrere
    Mesh-Objekte. Das Label ist ein reiner Anzeige-Alias; die echten
    Objektnamen bleiben unveraendert."""
    label: StringProperty(
        name="Label",
        description="Display name for this guide entry (alias only -- does "
                    "not rename the actual objects)",
        default="Guide",
    )
    objects: CollectionProperty(type=GCAPTURE_GuideObjRef)


class GCAPTURE_CollItem(PropertyGroup):
    """Ein Eintrag in der Ziel-Collection-Liste fuer die Auto-Skalierung
    der Kamera-Sphere."""
    coll: PointerProperty(
        name="Collection",
        type=bpy.types.Collection,
    )


# Relative Pfade (//...) fuer Ordner-Properties erlauben (v107). Die Option
# gibt es erst ab Blender 4.5; aeltere Versionen verweigern sonst die
# Registrierung des ganzen Addons (v110-Fix).
_GCAPTURE_PATH_OPTIONS = ({'PATH_SUPPORTS_BLEND_RELATIVE'}
                     if bpy.app.version >= (4, 5, 0) else set())


def _gcapture_on_resolution(settings, context):
    """Aufloesung geaendert: anwenden und als bestaetigt merken (v140)."""
    context.scene["gcapture_res_ok"] = True
    _gcapture_on_render_setting(settings, context)


def _gcapture_on_render_setting(settings, context):
    """Aenderung einer Render-Einstellung sofort anwenden (v127)."""
    try:
        _gcapture_apply_render_setup(context.scene, settings)
    except Exception as exc:
        print("[Gaussian Render Capture] render setting not applied:", exc)


class GCAPTURE_Settings(PropertyGroup):
    camera_name: StringProperty(
        name="Camera Name",
        description="Name of the created/reused camera",
        default="Orbit_Camera",
    )
    rename_guide: BoolProperty(
        name="Rename Guide",
        description="Rename the guide object when building the animation",
        default=True,
    )
    guide_target_name: StringProperty(
        name="Guide Name",
        description="Target name for the guide object",
        default="Camera_Array",
    )
    setup_guide_display: BoolProperty(
        name="Wireframe, Don't Render",
        description="Show the guide as wireframe in the viewport and "
                    "exclude it from rendering",
        default=True,
    )
    frame_start: IntProperty(
        name="Start Frame",
        description="First frame of the animation",
        default=1, min=0,
    )
    focal_length: bpy.props.FloatProperty(
        name="Focal Length (mm)",
        description="Camera focal length in millimeters",
        default=50.0, min=1.0, max=5000.0, soft_max=300.0,
        update=lambda self, context: _gcapture_on_param_change(self, context),
    )
    look_mode: EnumProperty(
        name="Look Target",
        description="Where each camera looks per face",
        items=[
            ('CENTER', "Geometry Center",
             "Camera looks at the bounding-box center of the geometry -- "
             "robust even with non-uniform scaling"),
            ('ORIGIN', "Object Origin (Pivot)",
             "Camera looks at the pivot/origin of the guide object"),
            ('NORMAL', "Face Normal (uniform sphere only)",
             "Camera looks opposite the face normal -- only correct for a "
             "uniformly scaled sphere, NOT for an ellipsoid"),
        ],
        default='CENTER',
    )
    set_active_camera: BoolProperty(
        name="Set as Active Camera",
        description="Make the capture camera the scene camera - at once and on "
                    "every Build",
        default=True, update=_gcapture_on_render_setting,
    )
    set_frame_range: BoolProperty(
        name="Set Scene Frame Range",
        description="Set the scene frame range to the camera keyframes - at "
                    "once and on every Build",
        default=True, update=_gcapture_on_render_setting,
    )
    set_resolution: BoolProperty(
        name="Set Render Resolution",
        description="Set the square render resolution - at once and on every "
                    "Build",
        default=True, update=_gcapture_on_render_setting,
    )
    resolution: IntProperty(
        name="Resolution (square)",
        description="Square render resolution in pixels (width = height), "
                    "applied at once",
        default=1000, min=16, max=16384, update=_gcapture_on_resolution,
    )
    wt_active: BoolProperty(
        name="Guide Active",
        description="Step-by-step guide is running",
        default=False,
    )
    wt_step: IntProperty(
        name="Guide Step",
        default=0, min=0,
    )
    show_advanced: BoolProperty(
        name="Show Advanced",
        description="Show advanced/less-common options (guide object list, "
                    "interior camera rejection, guide display)",
        default=False,
    )
    ga_fast_bbox: BoolProperty(
        name="Fast (bounding box)",
        description="Group & Align: use each object's bounding box (8 corners) "
                    "instead of all mesh vertices. Much faster on heavy models, "
                    "but slightly overestimates the bounds for rotated objects. "
                    "Leave off for exact centering",
        default=False,
    )
    constant_interp: BoolProperty(
        name="Step per Frame (Constant)",
        description="Keyframes set to CONSTANT -- exactly one viewpoint per "
                    "frame, no interpolation in between (for render/COLMAP). "
                    "Applied at once and on every Build",
        default=True, update=_gcapture_on_render_setting,
    )
    save_after_build: BoolProperty(
        name="Save Version after Build",
        description="After Build and after Live Camera Adjust > This Camera, "
                    "ask to save the scene as a version "
                    "(<Name>_v001.blend, _v002 ...). Saving also points the "
                    "render output into the dataset folder",
        default=True,
    )
    live_lock: BoolProperty(
        name="Live Camera Adjust",
        description="Live Camera Adjust is running: the camera of the frame "
                    "where it started follows the sphere while you scale (S) "
                    "or move (G) it. Finish with This Camera, All Cameras or "
                    "Cancel",
        default=False,
        update=lambda self, context: _gcapture_on_lock_toggle(self, context),
    )
    # --- COLMAP export settings ---
    exp_output_dir: StringProperty(
        name="Output Folder",
        description="Dataset folder of the COLMAP model. The add-on fills in "
                    "<Name>_COLMAP/<vNNN>/ next to the .blend and keeps it on "
                    "the loaded version. Enter your own folder to override "
                    "(// = relative to the .blend); an own folder is left alone",
        default="", subtype='DIR_PATH',
        options=_GCAPTURE_PATH_OPTIONS,
    )
    exp_image_dir: StringProperty(
        name="Image Folder",
        description="Folder with the rendered images. The add-on fills in "
                    "the folder of the render output and keeps it in step "
                    "with it. Enter your own folder to override (// = "
                    "relative to the .blend)",
        default="", subtype='DIR_PATH',
        options=_GCAPTURE_PATH_OPTIONS,
    )
    exp_image_mode: EnumProperty(
        name="Images",
        description="What to do with the rendered images for the dataset. "
                    "LichtFeld Studio and COLMAP need an 'images' folder next "
                    "to 'sparse'",
        items=[
            ('NONE', "Don't include",
             "Write only the sparse/ model, no images/ folder is created "
             "(e.g. for a quick pose/point-cloud check)"),
            ('COPY', "Copy into dataset",
             "Copy the rendered images into the images/ subfolder. Safe and "
             "portable, but duplicates the files (uses extra disk space)"),
            ('MOVE', "Move into dataset",
             "MOVE the rendered images into the images/ subfolder (no "
             "duplication). WARNING: the images are then GONE from the render "
             "folder. Only runs if ALL expected images are present -- "
             "otherwise it is skipped to avoid a half-emptied render folder"),
        ],
        default='NONE',
    )
    exp_point_objects: StringProperty(
        name="Point Objects",
        description="Comma-separated object names for the point cloud "
                    "(Point Source: Named Objects)",
        default="",
    )
    exp_point_source: EnumProperty(
        name="Point Source",
        description="Which meshes the start point cloud comes from",
        items=[('TARGETS', "Look Target Collections",
                "The meshes of the look-target collections (step 2)"),
               ('VISIBLE', "All Visible Meshes",
                "Every visible mesh that is rendered, except the add-on's "
                "helpers"),
               ('OBJECTS', "Named Objects",
                "Only the objects named in Point Objects")],
        default='TARGETS',
    )
    # Vom Addon zuletzt eingetragene Pfade (v129): stimmt ein Feld damit
    # ueberein, folgt es weiter automatisch.
    exp_output_dir_auto: StringProperty(options={'HIDDEN'})
    exp_image_dir_auto: StringProperty(options={'HIDDEN'})
    exp_reuse_points: BoolProperty(
        name="Reuse Point Cloud",
        description="If nothing that affects the point cloud has changed "
                    "since the last export (same objects, cameras and point "
                    "settings), copy the previous points3D.txt instead of "
                    "computing it again - handy when only a path changed",
        default=True,
    )
    exp_last_signature: StringProperty(
        name="Last Point Cloud Signature", default="", options={'HIDDEN'},
    )
    exp_last_points_file: StringProperty(
        name="Last points3D.txt", default="", options={'HIDDEN'},
    )
    exp_last_points: IntProperty(
        name="Last Export Points", default=0, min=0,
        description="Number of points written by the last COLMAP export",
    )
    exp_last_detail: StringProperty(
        name="Last Export Detail", default="",
        description="Breakdown of the last COLMAP export's point cloud",
    )
    exp_face_points_percent: FloatProperty(
        name="Face Points",
        description="How many face points to add, in percent of the vertex "
                    "points (only with Use Vertex Count)",
        default=10.0, min=0.1, max=500.0, soft_max=100.0, subtype='PERCENTAGE',
    )
    exp_use_vertex_count: BoolProperty(
        name="Use Vertex Count",
        description="Use every vertex of the model as start point (counted at "
                    "export time) and a share of that as Face Points. Off: enter "
                    "Max Points and Face Points yourself",
        default=True,
    )
    exp_max_points: IntProperty(
        name="Max Points",
        description="Upper limit for start points taken from the model's "
                    "vertices (a model cannot give more points than it has "
                    "vertices). Use the vertex button to enter the exact "
                    "vertex count. Face Points come on top of this. More "
                    "start points give LichtFeld and Postshot more to build "
                    "on",
        default=2000000, min=100, max=50000000, soft_max=5000000,
    )
    exp_target_radius: bpy.props.FloatProperty(
        name="Target Radius",
        description="Average camera distance from the orbit centre in the "
                    "exported dataset. Scenes far below 1 (a minifigure) or "
                    "far above a few dozen units (a whole city) densified "
                    "badly in Postshot - the splat count did not grow; a few "
                    "units worked reliably. The factor is written to "
                    "<vNNN>_gcapture_export.json next to the dataset folder",
        default=3.0, min=0.1, max=100.0,
    )
    exp_auto_scale: BoolProperty(
        name="Auto Scale",
        description="Scale cameras and point cloud of the export so the "
                    "average camera distance equals Target Radius (the images "
                    "stay the same). Off -> factor 1.0, original units. "
                    "The factor is written to <vNNN>_gcapture_export.json next to the "
                    "dataset folder, so a trained "
                    "splat can be scaled back to the model",
        default=True,
    )
    exp_zup_to_yup: BoolProperty(
        name="Z-up -> Y-up",
        description="Convert Blender's Z-up world to Postshot's Y-up "
                    "(model stands upright, no manual 90 deg rotation)",
        default=True,
    )
    exp_point_color: EnumProperty(
        name="Point Color",
        description="Color source for the initial point cloud",
        items=[
            ('NONE', "Neutral Gray",
             "Write neutral gray (200,200,200) -- Postshot learns color "
             "from images anyway"),
            ('VERTEX', "Vertex Color Layer",
             "Use the mesh color attribute if present, else fall back to "
             "the material base color"),
            ('MATERIAL', "Material Base Color",
             "Use the Principled BSDF base color of each vertex's material. "
             "For textured parts this takes the constant base color, not the "
             "texture -- fine for visual inspection (Postshot learns the real "
             "colors from the images anyway)"),
        ],
        default='NONE',
    )
    # --- Random face points (area-weighted surface sampling) ---
    exp_face_points: BoolProperty(
        name="Add Random Face Points",
        description="Additionally scatter points across the guide-target "
                    "mesh surfaces, area-weighted (large flat faces get "
                    "proportionally more points than the mesh topology alone "
                    "provides). Gives an even starting density independent of "
                    "vertex layout. Comes on top of Max Points. If LichtFeld "
                    "(MRNF) grows too few splats, try without: a dense, even "
                    "start cloud may leave it less to densify",
        default=True,
    )
    exp_face_points_count: IntProperty(
        name="Face Points",
        description="Approximate number of random surface points to add "
                    "(distributed across all target meshes by area), in "
                    "addition to the vertex points",
        default=100000, min=1000, max=50000000, soft_max=5000000,
    )
    exp_face_points_vischeck: EnumProperty(
        name="Visibility Filter",
        description="Discard occluded points (interior faces of nested parts "
                    "that no camera can see). Applies to the visible-vertex "
                    "main cloud AND the face points. Uses a depth map per "
                    "camera on the GPU (falls back to ray casting without "
                    "GPU); runs as a modal phase with live progress and "
                    "ESC-to-cancel",
        items=[
            ('NONE', "Off",
             "Keep all points, including hidden interior faces"),
            ('RAYCAST', "Visible Only",
             "Keep points that at least one camera sees in its image. Checked "
             "against a GPU depth map per camera"),
        ],
        default='RAYCAST',
    )
    # --- Point cloud crop box (avoid floaters outside a region) ---
    exp_crop_enable: BoolProperty(
        name="Crop Point Cloud",
        description="Only keep point cloud points inside the bounds of a "
                    "chosen object (e.g. an Empty or cube sized to the room). "
                    "Prevents floater seed points outside the region",
        default=False,
    )
    exp_crop_object: PointerProperty(
        name="Crop Bounds",
        description="Object whose world-space bounding box defines the crop "
                    "region. Use a cube or Empty scaled around the area to keep",
        type=bpy.types.Object,
    )
    exp_crop_margin: FloatProperty(
        name="Crop Margin",
        description="Extra margin added around the crop bounds (world units)",
        default=0.0, min=0.0, soft_max=5.0,
    )
    # --- Camera sphere generator ---
    sph_subdivisions: IntProperty(
        name="Subdivisions",
        description="Ico-Sphere subdivisions. Each step roughly quadruples "
                    "the face count (= number of cameras). 1=20, 2=80, "
                    "3=320, 4=1280 faces",
        default=2, min=1, max=7,
    )
    sph_target_colls: CollectionProperty(type=GCAPTURE_CollItem)
    sph_coll_index: IntProperty(default=0)
    sph_margin: FloatProperty(
        name="Framing Margin",
        description="Extra space around the object (1.0 = the model "
                    "touches the frame edge, 1.1 = ~10%% margin)",
        default=1.0, min=1.0, soft_max=4.0,
    )
    sph_fit_each: BoolProperty(
        name="Fill Each View",
        description="Every camera moves along its line of sight until the "
                    "model fills its frame (with the margin) - most detail "
                    "per image. Off: all cameras keep one distance from the "
                    "model's space diagonal (adjust with Live Camera Adjust)",
        default=True,
    )
    sph_center_view: BoolProperty(
        name="Center in Each View",
        description="With Fill Each View: every camera also shifts sideways "
                    "(keeping its direction) so the model sits in the middle "
                    "of its square image and fills it - like Frame All for "
                    "each camera",
        default=True,
    )
    sph_upper_only: BoolProperty(
        name="Upper Hemisphere Only",
        description="Keep only cameras on or above the object's vertical "
                    "center (no views from below). Saves frames; good for "
                    "objects on a ground/water plane",
        default=False,
    )
    render_headless: BoolProperty(
        name="Render in Background (script)",
        description="Cycles / EEVEE save the scene and write a script next "
                    "to it instead of rendering here. Double-click the script "
                    "to render all cameras without Blender's interface - "
                    "Blender stays free, and the render goes on when Blender "
                    "is closed",
        default=False,
    )
    render_script: StringProperty(
        name="Render Script",
        description="Last headless render script written",
        default="",
        subtype='FILE_PATH',
    )
    sph_set_as_guide: BoolProperty(
        name="Build Cameras on This Sphere",
        description="Cameras are built on the faces of this sphere, so you "
                    "can Build right away (replaces custom camera guides)",
        default=True,
    )
    sph_object: PointerProperty(
        name="Camera Sphere",
        description="The generated camera sphere (tracked so Update finds it "
                    "even after it was renamed by a Build)",
        type=bpy.types.Object,
    )
    sph_render_setup_done: BoolProperty(
        name="Render Setup Done",
        description="Internal: render settings for splatting have been "
                    "applied once for this file. Prevents re-applying them "
                    "on later sphere updates",
        default=False,
    )
    # --- Multiple guide objects ---
    guides: CollectionProperty(type=GCAPTURE_GuideItem)
    guide_index: IntProperty(
        default=0,
        update=_gcapture_on_guide_index_change,
    )
    use_guide_list: BoolProperty(
        name="Use Guide List",
        description="Build cameras from a list of guide meshes instead of "
                    "just the active object",
        default=False,
    )
    # --- Interior camera rejection ---
    skip_interior: BoolProperty(
        name="Skip Interior Cameras",
        description="Skip camera positions detected inside meshes or with "
                    "geometry poking through the near clip plane (ray-cast). "
                    "Useful for interior/room rigs",
        default=False,
    )


# ----------------------------------------------------------------------
# Kernlogik
# ----------------------------------------------------------------------
def face_centers_and_normals(obj):
    """Pro Face (zentrum_welt, normale_welt), evaluiert inkl. Modifier."""
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
    """Eigene Collection fuer Kamera und Camera Sphere (v136)."""
    coll = bpy.data.collections.get(_GCAPTURE_RIG_COLL)
    if coll is None or coll.library is not None:
        coll = bpy.data.collections.new(_GCAPTURE_RIG_COLL)
    if coll not in scene.collection.children_recursive:
        scene.collection.children.link(coll)
    return coll


def _gcapture_move_to_rig(scene, obj):
    """Objekt nur noch in der Rig-Collection fuehren (v136)."""
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
    """Render-Einstellungen aus Schritt 5 auf die Szene anwenden (v127) --
    dasselbe, was Build am Ende tut, soweit Kamera/Keyframes schon da sind."""
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
    """Liefert alle F-Curves der Action von 'obj' -- versionssicher.

    Blender 4.4+ hat 'Slotted Actions' eingefuehrt: action.fcurves wurde
    in einen Channelbag pro Slot verschoben und ist in Blender 5.0 ganz
    entfernt. Diese Funktion deckt beide Welten ab:
      1. neu: ueber action.layers[].strips[].channelbag(slot).fcurves
      2. alt: ueber action.fcurves (Legacy, < 4.4)
    """
    ad = obj.animation_data
    if not ad or not ad.action:
        return []
    action = ad.action

    # --- Neuer Weg (Slotted Actions, 4.4+/5.x) ---
    # Bevorzugt die offizielle Helferfunktion, falls vorhanden.
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

    # Manueller Weg ueber die Layer/Strip/Channelbag-Hierarchie.
    try:
        if len(action.layers) > 0:
            strip = action.layers[0].strips[0]
            slot = getattr(ad, "action_slot", None)
            if slot is not None:
                cbag = strip.channelbag(slot)
                if cbag is not None:
                    return list(cbag.fcurves)
            # Falls kein Slot greifbar: alle Channelbags des Strips nehmen.
            bags = getattr(strip, "channelbags", None)
            if bags:
                fcurves = []
                for cb in bags:
                    fcurves.extend(list(cb.fcurves))
                return fcurves
    except Exception:
        pass

    # --- Legacy-Weg (< 4.4) ---
    if hasattr(action, "fcurves"):
        try:
            return list(action.fcurves)
        except Exception:
            pass

    return []


def setup_guide_object(obj, settings):
    """Zeigt das Leitgitter als Drahtgitter und nimmt es aus dem Render.
    Seit v93 immer (vorher abschaltbar) und ohne Umbenennen -- rename_guide,
    guide_target_name und setup_guide_display bleiben nur als Properties
    fuer alte .blend-Dateien bestehen. Gibt den Objektnamen zurueck."""
    obj.display_type = 'WIRE'      # Viewport: nur Drahtgitter
    obj.hide_render = True          # vom finalen Rendering ausschliessen
    obj.show_in_front = False
    # Zusaetzlich aus allen Strahlen-Sichtbarkeiten nehmen, damit es
    # auch indirekt (Reflexionen/Schatten) nicht beitraegt.
    try:
        obj.visible_camera = False
        obj.visible_diffuse = False
        obj.visible_glossy = False
        obj.visible_transmission = False
        obj.visible_volume_scatter = False
        obj.visible_shadow = False
    except AttributeError:
        # Diese Properties existieren nur bei aktiver Cycles-Engine.
        pass

    return obj.name


def _gcapture_has_custom_guides(s):
    """True, wenn die Guide-Liste eigene Leitgitter enthaelt (andere als die
    von 'Camera Sphere' erzeugte Sphere)."""
    if not s.use_guide_list:
        return False
    for item in s.guides:
        for ref in item.objects:
            if (ref.obj is not None and ref.obj != s.sph_object
                    and ref.obj.users_scene):
                return True
    return False


def _collect_guide_objects(context, active_obj):
    """Liefert die Liste der zu verarbeitenden Leitgitter-Objekte.
    Bei aktivierter Guide-List die Collection, sonst das aktive Objekt."""
    s = context.scene.gcapture_settings
    if s.use_guide_list and len(s.guides) > 0:
        objs = []
        for item in s.guides:
            for ref in item.objects:
                o = ref.obj
                # Nur Objekte der aktuellen Szene (im Viewport geloeschte
                # Leitgitter koennen als verwaiste Daten weiterleben, v94).
                if (o and o.type == 'MESH' and o not in objs
                        and o.name in context.scene.objects):
                    objs.append(o)
        return objs
    if active_obj and active_obj.type == 'MESH':
        return [active_obj]
    return []


# Custom-Property an der Kamera-Sphere: ihr Mittelpunkt in Objekt-
# Koordinaten (= Zentrum der Ziel-Bounding-Box beim Erzeugen).
_SPH_CENTER_PROP = "gcapture_center"


def _gcapture_sphere_origin_to_center(obj):
    """Legt den Ursprung einer Camera Sphere in ihre Mitte, ohne sie zu
    verschieben (Spheres vor v95 hatten den Ursprung im Weltursprung). Nutzt
    den gespeicherten Mittelpunkt; ohne ihn bleibt alles, wie es ist.
    Rueckgabe: True, wenn umgestellt wurde."""
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
    """Liefert pro Face (center, normal, geo_center, origin) in Weltkoord.
    geo_center ist der gespeicherte Sphere-Mittelpunkt, falls das Leitgitter
    eine mit 'Camera Sphere' erzeugte Sphere ist, sonst der BBox-Mittelpunkt
    der Face-Zentren dieses Objekts. Die BBox der Face-Zentren taugt bei der
    oberen Halbkugel NICHT als Zielpunkt: sie liegt dort weit ueber dem
    Modell, die Kameras zielten daran vorbei (v81-Fix)."""
    faces = face_centers_and_normals(obj)
    if not faces:
        return []
    origin = obj.matrix_world.translation.copy()
    stored = obj.get(_SPH_CENTER_PROP)
    if stored is not None and len(stored) == 3:
        # In Objekt-Koordinaten gespeichert -> folgt Verschieben/Skalieren
        # der Sphere (z. B. beim Live Camera Lock).
        geo_center = obj.matrix_world @ Vector(stored)
    else:
        xs = [c.x for (c, _) in faces]
        ys = [c.y for (c, _) in faces]
        zs = [c.z for (c, _) in faces]
        geo_center = Vector((0.5 * (max(xs) + min(xs)),
                             0.5 * (max(ys) + min(ys)),
                             0.5 * (max(zs) + min(zs))))
    # Fill Each View (v96): Kamera je Face um den gespeicherten Faktor zur
    # Mitte ruecken. Folgt Verschieben/Skalieren der Sphere (Live Lock).
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
    # Center in Each View (v135): Kamera und Blickziel gemeinsam seitlich
    # versetzen -- die Blickrichtung bleibt, das Modell sitzt mittig.
    if obj.type == 'MESH':
        sa = obj.data.attributes.get(_SPH_SHIFT_ATTR)
        if (sa is not None and sa.domain == 'FACE'
                and len(sa.data) == len(faces)):
            rot = obj.matrix_world.to_3x3()
            for i, d in enumerate(sa.data):
                sv = rot @ Vector(d.vector)
                c, nrm, g, o = out[i]
                out[i] = (c + sv, nrm, g + sv, o)
    # Einzeln justierte Kameras (v123, Finish Live Adjust > Only this
    # camera): Position und Blickziel je Face in Objektkoordinaten.
    for i, pos, tgt in _sph_view_adjustments(obj, len(faces)):
        mw = obj.matrix_world
        out[i] = (mw @ pos, out[i][1], mw @ tgt, origin)
    return out


_SPH_ADJ_FLAG = "gcapture_adj"
_SPH_ADJ_POS = "gcapture_adj_pos"
_SPH_ADJ_TGT = "gcapture_adj_tgt"


def _sph_view_adjustments(obj, n_faces):
    """[(Face-Index, Position, Blickziel)] der einzeln justierten Kameras,
    beides in Objektkoordinaten (v123)."""
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
    """Lage der Sphere nach Create / Update (v133)."""
    base = obj.get("gcapture_base_loc")
    if base is not None and len(base) == 3:
        return Matrix.Translation(Vector(base))
    return Matrix.Translation(obj.matrix_world.translation)


def _sph_ensure_base(obj, s):
    """Spheres von vor v133 kennen ihre Lage nach Create / Update nicht:
    aus den Look-Target-Collections nachtragen, wie Create / Update sie
    berechnet (v134)."""
    if obj is None or obj.get("gcapture_base_loc") is not None:
        return
    try:
        bounds = _sph_world_bounds(_sph_collect_objects(s, exclude=obj))
    except Exception:
        bounds = None
    centre = bounds[2] if bounds else obj.matrix_world.translation
    obj["gcapture_base_loc"] = tuple(centre)


def _sph_has_adjustments(obj):
    """True, wenn Live Camera Adjust die Sphere veraendert oder Ansichten
    einzeln justiert hat (v133)."""
    if obj is None or obj.type != 'MESH':
        return False
    flag = obj.data.attributes.get(_SPH_ADJ_FLAG)
    if flag is not None and any(d.value for d in flag.data):
        return True
    base = _sph_base_matrix(obj)
    return any(abs(a - b) > 1e-5 for ra, rb in zip(obj.matrix_world, base)
               for a, b in zip(ra, rb))


def _sph_reset_adjustments(obj):
    """Sphere auf ihre Lage nach Create / Update zuruecksetzen und einzeln
    justierte Ansichten loeschen (v133). Rueckgabe: ob sich etwas aenderte."""
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
    """Speichert die Pose einer einzeln justierten Kamera an ihrem Face."""
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
    """Blickrichtung einer Kamera am Face, konsistent mit build_animation."""
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
    """Alle Faces aller aktuellen Leitgitter als flache Liste, in genau
    der Reihenfolge, die auch build_animation verwendet. Wenn guide_objs
    explizit uebergeben wird (z. B. die beim Lock gemerkten Objekte), wird
    diese Liste verwendet, statt ueber active_object zu sammeln."""
    if guide_objs is None:
        guide_objs = _collect_guide_objects(context, active_obj)
    faces = []
    for gobj in guide_objs:
        if gobj is not None:
            faces.extend(_guide_faces_with_targets(gobj))
    return faces


# ----------------------------------------------------------------------
# Live Camera Lock: bindet EINE Kamera (aktiver Frame -> Face-Index) live
# an das Leitgitter. Folgt Skalierung/Bewegung sofort, ohne die ganze
# Animation zu backen. Leichtgewichtiger Handler (eine Pose statt vieler).
# ----------------------------------------------------------------------
import bpy.app.handlers as _gcapture_handlers

_GCAPTURE_LOCK_BUSY = False
_GCAPTURE_LOCK_START_MW = {}    # Leitgitter-Name -> matrix_world beim Einschalten (v123)
# Beim Einschalten eingefrorener Face-Index (Variante 1). Der Handler
# haelt die Kamera immer an diesem Index, unabhaengig vom aktuellen Frame
# -- das umgeht das fragile Frame-Lesen im Handler-Kontext. None = nichts
# eingefroren.
_GCAPTURE_LOCK_FROZEN_IDX = None
# Diagnose: schreibt in die Blender-System-Konsole, was der Lock tut.
GCAPTURE_LOCK_DEBUG = False
# Beim Einschalten gemerkte Leitgitter-Objekte. Der Handler darf sich
# NICHT auf bpy.context.active_object verlassen -- das ist im Handler-
# Kontext unzuverlaessig (das skalierte Objekt ist dort nicht zwingend
# "aktiv"). Stattdessen merken wir die Guides beim Einschalten explizit.
_GCAPTURE_LOCK_GUIDES = []


def _gcapture_redraw_view3d():
    """Stoesst einen Redraw aller 3D-Viewports an. Nutzt tag_redraw() --
    das zeichnet nur neu und loest KEIN Depsgraph-Update aus (anders als
    cam.update_tag(), das im Handler eine Update-Schleife erzeugen wuerde).
    Noetig, weil das Setzen der Kamera per Skript den Viewport im Handler-
    Pfad sonst nicht neu zeichnet -- die Aenderung wuerde erst beim
    naechsten UI-Ereignis sichtbar."""
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
    """Setzt die aktive Kamera fuer den aktuellen Frame an das zum Frame
    gehoerende Face (Frame - frame_start -> Face-Index). 'scene' wird
    explizit uebergeben (im Handler aus dem scene-Argument), weil
    bpy.context im Handler-Kontext veraltete Werte liefern kann -- u.a.
    einen alten frame_current.

    tag_update: nur True aus dem Toggle-Callback (fuer den Redraw). Im
    Handler-Pfad MUSS es False bleiben, sonst loest cam.update_tag() ein
    neues Depsgraph-Update aus -> der Handler feuert erneut, diesmal mit
    der Kamera (statt der Sphere) in den Updates, was zu einem Frame-/
    Face-Versatz beim Skalieren fuehrt."""
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
        # Falls keine benannte Kamera existiert, die Szenenkamera nehmen.
        cam = scene.camera
        if GCAPTURE_LOCK_DEBUG:
            print("[GCAPTURE Lock] apply: falling back to scene.camera=%s"
                  % (cam.name if cam else "None"))
    if cam is None or cam.type != 'CAMERA':
        if GCAPTURE_LOCK_DEBUG:
            print("[GCAPTURE Lock] apply: NO usable camera -> abort")
        return False

    # Faces aus den beim Lock gemerkten Guides (robust gegen active_object).
    lock_guides = [o for o in _GCAPTURE_LOCK_GUIDES if o is not None]
    faces = _gcapture_all_guide_faces(context, active_obj,
                                 guide_objs=lock_guides if lock_guides else None)
    if not faces:
        return False

    # Variante 1: Wenn ein Index eingefroren ist (Lock aktiv), diesen
    # verwenden -- NICHT den aktuellen Frame lesen. Das macht den Lock
    # robust gegen den fragilen Frame-Kontext im Handler. Nur als Fallback
    # (z. B. direkter Aufruf ohne Einfrieren) wird der Frame herangezogen.
    if _GCAPTURE_LOCK_FROZEN_IDX is not None:
        idx = _GCAPTURE_LOCK_FROZEN_IDX
    else:
        idx = scene.frame_current - s.frame_start
    if idx < 0 or idx >= len(faces):
        return False  # Index ausserhalb des gueltigen Face-Bereichs

    center, normal, geo_center, origin = faces[idx]
    look_dir = _gcapture_look_dir_for(s, center, normal, geo_center, origin)
    cam.data.lens_unit = 'MILLIMETERS'
    cam.data.lens = s.focal_length
    cam.rotation_mode = 'QUATERNION'
    cam.location = center
    cam.rotation_quaternion = look_dir.to_track_quat('-Z', 'Y')

    # Den zum Index gehoerenden Keyframe AKTUALISIEREN (keyframe_insert),
    # statt nur direkt zu setzen. Grund: Hat die Kamera bereits Keyframes
    # (nach einem Build), ueberschreibt die Animation jedes direkte
    # cam.location beim naechsten Depsgraph-Eval -- die Kamera "bewegt sich
    # nicht". keyframe_insert schreibt den Live-Wert IN die Animation, also
    # gewinnt er. Frame = frame_start + idx (der zum Face gehoerende Frame).
    target_frame = s.frame_start + idx
    if cam.animation_data and cam.animation_data.action:
        cam.keyframe_insert(data_path="location", frame=target_frame)
        cam.keyframe_insert(data_path="rotation_quaternion", frame=target_frame)
        # Konstante Interpolation beibehalten, falls so gebaut.
        if s.constant_interp:
            for fcurve in iter_action_fcurves(cam):
                for kp in fcurve.keyframe_points:
                    if abs(kp.co[0] - target_frame) < 0.5:
                        kp.interpolation = 'CONSTANT'
    if GCAPTURE_LOCK_DEBUG:
        print("[GCAPTURE Lock] apply: set '%s' idx %d @frame %d, loc=(%.3f,%.3f,%.3f)"
              % (cam.name, idx, target_frame, center.x, center.y, center.z))
    # Depsgraph-Update nur taggen, wenn explizit gewuenscht (Toggle-
    # Einschalten, fuer den Redraw). Im Handler-Pfad NICHT -- sonst
    # Update-Schleife mit Frame-/Face-Versatz (siehe Docstring).
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

    # Beobachtete Leitgitter: die beim Einschalten GEMERKTEN Objekte, NICHT
    # ueber active_object (das ist im Handler unzuverlaessig -- genau das war
    # der Bug: das skalierte Camera_Array war nicht "aktiv", also matchte es
    # das watch-Set nicht und das Update galt als irrelevant).
    guide_objs = [o for o in _GCAPTURE_LOCK_GUIDES if o is not None]
    if not guide_objs:
        # Fallback: aus den aktuellen Einstellungen sammeln.
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

    # Variante 1: Der Face-Index ist eingefroren. Wir reagieren NUR auf
    # Transform/Geometry-Aenderungen der Leitgitter (Skalieren/Bewegen) --
    # kein Frame-Wechsel-Handling mehr noetig, da der Index fix ist. Das
    # eliminiert die fruehere Fehlerquelle (Frame-Lesen im Handler).
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
        # Viewport neu zeichnen, damit die Kamerabewegung sofort sichtbar
        # ist (tag_redraw loest KEIN Depsgraph-Update aus -> keine Schleife).
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
    """Schaltet den Live-Camera-Lock-Handler an/aus. Beim Einschalten wird
    der Face-Index des AKTUELLEN Frames eingefroren (Variante 1); der Lock
    haelt die Kamera danach an diesem Index, auch beim Skalieren/Bewegen
    des Leitgitters. Fuer ein anderes Face: Lock aus- und wieder
    einschalten (friert dann den neuen Frame ein)."""
    global _GCAPTURE_LOCK_BUSY, _GCAPTURE_LOCK_FROZEN_IDX, _GCAPTURE_LOCK_GUIDES
    if settings.live_lock:
        scene = context.scene
        # Aktuellen Frame-Index einfrieren.
        _GCAPTURE_LOCK_FROZEN_IDX = scene.frame_current - settings.frame_start
        # Leitgitter JETZT merken (active_object ist hier, im Toggle-
        # Kontext, noch zuverlaessig -- im Handler spaeter nicht mehr).
        _GCAPTURE_LOCK_GUIDES = _collect_guide_objects(context,
                                                  context.active_object)
        # Lage der Leitgitter beim Einschalten (v123): "Only this camera"
        # setzt sie beim Beenden zurueck.
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
        # Viewport-Redraw erzwingen -- im Property-update-Kontext wird sonst
        # nicht neu gezeichnet (Blender T74000), die Kamera saesse zwar
        # richtig, der Viewport zeigte aber noch den alten Stand.
        _gcapture_redraw_view3d()
    else:
        _GCAPTURE_LOCK_FROZEN_IDX = None
        _GCAPTURE_LOCK_GUIDES = []
        _gcapture_lock_remove()


# ----------------------------------------------------------------------
# Kamera-Sphere-Generator: Helfer
# ----------------------------------------------------------------------
def _sph_apply_render_settings(scene):
    """Setzt einmalig die Render-Einstellungen fuer Gaussian-Splatting-
    Captures. Jede Einstellung defensiv (try/except), damit ein in dieser
    Blender-Version abweichender Property-Pfad nicht den ganzen Operator
    abbricht -- die uebrigen werden trotzdem gesetzt. Gibt eine Liste der
    nicht setzbaren Punkte zurueck (fuer einen Hinweis an den Nutzer)."""
    skipped = []

    def tryset(setter, label):
        try:
            setter()
        except Exception:
            skipped.append(label)

    # Engine auf Cycles + GPU.
    tryset(lambda: setattr(scene.render, "engine", 'CYCLES'),
           "Cycles engine")
    tryset(lambda: setattr(scene.cycles, "device", 'GPU'),
           "Cycles GPU device")

    # Denoiser: Render + Viewport aktivieren.
    tryset(lambda: setattr(scene.cycles, "use_denoising", True),
           "Render denoiser")
    tryset(lambda: setattr(scene.cycles, "use_preview_denoising", True),
           "Viewport denoiser")
    # Denoiser-TYP auf OpenImageDenoise (OIDN) -- Render + Viewport.
    tryset(lambda: setattr(scene.cycles, "denoiser", 'OPENIMAGEDENOISE'),
           "Render denoiser = OpenImageDenoise")
    tryset(lambda: setattr(scene.cycles, "preview_denoiser", 'OPENIMAGEDENOISE'),
           "Viewport denoiser = OpenImageDenoise")
    # Denoise-Geraet auf GPU (Blender 4.2+: der Denoiser kann GPU nutzen,
    # auch wenn gerendert wird). Property-Name variiert je Version, daher
    # mehrere Varianten defensiv versuchen.
    tryset(lambda: setattr(scene.cycles, "denoising_use_gpu", True),
           "Render denoiser GPU")
    tryset(lambda: setattr(scene.cycles, "preview_denoising_use_gpu", True),
           "Viewport denoiser GPU")

    # Samples (Render). Viewport-Samples lassen wir unangetastet.
    tryset(lambda: setattr(scene.cycles, "samples", 512),
           "Render samples 512")

    # Film: Transparent + Transparent Glass.
    tryset(lambda: setattr(scene.render, "film_transparent", True),
           "Film transparent")
    tryset(lambda: setattr(scene.cycles, "film_transparent_glass", True),
           "Film transparent glass")

    # Welt-Hintergrund: Color Value auf 1 (weiss).
    def set_world_bg():
        world = scene.world
        if world is None:
            world = bpy.data.worlds.new("World")
            scene.world = world
        world.use_nodes = True
        bg = world.node_tree.nodes.get("Background")
        if bg is None:
            for node in world.node_tree.nodes:
                if node.type == 'BACKGROUND':
                    bg = node
                    break
        if bg is not None:
            bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        else:
            world.color = (1.0, 1.0, 1.0)
    tryset(set_world_bg, "World background color = 1")

    return skipped


def _sph_apply_camera_clip(s):
    """Setzt Clip Start/End der konfigurierten Kamera (fuer Splatting:
    Start klein, End sehr gross)."""
    cam = bpy.data.objects.get(s.camera_name)
    if cam is None or cam.type != 'CAMERA':
        cam = bpy.context.scene.camera
    if cam is not None and cam.type == 'CAMERA':
        cam.data.clip_start = 0.01
        cam.data.clip_end = 10000.0
        return True
    return False


def _sph_collect_objects(s, exclude=None):
    """Alle SICHTBAREN Mesh-Objekte aus den gewaehlten Ziel-Collections
    (rekursiv inkl. Kind-Collections). 'exclude' (Objekt) wird ausgelassen
    -- wichtig, damit die Kamera-Sphere selbst NICHT in die Groessen-
    berechnung eingeht. Nur sichtbare Objekte zaehlen (visible_get): so
    blaehen ausgeblendete Hilfsobjekte (Boden-Planes, Rigs, Hilfs-Meshes)
    die Bounding-Box nicht auf -- das war der Grund, warum die Sphere bei
    Modellen aus vielen Teilen viel zu gross wurde."""
    objs = []
    seen = set()
    excl_name = exclude.name if exclude is not None else None

    def walk(coll):
        for o in coll.objects:
            if (o.type == 'MESH' and o.name not in seen
                    and o.name != excl_name and o.visible_get()):
                seen.add(o.name)
                objs.append(o)
        for child in coll.children:
            walk(child)

    for item in s.sph_target_colls:
        if item.coll is not None:
            walk(item.coll)
    return objs


def _sph_world_bounds(objs):
    """(min_vec, max_vec, center, max_dim, biggest) der Welt-Raum-Bounding-
    Box ueber alle Objekte. max_dim ist die groesste Achsen-Ausdehnung.
    'biggest' = (name, dim) des Objekts mit der groessten Einzel-Diagonale
    (fuer Diagnose, welches Teil die Bounds aufblaeht). Liest die bound_box
    aus dem EVALUIERTEN Objekt (inkl. Modifier)."""
    import math as _m
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mn = Vector((_m.inf, _m.inf, _m.inf))
    mx = Vector((-_m.inf, -_m.inf, -_m.inf))
    found = False
    biggest_name = None
    biggest_dim = -1.0
    for o in objs:
        o_eval = o.evaluated_get(depsgraph)
        mw = o_eval.matrix_world
        omn = Vector((_m.inf, _m.inf, _m.inf))
        omx = Vector((-_m.inf, -_m.inf, -_m.inf))
        for corner in o_eval.bound_box:
            wc = mw @ Vector(corner)
            mn.x = min(mn.x, wc.x); mn.y = min(mn.y, wc.y); mn.z = min(mn.z, wc.z)
            mx.x = max(mx.x, wc.x); mx.y = max(mx.y, wc.y); mx.z = max(mx.z, wc.z)
            omn.x = min(omn.x, wc.x); omn.y = min(omn.y, wc.y); omn.z = min(omn.z, wc.z)
            omx.x = max(omx.x, wc.x); omx.y = max(omx.y, wc.y); omx.z = max(omx.z, wc.z)
            found = True
        # Einzel-Ausdehnung dieses Objekts.
        odim = max(omx.x - omn.x, omx.y - omn.y, omx.z - omn.z)
        if odim > biggest_dim:
            biggest_dim = odim
            biggest_name = o.name
    if not found:
        return None
    center = (mn + mx) * 0.5
    max_dim = max(mx.x - mn.x, mx.y - mn.y, mx.z - mn.z)
    # Raumdiagonale der Gesamt-Bounding-Box: das groesste Mass, das eine
    # Kamera aus IRGENDEINER Richtung sehen kann. Als Bezug fuer den Radius
    # macht das Framing formunabhaengig -- kompakte (Wuerfel) und laengliche
    # (Auto) Objekte werden mit demselben Margin gleich gut gerahmt.
    diag = math.sqrt((mx.x - mn.x) ** 2 + (mx.y - mn.y) ** 2 +
                     (mx.z - mn.z) ** 2)
    return (mn, mx, center, max_dim, diag, (biggest_name, biggest_dim))


def _sph_camera_min_fov(s, scene):
    """Kleinere der beiden FOV-Achsen (horizontal/vertikal) in Radiant --
    die einschraenkende Achse fuer 'Objekt komplett im Bild'. Basiert auf
    der im Addon eingestellten Brennweite. Sensorbreite 36mm.

    WICHTIG: Es wird BEWUSST ein QUADRATISCHES Seitenverhaeltnis (1:1)
    angenommen, NICHT das gerade eingestellte Render-Seitenverhaeltnis.
    Grund: Der Build rendert fuer Splatting quadratisch (set_resolution
    setzt res_x = res_y). Wuerde der Generator das aktuelle Seitenverhaeltnis
    nehmen, aenderte sich der Radius, sobald der Build die Aufloesung auf
    quadratisch umstellt -- die Sphere wuerde dann beim naechsten Erzeugen
    einmalig schrumpfen/wachsen. Mit fixem 1:1 ist der Generator von Anfang
    an konsistent mit dem Render-Ergebnis und bleibt stabil."""
    lens = max(s.focal_length, 1e-3)
    sensor_w = 36.0
    # Quadratisch: fov_h == fov_v, also genuegt eine Achse.
    fov = 2.0 * math.atan((sensor_w / 2.0) / lens)
    return fov


# Mesh-Attribut der Camera Sphere (Face-Domain): Faktor, um den die
# Kamera je Face zur Mitte rueckt (Fill Each View, v96).
_SPH_FIT_ATTR = "gcapture_fit"
_SPH_FIT_MIN = 0.6   # keine Kamera naeher als 60 % des Sphere-Abstands
_SPH_SHIFT_ATTR = "gcapture_shift"   # seitlicher Versatz je Kamera (v135)


def _sph_model_points(objs):
    """Alle Vertices der Objekte in Weltkoordinaten als (N,3)-Array."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    chunks = []
    for o in objs:
        oe = o.evaluated_get(depsgraph)
        me = oe.to_mesh()
        if me is None:
            continue
        n = len(me.vertices)
        if n:
            co = np.empty(n * 3, dtype=np.float32)
            me.vertices.foreach_get("co", co)
            co = co.reshape(n, 3)
            mw = np.array(oe.matrix_world, dtype=np.float64)
            chunks.append(co @ mw[:3, :3].T + mw[:3, 3])
        oe.to_mesh_clear()
    if not chunks:
        return None
    return np.vstack(chunks)


def _sph_required_distances(points, center, dirs, fov, margin):
    """Je Blickrichtung der Abstand zur Mitte, bei dem ALLE Punkte mit
    Rand ins (quadratische) Bild passen. dirs: Einheitsvektoren von der
    Mitte zur Kamera; die Kamera blickt auf die Mitte (-dir). Fuer einen
    Punkt p mit rel = p - Mitte gilt im Kamerabild |rel.right| <= t * Tiefe
    und |rel.up| <= t * Tiefe mit Tiefe = d - rel.u und t = tan(fov/2) /
    margin -> d >= rel.u + max(|rel.right|, |rel.up|) / t."""
    tt = math.tan(fov / 2.0) / max(margin, 1e-6)
    points, safety = _sph_reduce_points(points)
    safety *= (1.0 + 1.0 / tt)
    rel = points - np.array(center, dtype=np.float64)
    out = []
    for u in dirs:
        q = (-u).to_track_quat('-Z', 'Y')
        r = q @ Vector((1.0, 0.0, 0.0))
        up = q @ Vector((0.0, 1.0, 0.0))
        best = -np.inf
        for k in range(0, len(rel), 500000):   # speicherschonend
            blk = rel[k:k + 500000]
            du = blk @ np.array(u)
            lat = np.maximum(np.abs(blk @ np.array(r)),
                             np.abs(blk @ np.array(up)))
            best = max(best, float((du + lat / tt).max()))
        out.append(best + safety)
    return out


def _sph_framed_positions(points, center, dirs, fov, margin):
    """Je Blickrichtung (Einheitsvektor Mitte -> Kamera) die Kameraposition,
    bei der alle Punkte mit Rand mittig ins quadratische Bild passen, ohne
    die Blickrichtung zu aendern (v135). Rueckgabe je Richtung
    (Abstand entlang dir, seitlicher Versatz als Vector, relativ zur Mitte).
    Je Bildachse: a = rel.achse, z = rel.forward, M = max(a - t z),
    m = min(a + t z); Kamera p_a = (m + M) / 2, p_f <= (m - M) / (2 t)."""
    tt = math.tan(fov / 2.0) / max(margin, 1e-6)
    points, safety = _sph_reduce_points(points)
    safety *= (1.0 + 1.0 / tt)
    rel = points - np.array(center, dtype=np.float64)
    out = []
    for u in dirs:
        f = -u
        q = f.to_track_quat('-Z', 'Y')
        r = q @ Vector((1.0, 0.0, 0.0))
        up = q @ Vector((0.0, 1.0, 0.0))
        M1 = M2 = -np.inf
        m1 = m2 = np.inf
        for k in range(0, len(rel), 500000):   # speicherschonend
            blk = rel[k:k + 500000]
            z = blk @ np.array(f)
            a = blk @ np.array(r)
            b = blk @ np.array(up)
            M1 = max(M1, float((a - tt * z).max()))
            m1 = min(m1, float((a + tt * z).min()))
            M2 = max(M2, float((b - tt * z).max()))
            m2 = min(m2, float((b + tt * z).min()))
        p_f = min((m1 - M1) / (2.0 * tt), (m2 - M2) / (2.0 * tt))
        shift = r * ((m1 + M1) / 2.0) + up * ((m2 + M2) / 2.0)
        out.append((-p_f + safety, shift))
    return out


def _sph_center_shifts(points, center, dirs, fov, dists, shifts, iters=8):
    """Versatz so nachfuehren, dass das Modell bei der endgueltigen
    Entfernung (nach der 60-%-Grenze) in beiden Bildachsen mittig sitzt
    (v135). Die geschlossene Loesung zentriert nur die enge Achse exakt;
    perspektivisch braucht die andere Achse eine kurze Newton-Iteration:
    Mitte der Bildspanne -> Kamera um Mitte * t * Tiefe verschieben."""
    tt = math.tan(fov / 2.0)
    points, _ = _sph_reduce_points(points)
    P = np.asarray(points, dtype=np.float64)
    c0 = np.array(center, dtype=np.float64)
    out = []
    for u, d, sv in zip(dirs, dists, shifts):
        f = -u
        q = f.to_track_quat('-Z', 'Y')
        axes = (np.array(q @ Vector((1.0, 0.0, 0.0))),
                np.array(q @ Vector((0.0, 1.0, 0.0))))
        fa = np.array(f)
        pos = c0 + np.array(u) * d + np.array(sv)
        for ax in axes:
            for _ in range(iters):
                rel = P - pos
                z = rel @ fa
                v = (rel @ ax) / (tt * z)
                i_max, i_min = int(v.argmax()), int(v.argmin())
                off = 0.5 * (v[i_max] + v[i_min])
                if abs(off) < 1e-5:
                    break
                pos = pos + ax * (off * tt * 0.5 * (z[i_max] + z[i_min]))
        out.append(Vector(pos - c0 - np.array(u) * d))
    return out


def _sph_reduce_points(points, grid=384):
    """Verkleinert grosse Punktmengen fuer die Abstandsberechnung, ohne ein
    Ergebnis zu verfehlen: Je Rastersaeule (x/y-Zelle) zaehlen nur der
    tiefste und der hoechste Punkt -- jeder Punkt dazwischen liegt auf ihrer
    Verbindungsstrecke und kann fuer lineare Randbedingungen nie weiter
    aussen liegen. x/y werden auf die Zellmitte gesetzt; der dabei maximal
    entstehende Versatz (halbe Zelldiagonale) wird als Sicherheitsabstand
    zurueckgegeben. Kleine Mengen bleiben unveraendert (Rueckgabe 0.0)."""
    if len(points) <= 200000:
        return points, 0.0
    lo = points.min(axis=0)
    ext = float((points.max(axis=0) - lo)[:2].max())
    if ext <= 0.0:
        return points, 0.0
    cell = ext / grid
    ij = np.minimum(((points[:, :2] - lo[:2]) / cell).astype(np.int64),
                    grid - 1)
    key = ij[:, 0] * grid + ij[:, 1]
    zmin = np.full(grid * grid, np.inf)
    zmax = np.full(grid * grid, -np.inf)
    np.minimum.at(zmin, key, points[:, 2])
    np.maximum.at(zmax, key, points[:, 2])
    used = np.nonzero(np.isfinite(zmin))[0]
    cx = lo[0] + (used // grid + 0.5) * cell
    cy = lo[1] + (used % grid + 0.5) * cell
    red = np.vstack([np.column_stack([cx, cy, zmin[used]]),
                     np.column_stack([cx, cy, zmax[used]])])
    return red, cell * 0.7072


def _sph_required_radius(max_dim, fov, margin):
    """Sphere-Radius, bei dem ein Objekt der groessten Ausdehnung max_dim
    (mit Margin) komplett ins Bild passt: d = (max_dim*margin/2)/tan(fov/2).
    So ist das Objekt aus JEDER Richtung vollstaendig erfasst, weil fuer
    die laengste Ausdehnung dimensioniert wird."""
    half = (max_dim * margin) / 2.0
    t = math.tan(fov / 2.0)
    if t < 1e-6:
        return half  # Schutz gegen Division durch ~0
    return half / t


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
            # Einzelnes Objekt -> Eintrag mit Objektnamen als Label.
            item = s.guides.add()
            item.label = mesh_objs[0].name
            ref = item.objects.add()
            ref.obj = mesh_objs[0]
            msg = "Added guide '%s'." % mesh_objs[0].name
        else:
            # Mehrere Objekte -> ein Sammeleintrag "Guide" (durchnummeriert).
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
        # Bevorzugt die aktive Collection des View-Layers.
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
    """Sammelt die obersten Parent-Objekte aus den gegebenen Collections.
    'Oberstes' = Objekt hat keinen Parent ODER sein Parent liegt NICHT in den
    Collections (dann ist dieses Objekt das Oberelement seiner Gruppe).
    Bestehende Gruppen bleiben so erhalten -- ihre Kinder werden nicht
    einzeln gesammelt, nur das Gruppen-Oberelement. Rueckgabe: Liste
    eindeutiger Objekte."""
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
        # Oberstes, wenn kein Parent oder Parent nicht in der Auswahl liegt.
        if p is None or p not in all_objs:
            if o.name not in seen:
                seen.add(o.name)
                tops.append(o)
    return tops


def _ga_world_bounds(objs, depsgraph, fast_bbox=False):
    """Welt-Bounding-Box ueber alle objs UND ihre Nachkommen.

    fast_bbox=False (Standard, praezise): numpy-vektorisiert ueber die echten
    Mesh-Vertices -- exakt, auch bei rotierten Objekten, aber deutlich
    schneller als eine Python-Vertex-Schleife (foreach_get + Matrixmultipl.
    im Batch).

    fast_bbox=True (schnell, etwas ungenauer): nutzt nur die 8 Eckpunkte der
    objekteigenen Bounding-Box (obj.bound_box) pro Objekt. Bei rotierten
    Objekten wird die resultierende Welt-Box minimal zu gross.

    Rueckgabe (min_v, max_v) als mathutils.Vector oder (None, None)."""
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
                # 8 Eckpunkte der lokalen Bounding-Box.
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

            # Homogen transformieren: (N,4) @ (4,4)^T -> (N,4).
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
    """Rahmt die sichtbare Geometrie (Mesh) der Collections im 3D-Viewport ein
    (View Selected), im Viewport des Knopfs, sonst im ersten der Fenster.
    Die Auswahl wird danach wiederhergestellt (v117)."""
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

        # Empty-Position (Referenz): X/Y Mitte der BBox, Z am Boden (min Z).
        cx = (mn.x + mx.x) * 0.5
        cy = (mn.y + mx.y) * 0.5
        cz = mn.z
        ref = mathutils.Vector((cx, cy, cz))

        # Verschiebungsvektor, damit dieser Referenzpunkt (und damit das
        # Empty) am Ende auf dem Weltursprung 0/0/0 liegt. ALLE Top-Objekte
        # werden um denselben Vektor verschoben -> relative Anordnung bleibt,
        # Auto steht mittig ueber dem Ursprung, Raeder auf Z=0.
        d = -ref

        # Vorhandenes GCapture_Group-Empty wiederverwenden oder neu anlegen.
        empty = bpy.data.objects.get(self.EMPTY_NAME)
        if empty is None or empty.type != 'EMPTY':
            empty = bpy.data.objects.new(self.EMPTY_NAME, None)
            empty.empty_display_type = 'PLAIN_AXES'
            empty.empty_display_size = max((mx - mn).length * 0.1, 0.1)
            colls[0].objects.link(empty)

        # Zuerst alle Top-Objekte um d verschieben (Welt-Translation),
        # solange sie noch NICHT geparentet sind.
        for o in tops:
            if o == empty:
                continue
            o.location = o.location + d

        # Empty auf den Weltursprung.
        empty.location = mathutils.Vector((0.0, 0.0, 0.0))
        context.view_layer.update()

        # Parenten mit erhaltener (jetzt schon verschobener) Welt-Transform.
        for o in tops:
            if o == empty:
                continue
            if o.parent == empty:
                continue
            mw = o.matrix_world.copy()
            o.parent = empty
            o.matrix_parent_inverse = empty.matrix_world.inverted()
            o.matrix_world = mw

        # Sphere-Zielinfo: geometrisches Zentrum (nach Verschiebung) -- die
        # Sphere wird separat erstellt und sitzt volumenzentriert.
        sph_center_z = (mn.z + mx.z) * 0.5 + d.z
        self.report(
            {'INFO'},
            "Grouped %d top object(s) under '%s' at world origin. Ground on "
            "Z=0. Sphere center will be at Z=%.3f (volume center)."
            % (len(tops), self.EMPTY_NAME, sph_center_z))
        _gcapture_frame_collections(context, colls)
        return {'FINISHED'}


class GCAPTURE_OT_make_sphere(Operator):
    bl_idname = "gcapture.make_sphere"
    bl_label = "Create / Update Camera Sphere"
    bl_description = ("Create an Ico-Sphere camera array, auto-scaled so the "
                      "target collections' objects are fully framed from "
                      "every direction. Re-run to update subdivisions/scale")
    bl_options = {'REGISTER', 'UNDO'}

    SPHERE_NAME = "Camera_Sphere"

    @classmethod
    def poll(cls, context):
        return len(context.scene.gcapture_settings.sph_target_colls) > 0

    def execute(self, context):
        import bmesh as _bm
        s = context.scene.gcapture_settings
        scene = context.scene

        # Vorhandene Sphere ZUERST ermitteln (vor der Bounds-Berechnung),
        # damit sie aus der Groessenberechnung ausgeschlossen werden kann.
        # PRIMAER ueber die gespeicherte Referenz (sph_object); Name als
        # Fallback (ein Build koennte sie umbenannt haben).
        existing = s.sph_object
        if existing is None or existing.name not in bpy.data.objects:
            existing = bpy.data.objects.get(self.SPHERE_NAME)
        # Im Viewport geloeschte Sphere: Blender entfernt sie nur aus der
        # Szene, die gespeicherte Referenz (sph_object) haelt das Objekt am
        # Leben -- ohne Collection laesst es sich nicht selektieren (v94).
        # Liegt sie in keiner Szene mehr: endgueltig entfernen und neu
        # anlegen; liegt sie nur in einer anderen Szene: dort lassen.
        if existing is not None and existing.name not in context.scene.objects:
            if not existing.users_scene:
                if s.sph_object == existing:
                    s.sph_object = None
                bpy.data.objects.remove(existing, do_unlink=True)
            existing = None

        # Zielobjekte + Bounds -- die Sphere selbst ausschliessen, sonst
        # waechst der Radius bei jedem Update (akkumulierte Skalierung).
        objs = _sph_collect_objects(s, exclude=existing)
        if not objs:
            self.report({'ERROR'},
                        "No visible mesh objects in the target collections "
                        "(hidden objects are ignored).")
            return {'CANCELLED'}
        bounds = _sph_world_bounds(objs)
        if bounds is None:
            self.report({'ERROR'}, "Could not compute object bounds.")
            return {'CANCELLED'}
        mn, mx, center, max_dim, diag, biggest = bounds
        if diag < 1e-6:
            self.report({'ERROR'}, "Object bounds are degenerate (zero size).")
            return {'CANCELLED'}

        # Radius aus FOV + RAUMDIAGONALE + Margin. Die Diagonale (statt der
        # laengsten Achse) macht das Framing formunabhaengig: kompakte und
        # laengliche Objekte werden mit demselben Margin gleich gut gerahmt.
        fov = _sph_camera_min_fov(s, scene)
        radius = _sph_required_radius(diag, fov, s.sph_margin)
        fit_msg = ""

        # Radius in die Vertices, Zentrum in die Objekt-Position (v95): der
        # Ursprung liegt in der Sphere-Mitte, damit Skalieren mit S um die
        # Mitte wirkt (Live Camera Lock).
        mw = Matrix.Scale(radius, 4)

        # Sphere-Mesh in EINEM Durchgang aufbauen: Unit-Ico-Sphere erzeugen,
        # optional untere Haelfte entfernen, Vertices direkt mit mw in
        # Weltkoordinaten backen -- ALLES im bmesh, BEVOR es dem Objekt
        # zugewiesen wird. So gibt es kein Lesen-nach-Zuweisen und keine
        # Abhaengigkeit vom Update-Timing (das war die Fehlerquelle, durch
        # die die Sphere beim Wiederverwenden geschrumpft wirkte).
        new_mesh = bpy.data.meshes.new(self.SPHERE_NAME)
        bm = _bm.new()
        _bm.ops.create_icosphere(bm, subdivisions=s.sph_subdivisions,
                                 radius=1.0)
        if s.sph_upper_only:
            bm.verts.ensure_lookup_table()
            to_del = [v for v in bm.verts if v.co.z < -1e-5]
            if to_del:
                _bm.ops.delete(bm, geom=to_del, context='VERTS')

        # Framing auf die Sequenz eichen (v96): je Face (= Kamera) den
        # noetigen Abstand aus den echten Modellpunkten berechnen.
        fit_q = None
        shifts = None
        if s.sph_fit_each:
            pts = _sph_model_points(objs)
            if pts is not None:
                bm.faces.ensure_lookup_table()
                unit_c = [f.calc_center_median() for f in bm.faces]
                dirs = [c.normalized() for c in unit_c]
                # Mindestens ~1 Pixel Luft zum Bildrand (bei Margin 1.0
                # beruehrt das Modell den Rand sonst exakt -> Rundung).
                px_guard = 1.0 + 2.0 / max(s.resolution, 16)
                shifts = None
                # Mit Look Target "Origin" zielte die Kamera nach dem Versatz
                # nicht mehr parallel -> dort kein Versatz.
                if s.sph_center_view and s.look_mode != 'ORIGIN':
                    framed = _sph_framed_positions(pts, center, dirs, fov,
                                                   s.sph_margin * px_guard)
                    need = [d for d, _ in framed]
                    shifts = [sv for _, sv in framed]
                else:
                    need = _sph_required_distances(pts, center, dirs, fov,
                                                   s.sph_margin * px_guard)
                # Kamera sitzt im Face-Zentrum: Abstand = radius * |c_unit|.
                ratios = [d / max(c.length, 1e-9)
                          for d, c in zip(need, unit_c)]
                limit = max(range(len(ratios)), key=ratios.__getitem__)
                radius = ratios[limit]
                fit_q = [max(_SPH_FIT_MIN, min(1.0, rr / radius))
                         for rr in ratios]
                if shifts is not None:
                    # Endgueltige Entfernung je Kamera (nach der Grenze)
                    # und darauf zentrieren (v135).
                    finals = [q * radius * c.length
                              for q, c in zip(fit_q, unit_c)]
                    shifts = _sph_center_shifts(pts, center, dirs, fov,
                                                finals, shifts)
                fit_msg = " Each view filled%s; widest view: camera %d." % (
                    " and centred" if shifts is not None else "", limit + 1)
        # Vertices in Objekt-Koordinaten backen (Unit -> radius*v). Matrix
        # erst hier bilden: Fill Each View kann den Radius oben aendern.
        mw = Matrix.Scale(radius, 4)
        for v in bm.verts:
            v.co = mw @ v.co
        bm.to_mesh(new_mesh)
        bm.free()
        if fit_q is not None and len(fit_q) == len(new_mesh.polygons):
            attr = new_mesh.attributes.new(_SPH_FIT_ATTR, 'FLOAT', 'FACE')
            attr.data.foreach_set("value", fit_q)
            # Seitlicher Versatz je Kamera (v135). Die Sphere hat keine
            # Drehung/Skalierung -> Welt-Versatz = Objekt-Versatz.
            if (s.sph_fit_each and s.sph_center_view and shifts is not None
                    and len(shifts) == len(new_mesh.polygons)):
                sattr = new_mesh.attributes.new(_SPH_SHIFT_ATTR,
                                                'FLOAT_VECTOR', 'FACE')
                sattr.data.foreach_set(
                    "vector", [c for v in shifts for c in (v.x, v.y, v.z)])

        # Vorhandene Sphere wiederverwenden (Mesh ersetzen) oder neu anlegen.
        if existing is not None and existing.type == 'MESH':
            sphere_obj = existing
            if sphere_obj.name != self.SPHERE_NAME:
                sphere_obj.name = self.SPHERE_NAME
            old_mesh = sphere_obj.data
            sphere_obj.data = new_mesh
            if old_mesh.users == 0:
                bpy.data.meshes.remove(old_mesh)
        else:
            sphere_obj = bpy.data.objects.new(self.SPHERE_NAME, new_mesh)
            _gcapture_rig_collection(context.scene).objects.link(sphere_obj)

        _gcapture_move_to_rig(context.scene, sphere_obj)   # eigene Collection (v136)
        # Objekt-Transform: nur die Position (Zentrum); die Skalierung
        # steckt in den Vertices.
        sphere_obj.matrix_world = Matrix.Translation(center)
        # Lage nach Create / Update (v133): Build setzt die Sphere hierher
        # zurueck und verwirft damit Live-Camera-Adjust-Aenderungen.
        sphere_obj["gcapture_base_loc"] = tuple(center)
        # Mittelpunkt merken (Objekt-Koordinaten, also der Ursprung):
        # Zielpunkt der Kameras im CENTER-Modus, auch bei der Halbkugel.
        sphere_obj[_SPH_CENTER_PROP] = (0.0, 0.0, 0.0)

        # Anzeige als Wireframe, nicht rendern.
        sphere_obj.display_type = 'WIRE'
        sphere_obj.hide_render = True

        n_faces = len(sphere_obj.data.polygons)

        # Referenz auf die erzeugte Sphere speichern, damit ein spaeteres
        # Update sie auch nach einer Umbenennung (durch Build) wiederfindet.
        s.sph_object = sphere_obj

        # Als Leitgitter setzen -- FEST in die Guide-Liste, nicht ueber das
        # aktive Objekt. Sonst nimmt der Build das gerade aktive Objekt (z. B.
        # den Wuerfel), baut die Kameras auf DESSEN Faces und die Kamera klebt
        # an der Objektoberflaeche statt auf der Sphere zu sitzen.
        if s.sph_set_as_guide:
            # Guide-Liste leeren und nur die Sphere eintragen.
            s.guides.clear()
            item = s.guides.add()
            item.label = sphere_obj.name
            ref = item.objects.add()
            ref.obj = sphere_obj
            s.guide_index = 0
            s.use_guide_list = True
            # Auswahl/aktiv auf die Sphere (kosmetisch).
            for o in context.selected_objects:
                o.select_set(False)
            sphere_obj.select_set(True)
            context.view_layer.objects.active = sphere_obj
            # Look-Modus auf Zentrum (Sphere blickt nach innen).
            s.look_mode = 'CENTER'
            # Umbenennung beim Build verhindern: Die Sphere soll durchgehend
            # "Camera_Sphere" heissen, damit Update sie immer wiederfindet.
            s.rename_guide = False

        # Render-Einstellungen fuer Splatting: NUR EINMAL pro Datei setzen.
        # Bei spaeteren Updates bleiben die (evtl. vom Nutzer angepassten)
        # Einstellungen unangetastet.
        render_msg = ""
        if not s.sph_render_setup_done:
            skipped = _sph_apply_render_settings(scene)
            _sph_apply_camera_clip(s)
            s.sph_render_setup_done = True
            if skipped:
                render_msg = (" Render setup applied (could not set: %s)."
                              % ", ".join(skipped))
            else:
                render_msg = " Render setup applied."

        self.report({'INFO'},
                    "Camera sphere: %d cameras, radius %.2f (from %d object(s), "
                    "diagonal %.2f, max dim %.2f, biggest part: '%s' %.2f, "
                    "margin %.0f%%)%s.%s%s"
                    % (n_faces, radius, len(objs), diag, max_dim,
                       biggest[0], biggest[1],
                       (s.sph_margin - 1.0) * 100,
                       ", upper hemisphere" if s.sph_upper_only else "",
                       fit_msg, render_msg))
        return {'FINISHED'}


class GCAPTURE_OT_lock_start(Operator):
    bl_idname = "gcapture.lock_start"
    bl_label = "Live Camera Adjust"
    bl_description = ("Adjust the camera of the current frame live: selects the "
                      "sphere; press S in the viewport to scale it (G to move) "
                      "- the camera follows. Then apply it to this camera or "
                      "to all cameras")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        s = context.scene.gcapture_settings
        return (not s.live_lock and
                bool(_collect_guide_objects(context, context.active_object)))

    def invoke(self, context, event):
        s = context.scene.gcapture_settings
        sph = s.sph_object
        if sph is None or sph.name not in context.scene.objects:
            guides = _collect_guide_objects(context, context.active_object)
            sph = guides[0] if guides else None
        if sph is None:
            self.report({'ERROR'}, "No camera sphere in the scene.")
            return {'CANCELLED'}
        # Aeltere Sphere: Ursprung in die Mitte, damit S um die Mitte skaliert.
        _gcapture_sphere_origin_to_center(sph)
        # Lage nach Create / Update nachtragen, damit Build G/S zuruecksetzt.
        _sph_ensure_base(sph, s)
        # Sphere als einziges Objekt auswaehlen und aktiv setzen.
        for o in context.selected_objects:
            o.select_set(False)
        sph.select_set(True)
        context.view_layer.objects.active = sph
        # Lock einschalten (friert das Face des aktuellen Frames ein).
        s.live_lock = True

        # Kein automatisches Skalieren (v97): fuer Einsteiger verwirrend.
        # Die Sphere ist ausgewaehlt, der Nutzer drueckt selbst S.
        self.report({'INFO'}, "Live Camera Adjust on - press S in the viewport "
                    "to scale the sphere, G to move it. The camera follows.")
        return {'FINISHED'}


class GCAPTURE_OT_lock_finish(Operator):
    bl_idname = "gcapture.lock_finish"
    bl_label = "Finish Live Camera Adjust"
    bl_description = ("Finish Live Camera Adjust and apply the adjustment to "
                      "this camera only or to all cameras")
    bl_options = {'REGISTER', 'UNDO'}

    apply_to: EnumProperty(
        name="Apply to",
        items=[('THIS', "Only this camera",
                "Keep the adjusted view for this camera only: the sphere goes "
                "back to how it was; kept until the next Build Camera "
                "Animation"),
               ('ALL', "All cameras",
                "Keep the changed sphere and rebuild all camera poses from it"),
               ('CANCEL', "Cancel",
                "Undo the adjustment: the sphere and the camera go back to "
                "how they were")],
        default='ALL', options={'SKIP_SAVE'},
    )

    @classmethod
    def poll(cls, context):
        return context.scene.gcapture_settings.live_lock

    def execute(self, context):
        global _GCAPTURE_LOCK_BUSY
        s = context.scene.gcapture_settings
        if self.apply_to == 'CANCEL':
            # Sphere zuruecksetzen und die Kamera des eingefrorenen Frames
            # noch mit aktivem Lock neu setzen (Keyframe), dann Lock aus (v132).
            for g in _GCAPTURE_LOCK_GUIDES:
                if g is not None and g.name in _GCAPTURE_LOCK_START_MW:
                    g.matrix_world = _GCAPTURE_LOCK_START_MW[g.name]
            context.view_layer.update()
            _GCAPTURE_LOCK_BUSY = True
            try:
                _gcapture_lock_apply(context, scene=context.scene)
            finally:
                _GCAPTURE_LOCK_BUSY = False
            s.live_lock = False
            self.report({'INFO'}, "Live Camera Adjust cancelled - sphere and "
                        "camera are back.")
            return {'FINISHED'}
        if self.apply_to == 'THIS':
            # Erst den Lock aus (Handler weg), dann Pose speichern und die
            # Sphere zuruecksetzen -- sonst zoege der Handler die Kamera mit.
            idx = _GCAPTURE_LOCK_FROZEN_IDX
            guides = [g for g in _GCAPTURE_LOCK_GUIDES if g is not None]
            s.live_lock = False
            msg = self._keep_this_view(idx, guides)
            self.report({'INFO'}, msg)
            # Wie nach einem Build: Version speichern (v128).
            if (s.save_after_build and not bpy.app.background
                    and getattr(context, "window", None)):
                try:
                    bpy.ops.gcapture.save_version('INVOKE_DEFAULT')
                except Exception as exc:
                    print("[Gaussian Render Capture] Save prompt failed:", exc)
            return {'FINISHED'}
        # Lock ausschalten (loest _gcapture_on_lock_toggle aus -> Handler weg,
        # Frozen-Index zurueck).
        s.live_lock = False
        # Vollstaendige Neuberechnung ueber den modalen Build-Operator.
        bpy.ops.gcapture.build_animation('INVOKE_DEFAULT', keep_adjustments=True)
        self.report({'INFO'}, "Live adjust finished -> rebuilding all poses.")
        return {'FINISHED'}

    @staticmethod
    def _keep_this_view(idx, guides):
        """Pose der eingefrorenen Kamera am Face speichern und die Leitgitter
        auf ihre Lage beim Einschalten zuruecksetzen (v123)."""
        if idx is None or not guides:
            return "Live adjust finished."
        base = 0
        for g in guides:
            faces = _guide_faces_with_targets(g)
            if idx < base + len(faces):
                center, _, geo_center, _ = faces[idx - base]
                mw0 = _GCAPTURE_LOCK_START_MW.get(g.name, g.matrix_world.copy())
                inv = mw0.inverted()
                _sph_store_view_adjustment(g, idx - base, inv @ center,
                                           inv @ geo_center)
                for other in guides:
                    if other.name in _GCAPTURE_LOCK_START_MW:
                        other.matrix_world = _GCAPTURE_LOCK_START_MW[other.name]
                return ("Live adjust finished -> view %d kept for this camera "
                        "only; the sphere is back." % (idx + 1))
            base += len(faces)
        return "Live adjust finished."


class GCAPTURE_OT_build(Operator):
    bl_idname = "gcapture.build_animation"
    bl_label = "Build Camera Animation"
    bl_description = ("Builds a camera animation from the faces of the camera "
                      "sphere (one keyframe per face), fresh from the step 4 "
                      "settings - Live Camera Adjust changes are discarded. "
                      "Runs modally with a progress display (ESC to cancel)")
    bl_options = {'REGISTER', 'UNDO'}

    # Nur "All Cameras" (Live Camera Adjust) baut mit der veraenderten
    # Sphere neu; sonst frisch aus Schritt 3 (v133).
    keep_adjustments: BoolProperty(default=False, options={'SKIP_SAVE', 'HIDDEN'})

    _is_running = False

    @classmethod
    def poll(cls, context):
        if cls._is_running:
            return False
        s = context.scene.gcapture_settings
        # Waehrend Live Camera Lock aktiv ist gesperrt -- erst Finish Live
        # Adjust ausfuehren, sonst wuerden die live angepassten Keyframes
        # ueberschrieben.
        if s.live_lock:
            return False
        if s.use_guide_list and len(s.guides) > 0:
            return True
        obj = context.active_object
        return obj is not None and obj.type == 'MESH'

    def _setup(self, context):
        s = context.scene.gcapture_settings
        obj = context.active_object
        self._use_list = s.use_guide_list and len(s.guides) > 0
        if not self._use_list and (obj is None or obj.type != 'MESH'):
            raise RuntimeError("No active mesh object selected.")

        self._guide_objs = _collect_guide_objects(context, obj)
        if not self._guide_objs:
            raise RuntimeError("No camera sphere in the scene - create it "
                               "first (step 4: Create / Update Camera Sphere).")
        self._reset = False
        if not getattr(self, "keep_adjustments", False):
            sph = s.sph_object
            if sph is not None and sph in self._guide_objs:
                self._reset = _sph_reset_adjustments(sph)
                if self._reset:
                    context.view_layer.update()

        # Leitgitter als Drahtgitter anzeigen, aus dem Render nehmen.
        for gobj in self._guide_objs:
            setup_guide_object(gobj, s)

        self._cam = get_or_create_camera(s.camera_name)
        self._cam.data.lens_unit = 'MILLIMETERS'
        self._cam.data.lens = s.focal_length
        if self._cam.animation_data:
            self._cam.animation_data_clear()
        self._cam.rotation_mode = 'QUATERNION'

        # Alle Faces aller Leitgitter als flache Arbeitsliste sammeln.
        self._faces = []
        for gobj in self._guide_objs:
            self._faces.extend(_guide_faces_with_targets(gobj))
        self._total = len(self._faces)
        if self._total == 0:
            raise RuntimeError("Guides have no faces.")

        # Anzeigegroesse der Kamera im Viewport passend zur Sphere (v1.1.5):
        # Blenders 1 m verdeckt kleine Modelle. Nur solange der Nutzer sie
        # nicht selbst gesetzt hat (Blender-Standard oder von uns gesetzt).
        cam_data = self._cam.data
        if (abs(cam_data.display_size - 1.0) < 1e-6
                or cam_data.get("gcapture_display_auto")):
            radius = max((max(g.dimensions) * 0.5 for g in self._guide_objs),
                         default=0.0)
            if radius > 0.0:
                cam_data.display_size = max(radius * 0.1, 0.001)
                cam_data["gcapture_display_auto"] = True

        # Interior-Erkennung vorbereiten (einmalig). Nur mit eigenen
        # Leitgittern wirksam: die Kamera-Sphere liegt immer ausserhalb des
        # Modells, dort wuerde der Test nur bremsen (v93).
        self._skip_interior = s.skip_interior and _gcapture_has_custom_guides(s)
        self._s = s
        self._guide_set = set(self._guide_objs)
        self._bvh_cache = None
        self._aspect = 1.0
        if self._skip_interior:
            self._bvh_cache = _rc_build_visible_mesh_bvh_cache(
                context, self._guide_set)
            ry = context.scene.render.resolution_y
            self._aspect = (context.scene.render.resolution_x / ry) if ry else 1.0

        self._frame = s.frame_start
        self._n = 0
        self._rejected = 0
        self._done = 0
        self._start_time = time.time()

    def _process_one(self, context):
        s = self._s
        center, normal, geo_center, origin = self._faces[self._done]
        if s.look_mode == 'CENTER':
            look_dir = (geo_center - center)
        elif s.look_mode == 'ORIGIN':
            look_dir = (origin - center)
        else:
            look_dir = -normal
        if look_dir.length < 1e-9:
            look_dir = -normal if normal.length > 1e-9 else Vector((0, 0, -1))
        look_dir = look_dir.normalized()

        if self._skip_interior:
            quat = look_dir.to_track_quat('-Z', 'Y')
            right = quat @ Vector((1.0, 0.0, 0.0))
            up = quat @ Vector((0.0, 1.0, 0.0))
            if _rc_is_camera_inside_mesh(context, center, self._guide_set):
                self._rejected += 1
                return
            if _rc_geometry_within_near_clip(self._cam.data, center, look_dir,
                                             right, up, self._bvh_cache,
                                             self._aspect):
                self._rejected += 1
                return

        self._cam.location = center
        self._cam.rotation_quaternion = look_dir.to_track_quat('-Z', 'Y')
        self._cam.keyframe_insert(data_path="location", frame=self._frame)
        self._cam.keyframe_insert(data_path="rotation_quaternion",
                                  frame=self._frame)
        self._frame += 1
        self._n += 1

    def invoke(self, context, event):
        # Warnung, wenn Live-Camera-Adjust-Aenderungen verloren gingen (v133).
        s = context.scene.gcapture_settings
        if s.sph_object is not None:
            _sph_ensure_base(s.sph_object, s)     # aeltere Spheres (v134)
        if (not self.keep_adjustments
                and _sph_has_adjustments(s.sph_object)):
            return context.window_manager.invoke_props_dialog(self, width=360)
        return self.execute(context)

    def draw(self, context):
        col = self.layout.column()
        row = col.row()
        row.alert = True
        row.label(text="Live Camera Adjust changes will be discarded.",
                  icon='ERROR')
        col.label(text="Build starts again from the step 4 settings.")
        col.label(text="Cancel keeps the current camera animation.")

    def execute(self, context):
        try:
            self._setup(context)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        GCAPTURE_OT_build._is_running = True
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.02, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'ESC':
            self._cleanup(context)
            self.report({'WARNING'}, "Build cancelled.")
            return {'CANCELLED'}

        if event.type == 'TIMER':
            if self._done >= self._total:
                return self._finish(context)

            elapsed = time.time() - self._start_time
            time_info = ""
            if self._done > 0:
                per = elapsed / self._done
                rem = int(per * (self._total - self._done))
                tstr = ("%dm %ds" % (rem // 60, rem % 60)) if rem >= 60 else ("%ds" % rem)
                time_info = ", ~%s left" % tstr
            _gcapture_progress_set(
                "gcapture.build_animation", self._done / self._total,
                "Building cameras %d/%d%s" % (self._done, self._total, time_info))

            # Batchgroesse: ohne Interior-Check sind Posen sehr schnell ->
            # grosse Batches; mit Interior-Check (Ray-Cast) kleinere.
            batch = 5 if self._skip_interior else 100
            try:
                for _ in range(batch):
                    if self._done >= self._total:
                        break
                    self._process_one(context)
                    self._done += 1
            except Exception as exc:
                self._cleanup(context)
                self.report({'ERROR'}, "Build failed: %s" % exc)
                return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def _finish(self, context):
        s = self._s
        if self._n == 0:
            self._cleanup(context)
            self.report({'ERROR'},
                        "No camera poses generated (no faces, or all "
                        "rejected as interior).")
            return {'CANCELLED'}

        if s.constant_interp:
            for fcurve in iter_action_fcurves(self._cam):
                for kp in fcurve.keyframe_points:
                    kp.interpolation = 'CONSTANT'
        if s.set_active_camera:
            context.scene.camera = self._cam
            # In die Kameraansicht wechseln (entspricht Numpad 0) -- in allen
            # 3D-Viewports. region_3d.view_perspective ist der zuverlaessige
            # API-Weg dafuer.
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    for space in area.spaces:
                        if space.type == 'VIEW_3D':
                            space.region_3d.view_perspective = 'CAMERA'
        if s.set_frame_range:
            context.scene.frame_start = s.frame_start
            context.scene.frame_end = s.frame_start + self._n - 1
        if s.set_resolution:
            rnd = context.scene.render
            rnd.resolution_x = s.resolution
            rnd.resolution_y = s.resolution
            rnd.resolution_percentage = 100
        context.scene.frame_set(s.frame_start)

        cam_name = self._cam.name
        n = self._n
        rejected = self._rejected
        if self._use_list:
            total_objs = sum(len(it.objects) for it in s.guides)
            src = "%d guide entries (%d objects)" % (len(s.guides), total_objs)
        else:
            src = "'%s'" % self._guide_objs[0].name
        self._cleanup(context)

        msg = "Built %d camera poses from %s (camera: %s)." % (n, src, cam_name)
        if getattr(self, "_reset", False):
            msg += " Live Camera Adjust changes discarded (fresh from step 4)."
        if rejected:
            msg += " Rejected %d interior cameras." % rejected
        self.report({'INFO'}, msg)
        # Aufforderung, die Szene als Version zu speichern (v90). Nur mit
        # Fenster (nicht im Headless-Test).
        if (s.save_after_build and not bpy.app.background
                and getattr(context, "window", None)):
            try:
                bpy.ops.gcapture.save_version('INVOKE_DEFAULT')
            except Exception as exc:
                print("[Gaussian Render Capture] Save prompt failed:", exc)
        return {'FINISHED'}

    def _cleanup(self, context):
        if getattr(self, "_timer", None):
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        _gcapture_progress_clear("gcapture.build_animation")
        self._bvh_cache = None
        GCAPTURE_OT_build._is_running = False


# ----------------------------------------------------------------------
# UIList fuer die Guide-Eintraege (editierbares Label + Objektanzahl)
# ----------------------------------------------------------------------
class GCAPTURE_UL_guides(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon,
                  active_data, active_propname, index):
        n = len(item.objects)
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            # Editierbares Label (Doppelklick zum Umbenennen).
            layout.prop(item, "label", text="", emboss=False,
                        icon='MESH_ICOSPHERE')
            sub = layout.row()
            sub.alignment = 'RIGHT'
            sub.label(text=("%d objs" % n) if n != 1 else "1 obj")
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text=item.label)


class GCAPTURE_UL_colls(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon,
                  active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            if item.coll is not None:
                layout.label(text=item.coll.name, icon='OUTLINER_COLLECTION')
            else:
                layout.label(text="(empty)", icon='ERROR')
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text=item.coll.name if item.coll else "(empty)")


# ----------------------------------------------------------------------
# N-Panel
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Eigene Icons: farbige Nummern-Plaketten fuer die Schritte (v87)
# ----------------------------------------------------------------------
# Das Addon bleibt eine einzelne Datei: die Plaketten werden beim
# Registrieren mit numpy gezeichnet, als PNG in einen Cache-Ordner
# geschrieben und ueber bpy.utils.previews geladen. Kein bpy.data-Zugriff
# (beim Start eingeschraenkt), PNG-Kodierung per zlib.
import struct as _gcapture_struct
import tempfile as _gcapture_tempfile
import zlib as _gcapture_zlib

# Phasenfarben wie in der Workflow-Grafik: orange -> gelb -> gruen -> blau (sRGB 0..1).
_GCAPTURE_BADGE_RGB = {
    'PREP': (0.93, 0.50, 0.13),      # orange: Vorbereiten
    'CAMERAS': (0.95, 0.80, 0.16),   # gelb:   Kameras
    'OUTPUT': (0.33, 0.70, 0.28),    # gruen:  Rendern
    'COLMAP': (0.23, 0.65, 0.79),    # blau:   COLMAP-Export (v146)
    'NEUTRAL': (0.92, 0.92, 0.92),   # weiss:  keine Phase (Camera Settings)
}
# Ziffern als Strichzuege in einem Einheitsquadrat (x rechts, y oben).
_GCAPTURE_DIGIT_STROKES = {
    "1": [[(0.30, 0.78), (0.56, 0.96), (0.56, 0.04)]],
    "2": [[(0.18, 0.74), (0.28, 0.90), (0.50, 0.97), (0.72, 0.90),
           (0.82, 0.72), (0.76, 0.54), (0.18, 0.04), (0.84, 0.04)]],
    "3": [[(0.20, 0.94), (0.80, 0.94), (0.46, 0.57), (0.62, 0.57),
           (0.80, 0.44), (0.82, 0.24), (0.68, 0.07), (0.46, 0.03),
           (0.18, 0.12)]],
    "4": [[(0.66, 0.04), (0.66, 0.96), (0.12, 0.30), (0.88, 0.30)]],
    "5": [[(0.80, 0.94), (0.26, 0.94), (0.22, 0.56), (0.48, 0.61),
           (0.70, 0.56), (0.82, 0.38), (0.78, 0.14), (0.54, 0.03),
           (0.20, 0.10)]],
    "7": [[(0.18, 0.94), (0.82, 0.94), (0.40, 0.04)]],
    "6": [[(0.76, 0.92), (0.52, 0.97), (0.30, 0.86), (0.19, 0.60),
           (0.18, 0.32), (0.26, 0.12), (0.48, 0.03), (0.70, 0.10),
           (0.82, 0.30), (0.76, 0.50), (0.52, 0.58), (0.30, 0.52),
           (0.19, 0.36)]],
}
# Plaketten: Name -> (Phase, Ziffer; None = Punkt ohne Nummer, "" = leer).
_GCAPTURE_BADGES = {
    'gcapture_cam': ('PREP', "1"),        # Einstieg = Schritt 1 (v147)
    'gcapture_1': ('PREP', "2"),
    'gcapture_2': ('PREP', "3"),
    'gcapture_3': ('CAMERAS', "4"),
    'gcapture_4': ('CAMERAS', "5"),
    'gcapture_5': ('OUTPUT', "6"),
    'gcapture_6': ('COLMAP', "7"),
}
_GCAPTURE_ICON_VERSION = 6   # bei Aenderung am Aussehen erhoehen (Cache-Ordner)
_gcapture_previews = None


def _gcapture_badge_rgba(rgb, digit, size=64):
    """RGBA-Array (size, size, 4) uint8: gefuellter Kreis in Phasenfarbe,
    darauf die Ziffer fett in Dunkelgrau (oder ein kleiner Punkt).
    Kantenglaettung analytisch ueber den Abstand."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64) + 0.5
    x = xx / size
    y = 1.0 - yy / size
    px = 1.0 / size
    d_disc = np.hypot(x - 0.5, y - 0.5) - 0.47
    a_disc = np.clip(0.5 - d_disc / px, 0.0, 1.0)

    if digit:
        # Ziffernbox: Hoehe 0.56, Breite 0.40, zentriert.
        bx0, by0, bw, bh = 0.30, 0.22, 0.40, 0.56
        half_w = 0.052   # halbe Strichstaerke -> fett
        dmin = np.full_like(x, 1e9)
        for stroke in _GCAPTURE_DIGIT_STROKES[digit]:
            pts = [(bx0 + u * bw, by0 + v * bh) for (u, v) in stroke]
            for (ax, ay), (cx, cy) in zip(pts[:-1], pts[1:]):
                ex, ey = cx - ax, cy - ay
                ll = ex * ex + ey * ey
                tt = np.clip(((x - ax) * ex + (y - ay) * ey) / ll, 0.0, 1.0)
                dmin = np.minimum(dmin, np.hypot(x - ax - tt * ex,
                                                 y - ay - tt * ey))
        ink = np.clip(0.5 - (dmin - half_w) / px, 0.0, 1.0)
    elif digit is None:
        d_dot = np.hypot(x - 0.5, y - 0.5) - 0.13
        ink = np.clip(0.5 - d_dot / px, 0.0, 1.0)
    else:
        ink = np.zeros_like(x)   # leere Plakette

    ink_rgb = np.array((0.12, 0.12, 0.12))
    base = np.array(rgb)
    col = (base[None, None, :] * (1.0 - ink[..., None]) +
           ink_rgb[None, None, :] * ink[..., None])
    rgba = np.dstack([col, a_disc])
    return (np.clip(rgba, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _gcapture_write_png(path, rgba):
    """Minimaler PNG-Schreiber (RGBA, 8 Bit) ohne Zusatzbibliothek."""
    h, w = rgba.shape[:2]
    raw = b"".join(b"\x00" + rgba[row].tobytes() for row in range(h))

    def chunk(tag, data):
        crc = _gcapture_zlib.crc32(tag + data) & 0xFFFFFFFF
        return (_gcapture_struct.pack(">I", len(data)) + tag + data +
                _gcapture_struct.pack(">I", crc))
    png = (b"\x89PNG\r\n\x1a\n" +
           chunk(b"IHDR", _gcapture_struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)) +
           chunk(b"IDAT", _gcapture_zlib.compress(raw, 9)) +
           chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)


def _gcapture_icons_load():
    """Plaketten erzeugen (einmal pro Icon-Version) und laden. Scheitert
    etwas, bleibt _gcapture_previews None -> Panels fallen auf Blender-Icons
    zurueck."""
    global _gcapture_previews
    try:
        import bpy.utils.previews
        folder = os.path.join(_gcapture_tempfile.gettempdir(),
                              "gaussian_render_capture_icons_v%d"
                              % _GCAPTURE_ICON_VERSION)
        os.makedirs(folder, exist_ok=True)
        pcoll = bpy.utils.previews.new()
        for name, (phase, digit) in _GCAPTURE_BADGES.items():
            path = os.path.join(folder, name + ".png")
            if not os.path.isfile(path):
                _gcapture_write_png(path, _gcapture_badge_rgba(_GCAPTURE_BADGE_RGB[phase],
                                                     digit))
            pcoll.load(name, path, 'IMAGE')
        _gcapture_previews = pcoll
    except Exception as exc:
        print("[Gaussian Render Capture] Step icons unavailable:", exc)
        _gcapture_previews = None


def _gcapture_icons_unload():
    global _gcapture_previews
    if _gcapture_previews is not None:
        try:
            import bpy.utils.previews
            bpy.utils.previews.remove(_gcapture_previews)
        except Exception:
            pass
    _gcapture_previews = None


# Farben der Arbeitsphasen (Icons der Collection-Color-Tags). Blender
# erlaubt Addons keine eingefaerbten Boxen -- farbig sind nur Icons. Rot
# (COLLECTION_COLOR_01) bleibt fuer Warnungen (alert) reserviert.
# Reihenfolge wie eine Ampel: orange -> gelb -> gruen (v86).
_GCAPTURE_COLOR_PREP = 'COLLECTION_COLOR_02'     # orange: Vorbereiten
_GCAPTURE_COLOR_CAMERAS = 'COLLECTION_COLOR_03'  # gelb:   Kameras
_GCAPTURE_COLOR_OUTPUT = 'COLLECTION_COLOR_04'   # gruen:  Rendern
_GCAPTURE_COLOR_COLMAP = 'COLLECTION_COLOR_05'   # blau:   COLMAP-Export (v146)


# ----------------------------------------------------------------------
# Einsteigerhilfen (v90): Auswahl in eine Collection, Szenenversionen
# ----------------------------------------------------------------------
def _gcapture_version_files(folder, name):
    """Vorhandene Versionsnummern von <name>_vNNN*.blend im Ordner."""
    found = []
    if not folder or not os.path.isdir(folder):
        return found
    rx = re.compile(r"^%s_[vV](\d+)(?:[_\-.].*)?\.blend$" % re.escape(name))
    for fn in os.listdir(folder):
        m = rx.match(fn)
        if m:
            found.append(int(m.group(1)))
    return found


def _gcapture_next_version_path(folder, name, width=3):
    """Pfad der naechsten freien Version <name>_vNNN.blend im Ordner."""
    nums = _gcapture_version_files(folder, name)
    nxt = (max(nums) + 1) if nums else 1
    return os.path.join(folder, "%s_v%0*d.blend" % (name, width, nxt))


def _gcapture_clean_name(raw):
    """Szenenname ohne Endung, Versions-Token und unzulaessige Zeichen."""
    stem = os.path.splitext(os.path.basename(raw or ""))[0]
    name, _ = _exp_split_name_version(stem)
    name = re.sub(r'[\\/:*?"<>|\s]+', "_", name).strip("_.- ")
    return name or "Scene"


def _gcapture_managed_render_path(name, vtag):
    """Relativer Renderpfad direkt in den Datensatz:
    //<Name>_COLMAP/<vNNN>/images/<Name>_<vNNN>_"""
    return "//%s_COLMAP/%s/images/%s_%s_" % (name, vtag, name, vtag)


from bpy.app.handlers import persistent

_GCAPTURE_MANAGED_RX = re.compile(
    r"^//(?P<n>[^/]+)_COLMAP/(?P<v>[vV]\d+)/images/(?P=n)_(?P=v)_$")


def _gcapture_sync_render_version(filepath):
    """Setzt einen vom Addon verwalteten Renderpfad auf Name und Version der
    Datei (v119). Rueckgabe: Anzahl geaenderter Szenen."""
    stem = os.path.splitext(os.path.basename(filepath or ""))[0]
    name, vtag = _exp_split_name_version(stem)
    if not vtag:
        return 0
    name = _gcapture_clean_name(name)
    changed = 0
    for scene in bpy.data.scenes:
        rp = scene.render.filepath.replace("\\", "/")
        m = _GCAPTURE_MANAGED_RX.match(rp)
        if not m or (m.group("n"), m.group("v")) == (name, vtag):
            continue
        scene.render.filepath = _gcapture_managed_render_path(name, vtag)
        changed += 1
        print("[Gaussian Render Capture] render output follows %s: %s"
              % (os.path.basename(filepath), scene.render.filepath))
    return changed


_GCAPTURE_OUT_RX = re.compile(r"^//[^/]+_COLMAP/[vV]\d+/?$")


def _gcapture_render_path_is_default_or_managed(scene):
    """True, wenn der Renderpfad Blenders Standard ist oder vom Addon
    verwaltet wird -- dann darf Save Scene Version ihn setzen (v129)."""
    rp = (scene.render.filepath or "").replace("\\", "/").strip()
    return (rp in ("", "/tmp", "/tmp/", "//")
            or bool(_GCAPTURE_MANAGED_RX.match(rp)))


def _gcapture_auto_output_dir(filepath):
    """//<Name>_COLMAP/<vNNN>/ fuer die Datei (wie _exp_resolve_output_dir)."""
    if not filepath:
        return ""
    stem = os.path.splitext(os.path.basename(filepath))[0]
    name, vtag = _exp_split_name_version(stem)
    return "//%s_COLMAP/%s/" % (name or "Scene", vtag or "v001")


def _gcapture_auto_image_dir(scene):
    """Ordner des Renderpfads, wie er in der Szene steht (relativ bleibt
    relativ)."""
    rp = (scene.render.filepath or "").replace("\\", "/")
    if not rp.strip():
        return ""
    return rp if rp.endswith("/") else rp.rsplit("/", 1)[0] + "/"


def _gcapture_sync_path_fields(scene, filepath):
    """Output Folder und Image Folder zeigen die tatsaechlichen Pfade und
    folgen Version und Renderpfad, solange sie den vom Addon eingetragenen
    Wert haben (v129). Ungespeicherte Szenen bleiben leer (= automatisch)."""
    s = getattr(scene, "gcapture_settings", None)
    if s is None or not filepath:
        return
    out_auto = _gcapture_auto_output_dir(filepath)
    cur = (s.exp_output_dir or "").replace("\\", "/").strip()
    if not cur or cur == s.exp_output_dir_auto or _GCAPTURE_OUT_RX.match(cur):
        if s.exp_output_dir != out_auto:
            s.exp_output_dir = out_auto
        if s.exp_output_dir_auto != out_auto:
            s.exp_output_dir_auto = out_auto
    img_auto = _gcapture_auto_image_dir(scene)
    cur = (s.exp_image_dir or "").replace("\\", "/").strip()
    if img_auto and (not cur or cur == s.exp_image_dir_auto):
        if s.exp_image_dir != img_auto:
            s.exp_image_dir = img_auto
        if s.exp_image_dir_auto != img_auto:
            s.exp_image_dir_auto = img_auto


def _gcapture_sync_all(filepath):
    _gcapture_sync_render_version(filepath)
    for scene in bpy.data.scenes:
        _gcapture_sync_path_fields(scene, filepath)


_GCAPTURE_MSGBUS_OWNER = object()


def _gcapture_on_render_path_change(*args):
    for scene in bpy.data.scenes:
        _gcapture_sync_path_fields(scene, bpy.data.filepath)


def _gcapture_msgbus_subscribe():
    """Aenderungen am Renderpfad sofort in Image Folder uebernehmen (v129).
    Muss nach jedem Laden neu abonniert werden."""
    bpy.msgbus.clear_by_owner(_GCAPTURE_MSGBUS_OWNER)
    bpy.msgbus.subscribe_rna(key=(bpy.types.RenderSettings, "filepath"),
                             owner=_GCAPTURE_MSGBUS_OWNER, args=(),
                             notify=_gcapture_on_render_path_change)


# Uebernahme aus "Gaussian Render Scan" (bis 1.0.2, 23.09.2026): dieselben
# Einstellungen und Markierungen unter dem neuen Namen. Laeuft nach jedem
# Laden und beim Aktivieren; ohne Altdaten aendert sie nichts.
_GCAPTURE_LEGACY_KEYS = (("gscan_prepared", "gcapture_prepared"),
                         ("gscan_res_ok", "gcapture_res_ok"),
                         ("gscan_base_loc", "gcapture_base_loc"),
                         ("gscan_center", "gcapture_center"))
_GCAPTURE_LEGACY_ATTRS = ("fit", "shift", "adj", "adj_pos", "adj_tgt")
_GCAPTURE_LEGACY_NAMES = (("GScan_Group", "GCapture_Group", "objects"),
                          ("Scan_Rig", "Capture_Rig", "collections"))


def _gcapture_migrate_legacy():
    """Szenen aus Gaussian Render Scan uebernehmen. Die Einstellungen liegen
    ab Blender 5.0 in den System-Properties, davor im Dict der Szene.
    Rueckgabe: Anzahl uebernommener Eintraege."""
    n = 0
    for scene in bpy.data.scenes:
        sysp = getattr(scene, "bl_system_properties_get", None)
        g = sysp() if sysp else None
        dst = g if g is not None else scene
        for store in (g, scene):
            if store is None or "fco_settings" not in store:
                continue
            old = store["fco_settings"]
            if "gcapture_settings" not in dst and hasattr(old, "to_dict"):
                dst["gcapture_settings"] = old.to_dict()
                n += 1
            del store["fco_settings"]
    for data in (bpy.data.scenes, bpy.data.objects):
        for idb in data:
            for a, b in _GCAPTURE_LEGACY_KEYS:
                if a in idb:
                    if b not in idb:
                        idb[b] = idb[a]
                    del idb[a]
                    n += 1
    for me in bpy.data.meshes:
        for suffix in _GCAPTURE_LEGACY_ATTRS:
            attr = me.attributes.get("gscan_" + suffix)
            if attr is not None and me.attributes.get("gcapture_" + suffix) is None:
                attr.name = "gcapture_" + suffix
                n += 1
    # Objekt- und Collection-Namen nur, wenn die Datei vom alten Add-on stammt.
    if n:
        for a, b, kind in _GCAPTURE_LEGACY_NAMES:
            data = getattr(bpy.data, kind)
            idb = data.get(a)
            if idb is not None and data.get(b) is None:
                idb.name = b
                n += 1
        print("[Gaussian Render Capture] scene of Gaussian Render Scan taken "
              "over (%d entries)" % n)
    return n


def _gcapture_migrate_timer():
    try:
        _gcapture_migrate_legacy()
    except Exception as exc:
        print("[Gaussian Render Capture] migration:", exc)
    return None


@persistent
def _gcapture_on_load_post(*args):
    _GCAPTURE_PROGRESS.clear()   # ein neues File hat keinen laufenden Vorgang
    try:
        _gcapture_migrate_legacy()
    except Exception as exc:
        print("[Gaussian Render Capture] migration:", exc)
    _gcapture_sync_all(bpy.data.filepath)
    _gcapture_msgbus_subscribe()


@persistent
def _gcapture_on_save_pre(*args):
    # Blender uebergibt den Zielpfad (Save As); sonst der aktuelle.
    target = next((a for a in args if isinstance(a, str) and a), bpy.data.filepath)
    _gcapture_sync_all(target)


# Szene vorbereiten (v113): Werte aus der Vorlage des Nutzers (Setup.blend,
# 23.09.2026). Pfad relativ zur Szene -> Wert. Was eine Blender-Version nicht
# kennt, wird uebersprungen.
_GCAPTURE_PREPARE_SETTINGS = (
    ("render.engine", 'CYCLES'),
    ("cycles.samples", 1024),
    ("cycles.use_denoising", True),
    ("cycles.denoising_use_gpu", True),
    ("cycles.use_preview_denoising", True),
    ("cycles.sampling_pattern", 'TABULATED_SOBOL'),
    ("cycles.max_bounces", 32),
    ("cycles.diffuse_bounces", 32),
    ("cycles.glossy_bounces", 32),
    ("cycles.transmission_bounces", 32),
    ("cycles.volume_bounces", 32),
    ("cycles.transparent_max_bounces", 32),
    ("render.film_transparent", True),
    ("cycles.film_transparent_glass", True),
    ("view_settings.look", 'AgX - Base Contrast'),
    ("render.compositor_device", 'GPU'),
    ("eevee.fast_gi_thickness_near", 0.25),
    ("eevee.ray_tracing_options.screen_trace_thickness", 0.2),
    ("eevee.ray_tracing_options.use_backface_hit", False),
    ("eevee.ray_tracing_options.backface_radiance_scale", 0.0),
)

# EEVEE-Preset fuer Captures (v1.1.5): so ansichtsunabhaengig und so nah an
# Cycles wie moeglich -- Raytracing und Fast GI in voller Aufloesung, weiche
# Schatten mit vielen Strahlen, Overscan gegen Randeffekte, mehr Samples.
# Erste Fassung; per Messung (Cycles gegen EEVEE, Beetle) zu kalibrieren.
# Aendert sich die Tabelle, _GCAPTURE_EEVEE_PRESET erhoehen: dann wird sie
# beim naechsten EEVEE-Render erneut angewendet, sonst bleiben Aenderungen
# des Nutzers erhalten.
_GCAPTURE_EEVEE_PRESET = 1
_GCAPTURE_EEVEE_SETTINGS = (
    ("eevee.taa_render_samples", 256),
    ("eevee.use_raytracing", True),
    ("eevee.ray_tracing_method", 'SCREEN'),
    ("eevee.ray_tracing_options.resolution_scale", '1'),
    ("eevee.ray_tracing_options.screen_trace_quality", 1.0),
    ("eevee.ray_tracing_options.use_denoise", True),
    ("eevee.use_fast_gi", True),
    ("eevee.fast_gi_method", 'GLOBAL_ILLUMINATION'),
    ("eevee.fast_gi_resolution", '1'),
    ("eevee.fast_gi_ray_count", 4),
    ("eevee.fast_gi_step_count", 16),
    ("eevee.fast_gi_quality", 1.0),
    ("eevee.use_shadows", True),
    ("eevee.shadow_ray_count", 4),
    ("eevee.shadow_step_count", 16),
    ("eevee.shadow_resolution_scale", 1.0),
    ("eevee.light_threshold", 0.001),
    ("eevee.use_overscan", True),
    ("eevee.overscan_size", 10.0),
    ("render.film_transparent", True),
)


def _gcapture_apply_settings(scene, table):
    """(Pfad, Wert)-Tabelle auf die Szene anwenden; Rueckgabe: uebersprungene
    Pfade (in dieser Blender-Version unbekannt)."""
    skipped = []
    for path, value in table:
        owner_path, _, attr = path.rpartition(".")
        owner = scene
        try:
            for part in owner_path.split("."):
                owner = getattr(owner, part)
            setattr(owner, attr, value)
        except (AttributeError, TypeError, ValueError):
            skipped.append(path)
    return skipped


def _gcapture_apply_prepare(scene, dtype):
    """Cycles-Einstellungen von Prepare Scene. Blender 4.1: OpenImageDenoise
    zaehlt beim Start alle SYCL-Geraete auf und scheitert mit aktuellen
    Intel-Treibern (PI_ERROR_INVALID_VALUE) -- auch auf der CPU. Rendert
    Cycles dort mit OptiX, entrauscht dessen Denoiser (v1.1.5)."""
    skipped = _gcapture_apply_settings(scene, _GCAPTURE_PREPARE_SETTINGS)
    _gcapture_fix_denoiser(scene, dtype)
    return skipped


def _gcapture_fix_denoiser(scene, dtype):
    if bpy.app.version < (4, 2, 0) and dtype == 'OPTIX':
        try:
            scene.cycles.denoiser = 'OPTIX'
        except (AttributeError, TypeError):
            pass


def _gcapture_eevee_engine():
    """Kennung von EEVEE in dieser Blender-Version (4.2-4.x: EEVEE Next)."""
    items = {i.identifier for i in
             bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items}
    return 'BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in items else 'BLENDER_EEVEE'


# Reihenfolge der GPU-Backends: das erste mit einer GPU gewinnt.
_GCAPTURE_GPU_TYPES = ('OPTIX', 'CUDA', 'HIP', 'METAL', 'ONEAPI')

# Startobjekte der Blender-Standardszene: (Name, Typ).
_GCAPTURE_START_OBJECTS = (("Cube", 'MESH'), ("Camera", 'CAMERA'), ("Light", 'LIGHT'))


def _gcapture_setup_gpu():
    """Cycles-Einstellungen: bestes GPU-Backend waehlen, dessen GPUs an,
    CPU aus. Rueckgabe (Backend, [Geraetenamen]) oder (None, [])."""
    try:
        cp = bpy.context.preferences.addons["cycles"].preferences
    except (KeyError, AttributeError):
        return None, []
    # Kein refresh_devices(): das fragt alle Backends ab, und Blender 4.1
    # stuerzt mit aktuellen Intel-Treibern im oneAPI-Backend ab (sycl6.dll).
    # get_devices_for_type fragt nur das jeweilige Backend.
    for dtype in _GCAPTURE_GPU_TYPES:
        try:
            devs = list(cp.get_devices_for_type(dtype))
        except Exception:
            continue
        gpus = [d for d in devs if d.type != 'CPU']
        if not gpus:
            continue
        try:
            cp.compute_device_type = dtype
        except Exception:
            continue
        for d in devs:
            d.use = d.type != 'CPU'
        return dtype, [d.name for d in gpus]
    return None, []


def _gcapture_start_object_unchanged(obj):
    """Nur das unveraenderte Startobjekt: Cube mit 8 Vertices, Kamera und
    Licht an ihrer Startposition."""
    if obj.parent is not None or obj.children:
        return False
    if obj.type == 'MESH':
        return (len(obj.data.vertices) == 8 and not obj.modifiers
                and obj.matrix_world.to_translation().length < 1e-4)
    start = {'CAMERA': (7.3589, -6.9258, 4.9583), 'LIGHT': (4.0762, 1.0055, 5.9039)}
    pos = start.get(obj.type)
    return (pos is not None
            and (obj.matrix_world.to_translation() - Vector(pos)).length < 1e-3)


def _gcapture_scene_prepared(scene):
    """Prepare Scene lief in dieser Szene und sie rendert noch mit Cycles
    (v138; vorher genuegte Cycles + GPU, was viele Startdateien schon sind)."""
    return (bool(scene.get("gcapture_prepared"))
            and scene.render.engine in ('CYCLES', 'BLENDER_EEVEE',
                                        'BLENDER_EEVEE_NEXT'))


class GCAPTURE_OT_render_images(Operator):
    """Rendert alle Kameras in den Datensatz (v1.1.5): Cycles mit den
    Einstellungen von Prepare Scene, EEVEE mit dem Capture-Preset."""
    bl_idname = "gcapture.render_images"
    bl_label = "Render Images"
    bl_options = {'REGISTER'}

    engine: EnumProperty(
        name="Engine",
        items=[('CYCLES', "Cycles", "Path tracing - exact, slower"),
               ('EEVEE', "EEVEE", "Real-time engine - much faster, "
                "approximate lighting")],
        default='CYCLES')

    @classmethod
    def description(cls, context, props):
        if props.engine == 'EEVEE':
            return ("Render one image per camera into the dataset folder with "
                    "EEVEE - much faster than Cycles, lighting approximated. "
                    "The capture preset is applied once; later changes of "
                    "yours are kept")
        return ("Render one image per camera into the dataset folder with "
                "Cycles on the GPU - exact path tracing, with the settings "
                "of Prepare Scene")

    @classmethod
    def poll(cls, context):
        return (context.scene.camera is not None
                and not bpy.app.is_job_running('RENDER'))

    def _prepare(self, context):
        """Engine und Einstellungen setzen; Rueckgabe: Fehlertext oder None."""
        scene = context.scene
        s = scene.gcapture_settings
        if not bpy.data.filepath:
            return "Save the scene first (step 5, Save Version)"
        if _gcapture_wt_status_build_poses(context, s)[0] != 'DONE':
            return "Build the camera animation first (step 5)"
        if self.engine == 'EEVEE':
            scene.render.engine = _gcapture_eevee_engine()
            if scene.get("gcapture_eevee_preset") != _GCAPTURE_EEVEE_PRESET:
                skipped = _gcapture_apply_settings(scene, _GCAPTURE_EEVEE_SETTINGS)
                scene["gcapture_eevee_preset"] = _GCAPTURE_EEVEE_PRESET
                if skipped:
                    print("[Gaussian Render Capture] EEVEE preset skipped (not "
                          "in this Blender version): %s" % ", ".join(skipped))
        else:
            dtype, _ = _gcapture_setup_gpu()
            if not scene.get("gcapture_prepared"):
                _gcapture_apply_prepare(scene, dtype)
                scene["gcapture_prepared"] = True
            else:
                _gcapture_fix_denoiser(scene, dtype)
            scene.render.engine = 'CYCLES'
            scene.cycles.device = 'GPU' if dtype else 'CPU'
        _gcapture_apply_render_setup(scene, s)
        out = os.path.dirname(bpy.path.abspath(scene.render.filepath))
        if out:
            os.makedirs(out, exist_ok=True)
        return None

    def _write_script(self, context):
        """Szene speichern, daneben ein Skript fuer das Rendern ohne
        Oberflaeche schreiben (v1.1.5). Rueckgabe: Pfad des Skripts."""
        scene = context.scene
        dtype = None
        if self.engine == 'CYCLES':
            dtype, _ = _gcapture_setup_gpu()
        bpy.ops.wm.save_mainfile()
        path = _gcapture_write_render_script(
            bpy.data.filepath, self.engine, dtype, bpy.app.binary_path)
        scene.gcapture_settings.render_script = path
        return path

    def invoke(self, context, event):
        err = self._prepare(context)
        if err:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}
        if context.scene.gcapture_settings.render_headless:
            path = self._write_script(context)
            self.report({'INFO'}, "Scene saved; double-click %s to render"
                        % os.path.basename(path))
            return {'FINISHED'}
        # Blenders Renderfenster mit Fortschritt; Esc bricht ab.
        bpy.ops.render.render('INVOKE_DEFAULT', animation=True)
        return {'FINISHED'}

    def execute(self, context):
        err = self._prepare(context)
        if err:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}
        if context.scene.gcapture_settings.render_headless:
            self._write_script(context)
            return {'FINISHED'}
        bpy.ops.render.render(animation=True)
        return {'FINISHED'}


def _gcapture_write_render_script(blend, engine, dtype, blender):
    """Doppelklick-Skript neben der Szene: rendert alle Frames im Hintergrund
    mit derselben Blender-Version. Windows .cmd, macOS .command, sonst .sh."""
    folder, name = os.path.split(blend)
    stem = os.path.splitext(name)[0]
    eng = "cycles" if engine == 'CYCLES' else "eevee"
    tail = ["--", "--cycles-device", dtype] if (engine == 'CYCLES' and dtype) else []
    if sys.platform.startswith("win"):
        path = os.path.join(folder, "%s_render_%s.cmd" % (stem, eng))
        args = " ".join(tail)

        def q(s):
            return '"%s"' % s.replace("%", "%%")
        text = "\r\n".join([
            "@echo off",
            "rem Gaussian Render Capture - headless render of %s (%s)" % (name, eng),
            "rem Renders every camera into the dataset folder. Close this window to cancel.",
            'cd /d "%~dp0"',
            "%s -b %s -a %s" % (q(blender), q(name), args),
            "echo.",
            "echo Finished - press any key to close.",
            "pause > nul",
            ""])
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
    else:
        ext = "command" if sys.platform == "darwin" else "sh"
        path = os.path.join(folder, "%s_render_%s.%s" % (stem, eng, ext))
        import shlex
        text = "\n".join([
            "#!/bin/sh",
            "# Gaussian Render Capture - headless render of %s (%s)" % (name, eng),
            "# Renders every camera into the dataset folder. Ctrl+C cancels.",
            'cd "$(dirname "$0")"',
            " ".join([shlex.quote(blender), "-b", shlex.quote(name), "-a"] + tail),
            ""])
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.chmod(path, 0o755)
    return path


class GCAPTURE_OT_confirm_resolution(Operator):
    bl_idname = "gcapture.confirm_resolution"
    bl_label = "Confirm Resolution"
    bl_description = "Keep this render resolution for the rendered images"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        context.scene["gcapture_res_ok"] = True
        s = context.scene.gcapture_settings
        _gcapture_apply_render_setup(context.scene, s)
        self.report({'INFO'}, "Render resolution %d x %d" % (s.resolution,
                                                             s.resolution))
        return {'FINISHED'}


class GCAPTURE_OT_prepare_scene(Operator):
    bl_idname = "gcapture.prepare_scene"
    bl_label = "Prepare Scene"
    bl_description = ("Render with Cycles on the GPU (sets the Cycles device "
                      "in the Preferences), apply the capture render settings "
                      "and remove Blender's unchanged start cube, camera and "
                      "light")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        dtype, gpus = _gcapture_setup_gpu()
        skipped = _gcapture_apply_prepare(scene, dtype)
        scene.cycles.device = 'GPU' if dtype else 'CPU'
        scene["gcapture_prepared"] = True     # Schritt erledigt (v138)
        removed = []
        for name, otype in _GCAPTURE_START_OBJECTS:
            obj = bpy.data.objects.get(name)
            if obj is None or obj.type != otype or not _gcapture_start_object_unchanged(obj):
                continue
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if data is not None and data.users == 0:
                for coll in (bpy.data.meshes, bpy.data.cameras, bpy.data.lights):
                    if data.name in coll and coll[data.name] == data:
                        coll.remove(data)
                        break
            removed.append(name)
        msg = ("Cycles on the GPU (%s: %s)" % (dtype, ", ".join(gpus)) if dtype
               else "No GPU found - Cycles renders on the CPU")
        msg += ", capture render settings applied"
        if removed:
            msg += ", removed %s" % ", ".join(removed)
        if skipped:
            print("[Gaussian Render Capture] Prepare Scene skipped (not in this "
                  "Blender version): %s" % ", ".join(skipped))
        self.report({'INFO'} if dtype else {'WARNING'}, msg)
        return {'FINISHED'}


class GCAPTURE_OT_selection_to_collection(Operator):
    bl_idname = "gcapture.selection_to_collection"
    bl_label = "Put Selection into Collection"
    bl_description = ("Move the selected objects (with their children) into a "
                      "collection and use it as look target. An existing "
                      "collection of that name is used")
    bl_options = {'REGISTER', 'UNDO'}

    coll_name: StringProperty(name="Collection Name", default="Model")

    @classmethod
    def poll(cls, context):
        return bool(context.selected_objects)

    def invoke(self, context, event):
        obj = context.active_object or context.selected_objects[0]
        colls = [c for c in obj.users_collection
                 if c != context.scene.collection and c.library is None]
        self.coll_name = (colls[0].name if colls
                          else obj.name.split(".")[0] if obj else "Model")
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        s = context.scene.gcapture_settings
        skip = {s.sph_object, bpy.data.objects.get(s.camera_name)}

        def walk(o):
            yield o
            for c in o.children:
                yield from walk(c)
        objs = []
        for o in context.selected_objects:
            for x in walk(o):
                if x not in skip and x not in objs:
                    objs.append(x)
        if not objs:
            self.report({'WARNING'}, "Nothing to move.")
            return {'CANCELLED'}
        name = self.coll_name.strip() or "Model"
        coll = bpy.data.collections.get(name)
        existing = coll is not None and coll.library is None
        if not existing:
            coll = bpy.data.collections.new(name)
        scene_root = context.scene.collection
        if coll not in scene_root.children_recursive:
            scene_root.children.link(coll)
        for o in objs:
            for c in list(o.users_collection):
                if c != coll:
                    c.objects.unlink(o)
            if coll not in o.users_collection:
                coll.objects.link(o)
        if not any(it.coll == coll for it in s.sph_target_colls):
            s.sph_target_colls.add().coll = coll
            s.sph_coll_index = len(s.sph_target_colls) - 1
        self.report({'INFO'}, "Moved %d object(s) into %s '%s' (look target)."
                    % (len(objs), "existing" if existing else "new", coll.name))
        _gcapture_frame_collections(context, [coll])
        return {'FINISHED'}


class GCAPTURE_OT_save_version(Operator):
    bl_idname = "gcapture.save_version"
    bl_label = "Save Scene Version"
    bl_description = ("Save the scene as <Name>_v001.blend, later as a new "
                      "version or over the current one. Points the render "
                      "output into <Name>_COLMAP/<vNNN>/images")

    filepath: StringProperty(subtype='FILE_PATH', options={'SKIP_SAVE'})
    filter_glob: StringProperty(default="*.blend", options={'HIDDEN'})
    mode: EnumProperty(
        name="Save",
        items=[('NEW', "New Version", "Save as the next free version"),
               ('OVERWRITE', "Overwrite", "Save over the current version"),
               ('CUSTOM', "Custom Version", "Save with a version number of "
                "your choice")],
        default='NEW',
    )
    # Wahlknoepfe als Toggles ueber mode (v121): nur ein Toggle kann mit
    # alert rot erscheinen, ein gewaehlter Enum-Knopf bleibt blau.
    opt_overwrite: BoolProperty(
        name="Overwrite", options={'SKIP_SAVE'},
        get=lambda self: self.mode == 'OVERWRITE',
        set=lambda self, v: setattr(self, "mode", 'OVERWRITE') if v else None)
    opt_new: BoolProperty(
        name="New Version", options={'SKIP_SAVE'},
        get=lambda self: self.mode == 'NEW',
        set=lambda self, v: setattr(self, "mode", 'NEW') if v else None)
    opt_custom: BoolProperty(
        name="Custom Version", options={'SKIP_SAVE'},
        get=lambda self: self.mode == 'CUSTOM',
        set=lambda self, v: setattr(self, "mode", 'CUSTOM') if v else None)
    custom_version: IntProperty(
        name="Version", default=1, min=1, max=9999,
        description="Version number for 'Custom version'",
    )
    render_into_dataset: BoolProperty(
        name="Render into the dataset folder",
        description="Set the render output to <Name>_COLMAP/<vNNN>/images "
                    "(relative). Turn off if you render into your own "
                    "folder structure",
        default=True,
    )

    # --- Ziel berechnen ------------------------------------------------
    def _target(self):
        """(Zielpfad, Name, vtag) fuer eine bereits gespeicherte Datei."""
        cur = bpy.data.filepath
        folder = os.path.dirname(cur)
        stem = os.path.splitext(os.path.basename(cur))[0]
        name, vtag = _exp_split_name_version(stem)
        name = _gcapture_clean_name(name)
        if vtag and self.mode == 'OVERWRITE':
            return cur, name, vtag
        if self.mode == 'CUSTOM':
            width = max(3, len(vtag) - 1) if vtag else 3
            ctag = "v%0*d" % (width, self.custom_version)
            return (os.path.join(folder, "%s_%s.blend" % (name, ctag)),
                    name, ctag)
        width = max(3, len(vtag) - 1) if vtag else 3
        path = _gcapture_next_version_path(folder, name, width)
        return path, name, _exp_split_name_version(
            os.path.splitext(os.path.basename(path))[0])[1]

    def _versions(self):
        """(vtag der geladenen Datei, hoechste Nummer im Ordner, ist die
        geladene die hoechste). v120."""
        cur = bpy.data.filepath
        stem = os.path.splitext(os.path.basename(cur))[0]
        name, vtag = _exp_split_name_version(stem)
        nums = _gcapture_version_files(os.path.dirname(cur), _gcapture_clean_name(name))
        top = max(nums) if nums else 0
        latest = bool(vtag) and int(vtag[1:]) >= top
        return vtag, top, latest

    def invoke(self, context, event):
        if not bpy.data.filepath:
            s = context.scene.gcapture_settings
            first = next((it.coll.name for it in s.sph_target_colls if it.coll),
                         "Scene")
            self.filepath = _gcapture_clean_name(first) + "_v001.blend"
            self.render_into_dataset = _gcapture_render_path_is_default_or_managed(
                context.scene)
            context.window_manager.fileselect_add(self)
            return {'RUNNING_MODAL'}
        folder = os.path.dirname(bpy.data.filepath)
        stem = os.path.splitext(os.path.basename(bpy.data.filepath))[0]
        # Vorgabe: die geladene Version, wenn sie die hoechste ist; sonst die
        # naechste freie Nummer (v120). Ohne Version -> <Name>_v001.
        self.mode = 'OVERWRITE' if self._versions()[2] else 'NEW'
        # Eigener Renderpfad bleibt, sofern nicht ausdruecklich gewuenscht (v129).
        self.render_into_dataset = _gcapture_render_path_is_default_or_managed(
            context.scene)
        nums = _gcapture_version_files(folder, _gcapture_clean_name(
            _exp_split_name_version(stem)[0]))
        self.custom_version = (max(nums) + 1) if nums else 1
        return context.window_manager.invoke_props_dialog(self, width=380)

    def draw(self, context):
        layout = self.layout
        if not bpy.data.filepath:
            layout.prop(self, "render_into_dataset")
            return
        cur = os.path.basename(bpy.data.filepath)
        _, vtag = _exp_split_name_version(os.path.splitext(cur)[0])
        layout.label(text="Save the scene as a version:", icon='FILE_BLEND')
        saved_mode = self.mode
        self.mode = 'NEW'
        new_name = os.path.basename(self._target()[0])
        self.mode = saved_mode
        _, top, latest = self._versions()
        if vtag and not latest:
            layout.label(text="Loaded %s - the folder goes up to v%0*d."
                         % (vtag, max(3, len(vtag) - 1), top), icon='INFO')
        col = layout.column(align=True)
        if latest:
            orow = col.row(align=True)
            orow.alert = self.mode == 'OVERWRITE'     # Ueberschreiben: rot (v121)
            orow.prop(self, "opt_overwrite", text="Overwrite: %s" % cur, toggle=True)
        col.prop(self, "opt_new", text="New version: %s" % new_name, toggle=True)
        crow = col.row(align=True)
        crow.alert = (self.mode == 'CUSTOM'
                      and os.path.isfile(self._target()[0]))
        crow.prop(self, "opt_custom", text="Custom version:", toggle=True)
        sub = crow.row(align=True)
        sub.enabled = self.mode == 'CUSTOM'
        sub.prop(self, "custom_version", text="")
        if self.mode == 'CUSTOM':
            col.label(text="-> %s" % os.path.basename(self._target()[0]))

        # Warnungen: Ziel existiert bereits / Renderbilder passen nicht mehr.
        if self.mode in {'OVERWRITE', 'CUSTOM'}:
            path, name, ttag = self._target()
            if self.mode == 'CUSTOM' and os.path.isfile(path):
                row = layout.row()
                row.alert = True
                row.label(text="%s exists - it will be overwritten" % ttag,
                          icon='ERROR')
            images = os.path.join(os.path.dirname(path),
                                  "%s_COLMAP" % name, ttag, "images")
            if os.path.isdir(images) and any(os.scandir(images)):
                row = layout.row()
                row.alert = True
                row.label(text="Rendered images of %s will no longer "
                          "match the new cameras" % ttag, icon='ERROR')
        layout.prop(self, "render_into_dataset")
        if not self.render_into_dataset:
            layout.label(text="Render output stays: %s"
                         % context.scene.render.filepath, icon='OUTPUT')

    def execute(self, context):
        scene = context.scene
        if bpy.data.filepath:
            if self.mode == 'OVERWRITE' and not self._versions()[2]:
                self.mode = 'NEW'      # aeltere Version nie ueberschreiben
            path, name, vtag = self._target()
        else:
            if not self.filepath:
                self.report({'ERROR'}, "No file name given.")
                return {'CANCELLED'}
            folder = os.path.dirname(bpy.path.abspath(self.filepath))
            name = _gcapture_clean_name(self.filepath)
            # Eingetippte Version (z. B. Porsche_v007) uebernehmen, wenn die
            # Datei noch nicht existiert -- sonst naechste freie Version.
            typed = _exp_split_name_version(os.path.splitext(
                os.path.basename(self.filepath))[0])[1]
            path = None
            if typed:
                cand = os.path.join(folder, "%s_%s.blend" % (name, typed.lower()))
                if not os.path.isfile(cand):
                    path = cand
                else:
                    self.report({'WARNING'}, "%s exists - saved as the next "
                                "free version instead" % os.path.basename(cand))
            if path is None:
                width = max(3, len(typed) - 1) if typed else 3
                path = _gcapture_next_version_path(folder, name, width)
            vtag = _exp_split_name_version(
                os.path.splitext(os.path.basename(path))[0])[1]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.render_into_dataset:
            scene.render.filepath = _gcapture_managed_render_path(name, vtag)
            scene.gcapture_settings.exp_image_dir = ""
        # relative_remap=False: der neue "//"-Renderpfad gilt schon relativ
        # zum Zielordner und darf nicht umgerechnet werden. Neue Versionen
        # liegen im selben Ordner, eine ungespeicherte Szene hat noch keine
        # relativen Pfade -- es geht also nichts verloren.
        bpy.ops.wm.save_as_mainfile(filepath=path, relative_remap=False)
        msg = "Saved %s" % os.path.basename(path)
        if self.render_into_dataset:
            msg += " - renders go to %s_COLMAP/%s/images" % (name, vtag)
        else:
            rp = scene.render.filepath.replace("\\", "/")
            if (rp.startswith("//%s_COLMAP/" % name)
                    and "/%s/" % vtag not in rp):
                self.report({'WARNING'}, msg + " - render output still points "
                            "to another version's folder: %s" % rp)
                return {'FINISHED'}
        self.report({'INFO'}, msg)
        return {'FINISHED'}


# ----------------------------------------------------------------------
# Inhalte der Abschnitte: von Unterpanels UND Guide-Karte genutzt (v89)
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
    # Warnung: non-uniform scale + normal mode passen nicht zusammen.
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
        # Waehrend Live Camera Lock aktiv: Build gesperrt (erst Finish).
        brow.enabled = not s.live_lock
        _gcapture_action(brow, "gcapture.build_animation",
                    _gcapture_wt_status_build_poses(context, s)[0] != 'DONE',
                    'CON_CAMERASOLVER')
    layout = _gcapture_lock(outer)
    if not s.live_lock and _sph_has_adjustments(s.sph_object):
        wrow = layout.row()
        wrow.alert = True
        wrow.label(text="Adjusted views - Build discards them", icon='ERROR')
    # Wirkt beim naechsten Build -> gehoert hierher, nicht zu Schritt 5 (v126).
    layout.prop(s, "save_after_build")
    if s.live_lock:
        layout.label(text="Finish Live Camera Adjust before rebuilding",
                     icon='INFO')

    # Optional: Live Camera Lock (Feinjustierung). Einschalten startet
    # sofort das Skalieren der Sphere (v95).
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
        frozen = _GCAPTURE_LOCK_FROZEN_IDX
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
    # Nur die Aufloesung; die immer noetigen Schalter stehen unter
    # Advanced > Render Setup (v129).
    rrow = layout.row(align=True)
    rrow.enabled = s.set_resolution
    # Zu Beginn einmal blau: bewusst bestaetigen oder aendern (v140).
    if s.set_resolution and not context.scene.get("gcapture_res_ok"):
        split = rrow.split(factor=0.8, align=True)
        split.prop(s, "resolution")
        split.operator("gcapture.confirm_resolution", text="OK", depress=True)
    else:
        rrow.prop(s, "resolution")

    # Szene speichern + Renderziel (v90).
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

    # Rendern direkt aus dem Add-on (v1.1.5). Blau: der offene Schritt, und
    # zwar die Engine, auf der die Szene gerade steht.
    pending = _gcapture_wt_status_render(context, s)[0] != 'DONE'
    eevee = context.scene.render.engine != 'CYCLES'
    rcol = layout.column(align=True)
    rcol.enabled = bool(bpy.data.filepath)
    row = rcol.row(align=True)
    row.scale_y = 1.4
    # Beschriftung sagt, was der Klick tut: rendern oder Skript schreiben.
    verb = "Script" if s.render_headless else "Render"
    op = row.operator("gcapture.render_images", text="%s Cycles" % verb,
                      icon='CONSOLE' if s.render_headless else 'SHADING_RENDERED',
                      depress=pending and not eevee)
    op.engine = 'CYCLES'
    op = row.operator("gcapture.render_images", text="%s EEVEE" % verb,
                      icon='CONSOLE' if s.render_headless else 'SHADING_SOLID',
                      depress=pending and eevee)
    op.engine = 'EEVEE'
    rcol.prop(s, "render_headless")
    if s.render_headless and s.render_script:
        rcol.label(text="Double-click: %s" % os.path.basename(s.render_script),
                   icon='CONSOLE')
    if not s.wt_active:
        layout.label(text="Render farm: render the saved scene there.",
                     icon='INFO')


def _gcapture_section(layout, title, icon):
    """Kasten mit Ueberschrift, gliedert lange Schritte (v1.1.5)."""
    box = layout.box()
    box.label(text=title, icon=icon)
    return box.column()


def _gcapture_draw_export(layout, context):
    s = context.scene.gcapture_settings
    outer = layout
    layout = _gcapture_lock(outer)

    # --- Datensatz: wohin, und woher die Bilder kommen -------------------
    ds = _gcapture_section(layout, "Dataset", 'FILE_FOLDER')
    ecol = ds.column(align=True)
    # Bezeichnung ueber dem Feld: in der schmalen Seitenleiste wurde sie
    # sonst abgeschnitten, und der Pfad bekommt die volle Breite (v129).
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
    # Rendert die Szene schon in den Datensatz, sind Copy/Move ueberfluessig.
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
            # LichtFeld/COLMAP lesen images/ neben sparse/ -- ohne Bilder kein
            # Training. Mit eigenem Bildordner ist das der haeufigste Fehler.
            wrow = ds.row()
            wrow.alert = True       # immer rot: ohne Bilder kein Training (v125)
            wrow.label(text="Images are not in the dataset - choose Copy or Move",
                       icon='ERROR')

    # --- Startpunkte: Quelle, Menge, Filter, Wiederverwendung --------------
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
    # Vertex-Zahlen nur einmal je Neuzeichnen ermitteln (v110).
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

    # --- Massstab und Achsen ---------------------------------------------
    sc = _gcapture_section(layout, "Scale & Axes", 'ORIENTATION_GLOBAL')
    scol = sc.column(align=True)
    scol.prop(s, "exp_auto_scale")
    sub = scol.row(align=True)
    sub.enabled = s.exp_auto_scale
    sub.prop(s, "exp_target_radius")
    sc.prop(s, "exp_zup_to_yup")

    # Warnung, wenn im Zielordner schon ein Export liegt -- der Export
    # ueberschreibt sparse/0 (und images/) ohne Rueckfrage (v84).
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
            # Umbrechen statt abschneiden (v1.1.5).
            for line in _gcapture_textwrap.wrap(s.exp_last_detail, 42):
                rcol.label(text=line)


# ----------------------------------------------------------------------
# Guide: fuehrt Einsteiger Schritt fuer Schritt durch den Ablauf (v89)
# ----------------------------------------------------------------------
# Blender erlaubt Addons nicht, Panels auf-/zuzuklappen. Darum blendet der
# Guide die Unterpanels aus (poll) und zeigt im Hauptpanel eine Karte mit
# Erklaerung, den Bedienelementen des Schritts (dieselben _gcapture_draw_*-
# Funktionen wie die Unterpanels) und einer Statusanzeige.
import textwrap as _gcapture_textwrap


def _gcapture_wt_status_camera(context, s):
    if _gcapture_scene_prepared(context.scene):
        return 'DONE', "Scene prepared (Cycles on GPU)"
    return 'TODO', "Scene not prepared yet"


def _gcapture_draw_intro(layout, context):
    row = layout.row()
    row.scale_y = 1.2
    _gcapture_action(row, "gcapture.prepare_scene",
                not _gcapture_scene_prepared(context.scene), 'SCENE_DATA')
    layout.separator(factor=0.5)
    _gcapture_draw_camera(layout, context)


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


def _gcapture_wt_status_render(context, s):
    scene = context.scene
    expected = scene.frame_end - scene.frame_start + 1
    folder = _exp_resolve_image_dir(s)
    _, _, frames = _exp_detect_frames(folder)
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


_GCAPTURE_WT_STEPS = [
    dict(title="1. Scene & Camera", badge='gcapture_cam', icon='CAMERA_DATA',
         draw=_gcapture_draw_intro, status=_gcapture_wt_status_camera,
         goal="This guide turns your model into a Gaussian Splatting "
              "dataset: cameras on a sphere around the model, one rendered "
              "image per camera, then a COLMAP export for Postshot or "
              "LichtFeld Studio.",
         steps=["Press Prepare Scene: Cycles renders on your GPU with the "
                "capture render settings; Blender's start cube, camera and "
                "light are removed.",
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
                "Press Create / Update Camera Sphere."],
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
                "Press Render Cycles (exact, slower) or Render EEVEE (much "
                "faster, lighting approximated) - rendering starts at once. "
                "Or render the saved scene on a render farm."],
         check="the status line shows all images found.",
         note=["A change of the resolution applies at once."]),
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
         note=["Copy/Move only appear if you render into your own folder "
               "structure."]),
]

_GCAPTURE_WT_STATUS_ICON = {'DONE': 'CHECKMARK', 'TODO': 'ERROR',
                       'MISSING': 'ERROR',
                       'OPTIONAL': 'RADIOBUT_OFF', 'INFO': 'INFO'}


def _gcapture_wrap(layout, context, text, indent_px=0):
    """Blender bricht Labels nicht um: Text nach Panelbreite umbrechen.
    indent_px: Breite, die links schon belegt ist (Nummer der Anleitung)."""
    width = context.region.width if context.region else 300
    try:
        scale = context.preferences.system.ui_scale
    except Exception:
        scale = 1.0
    scale = scale if scale and scale > 0.1 else 1.0   # Hintergrund: 0
    chars = max(20, int((width - 40 - indent_px * scale) / (6.2 * scale)))
    col = layout.column(align=True)
    col.scale_y = 0.85
    for line in _gcapture_textwrap.wrap(text, chars):
        col.label(text=line)


def _gcapture_wt_draw_text(layout, context, step):
    """Guide-Text (v112): Zweck, nummerierte Schritte mit haengendem
    Einzug, Check und Hinweis."""
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
    # Check und Hinweise als Aufzaehlung (v131).
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


# Fortschritt langer Ablaeufe (Build, Export) im Panel an der Stelle des
# Knopfs statt in der Kopfzeile des Viewports (v1.1.4). Nur zur Laufzeit.
_GCAPTURE_PROGRESS = {}   # op_id -> (Anteil 0..1, Text, Zeitpunkt)
# Aeltere Eintraege gelten als verwaist (Absturz ohne Aufraeumen) und
# sperren das Panel nicht mehr.
_GCAPTURE_PROGRESS_STALE = 60.0


def _gcapture_redraw_viewports():
    try:
        for win in bpy.context.window_manager.windows:
            for area in win.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
    except Exception:
        pass


def _gcapture_progress_set(op_id, factor, text):
    _GCAPTURE_PROGRESS[op_id] = (max(0.0, min(1.0, float(factor))), text,
                                 time.time())
    _gcapture_redraw_viewports()


def _gcapture_progress_clear(op_id):
    if _GCAPTURE_PROGRESS.pop(op_id, None) is not None:
        _gcapture_redraw_viewports()


def _gcapture_busy():
    """True, solange Build oder Export laeuft (v1.1.4)."""
    now = time.time()
    return any(now - p[2] < _GCAPTURE_PROGRESS_STALE
               for p in _GCAPTURE_PROGRESS.values())


def _gcapture_lock(layout):
    """Spalte, die waehrend eines Laufs gesperrt ist; der Fortschrittsbalken
    selbst steht ausserhalb und bleibt lesbar."""
    col = layout.column()
    col.enabled = not _gcapture_busy()
    return col


def _gcapture_progress_draw(layout, op_id):
    """Zeichnet den Fortschritt von op_id, falls er laeuft -> True."""
    p = _GCAPTURE_PROGRESS.get(op_id)
    if p is None or time.time() - p[2] >= _GCAPTURE_PROGRESS_STALE:
        return False
    col = layout.box().column()
    if hasattr(col, "progress"):
        col.progress(factor=p[0], type='BAR', text=p[1])
    else:
        col.label(text="%d %%  %s" % (int(p[0] * 100), p[1]))
    col.label(text="Esc: cancel", icon='CANCEL')
    return True


def _gcapture_action(layout, op_id, pending, icon, text=None):
    """Knopf eines Pflichtschritts: blau (gedrueckt), solange der Schritt
    offen ist, danach neutral (v137)."""
    kw = {} if text is None else {"text": text}
    return layout.operator(op_id, icon=icon, depress=bool(pending), **kw)


def _gcapture_badge_label(layout, badge, fallback_icon):
    pv = _gcapture_previews.get(badge) if (badge and _gcapture_previews) else None
    if pv is not None:
        layout.label(text="", icon_value=pv.icon_id)
    elif fallback_icon:
        layout.label(text="", icon=fallback_icon)


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
    if step['draw'] in (_gcapture_draw_build, _gcapture_draw_export):
        step['draw'](card.column(), context)   # sperren selbst, Balken frei
    else:
        step['draw'](_gcapture_lock(card), context)
    card.separator()

    try:
        state, msg = step['status'](context, s)
    except Exception as exc:
        state, msg = 'INFO', "Status unavailable (%s)" % exc
    srow = card.row()
    srow.alert = state in {'TODO', 'MISSING'}   # Offenes rot (v124/v125)
    srow.label(text=msg, icon=_GCAPTURE_WT_STATUS_ICON.get(state, 'INFO'))

    nav = card.row(align=True)
    nav.scale_y = 1.3
    nav.enabled = not _gcapture_busy()
    back = nav.row(align=True)
    back.enabled = idx > 0
    op = back.operator("gcapture.walkthrough_nav", text="Back", icon='TRIA_LEFT')
    op.delta = -1
    nav.operator("gcapture.walkthrough_exit", text="Exit", icon='X')
    # Voraussetzungen erfuellt -> Next blau (v139).
    ready = state in {'DONE', 'OPTIONAL'}
    if idx < n - 1:
        op = nav.operator("gcapture.walkthrough_nav", text="Next", icon='TRIA_RIGHT',
                          depress=ready)
        op.delta = 1
    else:
        nav.operator("gcapture.walkthrough_exit", text="Finish", icon='CHECKMARK',
                     depress=ready)


class GCAPTURE_OT_walkthrough_start(Operator):
    bl_idname = "gcapture.walkthrough_start"
    bl_label = "Start Guide"
    bl_description = ("Step-by-step guide through the whole workflow, with an "
                      "explanation for every step. Can be restarted any time")

    def execute(self, context):
        s = context.scene.gcapture_settings
        s.wt_step = 0
        s.wt_active = True
        return {'FINISHED'}


class GCAPTURE_OT_walkthrough_nav(Operator):
    bl_idname = "gcapture.walkthrough_nav"
    bl_label = "Guide Step"
    bl_description = "Go to the previous or next step of the guide"

    delta: IntProperty(default=1)

    def execute(self, context):
        s = context.scene.gcapture_settings
        s.wt_step = max(0, min(s.wt_step + self.delta, len(_GCAPTURE_WT_STEPS) - 1))
        _gcapture_wt_show_props_tab(context, _GCAPTURE_WT_STEPS[s.wt_step].get('props_tab'))
        return {'FINISHED'}


def _gcapture_wt_show_props_tab(context, tab):
    """Schaltet die Properties-Editoren des Fensters auf den Reiter des
    Guide-Schritts (v124: Schritt 5 -> Output, dort steht der Ausgabepfad)."""
    if not tab or context.window is None:
        return
    for area in context.window.screen.areas:
        if area.type == 'PROPERTIES':
            try:
                area.spaces.active.context = tab
            except (TypeError, AttributeError):
                pass


class GCAPTURE_OT_walkthrough_exit(Operator):
    bl_idname = "gcapture.walkthrough_exit"
    bl_label = "Exit Guide"
    bl_description = "Close the guide and show the normal sections again"

    def execute(self, context):
        context.scene.gcapture_settings.wt_active = False
        return {'FINISHED'}


class GCAPTURE_PT_panel(Panel):
    """Hauptpanel. Die Abschnitte sind Unterpanels (v85) -- einklappbar,
    mit Phasenfarbe im Kopf. bl_idname bleibt unveraendert."""
    bl_label = "Gaussian Render Capture"
    bl_idname = "GCAPTURE_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Gaussian Render Capture"

    def draw(self, context):
        layout = self.layout
        s = context.scene.gcapture_settings
        if s.wt_active:
            _gcapture_draw_walkthrough(layout, context)
            return
        layout = _gcapture_lock(layout)
        row = layout.row()
        row.scale_y = 1.4
        row.operator("gcapture.walkthrough_start", icon='HELP')
        _gcapture_action(layout, "gcapture.prepare_scene",
                    not _gcapture_scene_prepared(context.scene), 'SCENE_DATA')
        col = layout.column(align=True)
        col.scale_y = 0.9
        col.label(text="First (in Blender): import your model", icon='INFO')
        col.label(text="and put it into its own Collection.")


class _GCAPTURE_SubPanel:
    """Gemeinsame Basis der Unterpanels: Kopf mit farbigem Phasen-Icon
    und Schritt-Symbol."""
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Gaussian Render Capture"
    bl_parent_id = "GCAPTURE_PT_panel"
    # Initial zugeklappt: der Nutzer klickt sich durch die Schritte (v89).
    bl_options = {'DEFAULT_CLOSED'}
    gcapture_color = None     # Rueckfall, falls die Plaketten fehlen
    gcapture_badge = None     # Name in _GCAPTURE_BADGES (farbige Nummern-Plakette)
    gcapture_icon = None

    @classmethod
    def poll(cls, context):
        # Waehrend der Guide laeuft, zeigt das Hauptpanel nur die Guide-Karte.
        return not context.scene.gcapture_settings.wt_active

    def draw_header(self, context):
        row = self.layout.row(align=True)
        badge = None
        if self.gcapture_badge and _gcapture_previews is not None:
            badge = _gcapture_previews.get(self.gcapture_badge)
        if badge is not None:
            row.label(text="", icon_value=badge.icon_id)
        elif self.gcapture_color:
            row.label(text="", icon=self.gcapture_color)
        if self.gcapture_icon:
            row.label(text="", icon=self.gcapture_icon)


class GCAPTURE_PT_camera(_GCAPTURE_SubPanel, Panel):
    bl_idname = "GCAPTURE_PT_camera"
    gcapture_badge = 'gcapture_cam'
    gcapture_icon = 'CAMERA_DATA'
    bl_label = "1. Scene & Camera"
    bl_order = 0
    gcapture_color = _GCAPTURE_COLOR_PREP

    def draw(self, context):
        _gcapture_draw_camera(_gcapture_lock(self.layout), context)


class GCAPTURE_PT_target(_GCAPTURE_SubPanel, Panel):
    bl_idname = "GCAPTURE_PT_target"
    gcapture_badge = 'gcapture_1'
    gcapture_icon = 'HIDE_OFF'
    bl_label = "2. Look Target (Collection)"
    bl_order = 1
    gcapture_color = _GCAPTURE_COLOR_PREP

    def draw(self, context):
        _gcapture_draw_target(_gcapture_lock(self.layout), context)


class GCAPTURE_PT_group(_GCAPTURE_SubPanel, Panel):
    bl_idname = "GCAPTURE_PT_group"
    gcapture_badge = 'gcapture_2'
    gcapture_icon = 'EMPTY_AXIS'
    bl_label = "3. Group & Align to Ground"
    bl_order = 2
    gcapture_color = _GCAPTURE_COLOR_PREP

    def draw(self, context):
        _gcapture_draw_group(_gcapture_lock(self.layout), context)


class GCAPTURE_PT_sphere(_GCAPTURE_SubPanel, Panel):
    bl_idname = "GCAPTURE_PT_sphere"
    gcapture_badge = 'gcapture_3'
    gcapture_icon = 'MESH_ICOSPHERE'
    bl_label = "4. Camera Sphere"
    bl_order = 3
    gcapture_color = _GCAPTURE_COLOR_CAMERAS

    def draw(self, context):
        _gcapture_draw_sphere(_gcapture_lock(self.layout), context)


class GCAPTURE_PT_build(_GCAPTURE_SubPanel, Panel):
    bl_idname = "GCAPTURE_PT_build"
    gcapture_badge = 'gcapture_4'
    gcapture_icon = 'KEYFRAME_HLT'
    bl_label = "5. Build Camera Animation"
    bl_order = 4
    gcapture_color = _GCAPTURE_COLOR_CAMERAS

    def draw(self, context):
        _gcapture_draw_build(self.layout, context)


class GCAPTURE_PT_render(_GCAPTURE_SubPanel, Panel):
    bl_idname = "GCAPTURE_PT_render"
    gcapture_badge = 'gcapture_5'
    gcapture_icon = 'RENDER_ANIMATION'
    bl_label = "6. Render Settings & Output"
    bl_order = 5
    gcapture_color = _GCAPTURE_COLOR_OUTPUT

    def draw(self, context):
        _gcapture_draw_render(_gcapture_lock(self.layout), context)


class GCAPTURE_PT_export(_GCAPTURE_SubPanel, Panel):
    bl_idname = "GCAPTURE_PT_export"
    gcapture_badge = 'gcapture_6'
    gcapture_icon = 'EXPORT'
    bl_label = "7. COLMAP Export (Postshot / LichtFeld)"
    bl_order = 6
    gcapture_color = _GCAPTURE_COLOR_COLMAP

    def draw(self, context):
        _gcapture_draw_export(self.layout, context)


class GCAPTURE_PT_advanced(_GCAPTURE_SubPanel, Panel):
    """Selten gebrauchte Optionen + Maintainer/Lizenz (zugeklappt)."""
    bl_idname = "GCAPTURE_PT_advanced"
    gcapture_icon = 'PREFERENCES'
    bl_label = "Advanced"
    bl_order = 7
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = _gcapture_lock(self.layout)
        s = context.scene.gcapture_settings
        # Eigene Kamera-Leitgitter statt/zusaetzlich zur Sphere (v93).
        glbox = layout.box()
        glbox.label(text="Custom Camera Guides", icon='MESH_ICOSPHERE')
        glbox.prop(s, "use_guide_list")
        if s.use_guide_list:
            row = glbox.row()
            row.template_list("GCAPTURE_UL_guides", "gcapture_guides", s, "guides",
                              s, "guide_index", rows=3)
            bcol = row.column(align=True)
            bcol.operator("gcapture.guide_add", text="", icon='ADD')
            bcol.operator("gcapture.guide_remove", text="", icon='REMOVE')
            bcol.operator("gcapture.guide_clear", text="", icon='TRASH')
            total_objs = sum(len(it.objects) for it in s.guides)
            glbox.label(text="%d entr%s, %d object(s)"
                        % (len(s.guides),
                           "y" if len(s.guides) == 1 else "ies",
                           total_objs))
        # Interior-Test nur mit eigenen Leitgittern sinnvoll.
        if _gcapture_has_custom_guides(s):
            glbox.prop(s, "skip_interior")

        # Render Setup (v129, vorher Schritt 5): wirken sofort und bei
        # jedem Build; fuer ein Capture immer an.
        rbox = layout.box()
        rbox.label(text="Render Setup (keep on for a capture)", icon='SCENE')
        rcol = rbox.column(align=True)
        rcol.prop(s, "constant_interp")
        rcol.prop(s, "set_active_camera")
        rcol.prop(s, "set_frame_range")
        rcol.prop(s, "set_resolution")

        # --- Maintainer & Lizenz (GPL-Attribution, Pflicht) ---
        layout.separator()
        foot = layout.column(align=True)
        foot.scale_y = 0.8
        foot.label(text="Prof. Michael Klein", icon='INFO')
        foot.label(text="Digital Film Design – Animation/VFX")
        foot.label(text="Mediadesign University of Applied Sciences")
        for url in ("www.mediadesign.de", "www.virtualrepublic.org",
                    "www.renderbricks.com",
                    "www.linkedin.com/in/virtualrepublic/"):
            lrow = foot.row()
            lrow.alignment = 'LEFT'
            op = lrow.operator("wm.url_open", text=url, icon='URL',
                               emboss=False)
            op.url = "https://" + url
        foot.label(text="Vibe-coded with Anthropic Claude")
        foot.label(text="(Claude Code) - not a programmer,")
        foot.label(text="a CGI artist")
        foot.label(text="GPL-3.0-or-later")
        foot.label(text="Started from Gauss Cannon by Arash")
        foot.label(text="Keshmirian (Warpgate Labs); its")
        foot.label(text="ray-casting helpers remain")


# ----------------------------------------------------------------------
# COLMAP-Export (integriert aus export_colmap_for_postshot_v02)
# ----------------------------------------------------------------------
# Achsenkorrektur Blender-Kamera -> COLMAP-Kamera (180 Grad um Kamera-X).
_BLENDER_TO_COLMAP = Matrix((
    (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, -1.0),
))
# Welt-Drehung Z-up -> Y-up (+90 Grad um Welt-X).
# (x, y, z) -> (x, -z, y): Blender-Z (oben) wird zu -Y. Postshot/COLMAP
# zaehlt Y nach unten, daher steht das Modell damit aufrecht. Die
# Variante -90 (x, z, -y) stellt die Achse zwar auch auf, dreht das
# Modell aber um 180 Grad (Kopfstand).
_WORLD_Z_UP_TO_Y_UP = Matrix((
    (1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 1.0, 0.0),
))
_EXP_KNOWN_EXTS = ("tif", "tiff", "png", "jpg", "jpeg", "exr", "tga", "bmp")


# ----------------------------------------------------------------------
# Ray-Casting-Helfer (Interior-Kamera-Erkennung)
# Abgeleitet aus "Gauss Cannon" von Arash Keshmirian (Warpgate Labs),
# GPL-3.0-or-later.
# Quelle: https://github.com/warpgatelabs/gauss-cannon
# An unsere Datenstrukturen angepasst; siehe Lizenzsektion im Header.
# ----------------------------------------------------------------------
def _rc_is_camera_inside_mesh(context, camera_pos, guide_objects):
    """True, wenn camera_pos innerhalb einer sichtbaren Nicht-Leitgitter-
    Mesh liegt. Odd-even-Regel: ungerade Anzahl Schnittpunkte entlang
    eines Strahls -> innen."""
    for obj in context.view_layer.objects:
        if (obj.type != 'MESH' or not obj.visible_get()
                or obj.hide_render or obj in guide_objects):
            continue
        mat_inv = obj.matrix_world.inverted_safe()
        pos_local = mat_inv @ camera_pos
        ray_dir = Vector((1.0, 0.0, 0.0))
        count = 0
        cur = pos_local.copy()
        # Begrenze die Iterationen, um Endlosschleifen sicher zu vermeiden.
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
    """Liste von Welt-Raum-BVHTrees aller sichtbaren Nicht-Leitgitter-
    Meshes (depsgraph-evaluiert, Modifier beruecksichtigt)."""
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
    """Welt-Raum-BVH des Volumens zwischen Kamera und Near-Plane.
    PERSP -> 5-Vertex-Pyramide, ORTHO -> 8-Vertex-Box."""
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
    """True, wenn sichtbare Geometrie in das Near-Clip-Volumen der Kamera
    ragt (zwei Tests: naechster Punkt im Near-Radius + BVH-Overlap)."""
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


def _exp_world_transform(s):
    return _WORLD_Z_UP_TO_Y_UP if s.exp_zup_to_yup else Matrix.Identity(3)


def _exp_write_scale_info(out_dir, s, scale, mean_r, n_poses, n_points):
    """Schreibt <vNNN>_gcapture_export.json NEBEN den Datensatzordner (v126,
    v145: nicht hinein -- Postshot liest jede .json im Datensatz als
    NeRF-Kameradatei und bricht ab). Massstab und
    Achsenumrechnung des Exports. Datensatz = scale * W @ Blender-Welt."""
    import json
    import datetime
    W = Matrix(_exp_world_transform(s)).to_4x4()
    fwd = Matrix.Diagonal((scale, scale, scale, 1.0)) @ W
    inv = fwd.inverted()
    info = {
        "addon": "Gaussian Render Capture %s" % ".".join(
            str(v) for v in bl_info["version"]),
        "blender": bpy.app.version_string,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "blend_file": os.path.basename(bpy.data.filepath) or None,
        "auto_scale": bool(s.exp_auto_scale),
        "target_radius": float(s.exp_target_radius) if s.exp_auto_scale else None,
        "mean_camera_radius_blender": mean_r,
        "scale": scale,
        "z_up_to_y_up": bool(s.exp_zup_to_yup),
        "dataset_from_blender": [list(r) for r in fwd],
        "blender_from_dataset": [list(r) for r in inv],
        "poses": n_poses,
        "points": n_points,
        "note": ("Dataset coordinates = scale * axis conversion @ Blender world "
                 "(about the world origin). To put a trained splat back onto the "
                 "model in Blender, apply blender_from_dataset - unless the "
                 "importer already converts Y-up to Z-up, then scale by 1/scale "
                 "only."),
    }
    ds = os.path.normpath(out_dir)
    old = os.path.join(ds, "gscan_export.json")        # Ablage bis v144
    if os.path.isfile(old):
        os.remove(old)
    # Gaussian Render Scan (bis 1.0.2) schrieb <vNNN>_gscan_export.json.
    legacy = os.path.join(os.path.dirname(ds),
                          os.path.basename(ds) + "_gscan_export.json")
    if os.path.isfile(legacy):
        os.remove(legacy)
    target = os.path.join(os.path.dirname(ds),
                          os.path.basename(ds) + "_gcapture_export.json")
    with open(target, "w",
              encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def _exp_resolve_image_dir(s):
    if s.exp_image_dir:
        return bpy.path.abspath(s.exp_image_dir)
    raw = bpy.context.scene.render.filepath
    path = bpy.path.abspath(raw)
    if not os.path.isdir(path):
        path = os.path.dirname(path)
    return path


def _exp_detect_frames(image_dir):
    if not image_dir or not os.path.isdir(image_dir):
        return None, None, []
    candidates = []
    for fn in os.listdir(image_dir):
        stem, dot, ext = fn.rpartition(".")
        if dot and ext.lower() in _EXP_KNOWN_EXTS:
            candidates.append((stem, ext.lower()))
    if not candidates:
        return None, None, []
    ext_counts = {}
    for _, e in candidates:
        ext_counts[e] = ext_counts.get(e, 0) + 1
    chosen_ext = max(ext_counts, key=ext_counts.get)
    stems = [st for st, e in candidates if e == chosen_ext]
    rx = re.compile(r"^(.*?)(\d+)$")
    parsed = [(m.group(1), m.group(2)) for m in (rx.match(st) for st in stems) if m]
    if not parsed:
        return None, None, []
    prefix_counts, width_counts = {}, {}
    for p, d in parsed:
        prefix_counts[p] = prefix_counts.get(p, 0) + 1
        width_counts[len(d)] = width_counts.get(len(d), 0) + 1
    chosen_prefix = max(prefix_counts, key=prefix_counts.get)
    chosen_width = max(width_counts, key=width_counts.get)
    frames = sorted(int(d) for p, d in parsed
                    if p == chosen_prefix and len(d) == chosen_width)
    pattern = "%s{n:0%dd}" % (chosen_prefix, chosen_width)
    return pattern, chosen_ext, frames


def _exp_image_size(image_path):
    """Liest die tatsaechliche Pixelgroesse einer Bilddatei ueber Blenders
    Bild-API (ohne Zusatzbibliothek). Liefert (w, h) oder None."""
    if not image_path or not os.path.isfile(image_path):
        return None
    img = None
    try:
        img = bpy.data.images.load(image_path, check_existing=False)
        w, h = int(img.size[0]), int(img.size[1])
        if w > 0 and h > 0:
            return (w, h)
        return None
    except Exception:
        return None
    finally:
        # Geladenes Bild wieder entfernen, damit die .blend nicht zumuellt.
        if img is not None:
            try:
                bpy.data.images.remove(img)
            except Exception:
                pass


def _exp_intrinsics(cam_data, scene, real_size=None):
    """fx, fy, cx, cy, w, h in Pixeln (PINHOLE).
    Wenn real_size (w, h) gegeben ist (echte Bildgroesse aus der Datei),
    werden Aufloesung und Principal Point daraus abgeleitet -- damit
    cameras.txt IMMER zu den real vorhandenen Bildern passt, unabhaengig
    von den Render-Settings. Die Brennweite wird auf die echte Bildbreite
    skaliert."""
    render = scene.render
    if real_size is not None:
        w, h = int(real_size[0]), int(real_size[1])
    else:
        w = int(render.resolution_x * render.resolution_percentage / 100.0)
        h = int(render.resolution_y * render.resolution_percentage / 100.0)

    # Blenders Kameramodell (v1.1.3): Pixel-Seitenverhaeltnis, Sensor-Fit
    # und Shift beziehen sich auf die effektive Bildgroesse (Pixel x Aspekt).
    # Bei AUTO bestimmt die laengere effektive Kante, die Sensorbreite gilt.
    pa_x = max(render.pixel_aspect_x, 1e-6)
    pa_y = max(render.pixel_aspect_y, 1e-6)
    size_x, size_y = w * pa_x, h * pa_y
    fit = cam_data.sensor_fit
    if fit == 'AUTO':
        fit = 'HORIZONTAL' if size_x >= size_y else 'VERTICAL'
        sensor = cam_data.sensor_width
    elif fit == 'HORIZONTAL':
        sensor = cam_data.sensor_width
    else:
        sensor = cam_data.sensor_height
    view = size_x if fit == 'HORIZONTAL' else size_y
    f_eff = cam_data.lens * view / sensor
    fx = f_eff / pa_x
    fy = f_eff / pa_y
    cx = w / 2.0 - cam_data.shift_x * view / pa_x
    cy = h / 2.0 + cam_data.shift_y * view / pa_y
    return fx, fy, cx, cy, w, h


def _exp_pose_w2c(cam, scale, W):
    m_c2w = cam.matrix_world
    R_c2w_bl = m_c2w.to_3x3()
    C = (W @ m_c2w.translation.copy()) * scale
    R_c2w_cm = R_c2w_bl @ _BLENDER_TO_COLMAP
    R_w2c = R_c2w_cm.transposed() @ W.transposed()
    T = -(R_w2c @ C)
    q = R_w2c.to_quaternion()
    return q.w, q.x, q.y, q.z, T.x, T.y, T.z


def _exp_srgb_byte(c):
    """Linear-Float (0..1) -> sRGB-kodiertes Byte (0..255).
    Blender-Materialfarben sind linear; COLMAP/Viewer erwarten sRGB."""
    c = max(0.0, min(1.0, c))
    if c <= 0.0031308:
        s = 12.92 * c
    else:
        s = 1.055 * (c ** (1.0 / 2.4)) - 0.055
    return int(round(max(0.0, min(1.0, s)) * 255))


def _exp_material_base_color(mat):
    """Liefert (r,g,b) 0..1 der Principled-BSDF-Base-Color eines Materials,
    oder None, wenn nicht ermittelbar."""
    if mat is None or not mat.use_nodes or mat.node_tree is None:
        # Material ohne Nodes: diffuse_color als Fallback.
        if mat is not None:
            dc = mat.diffuse_color
            return (dc[0], dc[1], dc[2])
        return None
    for node in mat.node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            inp = node.inputs.get("Base Color")
            if inp is not None:
                v = inp.default_value
                return (v[0], v[1], v[2])
    # Kein Principled gefunden: erstes Material mit diffuse_color.
    dc = mat.diffuse_color
    return (dc[0], dc[1], dc[2])


def _exp_object_material_colors(obj):
    """Liste von (r,g,b) je Material-Slot des Objekts (linear 0..1).
    Fehlende/unlesbare Slots -> mittleres Grau."""
    colors = []
    if not obj.material_slots:
        return [(0.5, 0.5, 0.5)]
    for slot in obj.material_slots:
        col = _exp_material_base_color(slot.material)
        colors.append(col if col is not None else (0.5, 0.5, 0.5))
    return colors


def _exp_vertex_color_lookup(mesh):
    """Baut, falls vorhanden, ein Dict vert_index -> (r,g,b) aus dem
    aktiven Color-Attribute. Liefert None, wenn keine Farbattribute da
    sind. Mittelt ueber alle Loops, die einen Vertex referenzieren."""
    ca = getattr(mesh, "color_attributes", None)
    if not ca or len(ca) == 0:
        return None
    layer = ca.active_color or ca[0]

    acc = {}
    cnt = {}
    if layer.domain == 'POINT':
        # Direkt pro Vertex.
        for i, d in enumerate(layer.data):
            c = d.color
            acc[i] = (c[0], c[1], c[2])
            cnt[i] = 1
    else:  # 'CORNER' -> ueber Loops auf Vertices mitteln
        for poly in mesh.polygons:
            for li in poly.loop_indices:
                vi = mesh.loops[li].vertex_index
                c = layer.data[li].color
                if vi in acc:
                    a = acc[vi]
                    acc[vi] = (a[0] + c[0], a[1] + c[1], a[2] + c[2])
                    cnt[vi] += 1
                else:
                    acc[vi] = (c[0], c[1], c[2])
                    cnt[vi] = 1
    return {vi: (acc[vi][0] / cnt[vi], acc[vi][1] / cnt[vi],
                 acc[vi][2] / cnt[vi]) for vi in acc}


def _exp_crop_bounds_world(s):
    """Liefert (min_vec, max_vec) der Welt-Raum-Bounding-Box des Crop-
    Objekts inkl. Margin, oder None wenn Crop aus/kein Objekt. Die Bounds
    sind in BLENDER-Weltkoordinaten -- der Test erfolgt VOR der Y-up-/
    Scale-Transformation der Punkte."""
    if not s.exp_crop_enable or s.exp_crop_object is None:
        return None
    obj = s.exp_crop_object
    mw = obj.matrix_world
    # bound_box: 8 lokale Eckpunkte; in Weltkoordinaten transformieren.
    corners = [mw @ Vector(c) for c in obj.bound_box]
    if not corners:
        return None
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    m = s.exp_crop_margin
    mn = Vector((min(xs) - m, min(ys) - m, min(zs) - m))
    mx = Vector((max(xs) + m, max(ys) + m, max(zs) + m))
    return (mn, mx)


def _exp_point_in_bounds(co_world, bounds):
    """True, wenn co_world (Blender-Weltkoord) innerhalb bounds liegt."""
    mn, mx = bounds
    return (mn.x <= co_world.x <= mx.x and
            mn.y <= co_world.y <= mx.y and
            mn.z <= co_world.z <= mx.z)


def _exp_build_scene_bvh(objs):
    """Einzelner Welt-Raum-BVHTree ueber alle angegebenen Meshes.
    Abgeleitet aus "Gauss Cannon" von Arash Keshmirian (Warpgate Labs),
    GPL-3.0-or-later;
    siehe Lizenzsektion im Header."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    vert_arrays = []
    all_polys = []
    offset = 0
    for obj in objs:
        if obj.type != 'MESH':
            continue
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
            all_polys.extend([i + offset for i in p.vertices]
                             for p in mesh.polygons)
            offset += n_verts
        finally:
            obj_eval.to_mesh_clear()
    if not vert_arrays or not all_polys:
        return None
    all_verts = np.vstack(vert_arrays)
    return BVHTree.FromPolygons(all_verts.tolist(), all_polys)


def _exp_visible_ray(p, bvh, cam_positions, eps=1e-3):
    """True, wenn der Weltpunkt p von mindestens einer Kamera frei gesehen
    wird (kein anderes Geometrie-Teil davor). Kameras nach Naehe sortiert,
    Early-Out beim ersten freien Sichtkontakt. Numerisch verifiziert."""
    cams = sorted(cam_positions, key=lambda c: (c - p).length_squared)
    for c in cams:
        direction = p - c
        d_p = direction.length
        if d_p < eps:
            return True
        direction = direction / d_p
        loc, nrm, idx, dist = bvh.ray_cast(c, direction)
        if loc is None or dist >= d_p - eps:
            return True
    return False


# ----------------------------------------------------------------------
# Sichtbarkeitsfilter auf der GPU (v105)
# ----------------------------------------------------------------------
# Statt je Punkt einen Strahl zu jeder Kamera zu schiessen (Python, bei
# Klemmbaustein-Modellen ~300 Strahlen je verdecktem Punkt, Vespa 5,4 Mio. Punkte:
# ~3 h), zeichnet die GPU das Modell aus jeder Kamera einmal als Tiefenbild;
# alle Punkte werden dann mit numpy gegen jedes Tiefenbild geprueft.
# Genauigkeit = Aufloesung des Tiefenbilds. Ohne GPU-Kontext (z. B.
# blender --background) faellt der Export auf den Ray-Cast zurueck.
_EXP_GPU_RES = 2048

# Compute-Shader: je Punkt (Texel der Punkt-Textur) in dieses Kamerabild
# projizieren und gegen den weitesten Tiefenwert der 3x3-Nachbarschaft
# pruefen -- dieselbe Regel wie _ExpGpuDepth.visible (numpy).
_EXP_VIS_COMPUTE_SRC = """
void main()
{
  ivec2 id = ivec2(gl_GlobalInvocationID.xy);
  ivec2 sz = imageSize(pts);
  if (id.x >= sz.x || id.y >= sz.y) { return; }
  if (imageLoad(vis, id).r > 0.5) { return; }
  vec4 p = imageLoad(pts, id);
  if (p.w < 0.5) { return; }
  vec3 rel = p.xyz - cam_pos.xyz;
  float d = dot(rel, cam_fwd.xyz);
  float f = params.x;
  float nr = params.y;
  float fr = params.z;
  int res = int(params.w);
  if (d <= nr) { return; }
  float xn = dot(rel, cam_right.xyz) / d * f;
  float yn = dot(rel, cam_up.xyz) / d * f;
  if (abs(xn) > 1.0 || abs(yn) > 1.0) { return; }
  int px = clamp(int((xn + 1.0) * 0.5 * float(res)), 0, res - 1);
  int py = clamp(int((yn + 1.0) * 0.5 * float(res)), 0, res - 1);
  float mx = 0.0;
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      ivec2 q = clamp(ivec2(px + dx, py + dy), ivec2(0), ivec2(res - 1));
      float z = texelFetch(depth_tx, q, 0).r;
      float lin = (z >= 1.0) ? 1e30 :
                  (2.0 * fr * nr) / (fr + nr - (2.0 * z - 1.0) * (fr - nr));
      mx = max(mx, lin);
    }
  }
  float tol = d * tol_v.x + tol_v.y;
  if (d <= mx + tol) { imageStore(vis, id, vec4(1.0)); }
}
"""


class _ExpGpuDepth:
    """Haelt Geometrie-Batch und Framebuffer fuer die Tiefenbilder."""

    def __init__(self, objs, res=_EXP_GPU_RES):
        import gpu
        from gpu_extras.batch import batch_for_shader
        depsgraph = bpy.context.evaluated_depsgraph_get()
        verts, tris, off = [], [], 0
        for o in objs:
            if o.type != 'MESH':
                continue
            oe = o.evaluated_get(depsgraph)
            me = oe.to_mesh()
            if me is None:
                continue
            try:
                me.calc_loop_triangles()
                n = len(me.vertices)
                nt = len(me.loop_triangles)
                if n and nt:
                    co = np.empty(n * 3, dtype=np.float32)
                    me.vertices.foreach_get("co", co)
                    mw = np.array(oe.matrix_world, dtype=np.float32)
                    verts.append(co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3])
                    tri = np.empty(nt * 3, dtype=np.int32)
                    me.loop_triangles.foreach_get("vertices", tri)
                    tris.append(tri.reshape(-1, 3) + off)
                    off += n
            finally:
                oe.to_mesh_clear()
        if not verts:
            raise RuntimeError("no geometry for the depth maps")
        self.verts = np.vstack(verts).astype(np.float32)
        self.tris = np.vstack(tris).astype(np.int32)
        self.shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        self.batch = batch_for_shader(self.shader, 'TRIS', {"pos": self.verts},
                                      indices=self.tris)
        self.res = res
        self.nb = 1          # Nachbarschaft (1 = 3x3, 0 = nur das Pixel)
        self.tol_rel = 2e-3  # Toleranz relativ zur Tiefe
        self.depth_tex = gpu.types.GPUTexture((res, res),
                                              format='DEPTH_COMPONENT32F')
        self.color_tex = gpu.types.GPUTexture((res, res), format='RGBA8')
        self.fb = gpu.types.GPUFrameBuffer(depth_slot=self.depth_tex,
                                           color_slots=self.color_tex)
        lo = self.verts.min(axis=0)
        hi = self.verts.max(axis=0)
        self.size = float(np.linalg.norm(hi - lo)) or 1.0
        self.center = (lo + hi) / 2.0
        self.radius = float(np.linalg.norm(self.verts - self.center,
                                           axis=1).max()) or 1.0
        self._cs = None      # Compute-Shader-Zustand (compute_begin)

    def near_far(self, cam_pos):
        """Tiefenbereich eng um das Modell (v111). Frueher reichte er vom
        Zehntausendstel bis zum Hundertfachen der Modellgroesse -- in 32-Bit-
        Genauigkeit wurde die Rueckrechnung der Tiefe dadurch ungenau."""
        dist = float(np.linalg.norm(np.array(cam_pos) - self.center))
        near = max(self.size * 1e-4, dist - self.radius * 1.01)
        far = dist + self.radius * 1.01
        return near, far

    def _render(self, cam_pos, look_dir, fov, near, far):
        """Zeichnet das Tiefenbild dieser Kamera in self.fb. Rueckgabe: die
        Kamera-Rotation (Quaternion)."""
        import gpu
        q = look_dir.to_track_quat('-Z', 'Y')
        cam_mw = Matrix.Translation(cam_pos) @ q.to_matrix().to_4x4()
        f = 1.0 / math.tan(fov / 2.0)
        proj = Matrix(((f, 0, 0, 0), (0, f, 0, 0),
                       (0, 0, (far + near) / (near - far),
                        2 * far * near / (near - far)),
                       (0, 0, -1, 0)))
        with self.fb.bind():
            self.fb.clear(color=(0, 0, 0, 0), depth=1.0)
            gpu.state.depth_test_set('LESS_EQUAL')
            gpu.state.depth_mask_set(True)
            gpu.state.face_culling_set('NONE')
            with gpu.matrix.push_pop():
                gpu.matrix.load_matrix(cam_mw.inverted())
                gpu.matrix.load_projection_matrix(proj)
                self.shader.uniform_float("color", (1, 1, 1, 1))
                self.batch.draw(self.shader)
            gpu.state.depth_test_set('NONE')
        return q

    # --- Punktpruefung als Compute-Shader (v111) ------------------------
    # Die Punkte liegen als RGBA32F-Textur auf der GPU, das Ergebnis als
    # R32F-Textur; je Kamera zeichnet die GPU das Tiefenbild und prueft alle
    # Punkte in einem Durchlauf. Vespa P200 (5,4 Mio. Punkte, 320 Kameras):
    # unter 1 s statt ~165 s mit numpy; gegen den exakten Ray-Cast verliert
    # er weniger sichtbare Punkte (80 statt 629 von ~7.000 strittigen).
    _CS_WIDTH = 4096

    def compute_begin(self, points):
        """Punkte hochladen und Shader bauen. Wirft bei Problemen eine
        Exception -- der Aufrufer faellt dann auf visible() (numpy) zurueck."""
        import gpu
        n = len(points)
        w = self._CS_WIDTH
        h = max(1, (n + w - 1) // w)
        if h > 16384:
            raise RuntimeError("too many points for one texture (%d)" % n)
        buf = np.zeros((h * w, 4), dtype=np.float32)
        buf[:n, :3] = points
        buf[:n, 3] = 1.0
        pts_tex = gpu.types.GPUTexture(
            (w, h), format='RGBA32F',
            data=gpu.types.Buffer('FLOAT', h * w * 4, buf.ravel()))
        vis_tex = gpu.types.GPUTexture((w, h), format='R32F')
        vis_tex.clear(format='FLOAT', value=(0.0,))
        info = gpu.types.GPUShaderCreateInfo()
        info.image(0, 'RGBA32F', 'FLOAT_2D', 'pts', qualifiers={'READ'})
        info.image(1, 'R32F', 'FLOAT_2D', 'vis', qualifiers={'READ', 'WRITE'})
        info.sampler(0, 'FLOAT_2D', 'depth_tx')
        for name in ('cam_pos', 'cam_fwd', 'cam_right', 'cam_up', 'params',
                     'tol_v'):
            info.push_constant('VEC4', name)
        info.local_group_size(16, 16, 1)
        info.compute_source(_EXP_VIS_COMPUTE_SRC)
        shader = gpu.shader.create_from_info(info)
        self._cs = dict(n=n, w=w, h=h, pts=pts_tex, vis=vis_tex,
                        shader=shader)

    def compute_camera(self, cam_pos, look_dir, fov):
        """Tiefenbild dieser Kamera zeichnen und alle noch nicht sichtbaren
        Punkte dagegen pruefen (auf der GPU, ohne Rueckholen)."""
        import gpu
        cs = self._cs
        near, far = self.near_far(cam_pos)
        q = self._render(cam_pos, look_dir, fov, near, far)
        r = q @ Vector((1.0, 0.0, 0.0))
        u = q @ Vector((0.0, 1.0, 0.0))
        fw = look_dir.normalized()
        sh = cs["shader"]
        sh.bind()
        sh.image('pts', cs["pts"])
        sh.image('vis', cs["vis"])
        sh.uniform_sampler('depth_tx', self.depth_tex)
        sh.uniform_float('cam_pos', (cam_pos.x, cam_pos.y, cam_pos.z, 0.0))
        sh.uniform_float('cam_fwd', (fw.x, fw.y, fw.z, 0.0))
        sh.uniform_float('cam_right', (r.x, r.y, r.z, 0.0))
        sh.uniform_float('cam_up', (u.x, u.y, u.z, 0.0))
        sh.uniform_float('params', (1.0 / math.tan(fov / 2.0), near, far,
                                    float(self.res)))
        sh.uniform_float('tol_v', (self.tol_rel, self.size * 5e-4, 0.0, 0.0))
        gpu.compute.dispatch(sh, (cs["w"] + 15) // 16, (cs["h"] + 15) // 16, 1)

    def compute_result(self):
        """bool je Punkt: von mindestens einer Kamera gesehen."""
        cs = self._cs
        v = np.array(cs["vis"].read(), dtype=np.float32).reshape(-1)
        return v[:cs["n"]] > 0.5

    def depth_map(self, cam_pos, look_dir, fov, near, far):
        """Lineare Tiefe (Abstand entlang der Blickachse) je Pixel, (R,R),
        Zeile 0 = unten. inf, wo nichts ist."""
        q = self._render(cam_pos, look_dir, fov, near, far)
        with self.fb.bind():
            buf = self.fb.read_depth(0, 0, self.res, self.res)
        zb = np.array(buf, dtype=np.float32).reshape(self.res, self.res)
        # Fensterstiefe -> lineare Augentiefe.
        z_ndc = zb.astype(np.float64) * 2.0 - 1.0
        lin = (2.0 * far * near) / (far + near - z_ndc * (far - near))
        lin[zb >= 1.0] = np.inf
        return lin, q

    def visible(self, points, cam_pos, look_dir, fov):
        """bool je Punkt: im Bild dieser Kamera und nicht verdeckt."""
        near, far = self.near_far(cam_pos)
        lin, q = self.depth_map(cam_pos, look_dir, fov, near, far)
        # Weitester Wert der 3x3-Nachbarschaft: Punkte an Kanten und in
        # schmalen Luecken nicht wegen der Pixelrasterung verlieren.
        fin = np.where(np.isfinite(lin), lin, np.float64(1e30))
        r = self.res
        mx = fin
        if self.nb:
            k = self.nb
            pad = np.pad(fin, k, mode='edge')
            mx = fin.copy()
            for dy in range(2 * k + 1):
                for dx in range(2 * k + 1):
                    mx = np.maximum(mx, pad[dy:dy + r, dx:dx + r])
        right = np.array(q @ Vector((1.0, 0.0, 0.0)))
        up = np.array(q @ Vector((0.0, 1.0, 0.0)))
        fwd = np.array(look_dir.normalized())
        rel = points - np.array(cam_pos)
        d = rel @ fwd
        f = 1.0 / math.tan(fov / 2.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            xn = (rel @ right) / d * f
            yn = (rel @ up) / d * f
        ok = (d > near) & (np.abs(xn) <= 1.0) & (np.abs(yn) <= 1.0)
        px = np.clip(((xn + 1.0) * 0.5 * r).astype(np.int64), 0, r - 1)
        py = np.clip(((yn + 1.0) * 0.5 * r).astype(np.int64), 0, r - 1)
        tol = d * self.tol_rel + self.size * 5e-4
        out = np.zeros(len(points), dtype=bool)
        idx = np.nonzero(ok)[0]
        out[idx] = d[idx] <= mx[py[idx], px[idx]] + tol[idx]
        return out


def _exp_sample_face_points(objs, count, scale, W, world_out=None,
                            crop_bounds=None):
    """Verteilt 'count' Punkte flaechengewichtet ueber die Oberflaechen der
    Objekte (gleichmaessige Startdichte unabhaengig vom Vertex-Layout).
    Rueckgabe: Liste (x,y,z) in Y-up + Scale; world_out bekommt parallel die
    Weltkoordinaten (fuer den Sichtbarkeitsfilter). Baryzentrisches
    sqrt-Sampling -> gleichverteilt im Dreieck.

    Seit v110: Dreiecke per foreach_get/numpy (vorher bmesh + Python-
    Schleife je Dreieck) und Crop-Box wie bei den Vertex-Punkten -- Punkte
    ausserhalb werden verworfen und so lange nachgezogen, bis 'count'
    erreicht ist (hoechstens 6 Runden)."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    tri_chunks = []
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
        finally:
            obj_eval.to_mesh_clear()
    if not tri_chunks or count <= 0:
        return []
    A = np.vstack(tri_chunks)                                   # (T,3,3)
    area = 0.5 * np.linalg.norm(np.cross(A[:, 1] - A[:, 0],
                                         A[:, 2] - A[:, 0]), axis=1)
    good = area > 0.0
    A, area = A[good], area[good]
    if not len(A):
        return []
    prob = area / area.sum()
    rng = np.random.default_rng()

    def draw(k):
        pick = rng.choice(len(A), size=int(k), p=prob)
        r1 = np.sqrt(rng.random(len(pick)))[:, None]
        r2 = rng.random(len(pick))[:, None]
        tri = A[pick]
        return (tri[:, 0] * (1.0 - r1) + tri[:, 1] * (r1 * (1.0 - r2))
                + tri[:, 2] * (r1 * r2))

    count = int(count)
    if crop_bounds is None:
        world = draw(count)
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
            inside = smp[np.all((smp >= c_mn) & (smp <= c_mx), axis=1)]
            frac = len(inside) / float(len(smp))
            got.append(inside)
            have += len(inside)
            if frac == 0.0:
                break
        world = np.vstack(got)[:count] if got else np.empty((0, 3))
    Wm = np.array(W, dtype=np.float64)
    out = (world @ Wm.T) * scale
    if world_out is not None:
        world_out.extend(map(Vector, world.tolist()))
    # tolist() -> echte Python-floats fuer points3D.txt (v109-Fix).
    return out.tolist()


def _exp_point_source_objects(scene, s, extra_exclude=()):
    """Mesh-Objekte, aus denen die Punktwolke entsteht. Explizit genannte
    Objekte (exp_point_objects) gelten unveraendert. Sonst: alle sichtbaren
    Meshes, die auch GERENDERT werden -- ohne Hilfsobjekte. Ausgeschlossen
    sind Objekte mit hide_render (z. B. die Kamera-Sphere), die Sphere und
    die Leitgitter der Guide-Liste, das Crop-Objekt sowie extra_exclude.
    Bis v81 landeten so die Vertices der Camera_Sphere als Punkte in der
    Luft in points3D.txt (v82-Fix)."""
    names = [x.strip() for x in s.exp_point_objects.split(",") if x.strip()]
    mode = s.exp_point_source
    # Szenen von vor v129: Namensliste gesetzt, Point Source nie gewaehlt.
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
    """(Vertices je Objekt, Vertices je eindeutigem Mesh, Objekte, Meshes) der
    Punktwolken-Objekte, ausgewertet inkl. Modifier -- dieselben Zahlen, die
    der Export nimmt. Liest die bereits ausgewerteten Meshes (kein to_mesh),
    daher auch im Panel-Draw billig. Die Punktwolke zaehlt jedes Objekt
    einzeln, Blenders Statistik jedes Mesh nur einmal (verknuepfte Kopien)."""
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
    """Obergrenze der Vertex-Punkte: mit Use Vertex Count keine (alle)."""
    return None if s.exp_use_vertex_count else s.exp_max_points


def _exp_srgb_bytes_np(rgb):
    """Vektorisierte Fassung von _exp_srgb_byte fuer ein (N,3)-Array
    linearer Farben (gleiche Schwelle, gleiche Koeffizienten, gleiches
    Runden). Rueckgabe (N,3) int."""
    c = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    srgb = np.where(c <= 0.0031308, 12.92 * c,
                    1.055 * np.power(c, 1.0 / 2.4) - 0.055)
    return np.round(np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.int64)


def _exp_vertex_first_material(mesh, n):
    """Materialindex je Vertex: der des ersten Polygons, das den Vertex
    enthaelt (Polygon-Reihenfolge), sonst 0 -- wie die fruehere Schleife."""
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
    # Blender legt die Loops polygonweise hintereinander ab (aufsteigende
    # loop_start), Loop-Reihenfolge = Polygon-Reihenfolge.
    poly_of_loop = np.repeat(np.arange(npoly), total)
    uniq, first = np.unique(lv, return_index=True)
    vm[uniq] = mi[poly_of_loop[first]]
    return vm


def _exp_gather_points(scene, scale, W, s, world_out=None):
    """Hauptwolke aus den Vertices der Punktwolken-Objekte, mit Crop, Farbe
    und Obergrenze. Seit v110 mit numpy statt einer Python-Schleife je
    Vertex (Vespa: 5,4 Mio. Vertices); Ergebnis wie zuvor."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    objs = _exp_point_source_objects(scene, s)
    if not objs:
        raise RuntimeError("No mesh objects found for the point cloud.")

    mode = s.exp_point_color
    crop_bounds = _exp_crop_bounds_world(s)
    if crop_bounds is not None:
        c_mn = np.array(tuple(crop_bounds[0]), dtype=np.float64)
        c_mx = np.array(tuple(crop_bounds[1]), dtype=np.float64)
    worlds, colors = [], []

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

            # Farbe je Vertex (linear 0..1); NaN = keine Farbe -> Grau.
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

            if crop_bounds is not None:
                keep = np.all((world >= c_mn) & (world <= c_mx), axis=1)
                world = world[keep]
                if rgb is not None:
                    rgb = rgb[keep]
            worlds.append(world)
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

    # Gemeinsam heruntersampeln (Positionen und Farben synchron).
    cap = _exp_max_points_effective(s)
    if cap is not None and len(world_all) > cap:
        step = len(world_all) / float(cap)
        idx = np.array([int(i * step) for i in range(cap)], dtype=np.int64)
        world_all = world_all[idx]
        col_all = col_all[idx]

    Wm = np.array(W, dtype=np.float64)
    pts = (world_all @ Wm.T) * scale
    if world_out is not None:
        world_out.extend(map(Vector, world_all.tolist()))
    # tolist(): echte Python-Zahlen fuer points3D.txt (siehe v109). Zeilen
    # bleiben Listen -- Tupel daraus zu bauen kostete bei 5 Mio. Punkten
    # mehrere Sekunden, der Schreibcode entpackt beides gleich.
    return pts.tolist(), col_all.tolist()


_EXP_SIGNATURE_VERSION = 2


def _exp_face_count(s, n_vertex_points):
    """Anzahl Face Points, wie der Export sie verwendet (auch fuer die
    Vorschau im Panel): mit Use Vertex Count ein Anteil der Vertex-Punkte,
    mindestens 100; sonst der eingegebene Wert."""
    if s.exp_use_vertex_count:
        return max(100, int(round(n_vertex_points * s.exp_face_points_percent
                                  / 100.0)))
    return s.exp_face_points_count


def _exp_point_signature(scene, s, cam_views, scale, W):
    """Fingerabdruck von allem, was die Punktwolke bestimmt. Gleicher
    Abdruck -> gleiche Punktwolke; dann uebernimmt der Export die zuletzt
    geschriebene points3D.txt (v106). Pfade gehen NICHT ein.

    v110: statt nur der Vertex-Anzahl die ausgewerteten Vertex-Positionen
    (Modifier, Edit Mode), bei Farbmodi die Farbquellen, der Face-Points-
    Anteil, Blickrichtungen und Brennweite der Kameras sowie der Weg des
    Sichtbarkeitsfilters (GPU oder Ray-Cast liefern leicht verschiedene
    Mengen)."""
    import hashlib
    h = hashlib.sha1()

    def add(*vals):
        h.update(repr(vals).encode("utf-8"))

    add("signature", _EXP_SIGNATURE_VERSION)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for o in _exp_point_source_objects(scene, s):
        oe = o.evaluated_get(depsgraph)
        add(o.name, [round(v, 6) for row in oe.matrix_world for v in row])
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


def _exp_write_cameras(path, fx, fy, cx, cy, w, h):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write("1 PINHOLE %d %d %s %s %s %s\n" % (
            w, h, repr(fx), repr(fy), repr(cx), repr(cy)))


def _exp_write_images(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write("# Number of images: %d\n" % len(entries))
        for (iid, qw, qx, qy, qz, tx, ty, tz, name) in entries:
            f.write("%d %s %s %s %s %s %s %s 1 %s\n" % (
                iid, repr(qw), repr(qx), repr(qy), repr(qz),
                repr(tx), repr(ty), repr(tz), name))
            f.write("\n")


def _exp_write_points(path, points, colors=None):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write("# Number of points: %d\n" % len(points))
        for i, (x, y, z) in enumerate(points):
            if colors is not None and i < len(colors):
                r, g, b = colors[i]
            else:
                r, g, b = 200, 200, 200
            f.write("%d %s %s %s %d %d %d 0\n" % (
                i + 1, repr(x), repr(y), repr(z), r, g, b))


def _exp_blend_basename():
    path = bpy.data.filepath
    if not path:
        return None
    stem, _ = os.path.splitext(os.path.basename(path))
    return stem


def _exp_split_name_version(stem):
    """Zerlegt den .blend-Basisnamen in (Name_ohne_Version, Versions-Token).
    Beispiel: 'Scene_v001' -> ('Scene', 'v001'); 'Bugatti_Chiron_v002_Addon'
    -> ('Bugatti_Chiron', 'v002'). Nimmt das LETZTE vN-Token. Ohne Version
    -> (stem, None). Trennzeichen vor der Version werden abgeschnitten."""
    matches = list(re.finditer(r'[vV]\d+', stem))
    if not matches:
        return stem, None
    m = matches[-1]
    name = stem[:m.start()].rstrip('_-. ')
    version = stem[m.start():m.end()]
    return name, version


def _exp_resolve_output_dir(s):
    """Liefert den Zielordner fuer das COLMAP-Modell.

    Ist das Feld 'Output Folder' LEER (Default), wird automatisch eine
    Struktur neben der .blend-Datei gebaut:
        <Name>_COLMAP / <vNNN> /
    also ein Ordner aus dem Szenennamen (bis vor das Versions-Token) plus
    '_COLMAP', und darin ein Unterordner mit dem Versions-Token. Beispiel:
    'Scene_v001.blend' -> 'Scene_COLMAP/v001/'. Fehlt eine Version, wird
    'v001' verwendet; ist die .blend noch nicht gespeichert, 'Scene_COLMAP'.

    Ist das Feld gefuellt, wird der Pfad as-is benutzt (// = relativ zur
    .blend wird aufgeloest)."""
    od = (s.exp_output_dir or "").strip()
    if od:
        return bpy.path.abspath(od)
    # Leeres Feld -> automatische Struktur neben der .blend.
    base = bpy.path.abspath("//") if bpy.data.filepath else os.getcwd()
    stem = _exp_blend_basename()
    if stem:
        name, version = _exp_split_name_version(stem)
    else:
        name, version = "Scene", None
    folder = "%s_COLMAP" % (name if name else "Scene")
    sub = version if version else "v001"
    return os.path.join(base, folder, sub)


def _exp_output_has_export(out_dir):
    """True, wenn im Zielordner bereits ein COLMAP-Export liegt (eine der
    sparse/0-Dateien). Bilder allein zaehlen nicht: seit v90 rendert die
    Szene direkt nach images/, das ist noch kein Export."""
    if not out_dir or not os.path.isdir(out_dir):
        return False
    sparse = os.path.join(out_dir, "sparse", "0")
    for fn in ("cameras.txt", "images.txt", "points3D.txt"):
        if os.path.isfile(os.path.join(sparse, fn)):
            return True
    return False


def _exp_same_dir(a, b):
    """True, wenn beide Pfade auf denselben Ordner zeigen."""
    if not a or not b:
        return False
    return (os.path.normcase(os.path.normpath(os.path.abspath(a))) ==
            os.path.normcase(os.path.normpath(os.path.abspath(b))))


class GCAPTURE_OT_path_relative(Operator):
    bl_idname = "gcapture.path_relative"
    bl_label = "Make Path Relative"
    bl_description = ("Store this folder relative to the .blend file (//...), "
                      "so the scene keeps working when the project is moved")
    bl_options = {'REGISTER', 'UNDO'}

    prop: StringProperty(default="exp_image_dir", options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return bool(bpy.data.filepath)

    def execute(self, context):
        s = context.scene.gcapture_settings
        raw = (getattr(s, self.prop, "") or "").strip()
        if not raw:
            self.report({'WARNING'}, "Folder is empty.")
            return {'CANCELLED'}
        if raw.startswith("//"):
            self.report({'INFO'}, "Already relative.")
            return {'FINISHED'}
        try:
            rel = bpy.path.relpath(bpy.path.abspath(raw))
        except Exception as exc:
            self.report({'ERROR'}, "Cannot make relative: %s" % exc)
            return {'CANCELLED'}
        if not rel.startswith("//"):
            self.report({'WARNING'}, "Folder is on another drive - keeping "
                        "the absolute path.")
            return {'CANCELLED'}
        setattr(s, self.prop, rel)
        self.report({'INFO'}, "Now relative: %s" % rel)
        return {'FINISHED'}


class GCAPTURE_OT_export_colmap(Operator):
    bl_idname = "gcapture.export_colmap"
    bl_label = "Export COLMAP (Postshot / LichtFeld)"
    bl_description = ("Exports cameras.txt / images.txt / points3D.txt plus "
                      "an images/ folder as a COLMAP model for Postshot and "
                      "LichtFeld Studio. "
                      "Runs modally with a progress display (ESC to cancel)")
    bl_options = {'REGISTER'}

    _is_running = False

    @classmethod
    def poll(cls, context):
        return context.scene.camera is not None and not cls._is_running

    # ----- Setup (einmalig) -----
    def _setup(self, context):
        scene = context.scene
        s = scene.gcapture_settings
        cam = scene.camera
        if cam is None or cam.type != 'CAMERA':
            raise RuntimeError("No active scene camera set.")

        self._cam = cam
        self._out_dir = _exp_resolve_output_dir(s)
        self._images_dir = os.path.join(self._out_dir, "images")
        self._sparse_dir = os.path.join(self._out_dir, "sparse", "0")
        # sparse immer; images nur, wenn Bilder eingebunden werden.
        os.makedirs(self._sparse_dir, exist_ok=True)
        if s.exp_image_mode != 'NONE':
            os.makedirs(self._images_dir, exist_ok=True)

        self._src_dir = _exp_resolve_image_dir(s)
        auto_pattern, auto_ext, auto_frames = _exp_detect_frames(self._src_dir)
        if not auto_pattern:
            raise RuntimeError("Could not detect filename pattern in: %s"
                               % self._src_dir)
        self._pattern = auto_pattern
        self._ext = (auto_ext or "png").lstrip(".")
        # WICHTIG: Die Posenzahl kommt aus dem SZENEN-FRAMEBEREICH (die
        # tatsaechlichen Kamera-Keyframes), NICHT aus den vorhandenen
        # Bilddateien. Sonst exportiert das Addon nur so viele Posen, wie
        # gerade Bilder im Ordner liegen -- bei noch nicht fertig
        # synchronisierten Renderfarm-Bildern fehlen dann Posen. Pattern und
        # Bildgroesse werden weiterhin aus einer Beispieldatei abgeleitet.
        self._frames = list(range(scene.frame_start, scene.frame_end + 1))
        # Fehlende Bilder ermitteln (nur Warnung, kein Weglassen der Pose).
        have = set(auto_frames or [])
        self._missing_images = [fr for fr in self._frames if fr not in have] \
            if have else []

        # Effektiven Bildmodus bestimmen. Verschieben ist destruktiv -- die
        # Bilder sind danach aus dem Render-Ordner weg. Sicherheitscheck:
        # MOVE nur, wenn ALLE erwarteten Bilder vorhanden sind. Fehlt eines
        # (z. B. Renderfarm noch nicht fertig), wird NICHT verschoben, sondern
        # sicher auf KOPIEREN zurueckgefallen, mit Warnung im Abschluss.
        self._image_mode = s.exp_image_mode
        self._move_blocked = False
        # Rendert die Szene direkt in <out>/images (Save Scene Version),
        # liegen die Bilder schon im Datensatz: nichts kopieren (v90).
        if _exp_same_dir(self._src_dir, self._images_dir):
            self._image_mode = 'INPLACE'
        if self._image_mode == 'MOVE' and self._missing_images:
            self._image_mode = 'COPY'
            self._move_blocked = True

        # Echte Bildgroesse -> Intrinsics + cameras.txt. Aus dem ersten
        # tatsaechlich VORHANDENEN Bild lesen (auto_frames), nicht aus
        # self._frames[0] -- dessen Bild koennte noch fehlen.
        real_size = None
        if self._src_dir:
            probe = (auto_frames[0] if auto_frames
                     else (self._frames[0] if self._frames else None))
            if probe is not None:
                first = self._pattern.format(n=probe) + "." + self._ext
                real_size = _exp_image_size(os.path.join(self._src_dir, first))
        self._real_size = real_size
        self._W = _exp_world_transform(s)
        fx, fy, cx, cy, w, h = _exp_intrinsics(cam.data, scene, real_size)
        self._res = (w, h)
        _exp_write_cameras(os.path.join(self._sparse_dir, "cameras.txt"),
                           fx, fy, cx, cy, w, h)

        self._orig_frame = scene.frame_current

        # Maszstab bestimmen (einmaliger Vorlauf ueber alle Frames).
        if s.exp_auto_scale:
            centers = []
            for fr in self._frames:
                scene.frame_set(fr)
                centers.append(cam.matrix_world.translation.copy())
            cxm = sum(c.x for c in centers) / len(centers)
            cym = sum(c.y for c in centers) / len(centers)
            czm = sum(c.z for c in centers) / len(centers)
            rs = [((c.x - cxm) ** 2 + (c.y - cym) ** 2 + (c.z - czm) ** 2) ** 0.5
                  for c in centers]
            mean_r = sum(rs) / len(rs) if rs else 0.0
            self._scale = s.exp_target_radius / mean_r if mean_r > 1e-9 else 1.0
            self._mean_r = mean_r
        else:
            self._scale = 1.0
            self._mean_r = None

        # Crop-Box-Bounds (Blender-Weltkoord).
        self._crop_bounds = _exp_crop_bounds_world(s)

        # Fingerabdruck der Punktwolke (v106): erlaubt das Wiederverwenden
        # der zuletzt geschriebenen points3D.txt, wenn sich nichts geaendert
        # hat (z. B. nur ein Pfad).
        try:
            faces = _gcapture_all_guide_faces(
                context, None, guide_objs=_collect_guide_objects(
                    context, context.active_object))
            views = [(tuple(f[0]), tuple(_gcapture_look_dir_for(s, *f)))
                     for f in faces]
            self._sig = _exp_point_signature(scene, s, views, self._scale,
                                             self._W)
        except Exception as exc:
            print("[Gaussian Render Capture] point signature failed:", exc)
            self._sig = ""

        # Arbeitslisten.
        self._entries = []
        self._copied = 0
        self._missing = []

        # Arbeitsliste: ein Eintrag ("pose", Bild-ID, Frame) pro Pose.
        self._work = [("pose", i, fr) for i, fr in enumerate(self._frames, start=1)]
        self._total = len(self._work)
        self._done = 0
        self._phase = 'work'  # 'work' -> Posen, 'filter' -> Sichtbarkeit
        self._start_time = time.time()
        self._s = s

    @staticmethod
    def _point_targets(scene, s, guide_objs):
        # Dieselbe Objektauswahl wie die Hauptwolke. guide_objs nur mit
        # Guide-Liste ausschliessen: ohne Liste ist es das AKTIVE Objekt --
        # oft das Modell selbst, das sonst aus Face Points und Filter-BVH fiele.
        return _exp_point_source_objects(
            scene, s, guide_objs if s.use_guide_list else ())

    # ----- Pro Arbeitseinheit -----
    def _process_pose(self, scene, idx, frame):
        s = self._s
        scene.frame_set(frame)
        qw, qx, qy, qz, tx, ty, tz = _exp_pose_w2c(self._cam, self._scale, self._W)
        img_name = self._pattern.format(n=frame) + "." + self._ext
        self._entries.append((idx, qw, qx, qy, qz, tx, ty, tz, img_name))
        if self._image_mode == 'INPLACE':
            if os.path.isfile(os.path.join(self._images_dir, img_name)):
                self._copied += 1
            else:
                self._missing.append(img_name)
        elif self._image_mode != 'NONE' and self._src_dir:
            src = os.path.join(self._src_dir, img_name)
            dst = os.path.join(self._images_dir, img_name)
            if os.path.isfile(src):
                if os.path.abspath(src) != os.path.abspath(dst):
                    if self._image_mode == 'MOVE':
                        shutil.move(src, dst)
                    else:
                        shutil.copy2(src, dst)
                self._copied += 1
            else:
                self._missing.append(img_name)

    # ----- Modal-Maschinerie -----
    def invoke(self, context, event):
        try:
            self._setup(context)
        except Exception as exc:
            self.report({'ERROR'}, "COLMAP export failed: %s" % exc)
            return {'CANCELLED'}

        if self._total == 0:
            self.report({'ERROR'}, "Nothing to export (no frames).")
            return {'CANCELLED'}

        GCAPTURE_OT_export_colmap._is_running = True
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.05, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'ESC':
            self._cleanup(context)
            self.report({'WARNING'}, "COLMAP export cancelled.")
            return {'CANCELLED'}

        if event.type == 'TIMER':
            scene = context.scene

            # Filter-Phase (modal, echtes ESC oben).
            if self._phase == 'filter':
                try:
                    return self._filter_tick(context)
                except Exception as exc:
                    self._cleanup(context)
                    self.report({'ERROR'}, "Visibility filter failed: %s" % exc)
                    return {'CANCELLED'}

            if self._done >= self._total:
                return self._finish(context)

            # Fortschritt im Header.
            elapsed = time.time() - self._start_time
            time_info = ""
            if self._done > 0:
                per = elapsed / self._done
                rem = int(per * (self._total - self._done))
                tstr = ("%dm %ds" % (rem // 60, rem % 60)) if rem >= 60 else ("%ds" % rem)
                time_info = ", ~%s left" % tstr
            _gcapture_progress_set(
                "gcapture.export_colmap", self._done / self._total,
                "Camera poses %d/%d%s" % (self._done, self._total, time_info))

            # Mehrere Pose-Frames pro Tick abarbeiten (schnell).
            try:
                batch = 0
                while self._done < self._total and batch < 25:
                    _, bidx, bframe = self._work[self._done]
                    self._process_pose(scene, bidx, bframe)
                    self._done += 1
                    batch += 1
            except Exception as exc:
                self._cleanup(context)
                self.report({'ERROR'}, "COLMAP export failed: %s" % exc)
                return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def _finish(self, context):
        scene = context.scene
        s = self._s
        # images.txt schreiben.
        _exp_write_images(os.path.join(self._sparse_dir, "images.txt"),
                          self._entries)

        # Unveraenderte Punktwolke uebernehmen (v106).
        target_points = os.path.join(self._sparse_dir, "points3D.txt")
        prev = bpy.path.abspath(s.exp_last_points_file or "")
        reuse = False
        if s.exp_reuse_points and self._sig:
            if s.exp_last_signature != self._sig:
                print("[Gaussian Render Capture] point cloud changed -> "
                      "recomputing")
            elif not prev or not os.path.isfile(prev):
                print("[Gaussian Render Capture] no stored point cloud (%s) -> "
                      "recomputing" % (prev or "none"))
            else:
                reuse = True
        if reuse:
            try:
                same = (os.path.normcase(os.path.abspath(prev))
                        == os.path.normcase(os.path.abspath(target_points)))
                if not same:
                    shutil.copy2(prev, target_points)
                self._cleanup(context)
                s.exp_last_points_file = target_points
                self.report({'INFO'}, "COLMAP export done: %d poses, point "
                            "cloud unchanged - %s (%s) -> %s"
                            % (len(self._entries),
                               "kept" if same else "reused "
                               + os.path.basename(prev),
                               s.exp_last_detail or "previous export",
                               self._out_dir))
                return {'FINISHED'}
            except Exception as exc:
                print("[Gaussian Render Capture] could not reuse point cloud:",
                      exc)

        # --- Punkte sammeln (schnell) ---
        # Sichtbare Vertices als Hauptwolke, Face-Points optional dazu. Wenn
        # ein Sichtbarkeitsfilter aktiv ist, werden die Punkte NICHT hier
        # synchron gefiltert (das blockierte ESC), sondern als modale Phase
        # haeppchenweise im Tick -> echtes ESC. Hier nur sammeln.
        vmode = s.exp_face_points_vischeck
        want_filter = (vmode == 'RAYCAST')
        guide_objs = _collect_guide_objects(bpy.context,
                                            bpy.context.active_object)
        targets = self._point_targets(scene, s, guide_objs)
        need_world = want_filter

        # 1) Hauptwolke.
        world_v = [] if need_world else None
        pts, cols = _exp_gather_points(scene, self._scale, self._W, s,
                                       world_out=world_v)
        world_main = world_v
        self._n_vert = len(pts)      # fuer die Abschlussmeldung (v103)
        self._n_face = 0
        self._filtered = False

        # 2) Face-Points (optional) dazu. Mit Use Vertex Count: 10 % der
        # gesammelten Vertex-Punkte (v101), sonst der eingegebene Wert.
        face_count = _exp_face_count(s, len(pts))
        if s.exp_face_points:
            world_f = [] if need_world else None
            fpts = _exp_sample_face_points(
                targets, face_count, self._scale, self._W,
                world_out=world_f, crop_bounds=self._crop_bounds)
            if fpts:
                self._n_face = len(fpts)
                neutral = (200, 200, 200)
                pts = list(pts) + list(fpts)
                if cols is not None:
                    cols = list(cols) + [neutral] * len(fpts)
                if need_world and world_main is not None and world_f is not None:
                    world_main = list(world_main) + list(world_f)

        # --- Filter-Phase modal starten oder direkt schreiben ---
        if want_filter and pts and world_main:
            faces = _gcapture_all_guide_faces(bpy.context, None,
                                         guide_objs=guide_objs)
            self._flt_cams = [f[0] for f in faces]
            # GPU-Tiefenbilder (v105); ohne GPU-Kontext Ray-Cast wie bisher.
            self._flt_gpu = None
            if not bpy.app.background:
                try:
                    self._flt_gpu = _ExpGpuDepth(targets)
                    self._flt_views = [
                        (f[0], _gcapture_look_dir_for(s, f[0], f[1], f[2], f[3]))
                        for f in faces]
                    self._flt_fov = _sph_camera_min_fov(s, scene)
                    self._flt_np = np.array([tuple(p) for p in world_main],
                                            dtype=np.float64)
                    self._flt_vis = np.zeros(len(world_main), dtype=bool)
                    self._flt_cam_i = 0
                    # Compute-Shader (v111); sonst numpy je Kamera.
                    self._flt_compute = False
                    try:
                        self._flt_gpu.compute_begin(self._flt_np)
                        self._flt_compute = True
                    except Exception as exc:
                        print("[Gaussian Render Capture] GPU compute unavailable, "
                              "checking points with numpy:", exc)
                except Exception as exc:
                    print("[Gaussian Render Capture] GPU visibility unavailable, "
                          "using ray casting:", exc)
                    self._flt_gpu = None
            self._flt_bvh = (None if self._flt_gpu is not None
                             else _exp_build_scene_bvh(targets))
            self._flt_world = world_main
            self._flt_pts = pts
            self._flt_cols = cols
            self._flt_idx = 0
            self._flt_kept_pts = []
            self._flt_kept_cols = []
            self._flt_total = len(pts)
            self._phase = 'filter'
            self._filtered = True
            self._start_time = time.time()
            return {'RUNNING_MODAL'}

        # Kein Filter -> direkt schreiben.
        return self._write_and_done(context, pts, cols, cancelled=False)

    def _filter_tick(self, context):
        """Eine modale Filter-Etappe: pruefe einen Batch Punkte gegen die
        Kameras (Ray-Cast). Echtes ESC, weil im modalen Tick. Wenn alle
        Punkte geprueft sind -> schreiben."""
        if getattr(self, "_flt_gpu", None) is not None:
            return self._filter_tick_gpu(context)
        BATCH = 4000
        end = min(self._flt_idx + BATCH, self._flt_total)
        bvh = self._flt_bvh
        cams = self._flt_cams
        for i in range(self._flt_idx, end):
            wp = self._flt_world[i]
            if _exp_visible_ray(wp, bvh, cams):
                self._flt_kept_pts.append(self._flt_pts[i])
                if self._flt_cols is not None:
                    self._flt_kept_cols.append(self._flt_cols[i])
        self._flt_idx = end

        # Header.
        _gcapture_progress_set(
            "gcapture.export_colmap", self._flt_idx / max(self._flt_total, 1),
            "Visibility filter %d/%d points, %d kept"
            % (self._flt_idx, self._flt_total, len(self._flt_kept_pts)))

        if self._flt_idx >= self._flt_total:
            cols = self._flt_kept_cols if self._flt_cols is not None else None
            return self._write_and_done(context, self._flt_kept_pts, cols,
                                        cancelled=False)
        return {'RUNNING_MODAL'}

    def _filter_tick_gpu(self, context):
        """Filter-Etappe auf der GPU: ein paar Kameras je Tick (Zeitbudget),
        Punkte, die schon eine Kamera sieht, werden nicht mehr geprueft."""
        t0 = time.time()
        n_cams = len(self._flt_views)
        if getattr(self, "_flt_compute", False):
            while self._flt_cam_i < n_cams and time.time() - t0 < 0.1:
                pos, look = self._flt_views[self._flt_cam_i]
                self._flt_gpu.compute_camera(pos, look, self._flt_fov)
                self._flt_cam_i += 1
            if self._flt_cam_i >= n_cams:
                self._flt_vis = self._flt_gpu.compute_result()
        while self._flt_cam_i < n_cams and time.time() - t0 < 0.25:
            pos, look = self._flt_views[self._flt_cam_i]
            todo = np.nonzero(~self._flt_vis)[0]
            if len(todo) == 0:
                self._flt_cam_i = n_cams
                break
            seen = self._flt_gpu.visible(self._flt_np[todo], pos, look,
                                         self._flt_fov)
            self._flt_vis[todo[seen]] = True
            self._flt_cam_i += 1
        if getattr(self, "_flt_compute", False):
            ftext = "Visibility filter (GPU): camera %d/%d" % (
                self._flt_cam_i, n_cams)
        else:
            ftext = ("Visibility filter (GPU): camera %d/%d, %d of %d points "
                     "visible" % (self._flt_cam_i, n_cams,
                                  int(self._flt_vis.sum()), self._flt_total))
        _gcapture_progress_set("gcapture.export_colmap",
                               self._flt_cam_i / max(n_cams, 1), ftext)
        if self._flt_cam_i >= n_cams:
            keep = np.nonzero(self._flt_vis)[0]
            pts = [self._flt_pts[i] for i in keep]
            cols = ([self._flt_cols[i] for i in keep]
                    if self._flt_cols is not None else None)
            self._flt_gpu = None
            return self._write_and_done(context, pts, cols, cancelled=False)
        return {'RUNNING_MODAL'}

    def _write_and_done(self, context, pts, cols, cancelled=False):
        s = self._s
        # Keine Gesamt-Obergrenze mehr (v98): Max Points begrenzt nur die
        # Vertex-Punkte (_exp_gather_points), Face Points kommen obendrauf.

        if cancelled:
            self.report({'WARNING'},
                        "Cancelled; exporting the points collected so far.")

        points_path = os.path.join(self._sparse_dir, "points3D.txt")
        _exp_write_points(points_path, pts, cols)
        try:
            _exp_write_scale_info(self._out_dir, s, self._scale,
                                  getattr(self, "_mean_r", None),
                                  len(self._entries), len(pts))
        except OSError as exc:
            self.report({'WARNING'}, "<vNNN>_gcapture_export.json not written: %s" % exc)
        if getattr(self, "_sig", ""):
            s.exp_last_signature = self._sig
            s.exp_last_points_file = points_path

        self._cleanup(context)

        verb = {'MOVE': "moved", 'INPLACE': "already in the dataset"}.get(
            self._image_mode, "copied")
        # Finale Punktzahl merken und im Panel anzeigen (v103).
        n_v = getattr(self, "_n_vert", len(pts))
        n_f = getattr(self, "_n_face", 0)
        detail = "{:,} vertex".format(n_v)
        if n_f:
            detail += " + {:,} face".format(n_f)
        if getattr(self, "_filtered", False):
            detail += " points, {:,} hidden removed".format(
                max(0, n_v + n_f - len(pts)))
        else:
            detail += " points, no visibility filter"
        s.exp_last_points = len(pts)
        s.exp_last_detail = "{} ({})".format(
            os.path.basename(os.path.normpath(self._out_dir)), detail)
        msg = ("COLMAP export done: %d poses, %dx%d (%s), %d images %s, "
               "{:,} points ({}), scale {:.6g} -> %s".format(
                   len(pts), detail, self._scale)
               % (len(self._entries), self._res[0], self._res[1],
                  "image file" if self._real_size is not None else "render settings",
                  self._copied, verb, self._out_dir))
        # Verschieben war angefordert, aber wegen fehlender Bilder auf
        # Kopieren zurueckgefallen (Sicherheitscheck) -> klar melden.
        if getattr(self, "_move_blocked", False):
            msg += (" | NOTE: Move was requested but some images were "
                    "missing, so images were COPIED instead (render folder "
                    "left intact)")
        # Warnung: alle Posen wurden geschrieben, aber fuer einige Frames
        # fehlt (noch) das Bild im Quellordner (z. B. Renderfarm nicht fertig
        # synchronisiert). Die Posen sind trotzdem im Export -- die Bilder
        # koennen spaeter ergaenzt werden.
        miss = getattr(self, "_missing_images", []) or self._missing
        if miss:
            ex = miss[0]
            ex = ex if isinstance(ex, str) else ("frame %d" % ex)
            msg += (" | WARNING: %d source image(s) missing (e.g. %s) -- "
                    "poses exported anyway, add the images later"
                    % (len(miss), ex))
            level = {'WARNING'}
        else:
            level = {'INFO'}
        self.report(level, msg)
        return {'FINISHED'}

    def _cleanup(self, context):
        if getattr(self, "_timer", None):
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        _gcapture_progress_clear("gcapture.export_colmap")
        try:
            context.scene.frame_set(self._orig_frame)
        except Exception:
            pass
        GCAPTURE_OT_export_colmap._is_running = False


# ----------------------------------------------------------------------
# Registrierung
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
    GCAPTURE_PT_advanced,
)


def register():
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
    # Beim Aktivieren ist bpy.data gesperrt -> kurz danach uebernehmen.
    bpy.app.timers.register(_gcapture_migrate_timer, first_interval=0.1)


def unregister():
    # Live-Lock-Handler sicher entfernen.
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


if __name__ == "__main__":
    register()
