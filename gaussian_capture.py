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

Code comments are in English.
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
    """Called when interactive parameters change (e.g. Focal Length).
    Applies the focal length to the (already existing) camera immediately
    -- regardless of live mode, since the focal length only affects the
    camera data block and does not require re-baking the poses."""
    cam = bpy.data.objects.get(settings.camera_name)
    if cam and cam.type == 'CAMERA':
        cam.data.lens_unit = 'MILLIMETERS'
        cam.data.lens = settings.focal_length


# ----------------------------------------------------------------------
# Settings (editable in the N panel, stored in the .blend)
# ----------------------------------------------------------------------
def _gcapture_on_guide_index_change(self, context):
    """Called when an entry in the guide list is selected. Selects the
    objects belonging to the entry in the Outliner/viewport and makes
    the first one the active object."""
    s = context.scene.gcapture_settings
    if not (0 <= s.guide_index < len(s.guides)):
        return
    item = s.guides[s.guide_index]
    objs = [r.obj for r in item.objects if r.obj]
    if not objs:
        return
    try:
        # Clear the existing selection, then select the entry's objects.
        for o in context.view_layer.objects:
            o.select_set(False)
        for o in objs:
            try:
                o.select_set(True)
            except RuntimeError:
                pass  # object may be in a hidden collection
        context.view_layer.objects.active = objs[0]
    except Exception:
        pass


class GCAPTURE_GuideObjRef(PropertyGroup):
    """Reference to a single mesh object within a guide entry."""
    obj: PointerProperty(
        name="Object",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'MESH',
    )


class GCAPTURE_GuideItem(PropertyGroup):
    """A guide mesh entry: editable label + one or more mesh objects.
    The label is purely a display alias; the real object names stay
    unchanged."""
    label: StringProperty(
        name="Label",
        description="Display name for this guide entry (alias only -- does "
                    "not rename the actual objects)",
        default="Guide",
    )
    objects: CollectionProperty(type=GCAPTURE_GuideObjRef)


class GCAPTURE_CollItem(PropertyGroup):
    """An entry in the target collection list for auto-scaling the
    camera sphere."""
    coll: PointerProperty(
        name="Collection",
        type=bpy.types.Collection,
    )


# Allow relative paths (//...) for folder properties (v107). The option
# only exists from Blender 4.5 on; older versions would otherwise refuse
# to register the whole add-on (v110 fix).
_GCAPTURE_PATH_OPTIONS = ({'PATH_SUPPORTS_BLEND_RELATIVE'}
                     if bpy.app.version >= (4, 5, 0) else set())


def _gcapture_on_resolution(settings, context):
    """Resolution changed: apply it and remember it as confirmed (v140)."""
    context.scene["gcapture_res_ok"] = True
    _gcapture_on_render_setting(settings, context)


def _gcapture_on_render_setting(settings, context):
    """Apply a change to a render setting immediately (v127)."""
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
    # Paths last entered by the add-on (v129): if a field matches them,
    # it keeps following automatically.
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
        default='MATERIAL',
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
        name="Command Line Render",
        description="Headless: instead of rendering here, the buttons save the "
                    "scene and write a double-click script next to it. The "
                    "script renders all cameras without Blender's interface - "
                    "Blender stays free, and the render goes on when Blender "
                    "is closed",
        default=False,
    )
    render_format: EnumProperty(
        name="Image Format",
        description="File format of the rendered images. Both are read by "
                    "Postshot and LichtFeld Studio",
        items=[('PNG', "PNG", "PNG, RGBA 8 bit"),
               ('TIFF', "TIFF", "TIFF, RGBA 8 bit, Deflate compression - "
                "smaller files, same images")],
        default='PNG',
        # Applied at once, so Blender's Output settings show the choice (1.2.0);
        # the render buttons apply it again in case it was changed there.
        update=lambda self, context: _gcapture_apply_image_format(
            context.scene, self.render_format),
    )
    clean_splat_file: StringProperty(
        name="Splat File",
        description="The trained splat (.ply from LichtFeld Studio or Postshot) "
                    "of this scene version. The folder button next to it opens "
                    "the dataset folder",
        # No FILE_PATH subtype: it adds Blender's own file button next to the
        # add-on's, which opens in the dataset folder (1.2.0).
        default="",
        update=lambda self, context: setattr(self, "clean_last_result", ""),
    )
    clean_min_views: IntProperty(
        name="Min. Views",
        description="A splat is removed when it lies in front of the model's "
                    "surface in at least this many cameras (splats no camera "
                    "sees are always removed)",
        default=2, min=1, max=20,
    )
    clean_last_result: StringProperty(options={'HIDDEN'})
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


# ----------------------------------------------------------------------
# Live Camera Lock: binds ONE camera (active frame -> face index) live
# to the guide mesh. Follows scaling/movement immediately without baking
# the whole animation. Lightweight handler (one pose instead of many).
# ----------------------------------------------------------------------
import bpy.app.handlers as _gcapture_handlers

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


# ----------------------------------------------------------------------
# Camera sphere generator: helpers
# ----------------------------------------------------------------------
def _sph_apply_render_settings(scene):
    """Sets the render settings for Gaussian splatting captures once.
    Each setting defensively (try/except), so that a property path that
    differs in this Blender version does not abort the whole operator --
    the others are set anyway. Returns a list of the items that could not
    be set (for a notice to the user)."""
    skipped = []

    def tryset(setter, label):
        try:
            setter()
        except Exception:
            skipped.append(label)

    # Engine to Cycles + GPU.
    tryset(lambda: setattr(scene.render, "engine", 'CYCLES'),
           "Cycles engine")
    tryset(lambda: setattr(scene.cycles, "device", 'GPU'),
           "Cycles GPU device")

    # Denoiser: enable Render + Viewport.
    tryset(lambda: setattr(scene.cycles, "use_denoising", True),
           "Render denoiser")
    tryset(lambda: setattr(scene.cycles, "use_preview_denoising", True),
           "Viewport denoiser")
    # Denoiser TYPE to OpenImageDenoise (OIDN) -- Render + Viewport.
    tryset(lambda: setattr(scene.cycles, "denoiser", 'OPENIMAGEDENOISE'),
           "Render denoiser = OpenImageDenoise")
    tryset(lambda: setattr(scene.cycles, "preview_denoiser", 'OPENIMAGEDENOISE'),
           "Viewport denoiser = OpenImageDenoise")
    # Denoising device on GPU (Blender 4.2+: the denoiser can use the GPU,
    # even when rendering). The property name varies by version, so
    # try several variants defensively.
    tryset(lambda: setattr(scene.cycles, "denoising_use_gpu", True),
           "Render denoiser GPU")
    tryset(lambda: setattr(scene.cycles, "preview_denoising_use_gpu", True),
           "Viewport denoiser GPU")

    # Samples (Render). Viewport samples are left untouched.
    tryset(lambda: setattr(scene.cycles, "samples", 512),
           "Render samples 512")

    # Film: Transparent + Transparent Glass.
    tryset(lambda: setattr(scene.render, "film_transparent", True),
           "Film transparent")
    tryset(lambda: setattr(scene.cycles, "film_transparent_glass", True),
           "Film transparent glass")

    # World background: Color Value to 1 (white).
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
    """Sets clip start/end of the configured camera (for splatting:
    start small, end very large)."""
    cam = bpy.data.objects.get(s.camera_name)
    if cam is None or cam.type != 'CAMERA':
        cam = bpy.context.scene.camera
    if cam is not None and cam.type == 'CAMERA':
        cam.data.clip_start = 0.01
        cam.data.clip_end = 10000.0
        return True
    return False


def _sph_collect_objects(s, exclude=None):
    """All VISIBLE mesh objects from the selected target collections
    (recursively incl. child collections). 'exclude' (object) is skipped
    -- important so that the camera sphere itself does NOT enter the size
    calculation. Only visible objects count (visible_get): this way hidden
    helper objects (ground planes, rigs, helper meshes) do not inflate the
    bounding box -- that was the reason why the sphere became far too
    large for models made of many parts."""
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
    """(min_vec, max_vec, center, max_dim, biggest) of the world-space bounding
    box over all objects. max_dim is the largest axis extent.
    'biggest' = (name, dim) of the object with the largest single diagonal
    (for diagnosing which part inflates the bounds). Reads the bound_box
    from the EVALUATED object (incl. modifiers)."""
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
        # Individual extent of this object.
        odim = max(omx.x - omn.x, omx.y - omn.y, omx.z - omn.z)
        if odim > biggest_dim:
            biggest_dim = odim
            biggest_name = o.name
    if not found:
        return None
    center = (mn + mx) * 0.5
    max_dim = max(mx.x - mn.x, mx.y - mn.y, mx.z - mn.z)
    # Space diagonal of the overall bounding box: the largest dimension a
    # camera can see from ANY direction. As the reference for the radius it
    # makes the framing shape-independent -- compact (cube) and elongated
    # (car) objects are framed equally well with the same margin.
    diag = math.sqrt((mx.x - mn.x) ** 2 + (mx.y - mn.y) ** 2 +
                     (mx.z - mn.z) ** 2)
    return (mn, mx, center, max_dim, diag, (biggest_name, biggest_dim))


def _sph_camera_min_fov(s, scene):
    """Smaller of the two FOV axes (horizontal/vertical) in radians --
    the limiting axis for 'object fully in frame'. Based on the focal
    length set in the add-on. Sensor width 36mm.

    IMPORTANT: a SQUARE aspect ratio (1:1) is DELIBERATELY assumed, NOT
    the currently set render aspect ratio. Reason: the build renders
    square for splatting (set_resolution sets res_x = res_y). If the
    generator used the current aspect ratio, the radius would change as
    soon as the build switches the resolution to square -- the sphere
    would then shrink/grow once on the next creation. With a fixed 1:1
    the generator is consistent with the render result from the start
    and stays stable."""
    lens = max(s.focal_length, 1e-3)
    sensor_w = 36.0
    # Square: fov_h == fov_v, so one axis is enough.
    fov = 2.0 * math.atan((sensor_w / 2.0) / lens)
    return fov


# Mesh attribute of the Camera Sphere (face domain): factor by which the
# camera moves toward the center per face (Fill Each View, v96).
_SPH_FIT_ATTR = "gcapture_fit"
_SPH_FIT_MIN = 0.6   # no camera closer than 60 % of the sphere distance
_SPH_SHIFT_ATTR = "gcapture_shift"   # lateral offset per camera (v135)


def _sph_model_points(objs):
    """All vertices of the objects in world coordinates as an (N,3) array."""
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
    """Per view direction, the distance from the center at which ALL points
    fit into the (square) image with margin. dirs: unit vectors from the
    center to the camera; the camera looks at the center (-dir). For a
    point p with rel = p - center, the camera image requires |rel.right| <=
    t * depth and |rel.up| <= t * depth with depth = d - rel.u and
    t = tan(fov/2) / margin -> d >= rel.u + max(|rel.right|, |rel.up|) / t."""
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
        for k in range(0, len(rel), 500000):   # memory-saving
            blk = rel[k:k + 500000]
            du = blk @ np.array(u)
            lat = np.maximum(np.abs(blk @ np.array(r)),
                             np.abs(blk @ np.array(up)))
            best = max(best, float((du + lat / tt).max()))
        out.append(best + safety)
    return out


def _sph_framed_positions(points, center, dirs, fov, margin):
    """Per view direction (unit vector center -> camera) the camera position
    at which all points fit centered into the square image with margin,
    without changing the view direction (v135). Returns per direction
    (distance along dir, lateral offset as Vector, relative to the center).
    Per image axis: a = rel.axis, z = rel.forward, M = max(a - t z),
    m = min(a + t z); camera p_a = (m + M) / 2, p_f <= (m - M) / (2 t)."""
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
        for k in range(0, len(rel), 500000):   # memory-saving
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
    """Adjust the offset so that the model sits centered in both image axes
    at the final distance (after the 60 % limit) (v135). The closed-form
    solution centers only the tight axis exactly; due to perspective the
    other axis needs a short Newton iteration: center of the image span ->
    move the camera by center * t * depth."""
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
    """Reduces large point sets for the distance calculation without missing
    a result: per grid column (x/y cell) only the lowest and the highest
    point count -- every point in between lies on their connecting segment
    and can never lie further out for linear constraints. x/y are set to
    the cell center; the maximum resulting offset (half the cell diagonal)
    is returned as a safety margin. Small sets stay unchanged (returns 0.0)."""
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
    """Sphere radius at which an object with the largest extent max_dim
    (with margin) fits completely into the image: d = (max_dim*margin/2)/tan(fov/2).
    This way the object is fully captured from EVERY direction, because
    the sizing is based on the longest extent."""
    half = (max_dim * margin) / 2.0
    t = math.tan(fov / 2.0)
    if t < 1e-6:
        return half  # guard against division by ~0
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

        # Determine the existing sphere FIRST (before the bounds calculation),
        # so that it can be excluded from the size calculation.
        # PRIMARILY via the stored reference (sph_object); name as
        # fallback (a build could have renamed it).
        existing = s.sph_object
        if existing is None or existing.name not in bpy.data.objects:
            existing = bpy.data.objects.get(self.SPHERE_NAME)
        # Sphere deleted in the viewport: Blender only removes it from the
        # scene; the stored reference (sph_object) keeps the object alive --
        # without a collection it cannot be selected (v94).
        # If it is in no scene anymore: remove it for good and recreate it;
        # if it is only in another scene: leave it there.
        if existing is not None and existing.name not in context.scene.objects:
            if not existing.users_scene:
                if s.sph_object == existing:
                    s.sph_object = None
                bpy.data.objects.remove(existing, do_unlink=True)
            existing = None

        # Target objects + bounds -- exclude the sphere itself, otherwise
        # the radius grows with every update (accumulated scaling).
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

        # Radius from FOV + SPACE DIAGONAL + margin. The diagonal (instead of
        # the longest axis) makes the framing shape-independent: compact and
        # elongated objects are framed equally well with the same margin.
        fov = _sph_camera_min_fov(s, scene)
        radius = _sph_required_radius(diag, fov, s.sph_margin)
        fit_msg = ""

        # Radius into the vertices, center into the object position (v95): the
        # origin lies at the sphere center so that scaling with S acts around
        # the center (Live Camera Lock).
        mw = Matrix.Scale(radius, 4)

        # Build the sphere mesh in ONE pass: create a unit ico sphere,
        # optionally remove the lower half, bake the vertices directly with mw
        # into world coordinates -- ALL in the bmesh, BEFORE it is assigned to
        # the object. This way there is no read-after-assign and no dependency
        # on update timing (that was the source of errors that made the
        # sphere appear to shrink when reused).
        new_mesh = bpy.data.meshes.new(self.SPHERE_NAME)
        bm = _bm.new()
        _bm.ops.create_icosphere(bm, subdivisions=s.sph_subdivisions,
                                 radius=1.0)
        if s.sph_upper_only:
            bm.verts.ensure_lookup_table()
            to_del = [v for v in bm.verts if v.co.z < -1e-5]
            if to_del:
                _bm.ops.delete(bm, geom=to_del, context='VERTS')

        # Calibrate the framing to the sequence (v96): per face (= camera)
        # compute the required distance from the real model points.
        fit_q = None
        shifts = None
        if s.sph_fit_each:
            pts = _sph_model_points(objs)
            if pts is not None:
                bm.faces.ensure_lookup_table()
                unit_c = [f.calc_center_median() for f in bm.faces]
                dirs = [c.normalized() for c in unit_c]
                # At least ~1 pixel of clearance to the image border (at margin 1.0
                # the model otherwise touches the border exactly -> rounding).
                px_guard = 1.0 + 2.0 / max(s.resolution, 16)
                shifts = None
                # With Look Target "Origin" the camera no longer aimed parallel
                # after the offset -> no offset there.
                if s.sph_center_view and s.look_mode != 'ORIGIN':
                    framed = _sph_framed_positions(pts, center, dirs, fov,
                                                   s.sph_margin * px_guard)
                    need = [d for d, _ in framed]
                    shifts = [sv for _, sv in framed]
                else:
                    need = _sph_required_distances(pts, center, dirs, fov,
                                                   s.sph_margin * px_guard)
                # Camera sits at the face center: distance = radius * |c_unit|.
                ratios = [d / max(c.length, 1e-9)
                          for d, c in zip(need, unit_c)]
                limit = max(range(len(ratios)), key=ratios.__getitem__)
                radius = ratios[limit]
                fit_q = [max(_SPH_FIT_MIN, min(1.0, rr / radius))
                         for rr in ratios]
                if shifts is not None:
                    # Final distance per camera (after the limit)
                    # and center on it (v135).
                    finals = [q * radius * c.length
                              for q, c in zip(fit_q, unit_c)]
                    shifts = _sph_center_shifts(pts, center, dirs, fov,
                                                finals, shifts)
                fit_msg = " Each view filled%s; widest view: camera %d." % (
                    " and centred" if shifts is not None else "", limit + 1)
        # Bake vertices into object coordinates (unit -> radius*v). Build the
        # matrix only here: Fill Each View can change the radius above.
        mw = Matrix.Scale(radius, 4)
        for v in bm.verts:
            v.co = mw @ v.co
        bm.to_mesh(new_mesh)
        bm.free()
        if fit_q is not None and len(fit_q) == len(new_mesh.polygons):
            attr = new_mesh.attributes.new(_SPH_FIT_ATTR, 'FLOAT', 'FACE')
            attr.data.foreach_set("value", fit_q)
            # Lateral offset per camera (v135). The sphere has no
            # rotation/scale -> world offset = object offset.
            if (s.sph_fit_each and s.sph_center_view and shifts is not None
                    and len(shifts) == len(new_mesh.polygons)):
                sattr = new_mesh.attributes.new(_SPH_SHIFT_ATTR,
                                                'FLOAT_VECTOR', 'FACE')
                sattr.data.foreach_set(
                    "vector", [c for v in shifts for c in (v.x, v.y, v.z)])

        # Reuse the existing sphere (replace the mesh) or create a new one.
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

        _gcapture_move_to_rig(context.scene, sphere_obj)   # dedicated collection (v136)
        # Object transform: only the location (center); the scale
        # is baked into the vertices.
        sphere_obj.matrix_world = Matrix.Translation(center)
        # Placement after Create / Update (v133): Build resets the sphere to
        # this and thereby discards Live Camera Adjust changes.
        sphere_obj["gcapture_base_loc"] = tuple(center)
        # Remember the center (object coordinates, i.e. the origin):
        # target point of the cameras in CENTER mode, also for the hemisphere.
        sphere_obj[_SPH_CENTER_PROP] = (0.0, 0.0, 0.0)

        # Display as wireframe, do not render.
        sphere_obj.display_type = 'WIRE'
        sphere_obj.hide_render = True

        n_faces = len(sphere_obj.data.polygons)

        # Store a reference to the created sphere so that a later
        # Update finds it again even after a rename (by Build).
        s.sph_object = sphere_obj

        # Set as guide mesh -- FIXED in the guide list, not via the
        # active object. Otherwise Build takes the currently active object (e.g.
        # the cube), builds the cameras on ITS faces and the camera sticks
        # to the object surface instead of sitting on the sphere.
        if s.sph_set_as_guide:
            # Clear the guide list and add only the sphere.
            s.guides.clear()
            item = s.guides.add()
            item.label = sphere_obj.name
            ref = item.objects.add()
            ref.obj = sphere_obj
            s.guide_index = 0
            s.use_guide_list = True
            # Select/activate the sphere (cosmetic).
            for o in context.selected_objects:
                o.select_set(False)
            sphere_obj.select_set(True)
            context.view_layer.objects.active = sphere_obj
            # Look mode to center (the sphere looks inward).
            s.look_mode = 'CENTER'
            # Prevent renaming during Build: the sphere should always be named
            # "Camera_Sphere" so that Update always finds it again.
            s.rename_guide = False

        # Render settings for splatting: set ONLY ONCE per file.
        # Later updates leave the settings (possibly adjusted by the user)
        # untouched.
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
        # Older sphere: origin to the center so that S scales around the center.
        _gcapture_sphere_origin_to_center(sph)
        # Add the placement after Create / Update so that Build resets G/S.
        _sph_ensure_base(sph, s)
        # Select the sphere as the only object and make it active.
        for o in context.selected_objects:
            o.select_set(False)
        sph.select_set(True)
        context.view_layer.objects.active = sph
        # Enable Lock (freezes the face of the current frame).
        s.live_lock = True

        # No automatic scaling (v97): confusing for beginners.
        # The sphere is selected; the user presses S themselves.
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
            # Reset the sphere and re-place the camera of the frozen frame
            # while Lock is still active (keyframe), then Lock off (v132).
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
            # First turn Lock off (handler gone), then save the pose and reset
            # the sphere -- otherwise the handler would drag the camera along.
            idx = _GCAPTURE_LOCK_FROZEN_IDX
            guides = [g for g in _GCAPTURE_LOCK_GUIDES if g is not None]
            s.live_lock = False
            msg = self._keep_this_view(idx, guides)
            self.report({'INFO'}, msg)
            # As after a Build: save a version (v128).
            if (s.save_after_build and not bpy.app.background
                    and getattr(context, "window", None)):
                try:
                    bpy.ops.gcapture.save_version('INVOKE_DEFAULT')
                except Exception as exc:
                    print("[Gaussian Render Capture] Save prompt failed:", exc)
            return {'FINISHED'}
        # Turn Lock off (triggers _gcapture_on_lock_toggle -> handler gone,
        # frozen index reset).
        s.live_lock = False
        # Full recalculation via the modal Build operator.
        bpy.ops.gcapture.build_animation('INVOKE_DEFAULT', keep_adjustments=True)
        self.report({'INFO'}, "Live adjust finished -> rebuilding all poses.")
        return {'FINISHED'}

    @staticmethod
    def _keep_this_view(idx, guides):
        """Save the pose of the frozen camera at the face and reset the guide
        meshes to their placement at the time Lock was enabled (v123)."""
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

    # Only "All Cameras" (Live Camera Adjust) rebuilds with the modified
    # sphere; otherwise fresh from step 3 (v133).
    keep_adjustments: BoolProperty(default=False, options={'SKIP_SAVE', 'HIDDEN'})

    _is_running = False

    @classmethod
    def poll(cls, context):
        if cls._is_running:
            return False
        s = context.scene.gcapture_settings
        # Blocked while Live Camera Lock is active -- run Finish Live
        # Adjust first, otherwise the live-adjusted keyframes would be
        # overwritten.
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

        # Display guide meshes as wireframe, exclude them from the render.
        for gobj in self._guide_objs:
            setup_guide_object(gobj, s)

        self._cam = get_or_create_camera(s.camera_name)
        self._cam.data.lens_unit = 'MILLIMETERS'
        self._cam.data.lens = s.focal_length
        if self._cam.animation_data:
            self._cam.animation_data_clear()
        self._cam.rotation_mode = 'QUATERNION'

        # Collect all faces of all guide meshes as a flat work list.
        self._faces = []
        for gobj in self._guide_objs:
            self._faces.extend(_guide_faces_with_targets(gobj))
        self._total = len(self._faces)
        if self._total == 0:
            raise RuntimeError("Guides have no faces.")

        # Viewport display size of the camera to match the sphere (v1.1.5):
        # Blender's 1 m hides small models. Only as long as the user has not
        # set it themselves (Blender default or set by us).
        cam_data = self._cam.data
        if (abs(cam_data.display_size - 1.0) < 1e-6
                or cam_data.get("gcapture_display_auto")):
            radius = max((max(g.dimensions) * 0.5 for g in self._guide_objs),
                         default=0.0)
            if radius > 0.0:
                cam_data.display_size = max(radius * 0.1, 0.001)
                cam_data["gcapture_display_auto"] = True

        # Prepare interior detection (once). Only effective with custom
        # guide meshes: the camera sphere always lies outside the
        # model, where the test would only slow things down (v93).
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
        # Warning when Live Camera Adjust changes were lost (v133).
        s = context.scene.gcapture_settings
        if s.sph_object is not None:
            _sph_ensure_base(s.sph_object, s)     # older spheres (v134)
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

            # Batch size: without interior check poses are very fast ->
            # large batches; with interior check (ray cast) smaller ones.
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
            # Switch to camera view (equivalent to Numpad 0) -- in all
            # 3D viewports. region_3d.view_perspective is the reliable
            # API way to do this.
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
        # Prompt to save the scene as a version (v90). Only with a
        # window (not in the headless test).
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
# UIList for the guide entries (editable label + object count)
# ----------------------------------------------------------------------
class GCAPTURE_UL_guides(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon,
                  active_data, active_propname, index):
        n = len(item.objects)
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            # Editable label (double-click to rename).
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
# Custom icons: colored number badges for the steps (v87)
# ----------------------------------------------------------------------
# The addon stays a single file: the badges are drawn with numpy on
# registration, written as PNG to a cache folder and loaded via
# bpy.utils.previews. No bpy.data access (restricted at startup),
# PNG encoding via zlib.
import struct as _gcapture_struct
import tempfile as _gcapture_tempfile
import zlib as _gcapture_zlib

# Phase colors as in the workflow graphic: orange -> yellow -> green -> blue (sRGB 0..1).
_GCAPTURE_BADGE_RGB = {
    'PREP': (0.93, 0.50, 0.13),      # orange: Prepare
    'CAMERAS': (0.95, 0.80, 0.16),   # yellow: Cameras
    'OUTPUT': (0.33, 0.70, 0.28),    # green:  Render
    'COLMAP': (0.23, 0.65, 0.79),    # blue:   COLMAP export (v146)
    'SPLAT': (0.62, 0.45, 0.85),     # violet: after training (1.2.0)
    'NEUTRAL': (0.92, 0.92, 0.92),   # white:  no phase (Camera Settings)
}
# Digits as stroke paths in a unit square (x right, y up).
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
    # 8 (1.2.0, step 8 Clean Splat): two closed loops, the upper one smaller.
    "8": [[(0.50 + 0.40 * math.cos(a * math.pi / 8.0),
            0.25 + 0.24 * math.sin(a * math.pi / 8.0)) for a in range(17)],
          [(0.50 + 0.33 * math.cos(a * math.pi / 8.0),
            0.74 + 0.22 * math.sin(a * math.pi / 8.0)) for a in range(17)]],
}
# Badges: name -> (phase, digit; None = dot without number, "" = empty).
_GCAPTURE_BADGES = {
    'gcapture_cam': ('PREP', "1"),        # Start = step 1 (v147)
    'gcapture_1': ('PREP', "2"),
    'gcapture_2': ('PREP', "3"),
    'gcapture_3': ('CAMERAS', "4"),
    'gcapture_4': ('CAMERAS', "5"),
    'gcapture_5': ('OUTPUT', "6"),
    'gcapture_6': ('COLMAP', "7"),
    'gcapture_7': ('SPLAT', "8"),
}
_GCAPTURE_ICON_VERSION = 8   # increase when the appearance changes (cache folder)
_gcapture_previews = None


def _gcapture_badge_rgba(rgb, digit, size=64):
    """RGBA array (size, size, 4) uint8: filled circle in the phase color,
    with the digit on top in bold dark gray (or a small dot).
    Analytic anti-aliasing via the distance."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64) + 0.5
    x = xx / size
    y = 1.0 - yy / size
    px = 1.0 / size
    d_disc = np.hypot(x - 0.5, y - 0.5) - 0.47
    a_disc = np.clip(0.5 - d_disc / px, 0.0, 1.0)

    if digit:
        # Digit box: height 0.56, width 0.40, centered.
        bx0, by0, bw, bh = 0.30, 0.22, 0.40, 0.56
        half_w = 0.052   # half stroke width -> bold
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
        ink = np.zeros_like(x)   # empty badge

    ink_rgb = np.array((0.12, 0.12, 0.12))
    base = np.array(rgb)
    col = (base[None, None, :] * (1.0 - ink[..., None]) +
           ink_rgb[None, None, :] * ink[..., None])
    rgba = np.dstack([col, a_disc])
    return (np.clip(rgba, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _gcapture_write_png(path, rgba):
    """Minimal PNG writer (RGBA, 8 bit) without an extra library."""
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
    """Create the badges (once per icon version) and load them. If something
    fails, _gcapture_previews stays None -> panels fall back to Blender
    icons."""
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


# Colors of the work phases (icons of the collection color tags). Blender
# does not allow add-ons colored boxes -- only icons are colored. Red
# (COLLECTION_COLOR_01) stays reserved for warnings (alert).
# Order like a traffic light: orange -> yellow -> green (v86).
_GCAPTURE_COLOR_PREP = 'COLLECTION_COLOR_02'     # orange: Prepare
_GCAPTURE_COLOR_CAMERAS = 'COLLECTION_COLOR_03'  # yellow: Cameras
_GCAPTURE_COLOR_OUTPUT = 'COLLECTION_COLOR_04'   # green:  Render
_GCAPTURE_COLOR_COLMAP = 'COLLECTION_COLOR_05'   # blue:   COLMAP export (v146)
_GCAPTURE_COLOR_SPLAT = 'COLLECTION_COLOR_06'    # violet: Clean Splat (1.2.0)


# ----------------------------------------------------------------------
# Beginner aids (v90): selection into a collection, scene versions
# ----------------------------------------------------------------------
def _gcapture_version_files(folder, name):
    """Existing version numbers of <name>_vNNN*.blend in the folder."""
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
    """Path of the next free version <name>_vNNN.blend in the folder."""
    nums = _gcapture_version_files(folder, name)
    nxt = (max(nums) + 1) if nums else 1
    return os.path.join(folder, "%s_v%0*d.blend" % (name, width, nxt))


def _gcapture_clean_name(raw):
    """Scene name without extension, version token and invalid characters."""
    stem = os.path.splitext(os.path.basename(raw or ""))[0]
    name, _ = _exp_split_name_version(stem)
    name = re.sub(r'[\\/:*?"<>|\s]+', "_", name).strip("_.- ")
    return name or "Scene"


def _gcapture_managed_render_path(name, vtag):
    """Relative render path directly into the dataset:
    //<Name>_COLMAP/<vNNN>/images/<Name>_<vNNN>_"""
    return "//%s_COLMAP/%s/images/%s_%s_" % (name, vtag, name, vtag)


from bpy.app.handlers import persistent

_GCAPTURE_MANAGED_RX = re.compile(
    r"^//(?P<n>[^/]+)_COLMAP/(?P<v>[vV]\d+)/images/(?P=n)_(?P=v)_$")


def _gcapture_sync_render_version(filepath):
    """Sets a render path managed by the addon to the file's name and
    version (v119). Returns: number of changed scenes."""
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
    """True if the render path is Blender's default or is managed by the
    addon -- then Save Scene Version may set it (v129)."""
    rp = (scene.render.filepath or "").replace("\\", "/").strip()
    return (rp in ("", "/tmp", "/tmp/", "//")
            or bool(_GCAPTURE_MANAGED_RX.match(rp)))


def _gcapture_auto_output_dir(filepath):
    """//<Name>_COLMAP/<vNNN>/ for the file (like _exp_resolve_output_dir)."""
    if not filepath:
        return ""
    stem = os.path.splitext(os.path.basename(filepath))[0]
    name, vtag = _exp_split_name_version(stem)
    return "//%s_COLMAP/%s/" % (name or "Scene", vtag or "v001")


def _gcapture_auto_image_dir(scene):
    """Folder of the render path as it is set in the scene (relative stays
    relative)."""
    rp = (scene.render.filepath or "").replace("\\", "/")
    if not rp.strip():
        return ""
    return rp if rp.endswith("/") else rp.rsplit("/", 1)[0] + "/"


def _gcapture_sync_path_fields(scene, filepath):
    """Output Folder and Image Folder show the actual paths and
    follow version and render path as long as they hold the value entered
    by the addon (v129). Unsaved scenes stay empty (= automatic)."""
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
    """Apply changes to the render path to Image Folder immediately (v129).
    Must be re-subscribed after every load."""
    bpy.msgbus.clear_by_owner(_GCAPTURE_MSGBUS_OWNER)
    bpy.msgbus.subscribe_rna(key=(bpy.types.RenderSettings, "filepath"),
                             owner=_GCAPTURE_MSGBUS_OWNER, args=(),
                             notify=_gcapture_on_render_path_change)


# Migration from "Gaussian Render Scan" (up to 1.0.2, 23.09.2026): the same
# settings and markers under the new name. Runs after every
# load and on activation; without legacy data it changes nothing.
_GCAPTURE_LEGACY_KEYS = (("gscan_prepared", "gcapture_prepared"),
                         ("gscan_res_ok", "gcapture_res_ok"),
                         ("gscan_base_loc", "gcapture_base_loc"),
                         ("gscan_center", "gcapture_center"))
_GCAPTURE_LEGACY_ATTRS = ("fit", "shift", "adj", "adj_pos", "adj_tgt")
_GCAPTURE_LEGACY_NAMES = (("GScan_Group", "GCapture_Group", "objects"),
                          ("Scan_Rig", "Capture_Rig", "collections"))


def _gcapture_migrate_legacy():
    """Migrate scenes from Gaussian Render Scan. From Blender 5.0 the settings
    live in the system properties, before that in the scene's dict.
    Returns: number of migrated entries."""
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
    # Object and collection names only if the file comes from the old add-on.
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
    _GCAPTURE_PROGRESS.clear()   # a new file has no running operation
    try:
        _gcapture_migrate_legacy()
    except Exception as exc:
        print("[Gaussian Render Capture] migration:", exc)
    _gcapture_sync_all(bpy.data.filepath)
    _gcapture_msgbus_subscribe()


@persistent
def _gcapture_on_save_pre(*args):
    # Blender passes the target path (Save As); otherwise the current one.
    target = next((a for a in args if isinstance(a, str) and a), bpy.data.filepath)
    _gcapture_sync_all(target)


# Prepare the scene (v113): values from the user's template (Setup.blend,
# 23.09.2026). Path relative to the scene -> value. Anything a Blender version
# does not know is skipped.
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

# EEVEE preset for captures (v1.1.5): as view-independent and as close to
# Cycles as possible -- raytracing and Fast GI at full resolution, soft
# shadows with many rays, overscan against edge artifacts, more samples.
# First version; to be calibrated by measurement (Cycles vs. EEVEE, Beetle).
# If the table changes, increase _GCAPTURE_EEVEE_PRESET: it is then
# applied again on the next EEVEE render, otherwise the user's changes
# are kept.
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
    """Apply a (path, value) table to the scene; returns: skipped
    paths (unknown in this Blender version)."""
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
    """Cycles settings of Prepare Scene. Blender 4.1: OpenImageDenoise
    enumerates all SYCL devices at startup and fails with current
    Intel drivers (PI_ERROR_INVALID_VALUE) -- even on the CPU. If Cycles
    renders with OptiX there, its denoiser does the denoising (v1.1.5)."""
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
    """Identifier of EEVEE in this Blender version (4.2-4.x: EEVEE Next)."""
    items = {i.identifier for i in
             bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items}
    return 'BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in items else 'BLENDER_EEVEE'


# Order of the GPU backends: the first one with a GPU wins.
_GCAPTURE_GPU_TYPES = ('OPTIX', 'CUDA', 'HIP', 'METAL', 'ONEAPI')

# Start objects of the Blender default scene: (name, type).
_GCAPTURE_START_OBJECTS = (("Cube", 'MESH'), ("Camera", 'CAMERA'), ("Light", 'LIGHT'))


def _gcapture_setup_gpu():
    """Cycles settings: choose the best GPU backend, its GPUs on,
    CPU off. Returns (backend, [device names]) or (None, [])."""
    try:
        cp = bpy.context.preferences.addons["cycles"].preferences
    except (KeyError, AttributeError):
        return None, []
    # No refresh_devices(): it queries all backends, and Blender 4.1
    # crashes with current Intel drivers in the oneAPI backend (sycl6.dll).
    # get_devices_for_type queries only the respective backend.
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
    """Only the unmodified start object: cube with 8 vertices, camera and
    light at their start positions."""
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
    """Prepare Scene ran in this scene and it still renders with Cycles
    (v138; before, Cycles + GPU sufficed, which many start files already are)."""
    return (bool(scene.get("gcapture_prepared"))
            and scene.render.engine in ('CYCLES', 'BLENDER_EEVEE',
                                        'BLENDER_EEVEE_NEXT'))


class GCAPTURE_OT_render_images(Operator):
    """Renders all cameras into the dataset (v1.1.5): Cycles with the
    Prepare Scene settings, EEVEE with the capture preset."""
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
            return ("Save the scene, then render one image per camera into "
                    "the dataset folder with EEVEE - much faster than Cycles, "
                    "lighting approximated. The capture preset is applied "
                    "once; later changes of yours are kept")
        return ("Save the scene, then render one image per camera into the "
                "dataset folder with Cycles on the GPU - exact path tracing, "
                "with the settings of Prepare Scene")

    @classmethod
    def poll(cls, context):
        return (context.scene.camera is not None
                and not bpy.app.is_job_running('RENDER'))

    def _prepare(self, context):
        """Set engine and settings; returns: error text or None."""
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
        _gcapture_apply_image_format(scene, s.render_format)
        out = os.path.dirname(bpy.path.abspath(scene.render.filepath))
        if out:
            os.makedirs(out, exist_ok=True)
        if s.render_headless:
            s.render_script = _gcapture_render_script_path(bpy.data.filepath,
                                                           self.engine)
        # Always save before rendering (v1.1.5): settings changed shortly
        # before are thus not lost; the script renders the saved file
        # anyway. bpy.data.is_dirty is not usable as a condition --
        # Blender does not mark values set via Python (engine, preset)
        # as unsaved.
        bpy.ops.wm.save_mainfile()
        self._saved = True
        return None

    def _write_script(self, context):
        """Write the script for rendering without UI next to the (just
        saved) scene (v1.1.5). Returns: path."""
        dtype = None
        if self.engine == 'CYCLES':
            dtype, _ = _gcapture_setup_gpu()
        # Absolute: on macOS Blender reports the path as relative when it was
        # started from the command line, but the script changes into the
        # scene folder only later (v1.1.5, found in CI: exit 127).
        return _gcapture_write_render_script(
            bpy.data.filepath, self.engine, dtype,
            os.path.abspath(bpy.app.binary_path))

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
        if self._saved:
            self.report({'INFO'}, "Scene saved - rendering")
        # Blender's render window with progress; Esc cancels.
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


def _gcapture_apply_image_format(scene, fmt):
    """Image format of the render buttons (1.2.0): PNG or TIFF, each RGBA
    8 bit. 16 bit brings no benefit for training (test 26.09.2026); EXR is
    left out -- LichtFeld reads neither DWAA/DWAB nor multilayer."""
    im = scene.render.image_settings
    if hasattr(im, "media_type"):          # Blender 5.x: set before the format
        im.media_type = 'IMAGE'
    im.file_format = fmt
    im.color_mode = 'RGBA'
    im.color_depth = '8'
    if fmt == 'TIFF':
        im.tiff_codec = 'DEFLATE'


def _gcapture_format_ext(scene):
    """File extension Blender writes for the selected image format."""
    fmt = scene.render.image_settings.file_format
    return {'PNG': "png", 'TIFF': "tif", 'OPEN_EXR': "exr",
            'OPEN_EXR_MULTILAYER': "exr", 'JPEG': "jpg",
            'TARGA': "tga", 'TARGA_RAW': "tga", 'BMP': "bmp"}.get(fmt)


def _gcapture_render_script_path(blend, engine):
    """Path of the render script next to the scene, per engine and system."""
    folder, name = os.path.split(blend)
    stem = os.path.splitext(name)[0]
    eng = "cycles" if engine == 'CYCLES' else "eevee"
    ext = ("cmd" if sys.platform.startswith("win") else
           "command" if sys.platform == "darwin" else "sh")
    return os.path.join(folder, "%s_render_%s.%s" % (stem, eng, ext))


def _gcapture_write_render_script(blend, engine, dtype, blender):
    """Double-click script next to the scene: renders all frames in the background
    with the same Blender version. Windows .cmd, macOS .command, otherwise .sh."""
    name = os.path.basename(blend)
    eng = "cycles" if engine == 'CYCLES' else "eevee"
    path = _gcapture_render_script_path(blend, engine)
    tail = ["--", "--cycles-device", dtype] if (engine == 'CYCLES' and dtype) else []
    if sys.platform.startswith("win"):
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
        scene["gcapture_prepared"] = True     # step done (v138)
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
    # Choice buttons as toggles via mode (v121): only a toggle can appear
    # red with alert, a selected enum button stays blue.
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

    # --- Compute target ------------------------------------------------
    def _target(self):
        """(target path, name, vtag) for an already saved file."""
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
        """(vtag of the loaded file, highest number in the folder, whether the
        loaded one is the highest). v120."""
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
        # Default: the loaded version if it is the highest; otherwise the
        # next free number (v120). Without version -> <Name>_v001.
        self.mode = 'OVERWRITE' if self._versions()[2] else 'NEW'
        # A custom render path stays unless explicitly requested (v129).
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
            orow.alert = self.mode == 'OVERWRITE'     # overwrite: red (v121)
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

        # Warnings: target already exists / render images no longer match.
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
                self.mode = 'NEW'      # never overwrite an older version
            path, name, vtag = self._target()
        else:
            if not self.filepath:
                self.report({'ERROR'}, "No file name given.")
                return {'CANCELLED'}
            folder = os.path.dirname(bpy.path.abspath(self.filepath))
            name = _gcapture_clean_name(self.filepath)
            # Use a typed-in version (e.g. Porsche_v007) if the
            # file does not exist yet -- otherwise the next free version.
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
        # relative_remap=False: the new "//" render path is already relative
        # to the target folder and must not be converted. New versions
        # live in the same folder, an unsaved scene has no relative
        # paths yet -- so nothing is lost.
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


# ----------------------------------------------------------------------
# Guide: walks beginners through the workflow step by step (v89)
# ----------------------------------------------------------------------
# Blender does not let add-ons expand/collapse panels. So the guide
# hides the subpanels (poll) and shows a card in the main panel with an
# explanation, the controls of the step (the same _gcapture_draw_*
# functions as the subpanels) and a status display.
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


def _cln_check(context, s):
    """What step 8 is missing (red lines) and the paths."""
    errs = []
    splat = bpy.path.abspath(s.clean_splat_file or "").strip()
    if not splat:
        errs.append("Pick the trained splat (.ply)")
    elif not os.path.isfile(splat):
        errs.append("Splat file not found: %s" % os.path.basename(splat))
    elif not splat.lower().endswith(".ply"):
        errs.append("Only .ply splats can be cleaned")
    ds = _exp_resolve_output_dir(s)
    js = _cln_export_json_path(ds)
    if not os.path.isfile(js):
        errs.append("No %s - export the dataset first (step 7)"
                    % os.path.basename(js))
    elif not os.path.isfile(os.path.join(ds, "sparse", "0", "images.txt")):
        errs.append("No cameras in %s - export the dataset first (step 7)"
                    % os.path.basename(os.path.normpath(ds)))
    stem = os.path.splitext(splat)[0]
    return errs, {"splat": splat, "dataset": ds, "json": js,
                  "out": stem + "_clean.ply"}


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
    if s.clean_last_result:
        layout.label(text=s.clean_last_result,
                     icon='ERROR' if "another scene version" in s.clean_last_result
                     else 'CHECKMARK')


def _gcapture_wt_status_clean(context, s):
    if s.clean_last_result:
        return 'DONE', s.clean_last_result
    return 'OPTIONAL', "Optional, after training: pick the splat and clean it"


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


# Progress of long operations (Build, Export) in the panel in place of
# the button instead of in the viewport header (v1.1.4). Runtime only.
_GCAPTURE_PROGRESS = {}   # op_id -> (fraction 0..1, text, timestamp)
# Older entries count as orphaned (crash without cleanup) and
# no longer lock the panel.
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
    """True while Build or Export is running (v1.1.4)."""
    now = time.time()
    return any(now - p[2] < _GCAPTURE_PROGRESS_STALE
               for p in _GCAPTURE_PROGRESS.values())


def _gcapture_lock(layout):
    """Column that is locked during a run; the progress bar
    itself sits outside it and stays readable."""
    col = layout.column()
    col.enabled = not _gcapture_busy()
    return col


def _gcapture_progress_draw(layout, op_id):
    """Draws the progress of op_id if it is running -> True."""
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
    """Button of a mandatory step: blue (pressed) while the step
    is open, neutral afterwards (v137)."""
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
    """Switches the window's Properties editors to the tab of the
    guide step (v124: step 5 -> Output, where the output path is)."""
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
    """Main panel. The sections are subpanels (v85) -- collapsible,
    with the phase color in the header. bl_idname stays unchanged."""
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
    """Common base of the subpanels: header with colored phase icon
    and step symbol."""
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Gaussian Render Capture"
    bl_parent_id = "GCAPTURE_PT_panel"
    # Initially collapsed: the user clicks through the steps (v89).
    bl_options = {'DEFAULT_CLOSED'}
    gcapture_color = None     # fallback if the badges are missing
    gcapture_badge = None     # Name in _GCAPTURE_BADGES (colored number badge)
    gcapture_icon = None

    @classmethod
    def poll(cls, context):
        # While the guide is running, the main panel shows only the guide card.
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


class GCAPTURE_PT_clean(_GCAPTURE_SubPanel, Panel):
    bl_idname = "GCAPTURE_PT_clean"
    gcapture_badge = 'gcapture_7'
    gcapture_icon = 'OUTLINER_OB_POINTCLOUD'
    bl_label = "8. Clean Splat (after training)"
    bl_order = 7
    gcapture_color = _GCAPTURE_COLOR_SPLAT

    def draw(self, context):
        _gcapture_draw_clean(self.layout, context)


class GCAPTURE_PT_advanced(_GCAPTURE_SubPanel, Panel):
    """Rarely used options + maintainer/license (collapsed)."""
    bl_idname = "GCAPTURE_PT_advanced"
    gcapture_icon = 'PREFERENCES'
    bl_label = "Advanced"
    bl_order = 8
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = _gcapture_lock(self.layout)
        s = context.scene.gcapture_settings
        # Custom camera guide meshes instead of/in addition to the sphere (v93).
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
        cbox = layout.box()
        cbox.label(text="Clean Splat", icon='OUTLINER_OB_POINTCLOUD')
        cbox.prop(s, "clean_min_views")
        # Interior test only makes sense with custom guide meshes.
        if _gcapture_has_custom_guides(s):
            glbox.prop(s, "skip_interior")

        # Render Setup (v129, previously step 5): takes effect immediately and
        # on every build; always on for a capture.
        rbox = layout.box()
        rbox.label(text="Render Setup (keep on for a capture)", icon='SCENE')
        rcol = rbox.column(align=True)
        rcol.prop(s, "constant_interp")
        rcol.prop(s, "set_active_camera")
        rcol.prop(s, "set_frame_range")
        rcol.prop(s, "set_resolution")

        # --- Maintainer & license (GPL attribution, mandatory) ---
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
# COLMAP export (integrated from export_colmap_for_postshot_v02)
# ----------------------------------------------------------------------
# Axis correction Blender camera -> COLMAP camera (180 degrees about camera X).
_BLENDER_TO_COLMAP = Matrix((
    (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, -1.0),
))
# World rotation Z-up -> Y-up (+90 degrees about world X).
# (x, y, z) -> (x, -z, y): Blender Z (up) becomes -Y. Postshot/COLMAP
# counts Y downward, so this keeps the model upright. The
# -90 variant (x, z, -y) also aligns the axis, but rotates the
# model by 180 degrees (upside down).
_WORLD_Z_UP_TO_Y_UP = Matrix((
    (1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 1.0, 0.0),
))
_EXP_KNOWN_EXTS = ("tif", "tiff", "png", "jpg", "jpeg", "exr", "tga", "bmp")


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


def _exp_world_transform(s):
    return _WORLD_Z_UP_TO_Y_UP if s.exp_zup_to_yup else Matrix.Identity(3)


def _exp_write_scale_info(out_dir, s, scale, mean_r, n_poses, n_points):
    """Writes <vNNN>_gcapture_export.json NEXT TO the dataset folder (v126,
    v145: not into it -- Postshot reads every .json in the dataset as a
    NeRF camera file and aborts). Scale and
    axis conversion of the export. Dataset = scale * W @ Blender world."""
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
    old = os.path.join(ds, "gscan_export.json")        # location up to v144
    if os.path.isfile(old):
        os.remove(old)
    # Gaussian Render Scan (up to 1.0.2) wrote <vNNN>_gscan_export.json.
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


def _exp_detect_frames(image_dir, prefer=None):
    """Detect the image sequence in the folder. prefer: extension of the
    configured format -- if present, it wins even against more images of
    another format from an earlier run (1.2.0)."""
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
    prefer = (prefer or "").lower()
    if prefer == "tif" and "tiff" in ext_counts and "tif" not in ext_counts:
        prefer = "tiff"
    chosen_ext = (prefer if prefer in ext_counts
                  else max(ext_counts, key=ext_counts.get))
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
    """Reads the actual pixel size of an image file via Blender's
    image API (no extra library). Returns (w, h) or None."""
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
        # Remove the loaded image again so the .blend does not get cluttered.
        if img is not None:
            try:
                bpy.data.images.remove(img)
            except Exception:
                pass


def _exp_intrinsics(cam_data, scene, real_size=None):
    """fx, fy, cx, cy, w, h in pixels (PINHOLE).
    If real_size (w, h) is given (actual image size from the file),
    resolution and principal point are derived from it -- so that
    cameras.txt ALWAYS matches the images actually present, regardless
    of the render settings. The focal length is scaled to the actual
    image width."""
    render = scene.render
    if real_size is not None:
        w, h = int(real_size[0]), int(real_size[1])
    else:
        w = int(render.resolution_x * render.resolution_percentage / 100.0)
        h = int(render.resolution_y * render.resolution_percentage / 100.0)

    # Blender's camera model (v1.1.3): pixel aspect ratio, sensor fit
    # and shift refer to the effective image size (pixels x aspect).
    # With AUTO the longer effective edge decides; the sensor width applies.
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
    """Linear float (0..1) -> sRGB-encoded byte (0..255).
    Blender material colors are linear; COLMAP/viewers expect sRGB."""
    c = max(0.0, min(1.0, c))
    if c <= 0.0031308:
        s = 12.92 * c
    else:
        s = 1.055 * (c ** (1.0 / 2.4)) - 0.055
    return int(round(max(0.0, min(1.0, s)) * 255))


def _exp_material_base_color(mat):
    """Returns (r,g,b) 0..1 of a material's Principled BSDF base color,
    or None if it cannot be determined."""
    if mat is None or not mat.use_nodes or mat.node_tree is None:
        # Material without nodes: diffuse_color as fallback.
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
    # No Principled found: first material with diffuse_color.
    dc = mat.diffuse_color
    return (dc[0], dc[1], dc[2])


def _exp_object_material_colors(obj):
    """List of (r,g,b) per material slot of the object (linear 0..1).
    Missing/unreadable slots -> medium gray."""
    colors = []
    if not obj.material_slots:
        return [(0.5, 0.5, 0.5)]
    for slot in obj.material_slots:
        col = _exp_material_base_color(slot.material)
        colors.append(col if col is not None else (0.5, 0.5, 0.5))
    return colors


def _exp_vertex_color_lookup(mesh):
    """Builds, if present, a dict vert_index -> (r,g,b) from the
    active color attribute. Returns None if there are no color
    attributes. Averages over all loops that reference a vertex."""
    ca = getattr(mesh, "color_attributes", None)
    if not ca or len(ca) == 0:
        return None
    layer = ca.active_color or ca[0]

    acc = {}
    cnt = {}
    if layer.domain == 'POINT':
        # Directly per vertex.
        for i, d in enumerate(layer.data):
            c = d.color
            acc[i] = (c[0], c[1], c[2])
            cnt[i] = 1
    else:  # 'CORNER' -> average over loops onto vertices
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
    """Returns (min_vec, max_vec) of the world-space bounding box of the
    crop object incl. margin, or None if crop is off/no object. The bounds
    are in BLENDER world coordinates -- the test happens BEFORE the Y-up/
    scale transformation of the points."""
    if not s.exp_crop_enable or s.exp_crop_object is None:
        return None
    obj = s.exp_crop_object
    mw = obj.matrix_world
    # bound_box: 8 local corners; transform into world coordinates.
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
    """True if co_world (Blender world coord) lies within bounds."""
    mn, mx = bounds
    return (mn.x <= co_world.x <= mx.x and
            mn.y <= co_world.y <= mx.y and
            mn.z <= co_world.z <= mx.z)


def _exp_build_scene_bvh(objs):
    """Single world-space BVHTree over all given meshes.
    Derived from "Gauss Cannon" by Arash Keshmirian (Warpgate Labs),
    GPL-3.0-or-later;
    see the license section in the header."""
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
    """True if the world point p is seen unobstructed by at least one camera
    (no other piece of geometry in front). Cameras sorted by proximity,
    early out at the first unobstructed line of sight. Numerically verified."""
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
# Visibility filter on the GPU (v105)
# ----------------------------------------------------------------------
# Instead of shooting a ray to every camera per point (Python, with
# construction-brick models ~300 rays per occluded point, Vespa 5.4 million points:
# ~3 h), the GPU draws the model once from each camera as a depth image;
# all points are then tested against each depth image with numpy.
# Accuracy = resolution of the depth image. Without a GPU context (e.g.
# blender --background) the export falls back to the ray cast.
_EXP_GPU_RES = 2048

# Compute shader: per point (texel of the point texture), project into
# this camera image and test against the farthest depth value of the 3x3
# neighborhood -- the same rule as _ExpGpuDepth.visible (numpy).
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

# Clean Splat (1.2.0): mirror image of the visibility test. Per splat count
# in how many cameras it lies in the image (seen) and in how many it lies in
# front of the NEAREST depth of the 3x3 neighbourhood (viol). p.w = sigma,
# < 0 = empty slot. The same rule as _cln_judge (numpy).
_CLN_CARVE_COMPUTE_SRC = """
void main()
{
  ivec2 id = ivec2(gl_GlobalInvocationID.xy);
  ivec2 sz = imageSize(pts);
  if (id.x >= sz.x || id.y >= sz.y) { return; }
  vec4 p = imageLoad(pts, id);
  if (p.w < 0.0) { return; }
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
  imageStore(seen, id, imageLoad(seen, id) + vec4(1.0));
  int px = clamp(int((xn + 1.0) * 0.5 * float(res)), 0, res - 1);
  int py = clamp(int((yn + 1.0) * 0.5 * float(res)), 0, res - 1);
  float mn = 1e30;
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      ivec2 q = clamp(ivec2(px + dx, py + dy), ivec2(0), ivec2(res - 1));
      float z = texelFetch(depth_tx, q, 0).r;
      float lin = (z >= 1.0) ? 1e30 :
                  (2.0 * fr * nr) / (fr + nr - (2.0 * z - 1.0) * (fr - nr));
      mn = min(mn, lin);
    }
  }
  if (d + 3.0 * p.w + tol_v.x * d + tol_v.y < mn) {
    imageStore(viol, id, imageLoad(viol, id) + vec4(1.0));
  }
}
"""


class _ExpGpuDepth:
    """Holds the geometry batch and framebuffer for the depth images."""

    def __init__(self, objs, res=_EXP_GPU_RES, geometry=None):
        import gpu
        from gpu_extras.batch import batch_for_shader
        depsgraph = bpy.context.evaluated_depsgraph_get()
        verts, tris, off = [], [], 0
        if geometry is not None:            # Clean Splat: ready triangles (1.2.0)
            verts, tris, objs = [geometry[0]], [geometry[1]], ()
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
        self.nb = 1          # neighborhood (1 = 3x3, 0 = only the pixel)
        self.tol_rel = 2e-3  # tolerance relative to depth
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
        self._cs = None      # compute shader state (compute_begin)

    def near_far(self, cam_pos):
        """Depth range tight around the model (v111). It used to range from
        a ten-thousandth to a hundred times the model size -- in 32-bit
        precision this made the depth reconstruction inaccurate."""
        dist = float(np.linalg.norm(np.array(cam_pos) - self.center))
        near = max(self.size * 1e-4, dist - self.radius * 1.01)
        far = dist + self.radius * 1.01
        return near, far

    def _render(self, cam_pos, look_dir, fov, near, far):
        """Draws the depth image of this camera into self.fb. Returns: the
        camera rotation (quaternion)."""
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

    # --- Point test as compute shader (v111) ----------------------------
    # The points live on the GPU as an RGBA32F texture, the result as an
    # R32F texture; per camera the GPU draws the depth image and tests all
    # points in one pass. Vespa P200 (5.4 million points, 320 cameras):
    # under 1 s instead of ~165 s with numpy; compared with the exact ray
    # cast it loses fewer visible points (80 instead of 629 of ~7,000 disputed).
    _CS_WIDTH = 4096

    def compute_begin(self, points):
        """Upload points and build the shader. Raises an exception on
        problems -- the caller then falls back to visible() (numpy)."""
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
        """Draw the depth image of this camera and test all points not yet
        visible against it (on the GPU, without reading back)."""
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
        """bool per point: seen by at least one camera."""
        cs = self._cs
        v = np.array(cs["vis"].read(), dtype=np.float32).reshape(-1)
        return v[:cs["n"]] > 0.5

    # --- Clean Splat (1.2.0) -------------------------------------------
    def carve_begin(self, points, sigma):
        """Upload the splats (xyz + sigma) and build the counting shader."""
        import gpu
        n = len(points)
        w = self._CS_WIDTH
        h = max(1, (n + w - 1) // w)
        if h > 16384:
            raise RuntimeError("too many splats for one texture (%d)" % n)
        buf = np.full((h * w, 4), -1.0, dtype=np.float32)
        buf[:n, :3] = points
        buf[:n, 3] = sigma
        pts_tex = gpu.types.GPUTexture(
            (w, h), format='RGBA32F',
            data=gpu.types.Buffer('FLOAT', h * w * 4, buf.ravel()))
        viol = gpu.types.GPUTexture((w, h), format='R32F')
        seen = gpu.types.GPUTexture((w, h), format='R32F')
        viol.clear(format='FLOAT', value=(0.0,))
        seen.clear(format='FLOAT', value=(0.0,))
        info = gpu.types.GPUShaderCreateInfo()
        info.image(0, 'RGBA32F', 'FLOAT_2D', 'pts', qualifiers={'READ'})
        info.image(1, 'R32F', 'FLOAT_2D', 'viol', qualifiers={'READ', 'WRITE'})
        info.image(2, 'R32F', 'FLOAT_2D', 'seen', qualifiers={'READ', 'WRITE'})
        info.sampler(0, 'FLOAT_2D', 'depth_tx')
        for name in ('cam_pos', 'cam_fwd', 'cam_right', 'cam_up', 'params',
                     'tol_v'):
            info.push_constant('VEC4', name)
        info.local_group_size(16, 16, 1)
        info.compute_source(_CLN_CARVE_COMPUTE_SRC)
        shader = gpu.shader.create_from_info(info)
        self._carve = dict(n=n, w=w, h=h, pts=pts_tex, viol=viol, seen=seen,
                           shader=shader)

    def carve_camera(self, cam_pos, look_dir, fov, tol_abs):
        """Draw this camera's depth and count all splats (on the GPU)."""
        import gpu
        cs = self._carve
        near, far = self.near_far(cam_pos)
        q = self._render(cam_pos, look_dir, fov, near, far)
        r = q @ Vector((1.0, 0.0, 0.0))
        u = q @ Vector((0.0, 1.0, 0.0))
        fw = look_dir.normalized()
        sh = cs["shader"]
        sh.bind()
        sh.image('pts', cs["pts"])
        sh.image('viol', cs["viol"])
        sh.image('seen', cs["seen"])
        sh.uniform_sampler('depth_tx', self.depth_tex)
        sh.uniform_float('cam_pos', (cam_pos.x, cam_pos.y, cam_pos.z, 0.0))
        sh.uniform_float('cam_fwd', (fw.x, fw.y, fw.z, 0.0))
        sh.uniform_float('cam_right', (r.x, r.y, r.z, 0.0))
        sh.uniform_float('cam_up', (u.x, u.y, u.z, 0.0))
        sh.uniform_float('params', (1.0 / math.tan(fov / 2.0), near, far,
                                    float(self.res)))
        sh.uniform_float('tol_v', (_CLN_TOL_REL, tol_abs, 0.0, 0.0))
        gpu.compute.dispatch(sh, (cs["w"] + 15) // 16, (cs["h"] + 15) // 16, 1)

    def carve_result(self):
        """(viol, seen) per splat as int32."""
        cs = self._carve
        viol = np.array(cs["viol"].read(), dtype=np.float32).reshape(-1)
        seen = np.array(cs["seen"].read(), dtype=np.float32).reshape(-1)
        return (np.rint(viol[:cs["n"]]).astype(np.int32),
                np.rint(seen[:cs["n"]]).astype(np.int32))

    def depth_map(self, cam_pos, look_dir, fov, near, far):
        """Linear depth (distance along the view axis) per pixel, (R,R),
        row 0 = bottom. inf where there is nothing."""
        q = self._render(cam_pos, look_dir, fov, near, far)
        with self.fb.bind():
            buf = self.fb.read_depth(0, 0, self.res, self.res)
        zb = np.array(buf, dtype=np.float32).reshape(self.res, self.res)
        # Window depth -> linear eye depth.
        z_ndc = zb.astype(np.float64) * 2.0 - 1.0
        lin = (2.0 * far * near) / (far + near - z_ndc * (far - near))
        lin[zb >= 1.0] = np.inf
        return lin, q

    def visible(self, points, cam_pos, look_dir, fov):
        """bool per point: in the image of this camera and not occluded."""
        near, far = self.near_far(cam_pos)
        lin, q = self.depth_map(cam_pos, look_dir, fov, near, far)
        # Farthest value of the 3x3 neighborhood: do not lose points at edges
        # and in narrow gaps due to pixel rasterization.
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


# ----------------------------------------------------------------------
# Clean Splat (1.2.0): remove the splats in empty space from a trained
# splat. Every pixel of the model depth seen by a dataset camera says: up to
# the surface the ray is empty. Idea of the maintainer (26.09.2026);
# prototype and measurement: _CLAUDE_/260926_priors_test (carve.py).
# ----------------------------------------------------------------------
def _gcapture_ply_read(path):
    """Read a standard 3DGS PLY: binary little endian, one element 'vertex',
    float properties only (LichtFeld, Postshot, ...). Returns a dict with
    header (lines without 'ply' and 'end_header'), names, rows (float32 N x P).
    Anything else -> ValueError with the reason."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError("%s is not a PLY file" % os.path.basename(path))
        header, names, n, fmt = [], [], None, None
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PLY header has no end_header")
            text = line.decode("ascii", "surrogateescape").strip()
            if text == "end_header":
                break
            header.append(text)
            parts = text.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1] if len(parts) > 1 else None
            elif parts[0] == "element":
                if parts[1] != "vertex" or n is not None:
                    raise ValueError("PLY has an element '%s' - only splats "
                                     "(one element 'vertex') can be cleaned"
                                     % parts[1])
                n = int(parts[2])
            elif parts[0] == "property":
                if parts[1] not in ("float", "float32"):
                    raise ValueError("PLY property '%s' is %s - only float "
                                     "properties are supported"
                                     % (parts[-1], parts[1]))
                names.append(parts[-1])
        if fmt != "binary_little_endian":
            raise ValueError("PLY format '%s' - only binary_little_endian is "
                             "supported" % fmt)
        if n is None or not names:
            raise ValueError("PLY has no splats (element 'vertex')")
        for need in ("x", "y", "z"):
            if need not in names:
                raise ValueError("PLY has no property '%s'" % need)
        rows = np.fromfile(f, dtype="<f4", count=n * len(names))
    if rows.size != n * len(names):
        raise ValueError("PLY is truncated (%d of %d values)"
                         % (rows.size, n * len(names)))
    return {"header": header, "names": names,
            "rows": rows.reshape(n, len(names))}


def _gcapture_ply_write(path, ply, keep, comment):
    """Write only the rows with keep=True; the header stays unchanged except
    for the new count and one comment line. Writes <path>.part first, then
    replaces -- if writing fails, no half-written file is left."""
    rows = ply["rows"][keep]
    out = ["ply"]
    for text in ply["header"]:
        if text.startswith("element vertex"):
            out.append("comment " + comment)
            out.append("element vertex %d" % len(rows))
        else:
            out.append(text)
    out.append("end_header")
    tmp = path + ".part"
    try:
        with open(tmp, "wb") as f:
            f.write(("\n".join(out) + "\n").encode("ascii", "surrogateescape"))
            f.write(np.ascontiguousarray(rows, dtype="<f4").tobytes())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _cln_export_json_path(ds_dir):
    """<vNNN>_gcapture_export.json sits NEXT TO the dataset folder (v145).
    Datasets from Gaussian Render Scan 1.0.x have the same file under its old
    names: <vNNN>_gscan_export.json next to the dataset (from 0.145) or
    gscan_export.json inside it (1.126.0-0.144). The first one that exists
    wins; otherwise the current name (for the message)."""
    ds = os.path.normpath(ds_dir)
    current = os.path.join(os.path.dirname(ds),
                           os.path.basename(ds) + "_gcapture_export.json")
    for path in (current,
                 os.path.join(os.path.dirname(ds),
                              os.path.basename(ds) + "_gscan_export.json"),
                 os.path.join(ds, "gscan_export.json")):
        if os.path.isfile(path):
            return path
    return current


def _cln_load_transform(json_path):
    """(blender_from_dataset as a 4x4 Matrix, scale) from the export JSON."""
    import json
    with open(json_path, encoding="utf-8") as f:
        info = json.load(f)
    mat, scale = info.get("blender_from_dataset"), info.get("scale")
    if not mat or not scale:
        raise ValueError("%s has no blender_from_dataset / scale"
                         % os.path.basename(json_path))
    return Matrix(mat), float(scale)


def _cln_dataset_views(ds_dir, bfd):
    """Cameras of the dataset in Blender world space: [(pos, fwd, fov)]. Exactly
    the poses the trainer used (including Live Camera Adjust, without
    rejected interior cameras). fov covers the whole image (larger axis,
    shift included) -- the depth map is square."""
    from mathutils import Quaternion
    sp = os.path.join(ds_dir, "sparse", "0")
    fovs = {}
    with open(os.path.join(sp, "cameras.txt"), encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) < 8 or p[0].startswith("#"):
                continue
            w, h = float(p[2]), float(p[3])
            fx, fy, cx, cy = map(float, p[4:8])
            half = max(max(cx, w - cx) / fx, max(cy, h - cy) / fy)
            fovs[p[0]] = 2.0 * math.atan(half)
    rot = bfd.to_3x3()
    views = []
    with open(os.path.join(sp, "images.txt"), encoding="utf-8") as f:
        for line in f:
            p = line.split()
            # Image line: ID QW QX QY QZ TX TY TZ CAMERA_ID NAME (the lines of the
            # 2D points have a multiple of 3 fields or are empty).
            if len(p) != 10 or p[0].startswith("#"):
                continue
            try:
                int(p[0])
                q = Quaternion(tuple(float(v) for v in p[1:5]))
                t = Vector(tuple(float(v) for v in p[5:8]))
            except ValueError:
                continue
            fov = fovs.get(p[8])
            if fov is None:
                continue
            r_w2c = q.to_matrix()
            c_ds = -(r_w2c.transposed() @ t)
            fwd_ds = r_w2c.transposed() @ Vector((0.0, 0.0, 1.0))
            pos = (bfd @ c_ds.to_4d()).to_3d()
            fwd = (rot @ fwd_ds).normalized()
            views.append((pos, fwd, fov))
    return views


_CLN_GEOM_TYPES = {'MESH', 'CURVE', 'SURFACE', 'FONT', 'META'}


def _cln_render_geometry(context):
    """Everything that is rendered, as triangles in world coordinates -- not
    only the look target: a rendered floor is not empty space. Evaluated as
    for the render: objects hidden in the viewport (eye or monitor icon) are
    shown, modifiers that are on for the render only are switched on, and
    subdivision uses its render levels -- all restored afterwards. Includes
    instances (collection instances, Geometry Nodes). Without the camera
    sphere, guides, crop object, Capture_Rig and hide_render objects."""
    scene = context.scene
    s = scene.gcapture_settings
    excl = set()
    if s.sph_object is not None:
        excl.add(s.sph_object.name)
    for item in s.guides:
        for ref in item.objects:
            if ref.obj is not None:
                excl.add(ref.obj.name)
    if s.exp_crop_object is not None:
        excl.add(s.exp_crop_object.name)
    rig = bpy.data.collections.get(_GCAPTURE_RIG_COLL)
    if rig is not None:
        excl.update(o.name for o in rig.all_objects)
    view_layer = context.view_layer
    undo = []                     # (restore function) in reverse order
    try:
        for o in scene.objects:
            if o.hide_render or o.name in excl:
                continue
            if o.hide_viewport:
                o.hide_viewport = False
                undo.append(lambda o=o: setattr(o, "hide_viewport", True))
            try:
                if o.hide_get(view_layer=view_layer):
                    o.hide_set(False, view_layer=view_layer)
                    undo.append(lambda o=o: o.hide_set(True, view_layer=view_layer))
            except RuntimeError:  # not in this view layer
                pass
            for md in o.modifiers:
                if md.show_render and not md.show_viewport:
                    md.show_viewport = True
                    undo.append(lambda md=md: setattr(md, "show_viewport", False))
                if md.type in {'SUBSURF', 'MULTIRES'} and md.levels != md.render_levels:
                    old = md.levels
                    md.levels = md.render_levels
                    undo.append(lambda md=md, old=old: setattr(md, "levels", old))
        if undo:
            view_layer.update()
        dg = context.evaluated_depsgraph_get()
        cache, verts, tris = {}, [], []
        off = 0
        for inst in dg.object_instances:
            ob = inst.object
            orig = ob.original
            if ob.type not in _CLN_GEOM_TYPES or orig.hide_render or orig.name in excl:
                continue
            # Key by the evaluated data: Geometry Nodes instances of different
            # generated meshes all carry the same temporary name.
            key = (ob.data.as_pointer() if ob.data is not None
                   else hash(orig.name))
            if key not in cache:
                co = tri = None
                me = ob.to_mesh()
                if me is not None:
                    try:
                        me.calc_loop_triangles()
                        if len(me.vertices) and len(me.loop_triangles):
                            co = np.empty(len(me.vertices) * 3, dtype=np.float32)
                            me.vertices.foreach_get("co", co)
                            co = co.reshape(-1, 3)
                            tri = np.empty(len(me.loop_triangles) * 3, dtype=np.int32)
                            me.loop_triangles.foreach_get("vertices", tri)
                            tri = tri.reshape(-1, 3)
                    finally:
                        ob.to_mesh_clear()
                cache[key] = (co, tri)
            co, tri = cache[key]
            if co is None:
                continue
            mwn = np.array(inst.matrix_world, dtype=np.float32)
            verts.append(co @ mwn[:3, :3].T + mwn[:3, 3])
            tris.append(tri + off)
            off += len(co)
    finally:
        for fn in reversed(undo):
            fn()
        if undo:
            view_layer.update()
    if not verts:
        raise ValueError("No rendered geometry in the scene")
    return (np.vstack(verts).astype(np.float32),
            np.vstack(tris).astype(np.int32))


def _cln_model_size(context, verts):
    """Bounding-box diagonal of the look target (step 2) -- the scale of the
    absolute tolerance. A big rendered floor must not widen it. Without a
    look target: all rendered geometry."""
    pts = []
    for it in context.scene.gcapture_settings.sph_target_colls:
        if it.coll is None:
            continue
        for o in it.coll.all_objects:
            if o.type in _CLN_GEOM_TYPES and not o.hide_render:
                mw = o.matrix_world
                pts.extend(tuple(mw @ Vector(c)) for c in o.bound_box)
    arr = np.array(pts) if pts else verts
    return float(np.linalg.norm(arr.max(axis=0) - arr.min(axis=0))) or 1.0


_CLN_TOL_REL = 0.01      # tolerance relative to the depth
_CLN_TOL_SIZE = 2e-3     # absolute tolerance: share of the model size
# Depth map resolution for GPU and ray casting alike. Coarse on purpose: the
# 3x3 neighbourhood then reaches about 1 % of the image sideways, so splats
# that hug an edge or a silhouette stay. Beetle, 26.09.2026 (1.92 M splats,
# 40 test views): 2048 px removed 12 % and cost 1 dB PSNR, 256 px removes
# 0.45 % at -0.1 dB and still 62 % of the floaters farther than 10 mm.
_CLN_DEPTH_RES = 256


def _cln_splat_points(ply, bfd, scale):
    """Splat centres in Blender world space and extent sigma (largest axis,
    Blender units). Without scale_* sigma is 0."""
    names, rows = ply["names"], ply["rows"]
    xyz = rows[:, [names.index(c) for c in ("x", "y", "z")]].astype(np.float64)
    mat = np.array(bfd, dtype=np.float64)
    points = xyz @ mat[:3, :3].T + mat[:3, 3]
    idx = [names.index("scale_%d" % i) for i in range(3)
           if "scale_%d" % i in names]
    if len(idx) == 3:
        sigma = np.exp(rows[:, idx].astype(np.float64)).max(axis=1) / scale
    else:
        sigma = np.zeros(len(points))
    return points, sigma


def _cln_camera_frame(pos, fwd, fov):
    """Camera axes as in _ExpGpuDepth._render (roll does not matter, only consistency)."""
    q = fwd.to_track_quat('-Z', 'Y')
    return (np.array(pos, dtype=np.float64),
            np.array(q @ Vector((1.0, 0.0, 0.0)), dtype=np.float64),
            np.array(q @ Vector((0.0, 1.0, 0.0)), dtype=np.float64),
            np.array(fwd.normalized(), dtype=np.float64),
            1.0 / math.tan(fov / 2.0))


def _cln_depth_cpu(bvh, frame, res):
    """Linear depth per pixel by ray casting (row 0 = bottom, inf = nothing)."""
    pos, right, up, fwd, f = frame
    origin = Vector(pos)
    c = (np.arange(res) + 0.5) / res * 2.0 - 1.0
    lin = np.full((res, res), np.inf)
    for py in range(res):
        row_dir = fwd + c[py] * up / f
        for px in range(res):
            d = row_dir + c[px] * right / f
            hit = bvh.ray_cast(origin, Vector(d))
            if hit[0] is not None:
                lin[py, px] = float(np.dot(np.array(hit[0]) - pos, fwd))
    return lin


def _cln_judge(points, sigma, lin, frame, near, tol_abs):
    """The same rule as _CLN_CARVE_COMPUTE_SRC: (in front of the surface, in the
    image) per splat. Nearest depth of the 3x3 neighbourhood, background = infinite."""
    pos, right, up, fwd, f = frame
    res = lin.shape[0]
    rel = points - pos
    d = rel @ fwd
    safe = np.where(d > near, d, 1.0)
    xn = (rel @ right) / safe * f
    yn = (rel @ up) / safe * f
    seen = (d > near) & (np.abs(xn) <= 1.0) & (np.abs(yn) <= 1.0)
    px = np.clip(((xn + 1.0) * 0.5 * res).astype(np.int64), 0, res - 1)
    py = np.clip(((yn + 1.0) * 0.5 * res).astype(np.int64), 0, res - 1)
    fin = np.where(np.isfinite(lin), lin, 1e30)
    pad = np.pad(fin, 1, mode='edge')
    zmin = fin.copy()
    for dy in range(3):
        for dx in range(3):
            zmin = np.minimum(zmin, pad[dy:dy + res, dx:dx + res])
    z = zmin[py, px]
    front = seen & (d + 3.0 * sigma + _CLN_TOL_REL * d + tol_abs < z)
    return front, seen


class _ClnJob:
    """One cleaning run: draw the depth per camera and count per splat how often
    it lies in front of the surface and how often in the image. GPU if
    possible; otherwise ray casting (blender --background, render nodes)."""

    def __init__(self, verts, tris, points, sigma, views, use_gpu, size=None):
        self.points, self.sigma, self.views = points, sigma, views
        self.total, self.done = len(views), 0
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        self.size = float(np.linalg.norm(hi - lo)) or 1.0
        self.center = (lo + hi) / 2.0
        self.radius = float(np.linalg.norm(verts - self.center, axis=1).max()) or 1.0
        # tolerance from the model (look target), the rest from all geometry
        self.tol_abs = _CLN_TOL_SIZE * (size or self.size)
        self.viol = np.zeros(len(points), dtype=np.int32)
        self.seen = np.zeros(len(points), dtype=np.int32)
        self.gpu = None
        if use_gpu:
            try:
                self.gpu = _ExpGpuDepth(None, res=_CLN_DEPTH_RES,
                                        geometry=(verts, tris))
                self.gpu.carve_begin(points, sigma)
            except Exception as exc:
                print("[Gaussian Render Capture] GPU carving unavailable, "
                      "using ray casting:", exc)
                self.gpu = None
        self.bvh = None
        if self.gpu is None:
            self.bvh = BVHTree.FromPolygons(verts.tolist(), tris.tolist(),
                                            all_triangles=True)

    @property
    def mode(self):
        return "GPU" if self.gpu is not None else "ray casting"

    def _near(self, pos):
        dist = float(np.linalg.norm(np.array(pos) - self.center))
        return max(self.size * 1e-4, dist - self.radius * 1.01)

    def step(self):
        pos, fwd, fov = self.views[self.done]
        if self.gpu is not None:
            self.gpu.carve_camera(pos, fwd, fov, self.tol_abs)
        else:
            frame = _cln_camera_frame(pos, fwd, fov)
            lin = _cln_depth_cpu(self.bvh, frame, _CLN_DEPTH_RES)
            front, seen = _cln_judge(self.points, self.sigma, lin, frame,
                                     self._near(pos), self.tol_abs)
            self.viol += front
            self.seen += seen
        self.done += 1

    def result(self, min_views):
        """bool per splat: remove."""
        if self.gpu is not None:
            self.viol, self.seen = self.gpu.carve_result()
        return (self.viol >= min_views) | (self.seen == 0)


class GCAPTURE_OT_clean_pick(Operator):
    """Pick the trained splat (.ply) - the file browser opens in the dataset folder"""
    bl_idname = "gcapture.clean_pick"
    bl_label = "Pick Splat"
    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(default="*.ply", options={'HIDDEN'})

    def invoke(self, context, event):
        s = context.scene.gcapture_settings
        cur = bpy.path.abspath(s.clean_splat_file or "")
        self.filepath = (cur if os.path.isfile(cur)
                         else _exp_resolve_output_dir(s) + os.sep)
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        context.scene.gcapture_settings.clean_splat_file = self.filepath
        return {'FINISHED'}


class GCAPTURE_OT_clean_splat(Operator):
    """Remove the splats in empty space: every camera of the dataset sees the
    model's depth, and a splat in front of the surface, or seen by no camera,
    goes. Writes <name>_clean.ply next to the splat - the original stays
    untouched. Use the scene version the dataset was exported from"""
    bl_idname = "gcapture.clean_splat"
    bl_label = "Clean Splat"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        s = context.scene.gcapture_settings
        return not _gcapture_busy() and not _cln_check(context, s)[0]

    def _setup(self, context):
        s = context.scene.gcapture_settings
        errs, paths = _cln_check(context, s)
        if errs:
            raise ValueError(errs[0])
        bfd, scale = _cln_load_transform(paths["json"])
        views = _cln_dataset_views(paths["dataset"], bfd)
        if not views:
            raise ValueError("No cameras in the dataset - export it again (step 7)")
        ply = _gcapture_ply_read(paths["splat"])
        limit = _ExpGpuDepth._CS_WIDTH * 16384      # texture limit of the GPU
        if len(ply["rows"]) > limit:
            raise ValueError("Too many splats (%d) - at most %d can be cleaned"
                             % (len(ply["rows"]), limit))
        points, sigma = _cln_splat_points(ply, bfd, scale)
        verts, tris = _cln_render_geometry(context)
        self._ply, self._paths = ply, paths
        self._min_views = s.clean_min_views
        self._job = _ClnJob(verts, tris, points, sigma, views,
                            use_gpu=not bpy.app.background,
                            size=_cln_model_size(context, verts))

    def _release(self):
        """Drop splats, geometry and GPU buffers -- Blender may keep the
        operator instance after the run."""
        self._job = None
        self._ply = None

    def _finish(self, context):
        s = context.scene.gcapture_settings
        cut = self._job.result(self._min_views)
        mode = self._job.mode
        n, k = len(cut), int(cut.sum())
        out = self._paths["out"]
        comment = ("gcapture_clean removed=%d of=%d min_views=%d addon=%s"
                   % (k, n, self._min_views,
                      ".".join(str(v) for v in bl_info["version"])))
        try:
            _gcapture_ply_write(out, self._ply, ~cut, comment)
        except OSError as exc:
            self._release()
            self.report({'ERROR'}, "Could not write %s: %s - close it in other "
                        "programs" % (os.path.basename(out), exc.strerror or exc))
            return {'CANCELLED'}
        self._release()
        msg = "Removed %d of %d splats (%.1f %%) - %s" % (
            k, n, 100.0 * k / max(n, 1), os.path.basename(out))
        if k > n / 2:
            msg += " - more than half: splat from another scene version?"
            self.report({'WARNING'}, msg)
        else:
            self.report({'INFO'}, msg)
        s.clean_last_result = "%s (%s)" % (msg, mode)
        print("[Gaussian Render Capture] Clean Splat (%s): %s" % (mode, msg))
        return {'FINISHED'}

    def _fail(self, context, exc):
        """Any error during a run: stop cleanly, report, keep no old result."""
        self._cleanup(context)
        self._release()
        context.scene.gcapture_settings.clean_last_result = ""
        self.report({'ERROR'}, "Clean Splat failed: %s" % exc)
        return {'CANCELLED'}

    def execute(self, context):
        context.scene.gcapture_settings.clean_last_result = ""
        try:
            self._setup(context)
            while self._job.done < self._job.total:
                self._job.step()
            return self._finish(context)
        except Exception as exc:
            return self._fail(context, exc)

    def invoke(self, context, event):
        context.scene.gcapture_settings.clean_last_result = ""
        try:
            self._setup(context)
        except Exception as exc:
            return self._fail(context, exc)
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.01, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'ESC':
            self._cleanup(context)
            self._release()
            self.report({'WARNING'}, "Clean Splat cancelled - nothing written.")
            return {'CANCELLED'}
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        try:
            job = self._job
            t0 = time.time()
            while job.done < job.total and time.time() - t0 < 0.1:
                job.step()
            _gcapture_progress_set("gcapture.clean_splat", job.done / job.total,
                                   "Clean Splat: camera %d/%d (%s)"
                                   % (job.done, job.total, job.mode))
            if job.done < job.total:
                return {'RUNNING_MODAL'}
            self._cleanup(context)
            return self._finish(context)
        except Exception as exc:
            return self._fail(context, exc)

    def _cleanup(self, context):
        if getattr(self, "_timer", None):
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        _gcapture_progress_clear("gcapture.clean_splat")


def _exp_sample_face_points(objs, count, scale, W, world_out=None,
                            crop_bounds=None, colors_out=None):
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
            if colors_out is not None:
                mi = np.empty(nt, dtype=np.int32)
                mesh.loop_triangles.foreach_get("material_index", mi)
                mc = np.array(_exp_object_material_colors(obj), dtype=np.float64)
                col_chunks.append(mc[np.clip(mi, 0, len(mc) - 1)])
        finally:
            obj_eval.to_mesh_clear()
    if not tri_chunks or count <= 0:
        return []
    A = np.vstack(tri_chunks)                                   # (T,3,3)
    C = np.vstack(col_chunks) if col_chunks else None           # (T,3)
    area = 0.5 * np.linalg.norm(np.cross(A[:, 1] - A[:, 0],
                                         A[:, 2] - A[:, 0]), axis=1)
    good = area > 0.0
    A, area = A[good], area[good]
    if C is not None:
        C = C[good]
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


def _exp_gather_points(scene, scale, W, s, world_out=None):
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

    # Downsample together (positions and colors in sync).
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
    # tolist(): real Python numbers for points3D.txt (see v109). Rows
    # stay lists -- building tuples from them cost several seconds with
    # 5 million points; the writing code unpacks both the same way.
    return pts.tolist(), col_all.tolist()


_EXP_SIGNATURE_VERSION = 2


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
    """Splits the .blend base name into (name_without_version, version token).
    Example: 'Scene_v001' -> ('Scene', 'v001'); 'Bugatti_Chiron_v002_Addon'
    -> ('Bugatti_Chiron', 'v002'). Takes the LAST vN token. Without a version
    -> (stem, None). Separators before the version are stripped."""
    matches = list(re.finditer(r'[vV]\d+', stem))
    if not matches:
        return stem, None
    m = matches[-1]
    name = stem[:m.start()].rstrip('_-. ')
    version = stem[m.start():m.end()]
    return name, version


def _exp_resolve_output_dir(s):
    """Returns the target folder for the COLMAP model.

    If the 'Output Folder' field is EMPTY (default), a structure is built
    automatically next to the .blend file:
        <Name>_COLMAP / <vNNN> /
    i.e. a folder named after the scene (up to the version token) plus
    '_COLMAP', and inside it a subfolder with the version token. Example:
    'Scene_v001.blend' -> 'Scene_COLMAP/v001/'. Without a version,
    'v001' is used; if the .blend is not saved yet, 'Scene_COLMAP'.

    If the field is filled in, the path is used as-is (// = relative to
    the .blend is resolved)."""
    od = (s.exp_output_dir or "").strip()
    if od:
        return bpy.path.abspath(od)
    # Empty field -> automatic structure next to the .blend.
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
    """True if the target folder already contains a COLMAP export (one of
    the sparse/0 files). Images alone do not count: since v90 the scene
    renders directly to images/, which is not an export yet."""
    if not out_dir or not os.path.isdir(out_dir):
        return False
    sparse = os.path.join(out_dir, "sparse", "0")
    for fn in ("cameras.txt", "images.txt", "points3D.txt"):
        if os.path.isfile(os.path.join(sparse, fn)):
            return True
    return False


def _exp_same_dir(a, b):
    """True if both paths point to the same folder."""
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

    # ----- Setup (one-time) -----
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
        # sparse always; images only if images are included.
        os.makedirs(self._sparse_dir, exist_ok=True)
        if s.exp_image_mode != 'NONE':
            os.makedirs(self._images_dir, exist_ok=True)

        self._src_dir = _exp_resolve_image_dir(s)
        auto_pattern, auto_ext, auto_frames = _exp_detect_frames(
            self._src_dir, prefer=_gcapture_format_ext(context.scene))
        if not auto_pattern:
            raise RuntimeError("Could not detect filename pattern in: %s"
                               % self._src_dir)
        self._pattern = auto_pattern
        self._ext = (auto_ext or "png").lstrip(".")
        # IMPORTANT: The pose count comes from the SCENE FRAME RANGE (the
        # actual camera keyframes), NOT from the existing
        # image files. Otherwise the add-on exports only as many poses as
        # there are images in the folder at the moment -- with render farm
        # images not yet fully synced, poses are then missing. Pattern and
        # image size are still derived from a sample file.
        self._frames = list(range(scene.frame_start, scene.frame_end + 1))
        # Determine missing images (warning only, the pose is not dropped).
        have = set(auto_frames or [])
        self._missing_images = [fr for fr in self._frames if fr not in have] \
            if have else []

        # Determine the effective image mode. Moving is destructive -- the
        # images are gone from the render folder afterwards. Safety check:
        # MOVE only if ALL expected images are present. If one is missing
        # (e.g. render farm not finished yet), nothing is moved; instead it
        # safely falls back to COPY, with a warning in the summary.
        self._image_mode = s.exp_image_mode
        self._move_blocked = False
        # If the scene renders directly into <out>/images (Save Scene Version),
        # the images are already in the dataset: copy nothing (v90).
        if _exp_same_dir(self._src_dir, self._images_dir):
            self._image_mode = 'INPLACE'
        if self._image_mode == 'MOVE' and self._missing_images:
            self._image_mode = 'COPY'
            self._move_blocked = True

        # Actual image size -> intrinsics + cameras.txt. Read from the first
        # image that is actually PRESENT (auto_frames), not from
        # self._frames[0] -- its image might still be missing.
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

        # Determine the scale (one-time pre-pass over all frames).
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

        # Crop box bounds (Blender world coords).
        self._crop_bounds = _exp_crop_bounds_world(s)

        # Point cloud fingerprint (v106): allows reusing the most recently
        # written points3D.txt if nothing has changed
        # (e.g. only a path).
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

        # Work lists.
        self._entries = []
        self._copied = 0
        self._missing = []

        # Work list: one entry ("pose", image ID, frame) per pose.
        self._work = [("pose", i, fr) for i, fr in enumerate(self._frames, start=1)]
        self._total = len(self._work)
        self._done = 0
        self._phase = 'work'  # 'work' -> poses, 'filter' -> visibility
        self._start_time = time.time()
        self._s = s

    @staticmethod
    def _point_targets(scene, s, guide_objs):
        # Same object selection as the main cloud. Exclude guide_objs only with
        # a guide list: without a list it is the ACTIVE object -- often the
        # model itself, which would otherwise drop out of face points and filter BVH.
        return _exp_point_source_objects(
            scene, s, guide_objs if s.use_guide_list else ())

    # ----- Per work unit -----
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

    # ----- Modal machinery -----
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

            # Filter phase (modal, real ESC above).
            if self._phase == 'filter':
                try:
                    return self._filter_tick(context)
                except Exception as exc:
                    self._cleanup(context)
                    self.report({'ERROR'}, "Visibility filter failed: %s" % exc)
                    return {'CANCELLED'}

            if self._done >= self._total:
                return self._finish(context)

            # Progress in the header.
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

            # Process several pose frames per tick (fast).
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
        # Write images.txt.
        _exp_write_images(os.path.join(self._sparse_dir, "images.txt"),
                          self._entries)

        # Reuse the unchanged point cloud (v106).
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

        # --- Collect points (fast) ---
        # Visible vertices as the main cloud, face points optionally added. If
        # a visibility filter is active, the points are NOT filtered here
        # synchronously (that blocked ESC), but as a modal phase
        # bit by bit in the tick -> real ESC. Only collect here.
        vmode = s.exp_face_points_vischeck
        want_filter = (vmode == 'RAYCAST')
        guide_objs = _collect_guide_objects(bpy.context,
                                            bpy.context.active_object)
        targets = self._point_targets(scene, s, guide_objs)
        need_world = want_filter

        # 1) Main cloud.
        world_v = [] if need_world else None
        pts, cols = _exp_gather_points(scene, self._scale, self._W, s,
                                       world_out=world_v)
        world_main = world_v
        self._n_vert = len(pts)      # for the completion message (v103)
        self._n_face = 0
        self._filtered = False

        # 2) Add face points (optional). With Use Vertex Count: 10 % of the
        # collected vertex points (v101), otherwise the entered value.
        face_count = _exp_face_count(s, len(pts))
        if s.exp_face_points:
            world_f = [] if need_world else None
            # Color like the vertex points: material of the face (v1.1.5).
            fcols = [] if (cols is not None and s.exp_point_color != 'NONE') else None
            fpts = _exp_sample_face_points(
                targets, face_count, self._scale, self._W,
                world_out=world_f, crop_bounds=self._crop_bounds,
                colors_out=fcols)
            if fpts:
                self._n_face = len(fpts)
                neutral = (200, 200, 200)
                pts = list(pts) + list(fpts)
                if cols is not None:
                    if fcols and len(fcols) == len(fpts):
                        cols = list(cols) + list(fcols)
                    else:
                        cols = list(cols) + [neutral] * len(fpts)
                if need_world and world_main is not None and world_f is not None:
                    world_main = list(world_main) + list(world_f)

        # --- Start the filter phase modally or write directly ---
        if want_filter and pts and world_main:
            faces = _gcapture_all_guide_faces(bpy.context, None,
                                         guide_objs=guide_objs)
            self._flt_cams = [f[0] for f in faces]
            # GPU depth images (v105); without a GPU context, ray cast as before.
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
                    # Compute shader (v111); else numpy per camera.
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

        # No filter -> write directly.
        return self._write_and_done(context, pts, cols, cancelled=False)

    def _filter_tick(self, context):
        """One modal filter stage: test a batch of points against the
        cameras (ray cast). Real ESC, since in the modal tick. When all
        points are tested -> write."""
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
        """Filter stage on the GPU: a few cameras per tick (time budget);
        points already seen by a camera are not tested again."""
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
        # No overall cap anymore (v98): Max Points only limits the
        # vertex points (_exp_gather_points); Face Points come on top.

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
        # Remember final point count, show it in the panel (v103).
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
        # Moving was requested but fell back to copying because of
        # missing images (safety check) -> report clearly.
        if getattr(self, "_move_blocked", False):
            msg += (" | NOTE: Move was requested but some images were "
                    "missing, so images were COPIED instead (render folder "
                    "left intact)")
        # Warning: all poses were written, but for some frames the image is
        # (still) missing from the source folder (e.g. render farm not finished
        # syncing). The poses are in the export anyway -- the images
        # can be added later.
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


if __name__ == "__main__":
    register()
