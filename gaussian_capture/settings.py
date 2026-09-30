"""Settings of the add-on (N panel, stored in the .blend): property groups and
their update callbacks."""

import bpy
from bpy.props import StringProperty, IntProperty, EnumProperty, FloatProperty, CollectionProperty, BoolProperty, PointerProperty
from bpy.types import PropertyGroup

from .core.live_lock import _gcapture_on_lock_toggle
from .core.render import (
    _gcapture_apply_image_format, _gcapture_get_transparent_glass, _gcapture_set_transparent_glass)
from .core.scene import _gcapture_apply_render_setup


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
    prep_see_through_glass: BoolProperty(
        name="Transparent Glass",
        description="Blender's Transparent Glass (Render Properties > Film), shown "
                    "and set here; the add-on never changes it otherwise. On: glass "
                    "becomes transparent against the empty background. Only for "
                    "objects that are mostly glass, or for splats shown in front of "
                    "other backgrounds - the panes then come out semi-transparent "
                    "and blotchy, a haze in the splat. Off (Blender's default, "
                    "recommended): glass shows what lies behind it - a car's "
                    "interior stays clean; where the background is seen through "
                    "the glass, the world color of the scene appears",
        # 1.3.1: mirrors the scene's Cycles setting instead of storing its own
        # value, so an imported scene shows what it really renders with.
        get=_gcapture_get_transparent_glass,
        set=_gcapture_set_transparent_glass,
    )
    render_format: EnumProperty(
        name="Image Format",
        description="File format of the rendered images. Both are read by "
                    "Postshot and LichtFeld Studio",
        items=[('PNG', "PNG", "PNG, RGBA 8 bit"),
               ('TIFF', "TIFF", "TIFF, RGBA 8 bit, LZW compression - "
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
