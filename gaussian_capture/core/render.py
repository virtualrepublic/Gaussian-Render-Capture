"""Prepare Scene settings, GPU setup, image format and the command line render script."""

import bpy
import os
import sys
from mathutils import Vector


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
    # Transparent Glass is not set here (1.3.1): the scene keeps its own value,
    # Blender's default off (opaque glass, see _gcapture_get_transparent_glass).
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


def _gcapture_get_transparent_glass(settings):
    """Option Transparent Glass = the scene's own Cycles setting (1.3.1,
    maintainer 30.09.2026): an imported scene shows what it renders with, a
    fresh one Blender's default off. Off is also the better choice for splats
    seen through windows: glass stays opaque in alpha and shows what lies
    behind it; transparent panes come out blotchy and leave a haze (Beetle,
    27.09.2026)."""
    cyc = getattr(settings.id_data, "cycles", None)
    return bool(getattr(cyc, "film_transparent_glass", False))


def _gcapture_set_transparent_glass(settings, value):
    """Writes the option straight into the scene's Cycles setting (1.3.1)."""
    cyc = getattr(settings.id_data, "cycles", None)
    if cyc is not None and hasattr(cyc, "film_transparent_glass"):
        cyc.film_transparent_glass = bool(value)


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
        im.tiff_codec = 'LZW'


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
