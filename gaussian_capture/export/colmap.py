"""COLMAP dataset: transform, scale info, frames, intrinsics, poses, colours,
crop, text writers, output folders."""

import bpy
import os
import re
from mathutils import Vector, Matrix

from .. import bl_info


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
