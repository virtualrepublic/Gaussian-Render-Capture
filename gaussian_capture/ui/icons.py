"""Coloured number badges of the steps (bpy.utils.previews)."""

import bpy
import os
import math
import numpy as np
import struct as _gcapture_struct
import tempfile as _gcapture_tempfile
import zlib as _gcapture_zlib


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
