"""Progress state and panel lock of long operators (drawing is in ui.guide)."""

import bpy
import time


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
