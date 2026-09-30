"""Small drawing helpers: action buttons, progress bar, panel lock, badges."""

import time

from ..core import progress
from ..ui import icons
from ..core.progress import _GCAPTURE_PROGRESS_STALE, _gcapture_busy


def _gcapture_lock(layout):
    """Column that is locked during a run; the progress bar
    itself sits outside it and stays readable."""
    col = layout.column()
    col.enabled = not _gcapture_busy()
    return col


def _gcapture_progress_draw(layout, op_id):
    """Draws the progress of op_id if it is running -> True."""
    p = progress._GCAPTURE_PROGRESS.get(op_id)
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
    pv = icons._gcapture_previews.get(badge) if (badge and icons._gcapture_previews) else None
    if pv is not None:
        layout.label(text="", icon_value=pv.icon_id)
    elif fallback_icon:
        layout.label(text="", icon=fallback_icon)
