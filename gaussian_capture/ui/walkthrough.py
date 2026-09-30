"""Operators of the guide: start, navigation, exit (they use the step table)."""

from bpy.props import IntProperty
from bpy.types import Operator

from ..ui.guide import _GCAPTURE_WT_STEPS


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
