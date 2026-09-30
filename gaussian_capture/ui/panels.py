"""Main panel and subpanels in the 3D viewport sidebar."""

from bpy.types import Panel

from ..ui import icons
from ..core.render import _gcapture_scene_prepared
from ..core.scene import _gcapture_has_custom_guides
from ..ui.guide import _gcapture_draw_clean, _gcapture_draw_walkthrough
from ..ui.icons import (
    _GCAPTURE_COLOR_CAMERAS, _GCAPTURE_COLOR_COLMAP, _GCAPTURE_COLOR_OUTPUT, _GCAPTURE_COLOR_PREP,
    _GCAPTURE_COLOR_SPLAT)
from ..ui.sections import (
    _gcapture_draw_build, _gcapture_draw_camera, _gcapture_draw_export, _gcapture_draw_group,
    _gcapture_draw_render, _gcapture_draw_sphere, _gcapture_draw_target)
from ..ui.widgets import _gcapture_action, _gcapture_lock


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
        layout.prop(s, "prep_see_through_glass")
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
        if self.gcapture_badge and icons._gcapture_previews is not None:
            badge = icons._gcapture_previews.get(self.gcapture_badge)
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
