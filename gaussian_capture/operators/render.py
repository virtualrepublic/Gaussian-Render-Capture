"""Render, resolution confirmation, Prepare Scene, selection to collection,
scene versions."""

import bpy
import os
from bpy.props import StringProperty, IntProperty, EnumProperty, BoolProperty
from bpy.types import Operator

from ..core.render import (
    _GCAPTURE_EEVEE_PRESET, _GCAPTURE_EEVEE_SETTINGS, _GCAPTURE_START_OBJECTS,
    _gcapture_apply_image_format, _gcapture_apply_prepare, _gcapture_apply_settings,
    _gcapture_eevee_engine, _gcapture_fix_denoiser, _gcapture_render_script_path,
    _gcapture_setup_gpu, _gcapture_start_object_unchanged, _gcapture_write_render_script)
from ..core.scene import _gcapture_apply_render_setup, _gcapture_wt_status_build_poses
from ..core.versions import (
    _exp_split_name_version, _gcapture_clean_name, _gcapture_managed_render_path,
    _gcapture_next_version_path, _gcapture_render_path_is_default_or_managed,
    _gcapture_version_files)
from ..operators.guides import _gcapture_frame_collections


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
