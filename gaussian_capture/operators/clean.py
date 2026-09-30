"""Clean Splat operators."""

import bpy
import os
import time
from bpy.props import StringProperty
from bpy.types import Operator

from .. import bl_info
from ..clean.carve import (
    _ClnJob, _cln_check, _cln_dataset_views, _cln_load_transform, _cln_model_size,
    _cln_render_geometry, _cln_result_text, _cln_splat_points)
from ..clean.ply import _gcapture_ply_read, _gcapture_ply_write
from ..core.gpu_depth import _ExpGpuDepth
from ..core.progress import _gcapture_busy, _gcapture_progress_clear, _gcapture_progress_set
from ..core.versions import _exp_resolve_output_dir


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
        warn = k > n / 2
        msg = _cln_result_text(k, n, os.path.basename(out), mode, warn)
        self.report({'WARNING'} if warn else {'INFO'}, msg)
        s.clean_last_result = msg
        print("[Gaussian Render Capture] Clean Splat: %s" % msg)
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
