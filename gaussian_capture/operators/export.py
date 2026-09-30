"""Export COLMAP and the relative-path helper."""

import bpy
import os
import shutil
import time
import numpy as np
from bpy.props import StringProperty
from bpy.types import Operator

from ..core.gpu_depth import _ExpGpuDepth
from ..core.progress import _gcapture_progress_clear, _gcapture_progress_set
from ..core.render import _gcapture_format_ext
from ..core.scene import _collect_guide_objects, _gcapture_all_guide_faces, _gcapture_look_dir_for
from ..core.sphere import _sph_camera_min_fov
from ..core.versions import _exp_resolve_output_dir, _exp_same_dir
from ..export.colmap import (
    _exp_crop_bounds_world, _exp_detect_frames, _exp_image_size, _exp_intrinsics, _exp_pose_w2c,
    _exp_resolve_image_dir, _exp_world_transform, _exp_write_cameras, _exp_write_images,
    _exp_write_points, _exp_write_scale_info)
from ..export.gpu_filter import _exp_visible_ray
from ..export.points import (
    _exp_face_count, _exp_gather_points, _exp_point_signature, _exp_point_source_objects,
    _exp_sample_face_points)
from ..export.raycast import _exp_build_scene_bvh


class GCAPTURE_OT_path_relative(Operator):
    bl_idname = "gcapture.path_relative"
    bl_label = "Make Path Relative"
    bl_description = ("Store this folder relative to the .blend file (//...), "
                      "so the scene keeps working when the project is moved")
    bl_options = {'REGISTER', 'UNDO'}

    prop: StringProperty(default="exp_image_dir", options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return bool(bpy.data.filepath)

    def execute(self, context):
        s = context.scene.gcapture_settings
        raw = (getattr(s, self.prop, "") or "").strip()
        if not raw:
            self.report({'WARNING'}, "Folder is empty.")
            return {'CANCELLED'}
        if raw.startswith("//"):
            self.report({'INFO'}, "Already relative.")
            return {'FINISHED'}
        try:
            rel = bpy.path.relpath(bpy.path.abspath(raw))
        except Exception as exc:
            self.report({'ERROR'}, "Cannot make relative: %s" % exc)
            return {'CANCELLED'}
        if not rel.startswith("//"):
            self.report({'WARNING'}, "Folder is on another drive - keeping "
                        "the absolute path.")
            return {'CANCELLED'}
        setattr(s, self.prop, rel)
        self.report({'INFO'}, "Now relative: %s" % rel)
        return {'FINISHED'}


class GCAPTURE_OT_export_colmap(Operator):
    bl_idname = "gcapture.export_colmap"
    bl_label = "Export COLMAP (Postshot / LichtFeld)"
    bl_description = ("Exports cameras.txt / images.txt / points3D.txt plus "
                      "an images/ folder as a COLMAP model for Postshot and "
                      "LichtFeld Studio. "
                      "Runs modally with a progress display (ESC to cancel)")
    bl_options = {'REGISTER'}

    _is_running = False

    @classmethod
    def poll(cls, context):
        return context.scene.camera is not None and not cls._is_running

    # ----- Setup (one-time) -----
    def _setup(self, context):
        scene = context.scene
        s = scene.gcapture_settings
        cam = scene.camera
        if cam is None or cam.type != 'CAMERA':
            raise RuntimeError("No active scene camera set.")

        self._cam = cam
        self._out_dir = _exp_resolve_output_dir(s)
        self._images_dir = os.path.join(self._out_dir, "images")
        self._sparse_dir = os.path.join(self._out_dir, "sparse", "0")
        # sparse always; images only if images are included.
        os.makedirs(self._sparse_dir, exist_ok=True)
        if s.exp_image_mode != 'NONE':
            os.makedirs(self._images_dir, exist_ok=True)

        self._src_dir = _exp_resolve_image_dir(s)
        auto_pattern, auto_ext, auto_frames = _exp_detect_frames(
            self._src_dir, prefer=_gcapture_format_ext(context.scene))
        if not auto_pattern:
            raise RuntimeError("Could not detect filename pattern in: %s"
                               % self._src_dir)
        self._pattern = auto_pattern
        self._ext = (auto_ext or "png").lstrip(".")
        # IMPORTANT: The pose count comes from the SCENE FRAME RANGE (the
        # actual camera keyframes), NOT from the existing
        # image files. Otherwise the add-on exports only as many poses as
        # there are images in the folder at the moment -- with render farm
        # images not yet fully synced, poses are then missing. Pattern and
        # image size are still derived from a sample file.
        self._frames = list(range(scene.frame_start, scene.frame_end + 1))
        # Determine missing images (warning only, the pose is not dropped).
        have = set(auto_frames or [])
        self._missing_images = [fr for fr in self._frames if fr not in have] \
            if have else []

        # Determine the effective image mode. Moving is destructive -- the
        # images are gone from the render folder afterwards. Safety check:
        # MOVE only if ALL expected images are present. If one is missing
        # (e.g. render farm not finished yet), nothing is moved; instead it
        # safely falls back to COPY, with a warning in the summary.
        self._image_mode = s.exp_image_mode
        self._move_blocked = False
        # If the scene renders directly into <out>/images (Save Scene Version),
        # the images are already in the dataset: copy nothing (v90).
        if _exp_same_dir(self._src_dir, self._images_dir):
            self._image_mode = 'INPLACE'
        if self._image_mode == 'MOVE' and self._missing_images:
            self._image_mode = 'COPY'
            self._move_blocked = True

        # Actual image size -> intrinsics + cameras.txt. Read from the first
        # image that is actually PRESENT (auto_frames), not from
        # self._frames[0] -- its image might still be missing.
        real_size = None
        if self._src_dir:
            probe = (auto_frames[0] if auto_frames
                     else (self._frames[0] if self._frames else None))
            if probe is not None:
                first = self._pattern.format(n=probe) + "." + self._ext
                real_size = _exp_image_size(os.path.join(self._src_dir, first))
        self._real_size = real_size
        self._W = _exp_world_transform(s)
        fx, fy, cx, cy, w, h = _exp_intrinsics(cam.data, scene, real_size)
        self._res = (w, h)
        _exp_write_cameras(os.path.join(self._sparse_dir, "cameras.txt"),
                           fx, fy, cx, cy, w, h)

        self._orig_frame = scene.frame_current

        # Determine the scale (one-time pre-pass over all frames).
        if s.exp_auto_scale:
            centers = []
            for fr in self._frames:
                scene.frame_set(fr)
                centers.append(cam.matrix_world.translation.copy())
            cxm = sum(c.x for c in centers) / len(centers)
            cym = sum(c.y for c in centers) / len(centers)
            czm = sum(c.z for c in centers) / len(centers)
            rs = [((c.x - cxm) ** 2 + (c.y - cym) ** 2 + (c.z - czm) ** 2) ** 0.5
                  for c in centers]
            mean_r = sum(rs) / len(rs) if rs else 0.0
            self._scale = s.exp_target_radius / mean_r if mean_r > 1e-9 else 1.0
            self._mean_r = mean_r
        else:
            self._scale = 1.0
            self._mean_r = None

        # Crop box bounds (Blender world coords).
        self._crop_bounds = _exp_crop_bounds_world(s)

        # Point cloud fingerprint (v106): allows reusing the most recently
        # written points3D.txt if nothing has changed
        # (e.g. only a path).
        try:
            faces = _gcapture_all_guide_faces(
                context, None, guide_objs=_collect_guide_objects(
                    context, context.active_object))
            views = [(tuple(f[0]), tuple(_gcapture_look_dir_for(s, *f)))
                     for f in faces]
            self._sig = _exp_point_signature(scene, s, views, self._scale,
                                             self._W)
        except Exception as exc:
            print("[Gaussian Render Capture] point signature failed:", exc)
            self._sig = ""

        # Work lists.
        self._entries = []
        self._copied = 0
        self._missing = []

        # Work list: one entry ("pose", image ID, frame) per pose.
        self._work = [("pose", i, fr) for i, fr in enumerate(self._frames, start=1)]
        self._total = len(self._work)
        self._done = 0
        self._phase = 'work'  # 'work' -> poses, 'filter' -> visibility
        self._start_time = time.time()
        self._s = s

    @staticmethod
    def _point_targets(scene, s, guide_objs):
        # Same object selection as the main cloud. Exclude guide_objs only with
        # a guide list: without a list it is the ACTIVE object -- often the
        # model itself, which would otherwise drop out of face points and filter BVH.
        return _exp_point_source_objects(
            scene, s, guide_objs if s.use_guide_list else ())

    # ----- Per work unit -----
    def _process_pose(self, scene, idx, frame):
        s = self._s
        scene.frame_set(frame)
        qw, qx, qy, qz, tx, ty, tz = _exp_pose_w2c(self._cam, self._scale, self._W)
        img_name = self._pattern.format(n=frame) + "." + self._ext
        self._entries.append((idx, qw, qx, qy, qz, tx, ty, tz, img_name))
        if self._image_mode == 'INPLACE':
            if os.path.isfile(os.path.join(self._images_dir, img_name)):
                self._copied += 1
            else:
                self._missing.append(img_name)
        elif self._image_mode != 'NONE' and self._src_dir:
            src = os.path.join(self._src_dir, img_name)
            dst = os.path.join(self._images_dir, img_name)
            if os.path.isfile(src):
                if os.path.abspath(src) != os.path.abspath(dst):
                    if self._image_mode == 'MOVE':
                        shutil.move(src, dst)
                    else:
                        shutil.copy2(src, dst)
                self._copied += 1
            else:
                self._missing.append(img_name)

    # ----- Modal machinery -----
    def invoke(self, context, event):
        try:
            self._setup(context)
        except Exception as exc:
            self.report({'ERROR'}, "COLMAP export failed: %s" % exc)
            return {'CANCELLED'}

        if self._total == 0:
            self.report({'ERROR'}, "Nothing to export (no frames).")
            return {'CANCELLED'}

        GCAPTURE_OT_export_colmap._is_running = True
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.05, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'ESC':
            self._cleanup(context)
            self.report({'WARNING'}, "COLMAP export cancelled.")
            return {'CANCELLED'}

        if event.type == 'TIMER':
            scene = context.scene

            # Filter phase (modal, real ESC above).
            if self._phase == 'filter':
                try:
                    return self._filter_tick(context)
                except Exception as exc:
                    self._cleanup(context)
                    self.report({'ERROR'}, "Visibility filter failed: %s" % exc)
                    return {'CANCELLED'}

            if self._done >= self._total:
                return self._finish(context)

            # Progress in the header.
            elapsed = time.time() - self._start_time
            time_info = ""
            if self._done > 0:
                per = elapsed / self._done
                rem = int(per * (self._total - self._done))
                tstr = ("%dm %ds" % (rem // 60, rem % 60)) if rem >= 60 else ("%ds" % rem)
                time_info = ", ~%s left" % tstr
            _gcapture_progress_set(
                "gcapture.export_colmap", self._done / self._total,
                "Camera poses %d/%d%s" % (self._done, self._total, time_info))

            # Process several pose frames per tick (fast).
            try:
                batch = 0
                while self._done < self._total and batch < 25:
                    _, bidx, bframe = self._work[self._done]
                    self._process_pose(scene, bidx, bframe)
                    self._done += 1
                    batch += 1
            except Exception as exc:
                self._cleanup(context)
                self.report({'ERROR'}, "COLMAP export failed: %s" % exc)
                return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def _finish(self, context):
        scene = context.scene
        s = self._s
        # Write images.txt.
        _exp_write_images(os.path.join(self._sparse_dir, "images.txt"),
                          self._entries)

        # Reuse the unchanged point cloud (v106).
        target_points = os.path.join(self._sparse_dir, "points3D.txt")
        prev = bpy.path.abspath(s.exp_last_points_file or "")
        reuse = False
        if s.exp_reuse_points and self._sig:
            if s.exp_last_signature != self._sig:
                print("[Gaussian Render Capture] point cloud changed -> "
                      "recomputing")
            elif not prev or not os.path.isfile(prev):
                print("[Gaussian Render Capture] no stored point cloud (%s) -> "
                      "recomputing" % (prev or "none"))
            else:
                reuse = True
        if reuse:
            try:
                same = (os.path.normcase(os.path.abspath(prev))
                        == os.path.normcase(os.path.abspath(target_points)))
                if not same:
                    shutil.copy2(prev, target_points)
                self._cleanup(context)
                s.exp_last_points_file = target_points
                self.report({'INFO'}, "COLMAP export done: %d poses, point "
                            "cloud unchanged - %s (%s) -> %s"
                            % (len(self._entries),
                               "kept" if same else "reused "
                               + os.path.basename(prev),
                               s.exp_last_detail or "previous export",
                               self._out_dir))
                return {'FINISHED'}
            except Exception as exc:
                print("[Gaussian Render Capture] could not reuse point cloud:",
                      exc)

        # --- Collect points in steps (1.3.0) ---
        # Visible vertices as the main cloud, face points optionally added,
        # then the visibility filter as a modal phase (real ESC). Each step
        # runs in a timer tick of its own and names itself in the progress
        # bar first - in one piece the Beetle took 48 s without a word.
        self._phase = 'filter'
        self._stage = 'vertices'
        self._start_time = time.time()
        _gcapture_progress_set("gcapture.export_colmap", 1.0,
                               "Start points: collecting the vertices ...")
        return {'RUNNING_MODAL'}

    def _stage_tick(self, context):
        """One step of collecting the start points (1.3.0); see _finish."""
        scene = context.scene
        s = self._s
        if self._stage == 'vertices':
            vmode = s.exp_face_points_vischeck
            want_filter = (vmode == 'RAYCAST')
            guide_objs = _collect_guide_objects(bpy.context,
                                                bpy.context.active_object)
            targets = self._point_targets(scene, s, guide_objs)
            need_world = want_filter

            # 1) Main cloud.
            world_v = [] if need_world else None
            glass_v = [] if need_world else None
            pts, cols = _exp_gather_points(scene, self._scale, self._W, s,
                                           world_out=world_v, glass_out=glass_v)
            world_main = world_v
            glass_main = glass_v
            self._n_vert = len(pts)      # for the completion message (v103)
            self._n_face = 0
            self._filtered = False
            self._c = dict(pts=pts, cols=cols, world_main=world_main,
                           glass_main=glass_main, targets=targets,
                           guide_objs=guide_objs, want_filter=want_filter,
                           need_world=need_world)
            self._stage = 'faces' if s.exp_face_points else 'prepare'
            _gcapture_progress_set(
                "gcapture.export_colmap", 1.0,
                "Start points: adding face points ..." if s.exp_face_points
                else ("Preparing the visibility filter ..." if want_filter
                      else "Writing the points ..."))
            return {'RUNNING_MODAL'}
        c = self._c
        pts, cols = c["pts"], c["cols"]
        world_main, glass_main = c["world_main"], c["glass_main"]
        targets, guide_objs = c["targets"], c["guide_objs"]
        want_filter, need_world = c["want_filter"], c["need_world"]
        if self._stage == 'faces':
            face_count = _exp_face_count(s, len(pts))
            if s.exp_face_points:
                world_f = [] if need_world else None
                glass_f = [] if need_world else None
                # Color like the vertex points: material of the face (v1.1.5).
                fcols = [] if (cols is not None and s.exp_point_color != 'NONE') else None
                fpts = _exp_sample_face_points(
                    targets, face_count, self._scale, self._W,
                    world_out=world_f, crop_bounds=self._crop_bounds,
                    colors_out=fcols, glass_out=glass_f)
                if fpts:
                    self._n_face = len(fpts)
                    neutral = (200, 200, 200)
                    pts = list(pts) + list(fpts)
                    if cols is not None:
                        if fcols and len(fcols) == len(fpts):
                            cols = list(cols) + list(fcols)
                        else:
                            cols = list(cols) + [neutral] * len(fpts)
                    if need_world and world_main is not None and world_f is not None:
                        world_main = list(world_main) + list(world_f)
                        glass_main = list(glass_main) + list(glass_f)
            c.update(pts=pts, cols=cols, world_main=world_main,
                     glass_main=glass_main)
            self._stage = 'prepare'
            _gcapture_progress_set(
                "gcapture.export_colmap", 1.0,
                "Preparing the visibility filter ..." if want_filter
                else "Writing the points ...")
            return {'RUNNING_MODAL'}
        # 'prepare': start the filter phase or write directly
        self._stage = None
        self._c = None
        if want_filter and pts and world_main:
            faces = _gcapture_all_guide_faces(bpy.context, None,
                                         guide_objs=guide_objs)
            self._flt_cams = [f[0] for f in faces]
            # GPU depth images (v105); without a GPU context, ray cast as before.
            # Glass (1.3.0): points behind glass are tested with the glass
            # removed (a car's interior seen through its windows), points ON
            # glass with the glass in place - only the panes seen from outside
            # keep theirs. Like merging a cloud without the glass objects
            # with one that has them (maintainer, 27.09.2026).
            self._flt_glass = np.array(glass_main, dtype=bool)
            on_glass = np.nonzero(self._flt_glass)[0]
            off_glass = np.nonzero(~self._flt_glass)[0]
            self._flt_gpu = None
            if not bpy.app.background:
                try:
                    self._flt_views = [
                        (f[0], _gcapture_look_dir_for(s, f[0], f[1], f[2], f[3]))
                        for f in faces]
                    self._flt_fov = _sph_camera_min_fov(s, scene)
                    self._flt_np = np.array([tuple(p) for p in world_main],
                                            dtype=np.float64)
                    self._flt_vis = np.zeros(len(world_main), dtype=bool)
                    self._flt_cam_i = 0
                    # (depth maps, indices of the points they test)
                    self._flt_groups = []
                    if len(off_glass):
                        try:
                            self._flt_groups.append(
                                (_ExpGpuDepth(targets, see_through=True), off_glass))
                        except RuntimeError:          # everything is glass
                            self._flt_vis[off_glass] = True
                    if len(on_glass):
                        self._flt_groups.append((_ExpGpuDepth(targets), on_glass))
                    self._flt_gpu = (self._flt_groups[0][0] if self._flt_groups
                                     else _ExpGpuDepth(targets))
                    # Compute shader (v111); else numpy per camera.
                    self._flt_compute = False
                    try:
                        for g, idx in self._flt_groups:
                            g.compute_begin(self._flt_np[idx])
                        self._flt_compute = True
                    except Exception as exc:
                        print("[Gaussian Render Capture] GPU compute unavailable, "
                              "checking points with numpy:", exc)
                except Exception as exc:
                    print("[Gaussian Render Capture] GPU visibility unavailable, "
                          "using ray casting:", exc)
                    self._flt_gpu = None
            if self._flt_gpu is None:
                # ray casting; None = no geometry in the way
                self._flt_bvh = _exp_build_scene_bvh(targets, see_through=True)
                self._flt_bvh_solid = (_exp_build_scene_bvh(targets)
                                       if len(on_glass) else None)
            else:
                self._flt_bvh = self._flt_bvh_solid = None
            self._flt_world = world_main
            self._flt_pts = pts
            self._flt_cols = cols
            self._flt_idx = 0
            self._flt_kept_pts = []
            self._flt_kept_cols = []
            self._flt_total = len(pts)
            self._phase = 'filter'
            self._filtered = True
            self._start_time = time.time()
            return {'RUNNING_MODAL'}

        # No filter -> write directly.
        return self._write_and_done(context, pts, cols, cancelled=False)

    def _filter_tick(self, context):
        """One modal filter stage: test a batch of points against the
        cameras (ray cast). Real ESC, since in the modal tick. When all
        points are tested -> write."""
        if getattr(self, "_stage", None):
            return self._stage_tick(context)
        if getattr(self, "_flt_gpu", None) is not None:
            return self._filter_tick_gpu(context)
        BATCH = 4000
        end = min(self._flt_idx + BATCH, self._flt_total)
        cams = self._flt_cams
        glass = getattr(self, "_flt_glass", None)
        for i in range(self._flt_idx, end):
            wp = self._flt_world[i]
            bvh = (self._flt_bvh_solid if glass is not None and glass[i]
                   else self._flt_bvh)
            if bvh is None or _exp_visible_ray(wp, bvh, cams):
                self._flt_kept_pts.append(self._flt_pts[i])
                if self._flt_cols is not None:
                    self._flt_kept_cols.append(self._flt_cols[i])
        self._flt_idx = end

        # Header.
        _gcapture_progress_set(
            "gcapture.export_colmap", self._flt_idx / max(self._flt_total, 1),
            "Visibility filter %d/%d points, %d kept"
            % (self._flt_idx, self._flt_total, len(self._flt_kept_pts)))

        if self._flt_idx >= self._flt_total:
            cols = self._flt_kept_cols if self._flt_cols is not None else None
            return self._write_and_done(context, self._flt_kept_pts, cols,
                                        cancelled=False)
        return {'RUNNING_MODAL'}

    def _filter_tick_gpu(self, context):
        """Filter stage on the GPU: a few cameras per tick (time budget);
        points already seen by a camera are not tested again."""
        t0 = time.time()
        n_cams = len(self._flt_views)
        groups = self._flt_groups
        if getattr(self, "_flt_compute", False):
            while self._flt_cam_i < n_cams and time.time() - t0 < 0.1:
                pos, look = self._flt_views[self._flt_cam_i]
                for g, idx in groups:
                    g.compute_camera(pos, look, self._flt_fov)
                self._flt_cam_i += 1
            if self._flt_cam_i >= n_cams:
                for g, idx in groups:
                    self._flt_vis[idx] = g.compute_result()
        while self._flt_cam_i < n_cams and time.time() - t0 < 0.25:
            pos, look = self._flt_views[self._flt_cam_i]
            left = False
            for g, idx in groups:
                todo = idx[~self._flt_vis[idx]]
                if len(todo):
                    left = True
                    seen = g.visible(self._flt_np[todo], pos, look, self._flt_fov)
                    self._flt_vis[todo[seen]] = True
            if not left:
                self._flt_cam_i = n_cams
                break
            self._flt_cam_i += 1
        if getattr(self, "_flt_compute", False):
            ftext = "Visibility filter (GPU): camera %d/%d" % (
                self._flt_cam_i, n_cams)
        else:
            ftext = ("Visibility filter (GPU): camera %d/%d, %d of %d points "
                     "visible" % (self._flt_cam_i, n_cams,
                                  int(self._flt_vis.sum()), self._flt_total))
        _gcapture_progress_set("gcapture.export_colmap",
                               self._flt_cam_i / max(n_cams, 1), ftext)
        if self._flt_cam_i >= n_cams:
            keep = np.nonzero(self._flt_vis)[0]
            pts = [self._flt_pts[i] for i in keep]
            cols = ([self._flt_cols[i] for i in keep]
                    if self._flt_cols is not None else None)
            self._flt_gpu = None
            self._flt_groups = []
            return self._write_and_done(context, pts, cols, cancelled=False)
        return {'RUNNING_MODAL'}

    def _write_and_done(self, context, pts, cols, cancelled=False):
        s = self._s
        # No overall cap anymore (v98): Max Points only limits the
        # vertex points (_exp_gather_points); Face Points come on top.

        if cancelled:
            self.report({'WARNING'},
                        "Cancelled; exporting the points collected so far.")

        points_path = os.path.join(self._sparse_dir, "points3D.txt")
        _exp_write_points(points_path, pts, cols)
        try:
            _exp_write_scale_info(self._out_dir, s, self._scale,
                                  getattr(self, "_mean_r", None),
                                  len(self._entries), len(pts))
        except OSError as exc:
            self.report({'WARNING'}, "<vNNN>_gcapture_export.json not written: %s" % exc)
        if getattr(self, "_sig", ""):
            s.exp_last_signature = self._sig
            s.exp_last_points_file = points_path

        self._cleanup(context)

        verb = {'MOVE': "moved", 'INPLACE': "already in the dataset"}.get(
            self._image_mode, "copied")
        # Remember final point count, show it in the panel (v103).
        n_v = getattr(self, "_n_vert", len(pts))
        n_f = getattr(self, "_n_face", 0)
        detail = "{:,} vertex".format(n_v)
        if n_f:
            detail += " + {:,} face".format(n_f)
        if getattr(self, "_filtered", False):
            detail += " points, {:,} hidden removed".format(
                max(0, n_v + n_f - len(pts)))
        else:
            detail += " points, no visibility filter"
        s.exp_last_points = len(pts)
        s.exp_last_detail = "{} ({})".format(
            os.path.basename(os.path.normpath(self._out_dir)), detail)
        msg = ("COLMAP export done: %d poses, %dx%d (%s), %d images %s, "
               "{:,} points ({}), scale {:.6g} -> %s".format(
                   len(pts), detail, self._scale)
               % (len(self._entries), self._res[0], self._res[1],
                  "image file" if self._real_size is not None else "render settings",
                  self._copied, verb, self._out_dir))
        # Moving was requested but fell back to copying because of
        # missing images (safety check) -> report clearly.
        if getattr(self, "_move_blocked", False):
            msg += (" | NOTE: Move was requested but some images were "
                    "missing, so images were COPIED instead (render folder "
                    "left intact)")
        # Warning: all poses were written, but for some frames the image is
        # (still) missing from the source folder (e.g. render farm not finished
        # syncing). The poses are in the export anyway -- the images
        # can be added later.
        miss = getattr(self, "_missing_images", []) or self._missing
        if miss:
            ex = miss[0]
            ex = ex if isinstance(ex, str) else ("frame %d" % ex)
            msg += (" | WARNING: %d source image(s) missing (e.g. %s) -- "
                    "poses exported anyway, add the images later"
                    % (len(miss), ex))
            level = {'WARNING'}
        else:
            level = {'INFO'}
        self.report(level, msg)
        return {'FINISHED'}

    def _cleanup(self, context):
        if getattr(self, "_timer", None):
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        _gcapture_progress_clear("gcapture.export_colmap")
        try:
            context.scene.frame_set(self._orig_frame)
        except Exception:
            pass
        GCAPTURE_OT_export_colmap._is_running = False
