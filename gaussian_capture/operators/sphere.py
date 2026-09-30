"""Camera sphere, Live Camera Adjust, build of the animation."""

import bpy
import time
from mathutils import Vector, Matrix
from bpy.props import EnumProperty, BoolProperty
from bpy.types import Operator

from ..core import live_lock
from ..core.live_lock import _gcapture_lock_apply
from ..core.progress import _gcapture_progress_clear, _gcapture_progress_set
from ..core.scene import (
    _SPH_CENTER_PROP, _collect_guide_objects, _gcapture_has_custom_guides, _gcapture_move_to_rig,
    _gcapture_rig_collection, _gcapture_sphere_origin_to_center, _guide_faces_with_targets,
    _sph_ensure_base, _sph_has_adjustments, _sph_reset_adjustments, _sph_store_view_adjustment,
    get_or_create_camera, iter_action_fcurves, setup_guide_object)
from ..core.sphere import (
    _SPH_FIT_ATTR, _SPH_FIT_MIN, _SPH_SHIFT_ATTR, _sph_apply_camera_clip,
    _sph_apply_render_settings, _sph_camera_min_fov, _sph_center_shifts, _sph_collect_objects,
    _sph_framed_positions, _sph_model_points, _sph_required_distances, _sph_required_radius,
    _sph_world_bounds)
from ..export.raycast import (
    _rc_build_visible_mesh_bvh_cache, _rc_geometry_within_near_clip, _rc_is_camera_inside_mesh)


class GCAPTURE_OT_make_sphere(Operator):
    bl_idname = "gcapture.make_sphere"
    bl_label = "Create / Update Camera Sphere"
    bl_description = ("Create an Ico-Sphere camera array, auto-scaled so the "
                      "target collections' objects are fully framed from "
                      "every direction. Re-run to update subdivisions/scale")
    bl_options = {'REGISTER', 'UNDO'}

    SPHERE_NAME = "Camera_Sphere"

    @classmethod
    def poll(cls, context):
        return len(context.scene.gcapture_settings.sph_target_colls) > 0

    def execute(self, context):
        import bmesh as _bm
        s = context.scene.gcapture_settings
        scene = context.scene

        # Determine the existing sphere FIRST (before the bounds calculation),
        # so that it can be excluded from the size calculation.
        # PRIMARILY via the stored reference (sph_object); name as
        # fallback (a build could have renamed it).
        existing = s.sph_object
        if existing is None or existing.name not in bpy.data.objects:
            existing = bpy.data.objects.get(self.SPHERE_NAME)
        # Sphere deleted in the viewport: Blender only removes it from the
        # scene; the stored reference (sph_object) keeps the object alive --
        # without a collection it cannot be selected (v94).
        # If it is in no scene anymore: remove it for good and recreate it;
        # if it is only in another scene: leave it there.
        if existing is not None and existing.name not in context.scene.objects:
            if not existing.users_scene:
                if s.sph_object == existing:
                    s.sph_object = None
                bpy.data.objects.remove(existing, do_unlink=True)
            existing = None

        # Target objects + bounds -- exclude the sphere itself, otherwise
        # the radius grows with every update (accumulated scaling).
        objs = _sph_collect_objects(s, exclude=existing)
        if not objs:
            self.report({'ERROR'},
                        "No visible mesh objects in the target collections "
                        "(hidden objects are ignored).")
            return {'CANCELLED'}
        bounds = _sph_world_bounds(objs)
        if bounds is None:
            self.report({'ERROR'}, "Could not compute object bounds.")
            return {'CANCELLED'}
        mn, mx, center, max_dim, diag, biggest = bounds
        if diag < 1e-6:
            self.report({'ERROR'}, "Object bounds are degenerate (zero size).")
            return {'CANCELLED'}

        # Radius from FOV + SPACE DIAGONAL + margin. The diagonal (instead of
        # the longest axis) makes the framing shape-independent: compact and
        # elongated objects are framed equally well with the same margin.
        fov = _sph_camera_min_fov(s, scene)
        radius = _sph_required_radius(diag, fov, s.sph_margin)
        fit_msg = ""

        # Radius into the vertices, center into the object position (v95): the
        # origin lies at the sphere center so that scaling with S acts around
        # the center (Live Camera Lock).
        mw = Matrix.Scale(radius, 4)

        # Build the sphere mesh in ONE pass: create a unit ico sphere,
        # optionally remove the lower half, bake the vertices directly with mw
        # into world coordinates -- ALL in the bmesh, BEFORE it is assigned to
        # the object. This way there is no read-after-assign and no dependency
        # on update timing (that was the source of errors that made the
        # sphere appear to shrink when reused).
        new_mesh = bpy.data.meshes.new(self.SPHERE_NAME)
        bm = _bm.new()
        _bm.ops.create_icosphere(bm, subdivisions=s.sph_subdivisions,
                                 radius=1.0)
        if s.sph_upper_only:
            bm.verts.ensure_lookup_table()
            to_del = [v for v in bm.verts if v.co.z < -1e-5]
            if to_del:
                _bm.ops.delete(bm, geom=to_del, context='VERTS')

        # Calibrate the framing to the sequence (v96): per face (= camera)
        # compute the required distance from the real model points.
        fit_q = None
        shifts = None
        if s.sph_fit_each:
            pts = _sph_model_points(objs)
            if pts is not None:
                bm.faces.ensure_lookup_table()
                unit_c = [f.calc_center_median() for f in bm.faces]
                dirs = [c.normalized() for c in unit_c]
                # At least ~1 pixel of clearance to the image border (at margin 1.0
                # the model otherwise touches the border exactly -> rounding).
                px_guard = 1.0 + 2.0 / max(s.resolution, 16)
                shifts = None
                # With Look Target "Origin" the camera no longer aimed parallel
                # after the offset -> no offset there.
                if s.sph_center_view and s.look_mode != 'ORIGIN':
                    framed = _sph_framed_positions(pts, center, dirs, fov,
                                                   s.sph_margin * px_guard)
                    need = [d for d, _ in framed]
                    shifts = [sv for _, sv in framed]
                else:
                    need = _sph_required_distances(pts, center, dirs, fov,
                                                   s.sph_margin * px_guard)
                # Camera sits at the face center: distance = radius * |c_unit|.
                ratios = [d / max(c.length, 1e-9)
                          for d, c in zip(need, unit_c)]
                limit = max(range(len(ratios)), key=ratios.__getitem__)
                radius = ratios[limit]
                fit_q = [max(_SPH_FIT_MIN, min(1.0, rr / radius))
                         for rr in ratios]
                if shifts is not None:
                    # Final distance per camera (after the limit)
                    # and center on it (v135).
                    finals = [q * radius * c.length
                              for q, c in zip(fit_q, unit_c)]
                    shifts = _sph_center_shifts(pts, center, dirs, fov,
                                                finals, shifts)
                fit_msg = " Each view filled%s; widest view: camera %d." % (
                    " and centred" if shifts is not None else "", limit + 1)
        # Bake vertices into object coordinates (unit -> radius*v). Build the
        # matrix only here: Fill Each View can change the radius above.
        mw = Matrix.Scale(radius, 4)
        for v in bm.verts:
            v.co = mw @ v.co
        bm.to_mesh(new_mesh)
        bm.free()
        if fit_q is not None and len(fit_q) == len(new_mesh.polygons):
            attr = new_mesh.attributes.new(_SPH_FIT_ATTR, 'FLOAT', 'FACE')
            attr.data.foreach_set("value", fit_q)
            # Lateral offset per camera (v135). The sphere has no
            # rotation/scale -> world offset = object offset.
            if (s.sph_fit_each and s.sph_center_view and shifts is not None
                    and len(shifts) == len(new_mesh.polygons)):
                sattr = new_mesh.attributes.new(_SPH_SHIFT_ATTR,
                                                'FLOAT_VECTOR', 'FACE')
                sattr.data.foreach_set(
                    "vector", [c for v in shifts for c in (v.x, v.y, v.z)])

        # Reuse the existing sphere (replace the mesh) or create a new one.
        if existing is not None and existing.type == 'MESH':
            sphere_obj = existing
            if sphere_obj.name != self.SPHERE_NAME:
                sphere_obj.name = self.SPHERE_NAME
            old_mesh = sphere_obj.data
            sphere_obj.data = new_mesh
            if old_mesh.users == 0:
                bpy.data.meshes.remove(old_mesh)
        else:
            sphere_obj = bpy.data.objects.new(self.SPHERE_NAME, new_mesh)
            _gcapture_rig_collection(context.scene).objects.link(sphere_obj)

        _gcapture_move_to_rig(context.scene, sphere_obj)   # dedicated collection (v136)
        # Object transform: only the location (center); the scale
        # is baked into the vertices.
        sphere_obj.matrix_world = Matrix.Translation(center)
        # Placement after Create / Update (v133): Build resets the sphere to
        # this and thereby discards Live Camera Adjust changes.
        sphere_obj["gcapture_base_loc"] = tuple(center)
        # Remember the center (object coordinates, i.e. the origin):
        # target point of the cameras in CENTER mode, also for the hemisphere.
        sphere_obj[_SPH_CENTER_PROP] = (0.0, 0.0, 0.0)

        # Display as wireframe, do not render.
        sphere_obj.display_type = 'WIRE'
        sphere_obj.hide_render = True

        n_faces = len(sphere_obj.data.polygons)

        # Store a reference to the created sphere so that a later
        # Update finds it again even after a rename (by Build).
        s.sph_object = sphere_obj

        # Set as guide mesh -- FIXED in the guide list, not via the
        # active object. Otherwise Build takes the currently active object (e.g.
        # the cube), builds the cameras on ITS faces and the camera sticks
        # to the object surface instead of sitting on the sphere.
        if s.sph_set_as_guide:
            # Clear the guide list and add only the sphere.
            s.guides.clear()
            item = s.guides.add()
            item.label = sphere_obj.name
            ref = item.objects.add()
            ref.obj = sphere_obj
            s.guide_index = 0
            s.use_guide_list = True
            # Select/activate the sphere (cosmetic).
            for o in context.selected_objects:
                o.select_set(False)
            sphere_obj.select_set(True)
            context.view_layer.objects.active = sphere_obj
            # Look mode to center (the sphere looks inward).
            s.look_mode = 'CENTER'
            # Prevent renaming during Build: the sphere should always be named
            # "Camera_Sphere" so that Update always finds it again.
            s.rename_guide = False

        # Render settings for splatting: set ONLY ONCE per file.
        # Later updates leave the settings (possibly adjusted by the user)
        # untouched.
        render_msg = ""
        if not s.sph_render_setup_done:
            skipped = _sph_apply_render_settings(scene)
            _sph_apply_camera_clip(s)
            s.sph_render_setup_done = True
            if skipped:
                render_msg = (" Render setup applied (could not set: %s)."
                              % ", ".join(skipped))
            else:
                render_msg = " Render setup applied."

        self.report({'INFO'},
                    "Camera sphere: %d cameras, radius %.2f (from %d object(s), "
                    "diagonal %.2f, max dim %.2f, biggest part: '%s' %.2f, "
                    "margin %.0f%%)%s.%s%s"
                    % (n_faces, radius, len(objs), diag, max_dim,
                       biggest[0], biggest[1],
                       (s.sph_margin - 1.0) * 100,
                       ", upper hemisphere" if s.sph_upper_only else "",
                       fit_msg, render_msg))
        return {'FINISHED'}


class GCAPTURE_OT_lock_start(Operator):
    bl_idname = "gcapture.lock_start"
    bl_label = "Live Camera Adjust"
    bl_description = ("Adjust the camera of the current frame live: selects the "
                      "sphere; press S in the viewport to scale it (G to move) "
                      "- the camera follows. Then apply it to this camera or "
                      "to all cameras")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        s = context.scene.gcapture_settings
        return (not s.live_lock and
                bool(_collect_guide_objects(context, context.active_object)))

    def invoke(self, context, event):
        s = context.scene.gcapture_settings
        sph = s.sph_object
        if sph is None or sph.name not in context.scene.objects:
            guides = _collect_guide_objects(context, context.active_object)
            sph = guides[0] if guides else None
        if sph is None:
            self.report({'ERROR'}, "No camera sphere in the scene.")
            return {'CANCELLED'}
        # Older sphere: origin to the center so that S scales around the center.
        _gcapture_sphere_origin_to_center(sph)
        # Add the placement after Create / Update so that Build resets G/S.
        _sph_ensure_base(sph, s)
        # Select the sphere as the only object and make it active.
        for o in context.selected_objects:
            o.select_set(False)
        sph.select_set(True)
        context.view_layer.objects.active = sph
        # Enable Lock (freezes the face of the current frame).
        s.live_lock = True

        # No automatic scaling (v97): confusing for beginners.
        # The sphere is selected; the user presses S themselves.
        self.report({'INFO'}, "Live Camera Adjust on - press S in the viewport "
                    "to scale the sphere, G to move it. The camera follows.")
        return {'FINISHED'}


class GCAPTURE_OT_lock_finish(Operator):
    bl_idname = "gcapture.lock_finish"
    bl_label = "Finish Live Camera Adjust"
    bl_description = ("Finish Live Camera Adjust and apply the adjustment to "
                      "this camera only or to all cameras")
    bl_options = {'REGISTER', 'UNDO'}

    apply_to: EnumProperty(
        name="Apply to",
        items=[('THIS', "Only this camera",
                "Keep the adjusted view for this camera only: the sphere goes "
                "back to how it was; kept until the next Build Camera "
                "Animation"),
               ('ALL', "All cameras",
                "Keep the changed sphere and rebuild all camera poses from it"),
               ('CANCEL', "Cancel",
                "Undo the adjustment: the sphere and the camera go back to "
                "how they were")],
        default='ALL', options={'SKIP_SAVE'},
    )

    @classmethod
    def poll(cls, context):
        return context.scene.gcapture_settings.live_lock

    def execute(self, context):
        s = context.scene.gcapture_settings
        if self.apply_to == 'CANCEL':
            # Reset the sphere and re-place the camera of the frozen frame
            # while Lock is still active (keyframe), then Lock off (v132).
            for g in live_lock._GCAPTURE_LOCK_GUIDES:
                if g is not None and g.name in live_lock._GCAPTURE_LOCK_START_MW:
                    g.matrix_world = live_lock._GCAPTURE_LOCK_START_MW[g.name]
            context.view_layer.update()
            live_lock._GCAPTURE_LOCK_BUSY = True
            try:
                _gcapture_lock_apply(context, scene=context.scene)
            finally:
                live_lock._GCAPTURE_LOCK_BUSY = False
            s.live_lock = False
            self.report({'INFO'}, "Live Camera Adjust cancelled - sphere and "
                        "camera are back.")
            return {'FINISHED'}
        if self.apply_to == 'THIS':
            # First turn Lock off (handler gone), then save the pose and reset
            # the sphere -- otherwise the handler would drag the camera along.
            idx = live_lock._GCAPTURE_LOCK_FROZEN_IDX
            guides = [g for g in live_lock._GCAPTURE_LOCK_GUIDES if g is not None]
            s.live_lock = False
            msg = self._keep_this_view(idx, guides)
            self.report({'INFO'}, msg)
            # As after a Build: save a version (v128).
            if (s.save_after_build and not bpy.app.background
                    and getattr(context, "window", None)):
                try:
                    bpy.ops.gcapture.save_version('INVOKE_DEFAULT')
                except Exception as exc:
                    print("[Gaussian Render Capture] Save prompt failed:", exc)
            return {'FINISHED'}
        # Turn Lock off (triggers _gcapture_on_lock_toggle -> handler gone,
        # frozen index reset).
        s.live_lock = False
        # Full recalculation via the modal Build operator.
        bpy.ops.gcapture.build_animation('INVOKE_DEFAULT', keep_adjustments=True)
        self.report({'INFO'}, "Live adjust finished -> rebuilding all poses.")
        return {'FINISHED'}

    @staticmethod
    def _keep_this_view(idx, guides):
        """Save the pose of the frozen camera at the face and reset the guide
        meshes to their placement at the time Lock was enabled (v123)."""
        if idx is None or not guides:
            return "Live adjust finished."
        base = 0
        for g in guides:
            faces = _guide_faces_with_targets(g)
            if idx < base + len(faces):
                center, _, geo_center, _ = faces[idx - base]
                mw0 = live_lock._GCAPTURE_LOCK_START_MW.get(g.name, g.matrix_world.copy())
                inv = mw0.inverted()
                _sph_store_view_adjustment(g, idx - base, inv @ center,
                                           inv @ geo_center)
                for other in guides:
                    if other.name in live_lock._GCAPTURE_LOCK_START_MW:
                        other.matrix_world = live_lock._GCAPTURE_LOCK_START_MW[other.name]
                return ("Live adjust finished -> view %d kept for this camera "
                        "only; the sphere is back." % (idx + 1))
            base += len(faces)
        return "Live adjust finished."


class GCAPTURE_OT_build(Operator):
    bl_idname = "gcapture.build_animation"
    bl_label = "Build Camera Animation"
    bl_description = ("Builds a camera animation from the faces of the camera "
                      "sphere (one keyframe per face), fresh from the step 4 "
                      "settings - Live Camera Adjust changes are discarded. "
                      "Runs modally with a progress display (ESC to cancel)")
    bl_options = {'REGISTER', 'UNDO'}

    # Only "All Cameras" (Live Camera Adjust) rebuilds with the modified
    # sphere; otherwise fresh from step 3 (v133).
    keep_adjustments: BoolProperty(default=False, options={'SKIP_SAVE', 'HIDDEN'})

    _is_running = False

    @classmethod
    def poll(cls, context):
        if cls._is_running:
            return False
        s = context.scene.gcapture_settings
        # Blocked while Live Camera Lock is active -- run Finish Live
        # Adjust first, otherwise the live-adjusted keyframes would be
        # overwritten.
        if s.live_lock:
            return False
        if s.use_guide_list and len(s.guides) > 0:
            return True
        obj = context.active_object
        return obj is not None and obj.type == 'MESH'

    def _setup(self, context):
        s = context.scene.gcapture_settings
        obj = context.active_object
        self._use_list = s.use_guide_list and len(s.guides) > 0
        if not self._use_list and (obj is None or obj.type != 'MESH'):
            raise RuntimeError("No active mesh object selected.")

        self._guide_objs = _collect_guide_objects(context, obj)
        if not self._guide_objs:
            raise RuntimeError("No camera sphere in the scene - create it "
                               "first (step 4: Create / Update Camera Sphere).")
        self._reset = False
        if not getattr(self, "keep_adjustments", False):
            sph = s.sph_object
            if sph is not None and sph in self._guide_objs:
                self._reset = _sph_reset_adjustments(sph)
                if self._reset:
                    context.view_layer.update()

        # Display guide meshes as wireframe, exclude them from the render.
        for gobj in self._guide_objs:
            setup_guide_object(gobj, s)

        self._cam = get_or_create_camera(s.camera_name)
        self._cam.data.lens_unit = 'MILLIMETERS'
        self._cam.data.lens = s.focal_length
        if self._cam.animation_data:
            self._cam.animation_data_clear()
        self._cam.rotation_mode = 'QUATERNION'

        # Collect all faces of all guide meshes as a flat work list.
        self._faces = []
        for gobj in self._guide_objs:
            self._faces.extend(_guide_faces_with_targets(gobj))
        self._total = len(self._faces)
        if self._total == 0:
            raise RuntimeError("Guides have no faces.")

        # Viewport display size of the camera to match the sphere (v1.1.5):
        # Blender's 1 m hides small models. Only as long as the user has not
        # set it themselves (Blender default or set by us).
        cam_data = self._cam.data
        if (abs(cam_data.display_size - 1.0) < 1e-6
                or cam_data.get("gcapture_display_auto")):
            radius = max((max(g.dimensions) * 0.5 for g in self._guide_objs),
                         default=0.0)
            if radius > 0.0:
                cam_data.display_size = max(radius * 0.1, 0.001)
                cam_data["gcapture_display_auto"] = True

        # Prepare interior detection (once). Only effective with custom
        # guide meshes: the camera sphere always lies outside the
        # model, where the test would only slow things down (v93).
        self._skip_interior = s.skip_interior and _gcapture_has_custom_guides(s)
        self._s = s
        self._guide_set = set(self._guide_objs)
        self._bvh_cache = None
        self._aspect = 1.0
        if self._skip_interior:
            self._bvh_cache = _rc_build_visible_mesh_bvh_cache(
                context, self._guide_set)
            ry = context.scene.render.resolution_y
            self._aspect = (context.scene.render.resolution_x / ry) if ry else 1.0

        self._frame = s.frame_start
        self._n = 0
        self._rejected = 0
        self._done = 0
        self._start_time = time.time()

    def _process_one(self, context):
        s = self._s
        center, normal, geo_center, origin = self._faces[self._done]
        if s.look_mode == 'CENTER':
            look_dir = (geo_center - center)
        elif s.look_mode == 'ORIGIN':
            look_dir = (origin - center)
        else:
            look_dir = -normal
        if look_dir.length < 1e-9:
            look_dir = -normal if normal.length > 1e-9 else Vector((0, 0, -1))
        look_dir = look_dir.normalized()

        if self._skip_interior:
            quat = look_dir.to_track_quat('-Z', 'Y')
            right = quat @ Vector((1.0, 0.0, 0.0))
            up = quat @ Vector((0.0, 1.0, 0.0))
            if _rc_is_camera_inside_mesh(context, center, self._guide_set):
                self._rejected += 1
                return
            if _rc_geometry_within_near_clip(self._cam.data, center, look_dir,
                                             right, up, self._bvh_cache,
                                             self._aspect):
                self._rejected += 1
                return

        self._cam.location = center
        self._cam.rotation_quaternion = look_dir.to_track_quat('-Z', 'Y')
        self._cam.keyframe_insert(data_path="location", frame=self._frame)
        self._cam.keyframe_insert(data_path="rotation_quaternion",
                                  frame=self._frame)
        self._frame += 1
        self._n += 1

    def invoke(self, context, event):
        # Warning when Live Camera Adjust changes were lost (v133).
        s = context.scene.gcapture_settings
        if s.sph_object is not None:
            _sph_ensure_base(s.sph_object, s)     # older spheres (v134)
        if (not self.keep_adjustments
                and _sph_has_adjustments(s.sph_object)):
            return context.window_manager.invoke_props_dialog(self, width=360)
        return self.execute(context)

    def draw(self, context):
        col = self.layout.column()
        row = col.row()
        row.alert = True
        row.label(text="Live Camera Adjust changes will be discarded.",
                  icon='ERROR')
        col.label(text="Build starts again from the step 4 settings.")
        col.label(text="Cancel keeps the current camera animation.")

    def execute(self, context):
        try:
            self._setup(context)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        GCAPTURE_OT_build._is_running = True
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.02, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'ESC':
            self._cleanup(context)
            self.report({'WARNING'}, "Build cancelled.")
            return {'CANCELLED'}

        if event.type == 'TIMER':
            if self._done >= self._total:
                return self._finish(context)

            elapsed = time.time() - self._start_time
            time_info = ""
            if self._done > 0:
                per = elapsed / self._done
                rem = int(per * (self._total - self._done))
                tstr = ("%dm %ds" % (rem // 60, rem % 60)) if rem >= 60 else ("%ds" % rem)
                time_info = ", ~%s left" % tstr
            _gcapture_progress_set(
                "gcapture.build_animation", self._done / self._total,
                "Building cameras %d/%d%s" % (self._done, self._total, time_info))

            # Batch size: without interior check poses are very fast ->
            # large batches; with interior check (ray cast) smaller ones.
            batch = 5 if self._skip_interior else 100
            try:
                for _ in range(batch):
                    if self._done >= self._total:
                        break
                    self._process_one(context)
                    self._done += 1
            except Exception as exc:
                self._cleanup(context)
                self.report({'ERROR'}, "Build failed: %s" % exc)
                return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def _finish(self, context):
        s = self._s
        if self._n == 0:
            self._cleanup(context)
            self.report({'ERROR'},
                        "No camera poses generated (no faces, or all "
                        "rejected as interior).")
            return {'CANCELLED'}

        if s.constant_interp:
            for fcurve in iter_action_fcurves(self._cam):
                for kp in fcurve.keyframe_points:
                    kp.interpolation = 'CONSTANT'
        if s.set_active_camera:
            context.scene.camera = self._cam
            # Switch to camera view (equivalent to Numpad 0) -- in all
            # 3D viewports. region_3d.view_perspective is the reliable
            # API way to do this.
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    for space in area.spaces:
                        if space.type == 'VIEW_3D':
                            space.region_3d.view_perspective = 'CAMERA'
        if s.set_frame_range:
            context.scene.frame_start = s.frame_start
            context.scene.frame_end = s.frame_start + self._n - 1
        if s.set_resolution:
            rnd = context.scene.render
            rnd.resolution_x = s.resolution
            rnd.resolution_y = s.resolution
            rnd.resolution_percentage = 100
        context.scene.frame_set(s.frame_start)

        cam_name = self._cam.name
        n = self._n
        rejected = self._rejected
        if self._use_list:
            total_objs = sum(len(it.objects) for it in s.guides)
            src = "%d guide entries (%d objects)" % (len(s.guides), total_objs)
        else:
            src = "'%s'" % self._guide_objs[0].name
        self._cleanup(context)

        msg = "Built %d camera poses from %s (camera: %s)." % (n, src, cam_name)
        if getattr(self, "_reset", False):
            msg += " Live Camera Adjust changes discarded (fresh from step 4)."
        if rejected:
            msg += " Rejected %d interior cameras." % rejected
        self.report({'INFO'}, msg)
        # Prompt to save the scene as a version (v90). Only with a
        # window (not in the headless test).
        if (s.save_after_build and not bpy.app.background
                and getattr(context, "window", None)):
            try:
                bpy.ops.gcapture.save_version('INVOKE_DEFAULT')
            except Exception as exc:
                print("[Gaussian Render Capture] Save prompt failed:", exc)
        return {'FINISHED'}

    def _cleanup(self, context):
        if getattr(self, "_timer", None):
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        _gcapture_progress_clear("gcapture.build_animation")
        self._bvh_cache = None
        GCAPTURE_OT_build._is_running = False
