"""GPU depth of the model per camera (compute shaders), shared by the
visibility filter of the export and by Clean Splat."""

import bpy
import math
import numpy as np
from mathutils import Vector, Matrix

from ..core.glass import _gcapture_see_through_slots


# ----------------------------------------------------------------------
# Visibility filter on the GPU (v105)
# ----------------------------------------------------------------------
# Instead of shooting a ray to every camera per point (Python, with
# construction-brick models ~300 rays per occluded point, Vespa 5.4 million points:
# ~3 h), the GPU draws the model once from each camera as a depth image;
# all points are then tested against each depth image with numpy.
# Accuracy = resolution of the depth image. Without a GPU context (e.g.
# blender --background) the export falls back to the ray cast.
_EXP_GPU_RES = 2048

# Compute shader: per point (texel of the point texture), project into
# this camera image and test against the farthest depth value of the 3x3
# neighborhood -- the same rule as _ExpGpuDepth.visible (numpy).
_EXP_VIS_COMPUTE_SRC = """
void main()
{
  ivec2 id = ivec2(gl_GlobalInvocationID.xy);
  ivec2 sz = imageSize(pts);
  if (id.x >= sz.x || id.y >= sz.y) { return; }
  if (imageLoad(vis, id).r > 0.5) { return; }
  vec4 p = imageLoad(pts, id);
  if (p.w < 0.5) { return; }
  vec3 rel = p.xyz - cam_pos.xyz;
  float d = dot(rel, cam_fwd.xyz);
  float f = params.x;
  float nr = params.y;
  float fr = params.z;
  int res = int(params.w);
  if (d <= nr) { return; }
  float xn = dot(rel, cam_right.xyz) / d * f;
  float yn = dot(rel, cam_up.xyz) / d * f;
  if (abs(xn) > 1.0 || abs(yn) > 1.0) { return; }
  int px = clamp(int((xn + 1.0) * 0.5 * float(res)), 0, res - 1);
  int py = clamp(int((yn + 1.0) * 0.5 * float(res)), 0, res - 1);
  float mx = 0.0;
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      ivec2 q = clamp(ivec2(px + dx, py + dy), ivec2(0), ivec2(res - 1));
      float z = texelFetch(depth_tx, q, 0).r;
      float lin = (z >= 1.0) ? 1e30 :
                  (2.0 * fr * nr) / (fr + nr - (2.0 * z - 1.0) * (fr - nr));
      mx = max(mx, lin);
    }
  }
  float tol = d * tol_v.x + tol_v.y;
  if (d <= mx + tol) { imageStore(vis, id, vec4(1.0)); }
}
"""

# Clean Splat (1.2.0): mirror image of the visibility test. Per splat count
# in how many cameras it lies in the image (seen) and in how many it lies in
# front of the NEAREST depth of the 3x3 neighbourhood (viol). p.w = sigma,
# < 0 = empty slot. The same rule as _cln_judge (numpy).
_CLN_CARVE_COMPUTE_SRC = """
void main()
{
  ivec2 id = ivec2(gl_GlobalInvocationID.xy);
  ivec2 sz = imageSize(pts);
  if (id.x >= sz.x || id.y >= sz.y) { return; }
  vec4 p = imageLoad(pts, id);
  if (p.w < 0.0) { return; }
  vec3 rel = p.xyz - cam_pos.xyz;
  float d = dot(rel, cam_fwd.xyz);
  float f = params.x;
  float nr = params.y;
  float fr = params.z;
  int res = int(params.w);
  if (d <= nr) { return; }
  float xn = dot(rel, cam_right.xyz) / d * f;
  float yn = dot(rel, cam_up.xyz) / d * f;
  if (abs(xn) > 1.0 || abs(yn) > 1.0) { return; }
  imageStore(seen, id, imageLoad(seen, id) + vec4(1.0));
  int px = clamp(int((xn + 1.0) * 0.5 * float(res)), 0, res - 1);
  int py = clamp(int((yn + 1.0) * 0.5 * float(res)), 0, res - 1);
  float mn = 1e30;
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      ivec2 q = clamp(ivec2(px + dx, py + dy), ivec2(0), ivec2(res - 1));
      float z = texelFetch(depth_tx, q, 0).r;
      float lin = (z >= 1.0) ? 1e30 :
                  (2.0 * fr * nr) / (fr + nr - (2.0 * z - 1.0) * (fr - nr));
      mn = min(mn, lin);
    }
  }
  if (d + 3.0 * p.w + tol_v.x * d + tol_v.y < mn) {
    imageStore(viol, id, imageLoad(viol, id) + vec4(1.0));
  }
}
"""


class _ExpGpuDepth:
    """Holds the geometry batch and framebuffer for the depth images."""

    def __init__(self, objs, res=_EXP_GPU_RES, geometry=None, see_through=False):
        import gpu
        from gpu_extras.batch import batch_for_shader
        depsgraph = bpy.context.evaluated_depsgraph_get()
        verts, tris, off = [], [], 0
        cache = {}
        if geometry is not None:            # Clean Splat: ready triangles (1.2.0)
            verts, tris, objs = [geometry[0]], [geometry[1]], ()
        for o in objs:
            if o.type != 'MESH':
                continue
            oe = o.evaluated_get(depsgraph)
            me = oe.to_mesh()
            if me is None:
                continue
            try:
                me.calc_loop_triangles()
                n = len(me.vertices)
                nt = len(me.loop_triangles)
                if n and nt:
                    co = np.empty(n * 3, dtype=np.float32)
                    me.vertices.foreach_get("co", co)
                    mw = np.array(oe.matrix_world, dtype=np.float32)
                    verts.append(co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3])
                    tri = np.empty(nt * 3, dtype=np.int32)
                    me.loop_triangles.foreach_get("vertices", tri)
                    tri = tri.reshape(-1, 3)
                    # see_through (1.3.0): glass does not block the view
                    skip = (_gcapture_see_through_slots(o, cache)
                            if see_through else None)
                    if skip:
                        mi = np.empty(nt, dtype=np.int32)
                        me.loop_triangles.foreach_get("material_index", mi)
                        glass = np.array(skip)[np.clip(mi, 0, len(skip) - 1)]
                        tri = tri[~glass]
                    tris.append(tri + off)
                    off += n
            finally:
                oe.to_mesh_clear()
        if not verts:
            raise RuntimeError("no geometry for the depth maps")
        self.verts = np.vstack(verts).astype(np.float32)
        self.tris = np.vstack(tris).astype(np.int32)
        self.shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        self.batch = batch_for_shader(self.shader, 'TRIS', {"pos": self.verts},
                                      indices=self.tris)
        self.res = res
        self.nb = 1          # neighborhood (1 = 3x3, 0 = only the pixel)
        self.tol_rel = 2e-3  # tolerance relative to depth
        self.depth_tex = gpu.types.GPUTexture((res, res),
                                              format='DEPTH_COMPONENT32F')
        self.color_tex = gpu.types.GPUTexture((res, res), format='RGBA8')
        self.fb = gpu.types.GPUFrameBuffer(depth_slot=self.depth_tex,
                                           color_slots=self.color_tex)
        lo = self.verts.min(axis=0)
        hi = self.verts.max(axis=0)
        self.size = float(np.linalg.norm(hi - lo)) or 1.0
        self.center = (lo + hi) / 2.0
        self.radius = float(np.linalg.norm(self.verts - self.center,
                                           axis=1).max()) or 1.0
        self._cs = None      # compute shader state (compute_begin)

    def near_far(self, cam_pos):
        """Depth range tight around the model (v111). It used to range from
        a ten-thousandth to a hundred times the model size -- in 32-bit
        precision this made the depth reconstruction inaccurate."""
        dist = float(np.linalg.norm(np.array(cam_pos) - self.center))
        near = max(self.size * 1e-4, dist - self.radius * 1.01)
        far = dist + self.radius * 1.01
        return near, far

    def _render(self, cam_pos, look_dir, fov, near, far):
        """Draws the depth image of this camera into self.fb. Returns: the
        camera rotation (quaternion)."""
        import gpu
        q = look_dir.to_track_quat('-Z', 'Y')
        cam_mw = Matrix.Translation(cam_pos) @ q.to_matrix().to_4x4()
        f = 1.0 / math.tan(fov / 2.0)
        proj = Matrix(((f, 0, 0, 0), (0, f, 0, 0),
                       (0, 0, (far + near) / (near - far),
                        2 * far * near / (near - far)),
                       (0, 0, -1, 0)))
        with self.fb.bind():
            self.fb.clear(color=(0, 0, 0, 0), depth=1.0)
            gpu.state.depth_test_set('LESS_EQUAL')
            gpu.state.depth_mask_set(True)
            gpu.state.face_culling_set('NONE')
            with gpu.matrix.push_pop():
                gpu.matrix.load_matrix(cam_mw.inverted())
                gpu.matrix.load_projection_matrix(proj)
                self.shader.uniform_float("color", (1, 1, 1, 1))
                self.batch.draw(self.shader)
            gpu.state.depth_test_set('NONE')
        return q

    # --- Point test as compute shader (v111) ----------------------------
    # The points live on the GPU as an RGBA32F texture, the result as an
    # R32F texture; per camera the GPU draws the depth image and tests all
    # points in one pass. Vespa P200 (5.4 million points, 320 cameras):
    # under 1 s instead of ~165 s with numpy; compared with the exact ray
    # cast it loses fewer visible points (80 instead of 629 of ~7,000 disputed).
    _CS_WIDTH = 4096

    def compute_begin(self, points):
        """Upload points and build the shader. Raises an exception on
        problems -- the caller then falls back to visible() (numpy)."""
        import gpu
        n = len(points)
        w = self._CS_WIDTH
        h = max(1, (n + w - 1) // w)
        if h > 16384:
            raise RuntimeError("too many points for one texture (%d)" % n)
        buf = np.zeros((h * w, 4), dtype=np.float32)
        buf[:n, :3] = points
        buf[:n, 3] = 1.0
        pts_tex = gpu.types.GPUTexture(
            (w, h), format='RGBA32F',
            data=gpu.types.Buffer('FLOAT', h * w * 4, buf.ravel()))
        vis_tex = gpu.types.GPUTexture((w, h), format='R32F')
        vis_tex.clear(format='FLOAT', value=(0.0,))
        info = gpu.types.GPUShaderCreateInfo()
        info.image(0, 'RGBA32F', 'FLOAT_2D', 'pts', qualifiers={'READ'})
        info.image(1, 'R32F', 'FLOAT_2D', 'vis', qualifiers={'READ', 'WRITE'})
        info.sampler(0, 'FLOAT_2D', 'depth_tx')
        for name in ('cam_pos', 'cam_fwd', 'cam_right', 'cam_up', 'params',
                     'tol_v'):
            info.push_constant('VEC4', name)
        info.local_group_size(16, 16, 1)
        info.compute_source(_EXP_VIS_COMPUTE_SRC)
        shader = gpu.shader.create_from_info(info)
        self._cs = dict(n=n, w=w, h=h, pts=pts_tex, vis=vis_tex,
                        shader=shader)

    def compute_camera(self, cam_pos, look_dir, fov):
        """Draw the depth image of this camera and test all points not yet
        visible against it (on the GPU, without reading back)."""
        import gpu
        cs = self._cs
        near, far = self.near_far(cam_pos)
        q = self._render(cam_pos, look_dir, fov, near, far)
        r = q @ Vector((1.0, 0.0, 0.0))
        u = q @ Vector((0.0, 1.0, 0.0))
        fw = look_dir.normalized()
        sh = cs["shader"]
        sh.bind()
        sh.image('pts', cs["pts"])
        sh.image('vis', cs["vis"])
        sh.uniform_sampler('depth_tx', self.depth_tex)
        sh.uniform_float('cam_pos', (cam_pos.x, cam_pos.y, cam_pos.z, 0.0))
        sh.uniform_float('cam_fwd', (fw.x, fw.y, fw.z, 0.0))
        sh.uniform_float('cam_right', (r.x, r.y, r.z, 0.0))
        sh.uniform_float('cam_up', (u.x, u.y, u.z, 0.0))
        sh.uniform_float('params', (1.0 / math.tan(fov / 2.0), near, far,
                                    float(self.res)))
        sh.uniform_float('tol_v', (self.tol_rel, self.size * 5e-4, 0.0, 0.0))
        gpu.compute.dispatch(sh, (cs["w"] + 15) // 16, (cs["h"] + 15) // 16, 1)

    def compute_result(self):
        """bool per point: seen by at least one camera."""
        cs = self._cs
        v = np.array(cs["vis"].read(), dtype=np.float32).reshape(-1)
        return v[:cs["n"]] > 0.5

    # --- Clean Splat (1.2.0) -------------------------------------------
    def carve_begin(self, points, sigma):
        """Upload the splats (xyz + sigma) and build the counting shader."""
        import gpu
        n = len(points)
        w = self._CS_WIDTH
        h = max(1, (n + w - 1) // w)
        if h > 16384:
            raise RuntimeError("too many splats for one texture (%d)" % n)
        buf = np.full((h * w, 4), -1.0, dtype=np.float32)
        buf[:n, :3] = points
        buf[:n, 3] = sigma
        pts_tex = gpu.types.GPUTexture(
            (w, h), format='RGBA32F',
            data=gpu.types.Buffer('FLOAT', h * w * 4, buf.ravel()))
        viol = gpu.types.GPUTexture((w, h), format='R32F')
        seen = gpu.types.GPUTexture((w, h), format='R32F')
        viol.clear(format='FLOAT', value=(0.0,))
        seen.clear(format='FLOAT', value=(0.0,))
        info = gpu.types.GPUShaderCreateInfo()
        info.image(0, 'RGBA32F', 'FLOAT_2D', 'pts', qualifiers={'READ'})
        info.image(1, 'R32F', 'FLOAT_2D', 'viol', qualifiers={'READ', 'WRITE'})
        info.image(2, 'R32F', 'FLOAT_2D', 'seen', qualifiers={'READ', 'WRITE'})
        info.sampler(0, 'FLOAT_2D', 'depth_tx')
        for name in ('cam_pos', 'cam_fwd', 'cam_right', 'cam_up', 'params',
                     'tol_v'):
            info.push_constant('VEC4', name)
        info.local_group_size(16, 16, 1)
        info.compute_source(_CLN_CARVE_COMPUTE_SRC)
        shader = gpu.shader.create_from_info(info)
        self._carve = dict(n=n, w=w, h=h, pts=pts_tex, viol=viol, seen=seen,
                           shader=shader)

    def carve_camera(self, cam_pos, look_dir, fov, tol_abs):
        """Draw this camera's depth and count all splats (on the GPU)."""
        import gpu
        cs = self._carve
        near, far = self.near_far(cam_pos)
        q = self._render(cam_pos, look_dir, fov, near, far)
        r = q @ Vector((1.0, 0.0, 0.0))
        u = q @ Vector((0.0, 1.0, 0.0))
        fw = look_dir.normalized()
        sh = cs["shader"]
        sh.bind()
        sh.image('pts', cs["pts"])
        sh.image('viol', cs["viol"])
        sh.image('seen', cs["seen"])
        sh.uniform_sampler('depth_tx', self.depth_tex)
        sh.uniform_float('cam_pos', (cam_pos.x, cam_pos.y, cam_pos.z, 0.0))
        sh.uniform_float('cam_fwd', (fw.x, fw.y, fw.z, 0.0))
        sh.uniform_float('cam_right', (r.x, r.y, r.z, 0.0))
        sh.uniform_float('cam_up', (u.x, u.y, u.z, 0.0))
        sh.uniform_float('params', (1.0 / math.tan(fov / 2.0), near, far,
                                    float(self.res)))
        sh.uniform_float('tol_v', (_CLN_TOL_REL, tol_abs, 0.0, 0.0))
        gpu.compute.dispatch(sh, (cs["w"] + 15) // 16, (cs["h"] + 15) // 16, 1)

    def carve_result(self):
        """(viol, seen) per splat as int32."""
        cs = self._carve
        viol = np.array(cs["viol"].read(), dtype=np.float32).reshape(-1)
        seen = np.array(cs["seen"].read(), dtype=np.float32).reshape(-1)
        return (np.rint(viol[:cs["n"]]).astype(np.int32),
                np.rint(seen[:cs["n"]]).astype(np.int32))

    def depth_map(self, cam_pos, look_dir, fov, near, far):
        """Linear depth (distance along the view axis) per pixel, (R,R),
        row 0 = bottom. inf where there is nothing."""
        q = self._render(cam_pos, look_dir, fov, near, far)
        with self.fb.bind():
            buf = self.fb.read_depth(0, 0, self.res, self.res)
        zb = np.array(buf, dtype=np.float32).reshape(self.res, self.res)
        # Window depth -> linear eye depth.
        z_ndc = zb.astype(np.float64) * 2.0 - 1.0
        lin = (2.0 * far * near) / (far + near - z_ndc * (far - near))
        lin[zb >= 1.0] = np.inf
        return lin, q

    def visible(self, points, cam_pos, look_dir, fov):
        """bool per point: in the image of this camera and not occluded."""
        near, far = self.near_far(cam_pos)
        lin, q = self.depth_map(cam_pos, look_dir, fov, near, far)
        # Farthest value of the 3x3 neighborhood: do not lose points at edges
        # and in narrow gaps due to pixel rasterization.
        fin = np.where(np.isfinite(lin), lin, np.float64(1e30))
        r = self.res
        mx = fin
        if self.nb:
            k = self.nb
            pad = np.pad(fin, k, mode='edge')
            mx = fin.copy()
            for dy in range(2 * k + 1):
                for dx in range(2 * k + 1):
                    mx = np.maximum(mx, pad[dy:dy + r, dx:dx + r])
        right = np.array(q @ Vector((1.0, 0.0, 0.0)))
        up = np.array(q @ Vector((0.0, 1.0, 0.0)))
        fwd = np.array(look_dir.normalized())
        rel = points - np.array(cam_pos)
        d = rel @ fwd
        f = 1.0 / math.tan(fov / 2.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            xn = (rel @ right) / d * f
            yn = (rel @ up) / d * f
        ok = (d > near) & (np.abs(xn) <= 1.0) & (np.abs(yn) <= 1.0)
        px = np.clip(((xn + 1.0) * 0.5 * r).astype(np.int64), 0, r - 1)
        py = np.clip(((yn + 1.0) * 0.5 * r).astype(np.int64), 0, r - 1)
        tol = d * self.tol_rel + self.size * 5e-4
        out = np.zeros(len(points), dtype=bool)
        idx = np.nonzero(ok)[0]
        out[idx] = d[idx] <= mx[py[idx], px[idx]] + tol[idx]
        return out


_CLN_TOL_REL = 0.01      # tolerance relative to the depth
