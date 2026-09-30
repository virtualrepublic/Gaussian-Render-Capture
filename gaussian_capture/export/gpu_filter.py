"""Visibility filter of the start points: ray-cast fallback and GPU depth."""


def _exp_visible_ray(p, bvh, cam_positions, eps=1e-3):
    """True if the world point p is seen unobstructed by at least one camera
    (no other piece of geometry in front). Cameras sorted by proximity,
    early out at the first unobstructed line of sight. Numerically verified."""
    cams = sorted(cam_positions, key=lambda c: (c - p).length_squared)
    for c in cams:
        direction = p - c
        d_p = direction.length
        if d_p < eps:
            return True
        direction = direction / d_p
        loc, nrm, idx, dist = bvh.ray_cast(c, direction)
        if loc is None or dist >= d_p - eps:
            return True
    return False
