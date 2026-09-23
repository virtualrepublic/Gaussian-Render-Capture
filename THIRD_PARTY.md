# Third-party notices

This project respects the rights of others. Everything below is used under its licence or only
named to describe the example and compatibility.

## Code

- **Gauss Cannon** by Warpgate Labs – GPL-3.0-or-later –
  <https://github.com/warpgatelabs/gauss-cannon>. The add-on started from it. Adapted from its
  `utils/ray_casting.py` and still part of `gaussian_capture.py`: `_rc_is_camera_inside_mesh`,
  `_rc_build_visible_mesh_bvh_cache`, `_rc_build_near_frustum_bvh`, `_rc_geometry_within_near_clip`
  (Skip Interior Cameras) and `_exp_build_scene_bvh` (ray-cast fallback of the visibility filter),
  plus the idea of a list of guide meshes. Everything else is original work.

## Example model, renders and screenshots

- **Example model** – LDraw model of set 10252 by **Roland Dahl (RolandD)**, "Redistributable under
  CCAL version 2.0" (CC BY 2.0), in
  [examples/10252_Volkswagen_Beetle](examples/10252_Volkswagen_Beetle/) with
  [CAreadme.txt](examples/10252_Volkswagen_Beetle/CAreadme.txt).
- **LDraw parts library** – LDraw.org, CC BY 2.0 / CC BY 4.0 – <https://library.ldraw.org/>. The
  example model is built from its parts; all renders and screenshots of the example show them.
- **Screenshots of other software** in the documentation, used to explain the workflow:
  Blender (Blender Foundation), Postshot (Jawset), LichtFeld Studio.

## Trademarks

LEGO® is a trademark of the LEGO Group. Volkswagen and Beetle are trademarks of Volkswagen AG.
Blender is a trademark of the Blender Foundation. Postshot is a product of Jawset. LichtFeld
Studio, COLMAP and SuperSplat belong to their respective owners. None of them sponsors,
authorises or endorses this project.
