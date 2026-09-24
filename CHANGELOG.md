# Changelog

All notable changes to this project are documented here. The project follows
[Semantic Versioning](https://semver.org/): PATCH for fixes, MINOR for new features, MAJOR for
changes that break existing scenes or settings. Only changes to the add-on itself are listed;
documentation changes are not.

## [1.1.5] – unreleased

- *Build Camera Animation* sizes the camera in the viewport to the sphere (10 % of its radius)
  instead of Blender's 1 m, which hid small models – unless you changed the size yourself.
- Step 6 renders the images itself: **Render Cycles** (exact path tracing, settings of *Prepare Scene*)
  or **Render EEVEE** (much faster, lighting approximated; a capture preset is applied once – ray tracing
  and fast GI at full resolution, soft shadows, overscan, 256 samples). Esc cancels.
- *Command Line Render* (headless): instead of rendering in Blender, the buttons save the scene and
  write a double-click script next to it that renders headless with the same Blender version
  (`.cmd` on Windows, `.command` on macOS, `.sh` on Linux).
- Start points take the material base colour by default; face points now carry it as well
  (before, they were always grey).
- Blender 4.1 with OptiX: Cycles denoises with OptiX – OpenImageDenoise fails there on systems
  with a current Intel graphics driver.

## [1.1.4] – 2026-09-24

- Progress of *Build Camera Animation* and *Export COLMAP* is shown as a progress bar in the
  panel, in place of the button (guide and normal panel), instead of the viewport header.
  While they run, all other settings and the guide navigation are locked; Esc cancels.

## [1.1.3] – 2026-09-24

- COLMAP export: focal length and principal point now follow Blender's camera model in every
  case – non-square pixel aspect and lens shift with an explicit sensor fit were off before.
  The normal workflow (square images, square pixels) was already exact.

## [1.1.2] – 2026-09-24

No functional changes – texts in the add-on and the documentation were revised in content and form.

## [1.1.1] – 2026-09-24

No functional changes – texts in the add-on and the documentation were revised in content and form.

## [1.1.0] – 2026-09-23

Renamed to **Gaussian Render Capture**: the add-on renders a capture – images plus camera poses –
and does not scan anything real.

- New file `gaussian_capture.py`, sidebar tab *Gaussian Render Capture*, operators `gcapture.*`,
  empty *GCapture_Group*, collection *Capture_Rig*, info file `<vNNN>_gcapture_export.json`.
- **Upgrade:** disable and remove *Gaussian Render Scan* (1.0.x), then install
  `gaussian_capture.py`. Scenes made with 1.0.x are taken over when opened – all settings, the
  markers of steps 1 and 5, the camera fit and live adjustments; save the scene to keep the new
  names. Exporting such a dataset again replaces `<vNNN>_gscan_export.json` with the new file.

## [1.0.2] – 2026-09-23

No functional changes – texts in the add-on and the documentation were revised in content and form.

## [1.0.1] – 2026-09-23

No functional changes – texts in the add-on and the documentation were revised in content and form.

## [1.0.0] – 2026-09-23

First public release.

- **Guide** in seven steps with a check per step; blue buttons show the next action, red status
  lines what is missing.
- **1 Scene & Camera** – *Prepare Scene*: Cycles on the fastest GPU backend with scan render
  settings; lens and look target of the scan camera.
- **2 Look Target** – *Put Selection into Collection*, reuses an existing collection, frames the
  geometry.
- **3 Group & Align to Ground** – for objects that stand on a ground.
- **4 Camera Sphere** – 20 to 1280 cameras; *Fill Each View* and *Center in Each View* make the
  model fill and centre every square image; upper hemisphere or full sphere.
- **5 Build Camera Animation** – one keyframe per camera, fresh from step 4; scene versions
  (`<Name>_v001.blend`) with their own dataset folder; *Live Camera Adjust* for one view or the
  whole sphere, with Cancel.
- **6 Render Settings & Output** – render output straight into the dataset folder of the version.
- **7 COLMAP Export** – cameras, poses and a start point cloud from all model vertices plus surface
  points; GPU visibility filter; crop box; reuse of an unchanged point cloud; auto scale with the
  factor in `<vNNN>_gscan_export.json`; works with Postshot and LichtFeld Studio.
- Requires Blender 4.1 or later.
