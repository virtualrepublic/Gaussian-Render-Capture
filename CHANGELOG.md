# Changelog

All notable changes to this project are documented here. The project follows
[Semantic Versioning](https://semver.org/): PATCH for fixes, MINOR for new features, MAJOR for
changes that break existing scenes or settings.

## [1.1.1] – 2026-09-24

- Credits: Gauss Cannon is by Arash Keshmirian (Warpgate Labs) – his name and copyright for the
  adapted ray-casting helpers added to the file header, the panel, README and THIRD_PARTY.md.

## [1.1.0] – 2026-09-23

Renamed to **Gaussian Render Capture**: the add-on renders a capture – images plus camera poses –
and does not scan anything real.

- New file `gaussian_capture.py`, sidebar tab *Gaussian Render Capture*, operators `gcapture.*`,
  empty *GCapture_Group*, collection *Capture_Rig*, info file `<vNNN>_gcapture_export.json`.
- **Upgrade:** disable and remove *Gaussian Render Scan* (1.0.x), then install
  `gaussian_capture.py`. Scenes made with 1.0.x are taken over when opened – all settings, the
  markers of steps 1 and 5, the camera fit and live adjustments; save the scene to keep the new
  names. Exporting such a dataset again replaces `<vNNN>_gscan_export.json` with the new file.
- Tests: Linux, macOS and Windows with Blender 4.1 and 5.2, including the takeover of a 1.0.2 scene.

## [1.0.2] – 2026-09-23

- Documentation: Linux (Ubuntu) and macOS (Apple Silicon) now tested with the automated test
  suite on Blender 4.1 and 5.2 – all tests pass, including the GPU visibility filter.

## [1.0.1] – 2026-09-23

- Documentation: supported platforms stated precisely – tested on Windows 11; Linux and macOS
  expected to work but not tested yet. Which trainer runs on which system.

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
- Blender 4.1 or later; tested on 4.1, 4.4, 4.5, 5.0, 5.2 and 5.3.
