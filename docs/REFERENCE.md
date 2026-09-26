# Reference – all settings

Grouped by the sections of the panel. Defaults in brackets. The tooltips in Blender say the same
in short.

## 1. Scene & Camera

| Setting / button | Default | What it does |
|---|---|---|
| **Prepare Scene** | – | Cycles on the fastest GPU backend (OptiX, CUDA, HIP, Metal, oneAPI; GPUs on, CPU off), capture render settings (1024 samples, GPU denoising, all bounces 32, transparent film and glass, AgX Base Contrast), removes Blender's unchanged start cube, camera and light. |
| Camera Name | Orbit_Camera | Name of the capture camera created by Build. |
| Start Frame | 1 | First frame of the camera animation. |
| Focal Length (mm) | 50 | Lens of the capture camera; the sphere adapts its size to it. |
| Look Target | Geometry Center | Where each camera looks: the centre of the model (recommended), the object origin, or along the face normal (only for a uniform sphere). |

## 2. Look Target

| Setting / button | What it does |
|---|---|
| **Put Selection into Collection** | Moves the selected objects (with their children) into a collection and adds it to the list; an existing collection of the entered name is used. Frames the geometry in the viewport. |
| List, **+ / – / trash** | The collections the cameras look at and the sphere is sized to. |

## 3. Group & Align to Ground

| Setting / button | Default | What it does |
|---|---|---|
| Fast (bounding box) | off | Uses the objects' bounding boxes instead of all vertices – faster, slightly less exact for rotated parts. |
| **Group & Align to Ground** | – | Groups the model under the empty *GCapture_Group* at the world origin, standing on Z = 0. Only for objects on a ground. |

## 4. Camera Sphere

| Setting | Default | What it does |
|---|---|---|
| Subdivisions | 2 | Number of cameras: 1 = 20, 2 = 80, 3 = 320, 4 = 1280. |
| Fill Each View | on | Each camera moves along its line of sight until the model fills its image. Off: one common distance for all. |
| Center in Each View | on | With Fill Each View: each camera also shifts sideways (same direction) so the model is centred in its image. |
| Framing Margin | 1.0 | Room around the model; 1.0 fills the image, 1.1 leaves about 10 %. |
| Upper Hemisphere Only | off | No views from below – for objects on a ground. |
| Build Cameras on This Sphere | on | Build puts the cameras on the faces of the new sphere (replaces custom camera guides). |
| **Create / Update Camera Sphere** | – | Creates or refits the sphere (collection *Capture_Rig*). Discards Live Camera Adjust changes. |

## 5. Build Camera Animation

| Setting / button | Default | What it does |
|---|---|---|
| **Build Camera Animation** | – | One keyframe per sphere face, fresh from the step 4 settings; warns first if Live Camera Adjust changes would be discarded. |
| Save Version after Build | on | Opens the save dialog after Build and after *This Camera*. |
| **Live Camera Adjust** | – | Locks the camera of the current frame to its face; S / G on the sphere adjusts it live. Finish with *This Camera*, *All Cameras* or *Cancel*. |

**Save Scene Version** – first save: `<Collection>_v001.blend`. Later: *Overwrite* (red; only for
the highest version in the folder), *New version* (next free number), *Custom version*.
*Render into the dataset folder* points the render output to `<Name>_COLMAP/<vNNN>/images`; it is
preset off when you render into a folder of your own.

## 6. Render Settings & Output

| Setting | Default | What it does |
|---|---|---|
| Resolution (square) | 1000 | Render resolution in pixels, width = height; applied at once. Confirm once with **OK**. |
| Image Format | PNG | **PNG** or **TIFF** (Deflate compression, smaller files), both RGBA 8 bit – set on every render; Postshot and LichtFeld Studio read both. If an earlier render in another format is still in the images folder, the export takes the chosen format. |
| **Render EEVEE** | – | Saves the scene, then starts rendering: one image per camera into the dataset with EEVEE – much faster, lighting approximated. The first time it applies a capture preset (256 samples, ray tracing and fast GI at full resolution, soft shadows, overscan); your later changes are kept. |
| **Render Cycles** | – | The same with Cycles on the GPU and the settings of *Prepare Scene*: exact path tracing, slower. |
| Command Line Render | off | Headless: the buttons read *Write EEVEE .cmd* / *Write Cycles .cmd*. They save the scene and write a double-click script next to it (`<Scene>_render_eevee.cmd` / `_cycles`; `.command` on macOS, `.sh` on Linux) that renders with the same Blender version, without its interface. |

Under *Advanced → Render Setup* (keep on for a capture): Step per Frame (constant keyframes), Set as
Active Camera, Set Scene Frame Range, Set Render Resolution. They apply at once and on every Build.

## 7. COLMAP Export

**Dataset**

| Setting | Default | What it does |
|---|---|---|
| Output Folder | `//<Name>_COLMAP/<vNNN>/` | Dataset folder; follows the loaded version. Enter your own to override. |
| Image Folder | folder of the render output | Where the rendered images are; follows the render output. |
| Images | Don't include | Only with your own render folder: *Copy* or *Move* the images into the dataset. |

**Start Points**

| Setting | Default | What it does |
|---|---|---|
| Point Source | Look Target Collections | Where the start points come from: the look-target collections, all visible meshes, or named objects. |
| Use Vertex Count | on | Every vertex of the model becomes a start point. Off: set *Max Points* yourself. |
| Add Random Face Points / Face Points | on / 10 % | Extra points spread over the surfaces, as a share of the vertex points. |
| Visibility Filter | Visible Only | Drops points no camera can see (GPU depth maps); Esc cancels. |
| Point Color | Material Base Color | Colour of the start points, face points included: the material base colour, a vertex colour layer, or grey. |
| Crop Point Cloud | off | Keeps only points inside the bounds of a chosen object (plus margin). |
| Reuse Point Cloud | on | Copies the last point cloud if nothing that affects it has changed. |

**Scale & Axes**

| Setting | Default | What it does |
|---|---|---|
| Auto Scale / Target Radius | on / 3.0 | Scales the dataset so the cameras are on average 3 units from the centre (Postshot densifies very small or large scenes badly). The factor goes into `<vNNN>_gcapture_export.json` next to the dataset folder. |
| Z-up → Y-up | on | Converts Blender's axes to the Y-up convention of the trainers. |

| Button | What it does |
|---|---|
| **Export COLMAP (Postshot / LichtFeld)** | Writes `sparse/0/` and, if chosen, the images into the dataset, and `<vNNN>_gcapture_export.json` next to it. While it runs, a progress bar replaces the button; Esc cancels. |

## 8. Clean Splat (after training)

| Setting | Default | What it does |
|---|---|---|
| Splat File | – | The trained splat (`.ply` from LichtFeld Studio or Postshot) of this scene version. The folder button opens the dataset folder. |
| **Clean Splat** | – | Removes every splat that lies in front of the model's surface in at least *Min. Views* cameras of the dataset, and every splat no camera sees. Writes `<name>_clean.ply` next to the file; the original stays untouched. Uses the cameras and the scale of the exported dataset (`<vNNN>_gcapture_export.json`) and every rendered object as the surface. While it runs, a progress bar replaces the button; Esc cancels and writes nothing. |

## Advanced

Custom camera guides (your own meshes instead of the sphere), interior camera rejection for room
rigs, Render Setup (see step 6), maintainer and licence.

| Setting | Default | What it does |
|---|---|---|
| Min. Views (Clean Splat) | 2 | A splat is removed when it lies in front of the surface in at least this many cameras. 1 removes more, higher values less. |
