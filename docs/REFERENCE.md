<div align="justify">

# Reference – all settings

Grouped by the sec­tions of the panel. Defaults in brack­ets. The tooltips in Blender say the same
in short.

## 1. Scene & Camera

| Setting / button | Default | What it does |
|---|---|---|
| **Prepare Scene** | – | Cycles on the fastest GPU back­end (OptiX, CUDA, HIP, Metal, oneAPI; GPUs on, CPU off), cap­ture render set­tings (1024 sam­ples, GPU denois­ing, all bounces 32, trans­par­ent film, AgX Base Con­trast), removes Blender's unchanged start cube, camera and light. |
| Trans­par­ent Glass | as the scene has it (Blender: off) | Blender's *Trans­par­ent Glass* (Render Prop­er­ties › Film), shown and set right below *Prepare Scene*; the add-on never changes it oth­er­wise – an opened scene keeps its value, a new one Blender's default off. Off: glass is ren­dered opaque in the alpha chan­nel and shows what lies behind it – a car's inte­rior stays clean in the splat; where the back­ground is seen through the glass, the scene's world colour appears. On: glass becomes trans­par­ent against the empty back­ground – for objects that are mostly glass, or for splats shown in front of other back­grounds; the panes then come out semi-trans­par­ent and blotchy. Applies at once. |
| Camera Name | Orbit_Camera | Name of the cap­ture camera cre­ated by Build. |
| Start Frame | 1 | First frame of the camera ani­ma­tion. |
| Focal Length (mm) | 50 | Lens of the cap­ture camera; the sphere adapts its size to it. |
| Look Target | Geom­e­try Center | Where each camera looks: the centre of the model (rec­om­mended), the object origin, or along the face normal (only for a uni­form sphere). |

## 2. Look Target

| Setting / button | What it does |
|---|---|
| **Put Selec­tion into Col­lec­tion** | Moves the selected objects (with their chil­dren) into a col­lec­tion and adds it to the list; an exist­ing col­lec­tion of the entered name is used. Frames the geom­e­try in the view­port. |
| List, **+ / – / trash** | The col­lec­tions the cam­eras look at and the sphere is sized to. |

## 3. Group & Align to Ground

| Setting / button | Default | What it does |
|---|---|---|
| Fast (bound­ing box) | off | Uses the objects' bound­ing boxes instead of all ver­tices – faster, slightly less exact for rotated parts. |
| **Group & Align to Ground** | – | Groups the model under the empty *GCapture_Group* at the world origin, stand­ing on Z = 0. Only for objects on a ground. |

## 4. Camera Sphere

| Setting | Default | What it does |
|---|---|---|
| Sub­di­vi­sions | 2 | Number of cam­eras: 1 = 20, 2 = 80, 3 = 320, 4 = 1280. |
| Fill Each View | on | Each camera moves along its line of sight until the model fills its image. Off: one common dis­tance for all. |
| Center in Each View | on | With Fill Each View: each camera also shifts side­ways (same direc­tion) so the model is cen­tred in its image. |
| Framing Margin | 1.0 | Room around the model; 1.0 fills the image, 1.1 leaves about 10 %. |
| Upper Hemi­sphere Only | off | No views from below – for objects on a ground. |
| Build Cameras on This Sphere | on | Build puts the cam­eras on the faces of the new sphere (replaces custom camera guides). |
| **Create / Update Camera Sphere** | – | Creates or refits the sphere (col­lec­tion *Capture_Rig*). Dis­cards Live Camera Adjust changes. |

## 5. Build Camera Animation

| Setting / button | Default | What it does |
|---|---|---|
| **Build Camera Ani­ma­tion** | – | One keyframe per sphere face, fresh from the step 4 set­tings; warns first if Live Camera Adjust changes would be dis­carded. |
| Save Version after Build | on | Opens the save dialog after Build and after *This Camera*. |
| **Live Camera Adjust** | – | Locks the camera of the cur­rent frame to its face; S / G on the sphere adjusts it live. Finish with *This Camera*, *All Cameras* or *Cancel*. |

**Save Scene Version** – first save: `<Collection>_v001.blend`. Later: *Over­write* (red; only for
the high­est ver­sion in the folder), *New ver­sion* (next free number), *Custom ver­sion*.
*Render into the dataset folder* points the render output to `<Name>_COLMAP/<vNNN>/images`; it is
preset off when you render into a folder of your own.

## 6. Render Settings & Output

| Setting | Default | What it does |
|---|---|---|
| Res­o­lu­tion (square) | 1000 | Render res­o­lu­tion in pixels, width = height; applied at once. Confirm once with **OK**. |
| Image Format | PNG | **PNG** or **TIFF** (LZW com­pres­sion, smaller files), both RGBA 8 bit – set on every render; Post­shot and LichtFeld Studio read both. If an ear­lier render in another format is still in the images folder, the export takes the chosen format. |
| **Render EEVEE** | – | Saves the scene, then starts ren­der­ing: one image per camera into the dataset with EEVEE – much faster, light­ing approx­i­mated. The first time it applies a cap­ture preset (256 sam­ples, ray trac­ing and fast GI at full res­o­lu­tion, soft shad­ows, over­scan); your later changes are kept. |
| **Render Cycles** | – | The same with Cycles on the GPU and the set­tings of *Prepare Scene*: exact path trac­ing, slower. |
| Command Line Render | off | Head­less: the but­tons read *Write EEVEE .cmd* / *Write Cycles .cmd*. They save the scene and write a double-click script next to it (`<Scene>_render_eevee.cmd` / `_cycles`; `.command` on macOS, `.sh` on Linux) that ren­ders with the same Blender ver­sion, with­out its inter­face. |

Under *Advanced → Render Setup* (keep on for a cap­ture): Step per Frame (con­stant keyframes), Set as
Active Camera, Set Scene Frame Range, Set Render Res­o­lu­tion. They apply at once and on every Build.

## 7. COLMAP Export

**Dataset**

| Setting | Default | What it does |
|---|---|---|
| Output Folder | `//<Name>_COLMAP/<vNNN>/` | Dataset folder; fol­lows the loaded ver­sion. Enter your own to over­ride. |
| Image Folder | folder of the render output | Where the ren­dered images are; fol­lows the render output. |
| Images | Don't include | Only with your own render folder: *Copy* or *Move* the images into the dataset. |

**Start Points**

| Setting | Default | What it does |
|---|---|---|
| Point Source | Look Target Col­lec­tions | Where the start points come from: the look-target col­lec­tions, all vis­i­ble meshes, or named objects. |
| Use Vertex Count | on | Every vertex of the model becomes a start point. Off: set *Max Points* your­self. |
| Add Random Face Points / Face Points | on / 10 % | Extra points spread over the sur­faces, as a share of the vertex points. |
| Vis­i­bil­ity Filter | Visible Only | Drops points no camera can see (GPU depth maps); Esc can­cels. |
| Point Color | Mate­rial Base Color | Colour of the start points, face points included: the mate­rial base colour, a vertex colour layer, or grey. |
| Crop Point Cloud | off | Keeps only points inside the bounds of a chosen object (plus margin). |
| Reuse Point Cloud | on | Copies the last point cloud if noth­ing that affects it has changed. |

**Scale & Axes**

| Setting | Default | What it does |
|---|---|---|
| Auto Scale / Target Radius | on / 3.0 | Scales the dataset so the cam­eras are on aver­age 3 units from the centre (Post­shot den­si­fies very small or large scenes badly). The factor goes into `<vNNN>_gcapture_export.json` next to the dataset folder. |
| Z-up → Y-up | on | Con­verts Blender's axes to the Y-up con­ven­tion of the train­ers. |

| Button | What it does |
|---|---|
| **Export COLMAP (Post­shot / LichtFeld)** | Writes `sparse/0/` and, if chosen, the images into the dataset, and `<vNNN>_gcapture_export.json` next to it. While it runs, a progress bar replaces the button; Esc can­cels. |

## 8. Clean Splat (after training)

| Setting | Default | What it does |
|---|---|---|
| Splat File | – | The trained splat (`.ply` from LichtFeld Studio or Post­shot) of this scene ver­sion. The folder button opens the dataset folder. |
| **Clean Splat** | – | Removes every splat that lies in front of the model's sur­face in at least *Min. Views* cam­eras of the dataset, and every splat no camera sees. Writes `<name>_clean.ply` next to the file; the orig­i­nal stays untouched. Uses the cam­eras and the scale of the exported dataset (`<vNNN>_gcapture_export.json`) and every ren­dered object as the sur­face. While it runs, a progress bar replaces the button; Esc can­cels and writes noth­ing. |

## Advanced

Custom camera guides (your own meshes instead of the sphere), inte­rior camera rejec­tion for room
rigs, Render Setup (see step 6), main­tainer and licence.

| Setting | Default | What it does |
|---|---|---|
| Min. Views (Clean Splat) | 2 | A splat is removed when it lies in front of the sur­face in at least this many cam­eras. 1 removes more, higher values less. |

</div>
