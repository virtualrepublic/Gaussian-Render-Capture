<div align="justify">

# Changelog

All notable changes to this project are doc­u­mented here. The project fol­lows
[Seman­tic Ver­sion­ing](https://semver.org/): PATCH for fixes, MINOR for new fea­tures, MAJOR for
changes that break exist­ing scenes or set­tings. Only changes to the add-on itself are listed;
doc­u­men­ta­tion changes are not.

## [1.3.1] – 2026-09-30

- The add-on ships as a ZIP (`gaussian_capture-1.3.1.zip`) instead of a single `.py` file:
  Blender 4.2 and later install it as an exten­sion, Blender 4.1 as an add-on, both with *Install
  from Disk*. On Blender 4.1 the ZIP replaces an older `gaussian_capture.py`; on 4.2 and later
  remove it first – enabling the new ver­sion next to it is refused with a mes­sage. Restart
  Blender after installing. Scenes and set­tings are unchanged.
- *Trans­par­ent Glass* shows and sets Blender's own set­ting (Render Prop­er­ties › Film): an opened
  scene keeps what it has, a new scene stays at Blender's default off. *Prepare Scene* and
  *Camera Sphere* no longer change it.

## [1.3.0] – 2026-09-27

- The vis­i­bil­ity filter of the start points looks through glass: sur­faces seen only through
  a window – e.g. a car's inte­rior – keep their start points, while points on the glass itself keep
  only the panes seen from out­side. Glass is recog­nised from the mate­rial (Glass, Trans­par­ent or
  Refrac­tion BSDF, or Prin­ci­pled BSDF with Trans­mis­sion of 0.5 or more, also inside node groups such
  as Mecabricks' mate­ri­als). Clean Splat still treats glass as a sur­face. Datasets are recom­puted
  once on the next export instead of reusing the stored points.
- *Prepare Scene* ren­ders glass opaque in the alpha chan­nel: panes show what lies behind them. With
  trans­par­ent glass they came out semi-trans­par­ent and blotchy, and the splat got a haze behind
  win­dows. The option *Trans­par­ent Glass* right below *Prepare Scene* switches trans­par­ent glass
  back on – for objects that are mostly glass, or splats shown in front of other back­grounds.
- The export shows each step after the camera poses – col­lect­ing the ver­tices, adding face points,
  pre­par­ing the vis­i­bil­ity filter – instead of stand­ing still for up to a minute on large models;
  Esc works in between.

## [1.2.0] – 2026-09-26

- Step 8 **Clean Splat**: removes the splats in empty space from a trained splat (`.ply` from
  LichtFeld Studio or Post­shot) using the model's depth per dataset camera, and writes
  `<name>_clean.ply` next to it. Splats in front of the sur­face in at least *Min. Views* cam­eras
  (default 2) and splats no camera sees are removed; the orig­i­nal file stays untouched.
  Datasets exported with Gauss­ian Render Scan 1.0.x (`gscan_export.json`) work as well.
- Step 6 has an image format choice: **PNG** or **TIFF**, both RGBA 8 bit (TIFF with LZW
  com­pres­sion). It shows in Blender's Output set­tings as soon as it is chosen and is applied on
  every render, also by *Command Line Render*. The export takes the chosen format when images of
  an ear­lier render in another format are still in the folder.

## [1.1.5] – 2026-09-24

- *Build Camera Ani­ma­tion* sizes the camera in the view­port to the sphere (10 % of its radius)
  instead of Blender's 1 m, which hid small models – unless you changed the size your­self.
- Step 6 ren­ders the images itself: **Render Cycles** (exact path trac­ing, set­tings of *Prepare Scene*)
  or **Render EEVEE** (much faster, light­ing approx­i­mated; a cap­ture preset is applied once – ray trac­ing
  and fast GI at full res­o­lu­tion, soft shad­ows, over­scan, 256 sam­ples). The scene is saved first,
  so set­tings changed just before are kept. Esc can­cels.
- *Command Line Render* (head­less): instead of ren­der­ing in Blender, the but­tons save the scene and
  write a double-click script next to it that ren­ders head­less with the same Blender ver­sion
  (`.cmd` on Windows, `.command` on macOS, `.sh` on Linux).
- Start points take the mate­rial base colour by default; face points now carry it as well
  (before, they were always grey).
- Blender 4.1 with OptiX: Cycles denoises with OptiX – OpenImageDenoise fails there on sys­tems
  with a cur­rent Intel graph­ics driver.

## [1.1.4] – 2026-09-24

- Progress of *Build Camera Ani­ma­tion* and *Export COLMAP* is shown as a progress bar in the
  panel, in place of the button (guide and normal panel), instead of the view­port header.
  While they run, all other set­tings and the guide nav­i­ga­tion are locked; Esc can­cels.

## [1.1.3] – 2026-09-24

- COLMAP export: focal length and prin­ci­pal point now follow Blender's camera model in every
  case – non-square pixel aspect and lens shift with an explicit sensor fit were off before.
  The normal work­flow (square images, square pixels) was already exact.

## [1.1.2] – 2026-09-24

No func­tional changes – texts in the add-on and the doc­u­men­ta­tion were revised in con­tent and form.

## [1.1.1] – 2026-09-24

No func­tional changes – texts in the add-on and the doc­u­men­ta­tion were revised in con­tent and form.

## [1.1.0] – 2026-09-23

Renamed to **Gauss­ian Render Capture**: the add-on ren­ders a cap­ture – images plus camera poses –
and does not scan any­thing real.

- New file `gaussian_capture.py`, side­bar tab *Gauss­ian Render Capture*, oper­a­tors `gcapture.*`,
  empty *GCapture_Group*, col­lec­tion *Capture_Rig*, info file `<vNNN>_gcapture_export.json`.
- **Upgrade:** dis­able and remove *Gauss­ian Render Scan* (1.0.x), then install
  `gaussian_capture.py`. Scenes made with 1.0.x are taken over when opened – all set­tings, the
  mark­ers of steps 1 and 5, the camera fit and live adjust­ments; save the scene to keep the new
  names. Export­ing such a dataset again replaces `<vNNN>_gscan_export.json` with the new file.

## [1.0.2] – 2026-09-23

No func­tional changes – texts in the add-on and the doc­u­men­ta­tion were revised in con­tent and form.

## [1.0.1] – 2026-09-23

No func­tional changes – texts in the add-on and the doc­u­men­ta­tion were revised in con­tent and form.

## [1.0.0] – 2026-09-23

First public release.

- **Guide** in seven steps with a check per step; blue but­tons show the next action, red status
  lines what is miss­ing.
- **1 Scene & Camera** – *Prepare Scene*: Cycles on the fastest GPU back­end with scan render
  set­tings; lens and look target of the scan camera.
- **2 Look Target** – *Put Selec­tion into Col­lec­tion*, reuses an exist­ing col­lec­tion, frames the
  geom­e­try.
- **3 Group & Align to Ground** – for objects that stand on a ground.
- **4 Camera Sphere** – 20 to 1280 cam­eras; *Fill Each View* and *Center in Each View* make the
  model fill and centre every square image; upper hemi­sphere or full sphere.
- **5 Build Camera Ani­ma­tion** – one keyframe per camera, fresh from step 4; scene ver­sions
  (`<Name>_v001.blend`) with their own dataset folder; *Live Camera Adjust* for one view or the
  whole sphere, with Cancel.
- **6 Render Set­tings & Output** – render output straight into the dataset folder of the ver­sion.
- **7 COLMAP Export** – cam­eras, poses and a start point cloud from all model ver­tices plus sur­face
  points; GPU vis­i­bil­ity filter; crop box; reuse of an unchanged point cloud; auto scale with the
  factor in `<vNNN>_gscan_export.json`; works with Post­shot and LichtFeld Studio.
- Requires Blender 4.1 or later.

</div>
