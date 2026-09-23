# Troubleshooting

**A status line is red.** It names what is missing – follow the blue button of the step. Red lines
are not errors, they are the to-do list.

**Prepare Scene stays blue although Blender already renders with Cycles on the GPU.** The step
counts as done only after Prepare Scene has run once in this scene – it also sets samples,
denoising, bounces and the transparent background.

**Put Selection into Collection is greyed out.** Nothing is selected. Select the parts of your
model in the viewport or the Outliner first.

**Build asks whether to discard Live Camera Adjust changes.** A Build starts from the step 4
settings. Press *Cancel* to keep your current camera animation; use *All Cameras* in Live Camera
Adjust if an adjustment should apply to all views.

**The save dialog offers only a new version, not Overwrite.** You loaded an older version while
newer ones exist in the folder. Older versions are never overwritten; the dialog offers the next
free number or a custom version.

**"No rendered images found" after rendering.** The render output does not point to the image
folder the add-on expects. Check the Output tab: with *Render into the dataset folder* the path is
`//<Name>_COLMAP/<vNNN>/images/<Name>_<vNNN>_`. If you render into your own folder, the Image
Folder in step 7 follows it, and the export needs *Copy* or *Move* to bring the images into the
dataset.

**"Images are not in the dataset – choose Copy or Move".** You rendered into your own folder.
Postshot and LichtFeld need an `images` folder next to `sparse` – choose *Copy into dataset* (safe)
or *Move into dataset* (no duplicates).

**The export takes long.** The visibility filter checks every point against every camera on the
graphics card; this needs a Blender window (in the background it falls back to a much slower
ray cast). Press Esc to cancel – the points collected so far are written.

**The splat is much smaller or larger than the model in Blender.** The export scales the dataset
(*Auto Scale*). The factor and the matrices to undo it are in `<vNNN>_gcapture_export.json` next to
the dataset folder.

**Postshot: "Failed to Camera Poses from JSON file ... missing frames member".** Postshot reads
every `.json` inside the dataset folder as a camera file. The add-on writes its info file next to
the dataset, never inside – if a `.json` ended up in the dataset folder, move it out.

**The splat looks noisy in Blender 5.3 EEVEE.** See [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

**Training in LichtFeld has reached the maximum number of splats – is more training useful?** Yes.
From then on no new splats are added, but the existing ones are still refined; fine details
usually settle in the second half of the run.
