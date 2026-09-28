<div align="justify">

# Troubleshooting

**A status line is red.** It names what is miss­ing – follow the blue button of the step. Red lines
are not errors, they are the to-do list.

**Prepare Scene stays blue although Blender already ren­ders with Cycles on the GPU.** The step
counts as done only after Prepare Scene has run once in this scene – it also sets sam­ples,
denois­ing, bounces and the trans­par­ent back­ground.

**Put Selec­tion into Col­lec­tion is greyed out.** Nothing is selected. Select the parts of your
model in the view­port or the Out­liner first.

**Build asks whether to dis­card Live Camera Adjust changes.** A Build starts from the step 4
set­tings. Press *Cancel* to keep your cur­rent camera ani­ma­tion; use *All Cameras* in Live Camera
Adjust if an adjust­ment should apply to all views.

**The save dialog offers only a new ver­sion, not Over­write.** You loaded an older ver­sion while
newer ones exist in the folder. Older ver­sions are never over­writ­ten; the dialog offers the next
free number or a custom ver­sion.

**"No ren­dered images found" after ren­der­ing.** The render output does not point to the image
folder the add-on expects. Check the Output tab: with *Render into the dataset folder* the path is
`//<Name>_COLMAP/<vNNN>/images/<Name>_<vNNN>_`. If you render into your own folder, the Image
Folder in step 7 fol­lows it, and the export needs *Copy* or *Move* to bring the images into the
dataset.

**"Images are not in the dataset – choose Copy or Move".** You ren­dered into your own folder.
Post­shot and LichtFeld need an `images` folder next to `sparse` – choose *Copy into dataset* (safe)
or *Move into dataset* (no dupli­cates).

**The export takes long.** The vis­i­bil­ity filter checks every point against every camera on the
graph­ics card; this needs a Blender window (in the back­ground it falls back to a much slower
ray cast). Press Esc to cancel – the points col­lected so far are writ­ten.

**The splat is much smaller or larger than the model in Blender.** The export scales the dataset
(*Auto Scale*). The factor and the matri­ces to undo it are in `<vNNN>_gcapture_export.json` next to
the dataset folder.

**Post­shot: "Failed to Camera Poses from JSON file ... miss­ing frames member".** Post­shot reads
every `.json` inside the dataset folder as a camera file. The add-on writes its info file next to
the dataset, never inside – if a `.json` ended up in the dataset folder, move it out.

**Render EEVEE / Render Cycles is greyed out.** Save the scene first (step 5, *Save Scene
Version*) – the render goes into the dataset folder of this ver­sion.

**The com­mand line script closes at once or finds no Blender.** It calls the Blender that wrote it.
If you moved or unin­stalled that Blender, write the script again with *Write EEVEE .cmd* /
*Write Cycles .cmd*.

**macOS: "cannot be opened because it is from an uniden­ti­fied devel­oper".** The `.command` script
was writ­ten on your Mac but not signed. Right-click it → *Open* and con­firm once; after­wards a
double-click works.

**Linux: a double-click opens the script in a text editor.** Your file man­ager does not run
scripts. Run it in a ter­mi­nal: `./<Scene>_render_eevee.sh` – or allow run­ning exe­cutable text
files in the file man­ager's pref­er­ences.

**Linux server: the EEVEE script fails with­out a dis­play.** EEVEE needs a graph­ics ses­sion. Start
it through a vir­tual dis­play: `xvfb-run -a ./<Scene>_render_eevee.sh`. Cycles runs with­out one.

**Blender 4.1: "OpenImageDenoise error … PI_ERROR_INVALID_VALUE".** A prob­lem of Blender 4.1 with
cur­rent Intel graph­ics driv­ers. With an NVIDIA card the add-on switches Cycles to the OptiX
denoiser; oth­er­wise use Blender 4.2 or later.

**The splat looks noisy in Blender 5.3 EEVEE.** See [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

**Train­ing in LichtFeld has reached the max­i­mum number of splats – is more train­ing useful?** Yes.
From then on no new splats are added, but the exist­ing ones are still refined; fine details
usu­ally settle in the second half of the run.

**Clean Splat: "No <vNNN>_gcapture_export.json".** The dataset of this scene ver­sion has not been
exported yet, or you opened another ver­sion. Export it in step 7, or open the ver­sion the splat
was trained from.

**Clean Splat removed more than half of the splats.** The splat belongs to another scene ver­sion,
or the model was moved or scaled after the export. Open the ver­sion the dataset was exported from;
the orig­i­nal file is untouched, delete the `_clean.ply`.

**Clean Splat: "Could not write … _clean.ply".** The file is open in a viewer (SuperSplat,
LichtFeld, Post­shot). Close it there and press Clean Splat again.

**Haze or blotches behind win­dows (e.g. a car's inte­rior).** Trans­par­ent glass ren­ders the panes
semi-trans­par­ent and blotchy, dif­fer­ently in every view. Keep *Trans­par­ent Glass* (right below *Prepare
Scene*) off, press *Prepare Scene* and render again. Export the dataset again as well: the start points behind glass are
kept since ver­sion 1.3.0.

**The glass looks grey or white in the splat.** With *Trans­par­ent Glass* off, glass shows the scene's
world colour where you look through it into the back­ground. Use a neu­tral world colour, or switch
*Trans­par­ent Glass* on for objects that are mostly glass.

</div>
