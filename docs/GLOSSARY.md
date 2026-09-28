<div align="justify">

# Glossary

**3D Gauss­ian Splat­ting (3DGS)** – a way to rep­re­sent a scene as mil­lions of small, coloured,
semi-trans­par­ent ellip­soids ("splats"). They are *trained* from images with known camera
posi­tions until their ren­der­ing matches the images. Intro­duced by Kerbl et al., SIGGRAPH 2023.

**Train­ing** – the opti­mi­sa­tion that turns a dataset into a splat: posi­tions, sizes, rota­tions,
opac­i­ties and colours of the splats are adjusted step by step. Train­ers: Post­shot, LichtFeld
Studio.

**Den­si­fi­ca­tion** – during train­ing the trainer adds splats where detail is miss­ing, up to a
max­i­mum. After that the exist­ing splats are only refined.

**COLMAP dataset** – the folder a trainer reads: `images/` with the photos or ren­ders and
`sparse/0/` with `cameras.txt` (lens, image size), `images.txt` (one camera pose per image) and
`points3D.txt` (start points). Named after the COLMAP pho­togram­me­try soft­ware, whose text format
it uses.

**Camera pose** – posi­tion and ori­en­ta­tion of a camera for one image.

**Struc­ture from Motion (SfM)** – the step that com­putes camera poses and start points from real
photos. Not needed here: Blender knows the poses exactly.

**Start point cloud** – the points in `points3D.txt`. The trainer starts one splat per point; a
good start cloud (all vis­i­ble sur­faces, no hidden inner parts) helps the train­ing.

**Camera sphere** – the ico-sphere around the model; each face becomes one camera.

**Sub­di­vi­sions** – how finely the sphere is divided: 1 = 20, 2 = 80, 3 = 320, 4 = 1280 cam­eras.

**Fill Each View / Center in Each View** – each camera moves as close as pos­si­ble and side­ways so
the model fills and cen­tres its square image.

**Scene ver­sion** – `<Name>_v001.blend`, `_v002` … Each ver­sion ren­ders and exports into its own
dataset folder `<Name>_COLMAP/<vNNN>/`.

**Auto Scale** – the export scales the dataset so the cam­eras are on aver­age 3 units from the
centre; the factor is stored in `<vNNN>_gcapture_export.json` next to the dataset folder.

**EEVEE / Cycles** – Blender's two render engines. Cycles traces light paths phys­i­cally (exact,
slower); EEVEE approx­i­mates light­ing in real time (much faster). For splats every view must be lit
con­sis­tently – Cycles guar­an­tees that, EEVEE comes close with the cap­ture preset.

**Command Line Render (head­less)** – ren­der­ing with­out Blender's inter­face, started from a script
(`blender -b <scene> -a`). The add-on writes such a script next to the scene; double-click it.

**Z-up / Y-up** – Blender uses Z as "up", many splat­ting tools use Y. The export con­verts it.

**Floater** – a splat in empty space, e.g. inside a car body seen through the win­dows or in the air
around the model. Step 8 removes them.

**Free-space carv­ing** – remov­ing splats by the depth of the model: along each camera ray the space
up to the first sur­face is empty, so a splat there cannot belong to the model.

</div>
