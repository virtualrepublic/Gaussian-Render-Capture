# Glossary

**3D Gaussian Splatting (3DGS)** – a way to represent a scene as millions of small, coloured,
semi-transparent ellipsoids ("splats"). They are *trained* from images with known camera
positions until their rendering matches the images. Introduced by Kerbl et al., SIGGRAPH 2023.

**Training** – the optimisation that turns a dataset into a splat: positions, sizes, rotations,
opacities and colours of the splats are adjusted step by step. Trainers: Postshot, LichtFeld
Studio.

**Densification** – during training the trainer adds splats where detail is missing, up to a
maximum. After that the existing splats are only refined.

**COLMAP dataset** – the folder a trainer reads: `images/` with the photos or renders and
`sparse/0/` with `cameras.txt` (lens, image size), `images.txt` (one camera pose per image) and
`points3D.txt` (start points). Named after the COLMAP photogrammetry software, whose text format
it uses.

**Camera pose** – position and orientation of a camera for one image.

**Structure from Motion (SfM)** – the step that computes camera poses and start points from real
photos. Not needed here: Blender knows the poses exactly.

**Start point cloud** – the points in `points3D.txt`. The trainer starts one splat per point; a
good start cloud (all visible surfaces, no hidden inner parts) helps the training.

**Camera sphere** – the ico-sphere around the model; each face becomes one camera.

**Subdivisions** – how finely the sphere is divided: 1 = 20, 2 = 80, 3 = 320, 4 = 1280 cameras.

**Fill Each View / Center in Each View** – each camera moves as close as possible and sideways so
the model fills and centres its square image.

**Scene version** – `<Name>_v001.blend`, `_v002` … Each version renders and exports into its own
dataset folder `<Name>_COLMAP/<vNNN>/`.

**Auto Scale** – the export scales the dataset so the cameras are on average 3 units from the
centre; the factor is stored in `<vNNN>_gcapture_export.json` next to the dataset folder.

**Z-up / Y-up** – Blender uses Z as "up", many splatting tools use Y. The export converts it.
