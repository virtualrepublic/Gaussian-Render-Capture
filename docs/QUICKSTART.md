<div align="justify">

# Quick start

The short­est way from a model to a Gauss­ian Splat. Each line is one click or one deci­sion; the
[tuto­rial](TUTORIAL.md) explains every step with pic­tures.

| # | In Blender | You see |
|---|---|---|
| 1 | [Down­load](https://github.com/virtualrepublic/Gaussian-Render-Capture/releases/latest/download/gaussian_capture.py) and install `gaussian_capture.py` (*Pref­er­ences → Add-ons → Install from Disk*), enable it | Sidebar tab **Gauss­ian Render Capture** (press N) |
| 2 | Import your model | – |
| 3 | **Start Guide** | Guide card, step 1 of 8 |
| 4 | **Prepare Scene** | *Scene pre­pared (Cycles on GPU)* |
| 5 | Select the model, **Put Selec­tion into Col­lec­tion** – or add its col­lec­tion with **+** | *Target: …* |
| 6 | Object on a ground: **Group & Align to Ground** – float­ing object: **Next** | *Grouped under 'GCapture_Group'* |
| 7 | Sub­di­vi­sions 3, **Create / Update Camera Sphere** (on a ground: *Upper Hemi­sphere Only*) | *Camera_Sphere: 320 cam­eras* |
| 8 | **Build Camera Ani­ma­tion**, save as `<Name>_v001.blend` | *320 camera poses* |
| 9 | Res­o­lu­tion **OK**, then **Render EEVEE** (fast) or **Render Cycles** (exact) | *All 320 images found* |
| 10 | **Export COLMAP (Post­shot / LichtFeld)** | *Dataset writ­ten* |
| 11 | Open `<Name>_COLMAP/v001` in Post­shot or LichtFeld Studio and train | the splat |
| 12 | Optional: export the splat as `.ply`, pick it in step 8, **Clean Splat** | *Removed … splats – <name>_clean.ply* |

Blue but­tons show the next action, red status lines show what is still miss­ing, and **Next**
turns blue when a step is done.

</div>
