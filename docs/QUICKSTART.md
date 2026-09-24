# Quick start

The shortest way from a model to a Gaussian Splat. Each line is one click or one decision; the
[tutorial](TUTORIAL.md) explains every step with pictures.

| # | In Blender | You see |
|---|---|---|
| 1 | [Download](https://github.com/virtualrepublic/Gaussian-Render-Capture/releases/latest/download/gaussian_capture.py) and install `gaussian_capture.py` (*Preferences → Add-ons → Install from Disk*), enable it | Sidebar tab **Gaussian Render Capture** (press N) |
| 2 | Import your model | – |
| 3 | **Start Guide** | Guide card, step 1 of 7 |
| 4 | **Prepare Scene** | *Scene prepared (Cycles on GPU)* |
| 5 | Select the model, **Put Selection into Collection** – or add its collection with **+** | *Target: …* |
| 6 | Object on a ground: **Group & Align to Ground** – floating object: **Next** | *Grouped under 'GCapture_Group'* |
| 7 | Subdivisions 3, **Create / Update Camera Sphere** (on a ground: *Upper Hemisphere Only*) | *Camera_Sphere: 320 cameras* |
| 8 | **Build Camera Animation**, save as `<Name>_v001.blend` | *320 camera poses* |
| 9 | Resolution **OK**, then **Render Cycles** (or **Render EEVEE**, faster) | *All 320 images found* |
| 10 | **Export COLMAP (Postshot / LichtFeld)** | *Dataset written* |
| 11 | Open `<Name>_COLMAP/v001` in Postshot or LichtFeld Studio and train | the splat |

Blue buttons show the next action, red status lines show what is still missing, and **Next**
turns blue when a step is done.
