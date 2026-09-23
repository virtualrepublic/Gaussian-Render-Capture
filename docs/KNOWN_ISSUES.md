# Known issues

Problems outside the add-on that affect the Gaussian splatting workflow. The research
notes behind them are kept outside this repository.

## Blender 5.3: Gaussian splats render noisy or with black patches in EEVEE

Blender 5.3 can import Gaussian splat PLY and SPZ files and render them. EEVEE draws the
splats with dithered (hashed) transparency: every splat is either drawn fully or skipped per
sample, and the image only converges over many samples.

- Splat sets made of many semi-transparent splats converge much more slowly. LichtFeld
  Studio's MRNF strategy produces such sets: on a brick model of a car, 91 % of its splats were
  below 30 % opacity against 31 % for Postshot (Splat3), and EEVEE needed about ten times the
  samples for the same noise level.
- Independently, EEVEE rendered black patches on surfaces that Cycles renders correctly:
  [blender/blender#164171](https://projects.blender.org/blender/blender/issues/164171). An
  erroneous clamp was fixed (PR #164192); a remaining glitch is being investigated (status
  22 Sept 2026). The slower convergence of translucent splat sets is separate and persists
  in builds with that fix.

**Until fixed:** for renders in Blender, train with Postshot, or raise the EEVEE samples
considerably (512–1024), or render in Cycles. The COLMAP dataset from this add-on works the
same in both trainers.
