<div align="justify">

# Known issues

Prob­lems out­side the add-on that affect the Gauss­ian splat­ting work­flow. The research
notes behind them are kept out­side this repos­i­tory.

## Blender 5.3: Gaussian splats render noisy or with black patches in EEVEE

Blender 5.3 can import Gauss­ian splat PLY and SPZ files and render them. EEVEE draws the
splats with dithered (hashed) trans­parency: every splat is either drawn fully or skipped per
sample, and the image only con­verges over many sam­ples.

- Splat sets made of many semi-trans­par­ent splats con­verge much more slowly. LichtFeld
  Studio's MRNF strat­egy pro­duces such sets: on a brick model of a car, 91 % of its splats were
  below 30 % opac­ity against 31 % for Post­shot (Splat3), and EEVEE needed about ten times the
  sam­ples for the same noise level.
- Inde­pen­dently, EEVEE ren­dered black patches on sur­faces that Cycles ren­ders cor­rectly:
  [blender/blender#164171](https://projects.blender.org/blender/blender/issues/164171). An
  erro­neous clamp was fixed (PR #164192); a remain­ing glitch is being inves­ti­gated (status
  22 Sept 2026). The slower con­ver­gence of translu­cent splat sets is sep­a­rate and per­sists
  in builds with that fix.

**Until fixed:** for ren­ders in Blender, train with Post­shot, or raise the EEVEE sam­ples
con­sid­er­ably (512–1024), or render in Cycles. The COLMAP dataset from this add-on works the
same in both train­ers.

</div>
