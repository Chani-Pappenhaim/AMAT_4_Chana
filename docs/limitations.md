# Limitations

Current, honest state as of Layer 7 (in progress). A negative result is a
result - nothing here is hidden or tuned away, per the project's own rules.

## Structural (not fixable by more tuning within this method)
- Single-scale generation cannot preserve global organization - a single
  patch has no way to know it belongs to a larger coherent structure. This
  is why the coarse-to-fine multiscale sampler exists; it is not a bug to
  keep chasing at Layer 4.
- Variance erodes across chained coarse-to-fine refinement stages even
  after the single-stage partial-step fix (Layer 4/5 finding, confirmed
  again at Layer 6: more steps per scale makes it worse, not better).
- The source mask cannot be transferred to a generated image as-is: mean
  geometric displacement measured at ~27px (analytic control, n=5 trials),
  ~17px with only moderate match confidence (real EMPS image, n=6
  instances, local template matching - no known transform exists on real
  data the way it does in the analytic control), and ~12px on RODARE
  (n=333 connected-component pseudo-instances - an even weaker claim than
  EMPS's, since RODARE's binary mask cannot distinguish individual touching
  carbides in the first place, before generation is involved at all).

## Not yet measured (open, tracked for Layer 7 completion)
- The downstream 3-arm experiment has been run once (Days 21-22, 10
  TRAIN/10 VALIDATION sources, EMPS only) with a genuine methodology
  (pre-registered thresholds, equal update budget, group bootstrap) - but
  the classifier itself (a from-scratch 2-layer MLP, 500 steps) barely
  beat the majority-class baseline (53.22% vs 53.11%), so the INCONCLUSIVE
  verdict reflects a weak classifier as much as it reflects the synthetic
  data's actual usefulness. A stronger classifier and/or larger source
  pool is needed before this comparison means much on its own.
- The downstream 3-arm experiment has NOT been repeated on RODARE - only
  26 usable, non-quarantined field images exist there (see the ANP_3 fix
  below - smaller than the 36 total files once assumed, since 10 of those
  36 turned out to be quarantined), too small for the group-bootstrap
  methodology used on EMPS (310 images / 210 DOI groups) to mean much.
  Left open rather than run with inadequate statistical power.
- **Fixed during the spec audit (previously undetected): the RODARE
  "other field" comparison used quarantined ANP_3 data.** The spec
  explicitly quarantines ANP_3 ("shares an instrument serial with
  ANP-10, so it is not a third microscope"). `rodare_arm.ipynb`
  originally sourced its own-field/other-field comparison's "other field"
  from `preprocessed/hold-out/`, assumed to be a generic held-out split -
  inspecting the raw `data.zip` showed `hold-out/` is built entirely from
  `cloud/ANP_3/` (WD6mm_31-40). Fixed by sourcing the other field from
  `preprocessed/images/` instead (a different, non-quarantined field).
  Re-ran the notebook: own/other-field copying rates unchanged at 0.0%/
  0.0% and the label-validity/displacement numbers (computed from the own
  field only) are unaffected - the fix corrects which data was compared
  against, not the headline numbers, but the earlier comparison should not
  have been trusted as "a second legitimate field" until this was found.
- RODARE's evaluator/copying/label-validity numbers (Day 23) needed an
  extreme downsize (0.12x) for CPU feasibility on its much larger fields,
  which plausibly explains why several numbers looked BETTER than EMPS's
  (lower copying rate, higher relative diversity) rather than worse as the
  spec predicts for a denser dataset - a measurement-scale artifact, not a
  finding that the method works better on real SEM. RODARE's label is also
  a binary mask, not per-instance like EMPS, so its "instance count" is a
  connected-component approximation (454 components measured, vs the
  spec's stated ~800 true instances) - a real, format-driven limitation on
  how precisely density-regime effects can be measured there.
- The approximate (mean-prefiltered) denoiser has only been measured on
  small banks (a 64x64-image pyramid) where it is a net loss; Day 17's
  large-bank result (16,256 patches, real speedup) has not yet been
  re-confirmed inside the full sampling pipeline at that scale.

## Data-inherent
- EMPS has no per-image SEM/TEM modality label - all EMPS-derived claims
  say "electron microscopy", never "SEM" (enforced by `src/data/claims.py`).
- EMPS's 465 images come from 322 distinct papers with no shared
  acquisition metadata (dose, scan rate, instrument, accelerating voltage) -
  any population-level statistic from it is a rough estimate across a
  heterogeneous set, not a single-instrument specification.
- The real-EMPS label-validity measurement (Layer 5) is n=6 instances from
  one source image - one measured data point, not a population statistic.
  The analytic control's n_trials=5 controlled-geometry result remains the
  more statistically grounded number for the pixel-scale of the effect.
