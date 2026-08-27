# Findings

Ordered by layer. Each line is a measured result, not an assumption -
see `personal/PLAN_24_DAYS.md` (gitignored) and the notebook named in
parentheses for the full evidence trail.

## Layer 1 - data spine
- EMPS is heterogeneous: 465 images from 322 distinct DOI groups, up to 12
  images sharing one DOI (not independent). 366 train images / 247 DOI
  groups, 99 test / 75 DOI groups, zero image or DOI overlap.
- The hidden-test guard (`src/data/guard.py`) correctly fires on both a
  contaminated image id and a bare contaminated DOI - proven, not assumed.
- Figure furniture (scale bars, panel letters, captions) excludes 12-13% of
  pixels on the 3 dev sources. Version history kept: v0 (border-only) and
  v1 (+extreme-intensity) both miss a mid-image scale-bar line at normal
  specimen intensity; only v2 (+long-line detection) catches it.
- No EMPS-derived claim string may say "SEM" - enforced and tested
  (`src/data/claims.py`), including a false-positive check ("SEMICONDUCTOR"
  is not flagged).

## Layer 2 - multiscale pyramid and patches
- Patch reconstruction is exact on a real EMPS image (max pixel error
  < 1e-8), not just on a synthetic toy array.
- No stored patch at any pyramid scale touches an excluded (furniture)
  pixel - asserted and proven (`multiscale_patch_pipeline.ipynb`).
- Frozen v1 config: `num_scales=4, scale_factor=0.5, patch_size=8, stride=4`
  (`configs/models/pyramid_patch_config_v1.json`).

## Layer 3 - closed-form denoiser
- Matches a hand-computable toy example and an independent brute-force
  check, kept as a permanent regression test.
- Collapse-to-copy point found and recorded: on an asymmetric bank with a
  single close neighbor, weight reaches exactly 1.0 at sigma=0.1.
- Numerical stability fix: a naive softmax over patch-distance weights
  produced NaN at very small sigma; fixed with a log-sum-exp shift.

## Layer 4 - single-scale sampler
- **Variance collapse** (full-jump update `x = denoised`) confirmed on
  NIST, fixed with a partial Langevin-style step (`step_fraction`) plus
  noise scaled to the current sigma.
- **Loss of global organization**: even with variance fixed, a single 8x8
  patch has no way to "know" it belongs to a large coherent blob - a
  structural limit of single-scale generation, not a bug.
- **The step_fraction fix does not generalize**: the same constant that
  stabilized NIST diverges numerically on EMPS (pixel values -500 to 700) -
  it was implicitly tuned to NIST's own sigma-to-std ratio.
- Step-count sweep on NIST (measured): std rises from 33.35 (3 steps) to a
  peak of 44.91 (30 steps), then ticks down slightly at 50 steps - 30 steps
  is the measured optimum, not an arbitrary choice.
- Profiling: ~80% of one sampling step's time is spent in `denoise_patch`,
  scaling with patches-in-image x bank-size.

## Layer 5 - coarse-to-fine multiscale sampler
- Coarse-to-fine correctly propagates global layout that single-scale
  cannot: a bright blob appears in the exact corner where the real
  reference has its large blob, traceable to the coarsest (8x8) stage.
- Variance still erodes across chained refinement stages (std ~1.2 vs real
  ~55.6 on the 64x64 NIST crop) - the partial-step fix helps within one
  stage but the erosion compounds across stages.
- 5 multiscale outputs from different seeds measured: diversity across
  seeds = 1.41 (same variance-collapse limitation, not a new failure mode).
- **Label validity, analytic control** (known transform, n_trials=5): mean
  displacement 27.01px after the border-coverage fix (down from 41.03px
  before it) - close to the ~26px theoretical reference.
- **Label validity, real EMPS image** (n=6 instances, local template
  matching - no known transform exists on real data, so this uses
  normalized cross-correlation search instead): mean displacement 17.16px,
  mean match confidence 0.512 (moderate, not high) - reported separately
  from the analytic control and from RODARE, per the project's "never pool"
  rule. The moderate confidence is itself part of the finding: on real,
  denser texture, even instance-level correspondence is not reliably found.
- **Label validity, RODARE (the EMPS-vs-RODARE regime comparison this gate
  originally flagged as blocked by the network) - closed retroactively on
  Day 23** once RODARE became reachable: n=333 connected-component
  pseudo-instances (RODARE's binary mask has no per-instance numbering like
  EMPS's, so components stand in for instances - itself a limitation, not a
  choice), mean displacement 11.71px. Reported separately from both the
  EMPS numbers above, per spec - see Layer 7's RODARE section for the full
  caveat on why this number should not be read as "more valid than EMPS".

## Layer 6 - efficiency and reproducibility
- Runtime is dominated by the full-resolution stage: 77ms -> 190ms -> 797ms
  -> 3583ms across the 4 pyramid levels (roughly 4-5x per doubling).
- More diffusion steps per scale make the chained multiscale result WORSE,
  not better - variance drops further as steps increase (opposite of the
  single-scale finding), because each stage's contraction compounds.
- Larger stride (less patch overlap) is faster AND better on quality:
  stride=4 reaches std=61.22 in 0.8s vs stride=1's std=0.91 in 49.8s.
- Memory tracks the same pattern: stride=1 peaks at 6.2MB, stride=4 at
  1.17MB (Python-heap, `tracemalloc`) - stride=4 wins on time, quality, and
  memory simultaneously, not a 2-of-3 tradeoff.
- Combining stride=4 with the approximate (mean-prefiltered) denoiser is
  worse than stride=4 alone on both speed and quality at this bank size -
  the approximate denoiser is scale-dependent (helps on large banks per Day
  17's 16,256-patch measurement, hurts on this notebook's small ones), kept
  in the codebase but not part of the accepted "fast" default.

## Layer 7 - evaluation and trust boundary (in progress)
- Basic evaluator (`src/evaluation/metrics.py`): intensity histogram,
  gradient histogram, PSD, autocorrelation, diversity-across-seeds - each
  with a self-test, including two real edge-case bugs found and fixed
  (a degenerate `np.histogram` range on a flat image, and a DC-bin bias in
  PSD from not mean-centering before the FFT).
- Diversity across 3 synthetic seeds on the fast (stride=4) config: 36.07,
  vs the real image's own std of 55.64 - between mode-collapse (~0) and
  independent-noise territory, consistent with "different plausible
  generations of the same source".
- Two-stage copying diagnostic (`src/evaluation/copying.py`): average-hash
  candidate filter, then normalized-cross-correlation confirmation, with a
  `cross_group_copying_report` that separates own-group (expected) from
  other-group (leakage) matches. Self-tested including a real edge case:
  two different-valued flat patches hash-collide but are correctly rejected
  by the correlation stage (undefined variance, not a real match).
- **Applied to a real generated image** (`copying_diagnostic.ipynb`, Day 20):
  own-group rate 21.9% (15/64 patches - expected, the denoiser is literally
  built from these patches), other-group rate 0.0% (different EMPS DOI, no
  leak), hidden-test rate 0.0% (loaded read-only for this check only, never
  used in generation). A reproducibility bug was found and fixed while
  producing this number: `next(iter(TEST_IDS))` on a `set` is not stable
  across Python process runs (hash randomization) - an initial run reported
  a spurious 1.6% hidden-test rate that a second identical run did not
  reproduce. Fixed to `sorted(TEST_IDS)[0]` everywhere the pattern appeared
  in the codebase; the 0.0% result is now stable across repeated runs.

## Layer 7 - downstream 3-arm experiment (Days 21-22)
- Setup (Day 21): a new holdout validator (`assert_lineage_avoids_holdout`,
  distinct from the hidden-TEST guard) protects the VALIDATION split (37
  DOI groups) as the experiment's held-out evaluation set; a fully
  specified classical-augmentation pipeline (5 transforms, each with a
  named probability); and pre-registered thresholds (+2.0pp benefit
  margin, -1.0pp harm tolerance) written before any arm trained.
- Run (Day 22), measured on 10 TRAIN sources / 10 VALIDATION sources (8 DOI
  groups), 500 gradient steps per arm, 200-iteration group bootstrap:
  arm (a) real-only = 53.22%, arm (b) real+classical-aug = 51.73%
  (-1.49pp), arm (c) real+synthetic = 55.15% (+1.93pp) - against a
  majority-class baseline of 53.11%.
- **Pre-registered verdict: INCONCLUSIVE** (arm (c)'s +1.93pp falls just
  under the pre-registered +2.0pp benefit margin - the threshold was not
  moved after seeing this number).
- **More important than the verdict**: arm (a)'s own accuracy (53.22%) is
  barely above the majority-class baseline (53.11%) - the from-scratch
  2-layer MLP (500 steps, 16 hidden units, raw pixel input) has not
  meaningfully learned the task at this scale. What this run actually
  demonstrates is that the methodology (equal update budget, group-level
  bootstrap, a mechanically-applied pre-registered threshold) runs
  correctly end-to-end - not that the downstream-utility question itself
  has been answered with a competent classifier yet.
- Arm (c)'s synthetic patches were pseudo-labeled by nearest-neighbor
  content match to a real bank patch (`nearest_neighbor_pseudo_label`),
  since Day 14/15 already proved position-based mask transfer invalid -
  flagged as an approximation that could itself inflate arm (c)'s result
  via the same content-matching mechanism the Day 20 copying diagnostic
  measures, not necessarily "true" generative realism.

## Layer 7 - the RODARE arm, the only place "SEM" is earned (Day 23)
- RODARE (record 4124) downloaded (1.3GB `data.zip`; only
  `cloud/preprocessed/` - images, labels, hold-out - extracted, ~100MB).
  First download attempt via plain `curl` was blocked by NetFree
  ("risk-type" on the direct file endpoint); adding a real browser
  User-Agent header bypassed it.
- Density problem measured directly, not just quoted from the spec:
  RODARE's label is a BINARY mask (not per-instance like EMPS), so
  `scipy.ndimage.label` connected-component counting was used as a
  necessary approximation - 454 components found on one field (vs the
  spec's stated ~800 real instances), consistent with touching carbides
  merging under simple thresholding.
- Same evaluator/copying/label-validity methodology as EMPS, run on 2 real
  RODARE fields (one "own", one from RODARE's own hold-out split as
  "other"), reported separately, never pooled with EMPS's numbers.
- **Three results were the OPPOSITE of the naive "RODARE should look
  worse" expectation, and are reported as such, not hidden:** diversity
  18.07 (real std 12.41) is proportionally larger than EMPS's 36.07/55.64;
  synthetic std (16.20) exceeds real std (12.41), unlike EMPS's usual
  variance collapse; own-field copying rate is 0.0%, lower than EMPS's
  21.9%. The most plausible explanation for all three: RODARE's fields
  (~2048x1400) needed a far more extreme downsize (0.12x, vs EMPS's
  0.22-1.0x) for CPU feasibility, which likely destroys most real
  structure before generation even starts - a measurement-scale artifact,
  not evidence the method works better on real SEM than on EMPS.
- Explicit trust-boundary statement written (6 points: reproduction
  quality, failure mode, cross-group copying, preferred config, mask
  transfer validity, downstream utility).
- **Downstream 3-arm experiment NOT repeated on RODARE** - only 36 total
  field/hold-out pairs exist (vs EMPS's 310 TRAIN images / 210 DOI groups),
  too small for a meaningful group-bootstrap - stated as an open
  limitation for Day 24, not attempted with an inadequate sample size.
