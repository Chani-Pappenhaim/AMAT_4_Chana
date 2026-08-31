# Findings

Ordered by layer. Each line is a measured result, not an assumption -
see `personal/PLAN_24_DAYS.md` (gitignored) and the notebook named in
parentheses for the full evidence trail.

## Layer 1 - data spine
- EMPS is heterogeneous: 465 images from 322 distinct DOI groups, up to 12
  images sharing one DOI (not independent). 366 train images / 247 DOI
  groups, 99 test / 75 DOI groups, zero image or DOI overlap.
- The hidden-test guard (`src/data/hidden_test_guard.py`) correctly fires on both a
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
- **Patch-grid boundary artifacts, two candidate fixes tested on all 5 real
  min-dataset images (not tuned to any single one)**: (1) widening the
  absolute overlap by increasing patch_size 4->6 at fixed stride=2 (matching
  the reference repo's and the frozen L2 config's overlap width) - REJECTED,
  made blockiness measurably worse: introduced a vertical-stripe artifact in
  391ef00939.png and a brick/tiling pattern in 802e607c7c.png that neither
  image showed before, most likely because a larger patch is more
  discriminative in the softmax match, pushing reconstruction closer to
  copying a few bank patches on a still-small stride. (2) jittering the
  patch grid's phase every denoising step instead of a fixed grid (see
  sample_single_scale's `jitter` param) - ADOPTED as the CLI's new default:
  no regression on any of the 5 images, and a real if
  modest reduction in blockiness on the two images that showed it worst
  (391ef00939.png, 802e607c7c.png). Both candidates and the unchanged
  baseline are preserved under
  results/generated/min data set/grid_artifact_fix/ for direct comparison.
- **Shape loss (real discrete particles -> soft amorphous blobs), root-caused
  and partially fixed via a per-level patch size, NOT adopted as a default**:
  even after the grid-jitter fix above, synthetic output for images with
  regular/periodic particle arrangements (e.g. 434e287439.png's hexagonally
  packed spheres) lost the particles entirely - std matched the real image
  almost exactly, but the spatial layout was a handful of soft blobs, not
  discrete circles. Stage-by-stage tracing (diagnostic_stagebystage_baseline_
  434e287439.png in grid_artifact_fix/) showed this is NOT a resolution
  problem - the real coarsest pyramid level (30x36px) still shows a clearly
  distinguishable circle lattice by eye. The failure is in the FIRST
  noise-to-structure step (generate_coarse_sketch): at that tiny level each
  real particle is only ~7-8px across, but patch_size=4 there (same as every
  other level) means a patch is smaller than half a particle, so it can only
  match/reproduce local blur fragments, never a whole discrete particle or
  its spacing to neighbors. Every later refine_at_scale step then only ADDS
  detail on top of that wrong layout by design (so a later scale cannot
  drift an already-good coarse structure) - meaning a wrong coarse layout is
  never corrected, only sharpened. This is the classical finding from Efros
  & Leung 1999 ("Texture Synthesis by Non-parametric Sampling"): a matching
  window smaller than a texture's own regular structure loses that
  structure, producing something locally plausible but globally incoherent.
  Fix tested: sample_coarse_to_fine's new coarse_patch_size/coarse_stride
  params (default None = unchanged) use a LARGER patch (8, matching the
  frozen L2 config's own patch_size, and roughly one real particle's width
  at that level) only for the coarsest sketch, leaving every finer level's
  patch_size untouched - a deliberately narrower lever than the earlier,
  rejected "widen patch_size everywhere" candidate, specifically so it
  cannot reintroduce that candidate's fine-level blocking regression.
  Result across all 5 real images (see 03_coarsepatch8_*.png in
  grid_artifact_fix/), tested broadly per the same anti-overfitting
  discipline as above: **434e287439.png (the one clearly regular/periodic
  image) improved dramatically** - distinct, separated, roughly correctly-
  sized circles instead of 3-4 giant blobs (compare diagnostic_stagebystage_
  coarsepatch8_434e287439.png stage 0 to the baseline's stage 0). The three
  irregular-but-discrete-particle images (2807b90ea9.png rods, 391ef00939.png
  and 586dc6cc59.png cubes/diamonds) showed **no clear improvement** -
  still amorphous blobs, just with a differently-shaped blob boundary.
  802e607c7c.png (sparse dots on a mostly-textureless/stochastic background,
  the one real image with no periodic structure at all) **regressed**: a
  new waffle-like grid texture appeared in the background and fewer of the
  real dark particles survived as distinct dots. This is consistent with
  Efros & Leung's own finding that a larger window only helps *regular*
  texture and can actively hurt *stochastic* texture by over-constraining
  it. Net: the mechanism is real and evidenced, but its benefit is
  conditional on the image's own particle regularity, not universal -
  exactly the kind of image-dependent tradeoff the "don't overfit" rule
  warns against baking in as a blanket default. Left as an opt-in flag
  (`--coarse-patch-size`/`--coarse-stride`, both default None = off) rather
  than promoted to the CLI default. A natural next step (not attempted -
  scope/time) is making coarse_patch_size adaptive per image (e.g. measured
  from the image's own dominant particle scale) instead of a fixed constant.
- **Shape loss, continued - a SECOND cause was bigger than the first, found
  by tracing the mechanism instead of tuning the fix above**: even with
  coarse_patch_size, particles were still soft. Re-read the reference
  implementation this project's coarse-to-fine design is closest to (GPNN -
  Granot et al., CVPR 2022, "Drop the GAN: In Defense of Patch Nearest
  Neighbors as Single Image Generative Models", github.com/iyttor/GPNN,
  model/gpnn.py) to check whether this pipeline's patch-estimation step
  actually matches its own inspiration. It does not: this codebase's
  denoiser (denoise_patch/denoise_patches_batch) returns a SOFTMAX-WEIGHTED
  AVERAGE of the topk=8 nearest bank patches, whereas GPNN uses hard
  nearest-neighbour selection (`NNs = torch.argmin(norm_dist, dim=1)`) -
  every estimate is one exact real patch, never a blend. Averaging several
  real patches whose particle edges sit even 1-2px apart produces a patch
  with NO sharp edge at all, and with patch_size=4/stride=2 each output
  pixel is already the blend of ~4 overlapping patches on top of that.
  **Fix**: `denoiser.select_patches_nn` implements GPNN's normalized hard-NN
  selection (`sampler`'s new `nn_alpha` param / CLI `--nn-alpha`, alpha_rel
  in GPNN's formulation). One correction made to GPNN's own stated
  rationale while implementing this, recorded so the mistake doesn't get
  repeated: the normalized distance is often described as increasing
  output diversity by preventing a few dominant bank patches from
  "hogging" every match. MEASURED on real data (4725 queries vs 4725-patch
  bank) this is backwards - normalization *concentrates* selection (distinct
  patches used: 2416 at alpha_rel=1e12 (~plain NN) down to 1531 at
  alpha_rel=0.01; max single-patch reuse 14 -> 80). What it actually does is
  bias completeness toward rare/hard-to-match patches at the cost of
  coherence elsewhere - a real, useful, but different effect than the one
  usually cited for it.
- **Shape loss, THIRD cause, found by re-examining the reconstruction step
  itself rather than assuming hard-NN selection was sufficient**: even
  after nn_alpha returns exact sharp real patches, `reconstruct_from_patches`
  stitches overlapping patches back together with a plain weighted MEAN
  (stride=2 < patch_size=4, so every output pixel is covered by ~4
  patches). Averaging patches that disagree about where an edge sits still
  turns that edge into a ramp - a second, independent blurring stage,
  downstream of the first. **Fix**: `reconstruct_from_patches` gained
  `robust_norm` (sampler/CLI: `--robust-norm`), the IRLS aggregation of
  Kwatra et al. 2005 ("Texture Optimization for Example-based Synthesis") -
  minimizes an r<2 norm instead of the r=2 mean, so patches that disagree
  with the consensus lose influence instead of dragging the result toward
  their midpoint. Verified in isolation on a contested step edge (half the
  overlapping patches shifted 1px): plain mean gives slope 75/100 of true;
  r=0.8 IRLS recovers 99.9/100. On the real 5-image set, though,
  `robust_norm=0.8` measured NO further gain once nn_alpha and the fourth
  fix below were both already active (see the ablation table below) - the
  edge-disagreement it targets barely exists anymore once patches are
  exact and the coarse layout is right. Kept as an opt-in flag with the
  mechanism proven and unit-tested, not because it failed, but because it
  had nothing left to fix in this pipeline's current state.
- **Shape loss, FOURTH cause - the largest single measured fix, found by
  quantifying a failure mode none of the three fixes above were built to
  address**: even with sharp exact patches and IRLS aggregation, several
  synthetic outputs still showed particles MERGING into connected
  "continents" instead of staying separate - a different symptom than
  blur. Quantified it directly instead of continuing to iterate by eye:
  measured each real image's own particle-vs-background area fraction
  (pixels below the midpoint of that image's 5th/95th percentile range)
  and its histogram skew. All 5 real images are markedly asymmetric -
  18-39% particle area, skew -1.3 to -4.3 (sparse dark particles on a
  bright support) - while the `08_nn10_coarse8` synthetic outputs were
  42-72% "particle" with near-zero skew. Root cause: this sampler's
  variance-preserving renormalization (needed to stop the chained
  refinement stages from collapsing variance, see Layer 5's std-erosion
  finding above) only constrains the first two moments (mean, std) of the
  output. An asymmetric bimodal histogram is not determined by its first
  two moments, so the sampler was free to satisfy mean/std with a balanced
  ~50/50 texture instead - exactly the observed failure, and invisible to
  every metric used so far (fidelity/completeness/edge_ratio are all local,
  patch- or gradient-scale; none of them see a wrong GLOBAL area fraction).
  **Fix**: `sampler.match_histogram` (new sampler/CLI param
  `histogram_match`/`--histogram-match`) - exact histogram specification
  (rank-preserving remap so structure is untouched, only the tone curve
  changes) applied at every pyramid level against that level's own real
  histogram. This is the classical multi-scale texture-synthesis technique
  of Heeger & Bergen 1995 ("Pyramid-Based Texture Analysis/Synthesis"), who
  match histograms at every pyramid level for exactly this reason. Fully
  per-image and data-driven (the reference is that image's own pyramid
  level; no tuned constant), so it cannot encode a preference for any one
  image's appearance.
- **Combined result and CLI defaults, measured across all 5 real images**
  (bidirectional fidelity/completeness per Simakov et al. 2008,
  edge_ratio = Sobel gradient magnitude ratio, area_gap = |synthetic - real|
  particle-area-fraction; lower fidelity/completeness/area_gap is better,
  edge closer to 1.00 is better; full per-image breakdown and every
  intermediate variant preserved under
  results/generated/min data set/grid_artifact_fix/):

  | variant                                    | fidelity | complete | edge | area_gap |
  |---------------------------------------------|---------:|---------:|-----:|---------:|
  | baseline (pre-jitter)                        |    0.443 |    0.416 | 0.80 |    0.373 |
  | grid jitter (prior CLI default)              |    0.406 |    0.394 | 0.66 |    0.368 |
  | coarse_patch_size=8 alone (opt-in)            |    0.404 |    0.392 | 0.68 |    0.302 |
  | nn_alpha=1.0 alone                            |    0.454 |    0.383 | 0.92 |    0.346 |
  | nn_alpha=1.0 + histogram_match (**new default**) | 0.325 |    0.371 | 0.78 |    0.002 |
  | + coarse_patch_size=8 on top                  |    0.324 |    0.373 | 0.78 |    0.002 |
  | + robust_norm=0.8 on top                      |    0.320 |    0.372 | 0.75 |    0.002 |

  histogram_match is by far the largest single lever (area_gap 0.373 ->
  0.002, a ~99% reduction; fidelity 0.44 -> ~0.33, ~25% better) - it fixes
  a failure the other three mechanisms cannot touch, since none of them
  constrain the global tone distribution. coarse_patch_size and robust_norm
  each measured essentially no further gain once nn_alpha+histogram_match
  were both active (differences within noise across all three metrics) -
  both are real, correctly-targeted, unit-tested mechanisms, but in this
  pipeline's current state they had nothing significant left to fix, so
  neither was promoted to a default. **New CLI defaults: `--nn-alpha 1.0`
  and histogram matching on** (`--no-histogram-match` to disable). Grid
  jitter's default was RE-TESTED rather than assumed still correct once
  hard-NN selection existed (jitter's original justification was reducing
  softmax-averaging blur, which hard-NN also addresses, so the two
  mechanisms could plausibly have started fighting each other): jitter+nn10
  beats no-jitter+nn10 on every one of the four metrics (fidelity 0.454 vs
  0.487, completeness 0.383 vs 0.398, edge 0.92 vs 1.04 [closer to 1.00,
  not overshooting], area_gap 0.346 vs 0.349) - kept as default, now
  additionally justified independent of the softmax-blur mechanism it was
  originally adopted for.
- **What is still NOT solved, stated plainly**: at these numbers, this is
  real, measured, broad-based progress (not a point fix for one image) -
  but it is not "solved". Per-image fidelity/completeness still range
  0.20-0.59 across the 5 images (802e607c7c, the sparse-dot/no-periodic-
  structure image, remains the hardest case on every metric), and several
  outputs still show particles fused along one axis into short chains
  rather than fully separated individuals (see 15_new_default_*.png) -
  histogram_match fixes the AREA fraction, not each individual particle's
  boundary against its neighbors. Candidate next steps, not yet attempted:
  (1) an explicit particle-separation/connected-component penalty during
  denoising, rather than relying on histogram + patch metrics as a proxy
  for it; (2) adaptive per-image coarse_patch_size (carried over from
  above); (3) GPNN's own full pipeline (rotation/reflection-augmented patch
  bank, coarser-to-finer noise injection only at the coarsest level) rather
  than this project's diffusion-derived per-level noise schedule, which
  this investigation adopted only the NN-selection piece of.

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
