# Research log

## 2026-08-18 — W1.T1/W1.T2 equivalent: real EMPS data + diagnostic notebooks

**Question:** What does a real SEM/electron-microscopy image actually look like,
statistically, and what artifacts (noise, edge effects, charging, drift) can be
detected in it without acquisition metadata?

**Hypothesis:** A synthetic generator needs to match not just object shape but
the underlying pixel statistics (intensity distribution, edge-yield curve,
noise structure) to be considered successful.

**What was changed/built:**
- Downloaded EMPS (465 real images, 322 source papers, MIT license).
- Built `sem_image_formation.ipynb`: dataset examples, per-image
  statistics over a 40-image sample, histograms, zoomed crops, SEM-vs-photo
  comparison table.
- Built `sem_artifact_diagnostics.ipynb`: row/column intensity
  profile, Sobel gradient magnitude, local mean/variance (9x9 window), FFT
  magnitude, plus an independent pedagogical mini-experiment (synthetic line
  shift / drift / low-dose Poisson noise, explicitly labeled as unvalidated).

**Result:**
- Sample statistics (n=40): mean intensity ~122, std ~45 (0-255 scale).
- On the specific image analyzed in D2: row-profile range 51-189, max
  single-step jump 14 levels (gradual, not a step); gradient mean ~11.5, 95th
  pct ~34; local variance mean ~129, max ~8173 (concentrated in a few
  regions); FFT peak (excl. DC) ~13.9 vs mean ~6.5 (mild, not extreme).

**What failed/surprised:** No strong periodic-artifact signature (no sharp
isolated FFT peak) was found in the one image inspected in detail - a
legitimate negative finding, not a bug. This is a single-image spot check,
not yet run across the dataset.

**Conclusion:** Baseline established. The observation table conclusion for
this image is "no strong drift/charging evidence" with medium confidence
(no acquisition metadata available to confirm/deny).

**Next experiment:** Source/specimen/DOI-safe split of EMPS (test quarantine),
then a first training-free real->synthetic vertical slice
(`generator_v0`, deterministic, seeded).

## 2026-08-19 — Data -> generator input (source-safe split + loader)

**Question:** how to guarantee the generator never touches a TEST-split image.

**What was found:** EMPS ships its own train.csv/test.csv (by filename, 366/99
images). Verified these are already DOI-safe - zero DOI overlap. Reused this
instead of building a split from scratch.

**What was built:**
- `src/data/split.py::build_source_split` - carves a validation set out of the
  official TRAIN pool by DOI (not by image), writes
  `configs/experiments/source_split_v1.json`.
- `src/data/loader.py::load_generator_source(source_id)` - the only path into
  the generator; asserts the id is not in TEST_IDS.

**Result:** train 310 images / 210 sources, validation 56 images / 37 sources,
test 99 images / 75 sources. Zero cross-split source overlap (assertion
passes). Rejection of a real TEST id verified to actually raise.

**Conclusion:** split + loader gate are in place and tested.
**Next experiment:** build `generator_v0` using the 3 selected TRAIN dev
sources, then run the "does source image matter?" comparison.

## 2026-08-19 to 2026-08-26 - Layers 1 through 6 built and gated

Full per-day detail lives in `personal/PLAN_24_DAYS.md` (gitignored - not
duplicated here). Summary of what each layer left behind, per the project's
own "every layer must leave an artifact" rule - see `docs/findings.md` for
the measured numbers behind each line:

- **L1** - manifest with grouping/split/furniture/modality, hidden-test
  guard proven to fire on contamination.
- **L2** - `build_pyramid`, `extract_patches`/`reconstruct_from_patches`,
  exact reconstruction on a real image.
- **L3** - `denoise_patch`, hand-checked + brute-force verified, collapse-
  to-copy point recorded.
- **L4** - `sample_single_scale`: variance-collapse fix, global-organization
  limit found and explained (not patched), the fix's own non-generalization
  to EMPS found and explained rather than re-patched a third time.
- **L5** - `sample_coarse_to_fine`: propagates global layout single-scale
  cannot; label-validity measured on both an analytic control and (later,
  see 2026-08-27 below) a real EMPS image.
- **L6** - stride=4 identified as a genuine win on time, quality, AND memory
  simultaneously; the mean-prefiltered approximate denoiser classified as
  scale-dependent (loses on small banks, won on Day 17's large one) rather
  than accepted or rejected outright.

**Next experiment:** Layer 7 - basic evaluator, then the two-stage copying
diagnostic, then the downstream 3-arm experiment and the RODARE SEM-claim
arm.

## 2026-08-27 - Layer 7 started, then an unplanned systematic gap-closing pass

**Question:** before continuing Layer 7 (copying diagnostic, Day 20), does
everything already built for Layers 1-6 actually match the full spec
(`AMAT4_STUDENTS_Training_Free_SID_Layered_Tasks.md`), or only what came up
in conversation along the way?

**What was found:** a systematic audit (every "Deliver" bullet in every
layer, checked against the actual repo files, not against the personal
plan's own checkmarks) found 9 real-or-minor gaps: no frozen pyramid/patch
config, no `eligible` manifest column, tables/figures existing only as
notebook output rather than saved files, no furniture-policy version
history, no per-scale patch-tile visualization (only pyramid-level
visualization), no step-count sweep for the single-scale sampler, only 1
(not 5) multiscale synthetic output, no real-EMPS label-validity
measurement (analytic control only), and no memory profiling.

**What was built to close them** (see `docs/findings.md` for the numbers):
frozen config (`configs/models/pyramid_patch_config_v1.json`); `eligible`
column; persisted CSV/PNG artifacts across L2/L5/L6 notebooks; furniture
`v0`/`v1`/`v2` version functions with a self-test proving the v0/v1 gap
(mid-image scale-bar line at normal intensity); an 8-patch fine/medium/
coarse tile gallery; a step-count sweep on NIST (30 steps confirmed as the
measured optimum); a 5-seed multiscale gallery with diversity measured;
`src/evaluation/label_validity.py` (local template-matching label-validity
for real data, since real EMPS has no known geometric transform the way the
analytic control does) run on a real 6-instance EMPS image; and a
`tracemalloc`-based memory table alongside the existing runtime table.

**What was NOT closed today:** the RODARE arm (record 4124, `data.zip`,
1.3GB) - the direct file-download endpoint is blocked by NetFree at the
network level even though the record page itself loads; manual browser
download is in progress. The copying-diagnostic MODULE
(`src/evaluation/copying.py`) is built and self-tested, but has not yet
been applied to real generated images with own-group/other-group reporting
- that remains Day 20's actual open item.

**Conclusion:** Layers 1-6 now match the full spec, not just the
conversation-driven subset of it - see `docs/limitations.md` for what
remains open going into the rest of Layer 7.
**Next experiment:** apply the copying diagnostic to real generated images
(Day 20), then the downstream 3-arm experiment, then RODARE once the
download completes.
