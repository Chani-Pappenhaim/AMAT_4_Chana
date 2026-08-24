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
