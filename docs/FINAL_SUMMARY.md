# Final summary - Day 24

What the team can state, with evidence, per the spec's Layer 7 gate and the
"Final project comparison" schema. Every number here is quoted from
`docs/findings.md` (the underlying notebook is named alongside it) - this
file does not introduce any new measurement, it only assembles the
evidence into the 6-point statement the spec asks for at the end.

## 1. What the generator reproduces well

- Coarse-to-fine generation propagates real global layout that single-scale
  generation cannot: a bright blob appears in the exact corner where the
  real reference has its large blob (`coarse_to_fine_sampler.ipynb`).
- Intensity/gradient/PSD statistics are directly comparable between real
  and synthetic output via a shared evaluator (`src/evaluation/metrics.py`,
  `basic_evaluator.ipynb`), with self-tested edge-case handling (flat-image
  histogram, PSD DC-bin bias).
- Seed diversity is neither collapsed nor unrelated-noise-like on EMPS:
  36.07 vs the real image's own std of 55.64.
- The denoiser itself matches a hand-computable toy example and an
  independent brute-force check exactly (`src/models/denoiser.py`).

## 2. What it fails to reproduce

- Single-scale generation cannot preserve global organization - a
  structural limit, not a tuning problem (Layer 4).
- Variance erodes across chained coarse-to-fine refinement stages, and MORE
  diffusion steps per scale makes this WORSE, not better (Layers 5-6).
- The approximate (mean-prefiltered) denoiser is a net loss on small
  patch banks (only helps on Day 17's large 16,256-patch bank) - scale-
  dependent, not universally useful.
- On RODARE, several evaluator numbers looked "better" than EMPS's, most
  plausibly because the extreme downsize (0.12x) needed for CPU feasibility
  destroyed real structure before generation even started - a measurement
  artifact, not genuine superiority on real SEM.

## 3. How much it copies, and whether any copying crosses groups

Two-stage diagnostic (`src/evaluation/copying.py`, `copying_diagnostic.ipynb`,
Day 20), applied to a real generated EMPS image:

| | own-group | other-group | hidden-test |
|---|---:|---:|---:|
| EMPS | 21.9% (expected - denoiser built from these patches) | 0.0% | 0.0% |
| RODARE | 0.0% | 0.0% | n/a (RODARE has no hidden-test split) |

No confirmed cross-group or hidden-test leakage in either dataset. Caveat,
stated explicitly at the time: this is a high-specificity, not
high-sensitivity, gate - it does not prove all semantic similarity is
absent.

## 4. Which configuration is preferred, and why

`stride=4` (non-overlapping patches), `mean_tolerance=None` (exact
denoiser) - measured, not assumed, to win on THREE axes simultaneously
(`efficiency_profiling.ipynb`, `exact_vs_fast_config.ipynb`, Day 16-18):

- **Time**: 4.21x faster than the dense-overlap reference.
- **Quality**: std=61.22 vs the reference's collapsed std=1.23 (real=55.64).
- **Memory**: 1.17MB peak vs the reference's 6.2MB (`tracemalloc`).

The mean-prefiltered approximate denoiser is kept in the codebase as an
available, scale-dependent option (helps on large banks per Day 17), not
part of this default.

## 5. Whether the source mask may be transferred, and in which density regime

**No, not as-is, in any regime measured:**

| regime | n | mean displacement | method |
|---|---:|---:|---|
| Analytic control (known transform) | 5 trials | 27.01px | exact geometric ground truth |
| EMPS (real, sparse, 2-9 instances/image) | 6 instances | 17.16px | local NCC template matching |
| RODARE (real, dense, ~800 instances/image) | 333 pseudo-instances | 11.71px | same, on connected-component blobs (RODARE's mask has no per-instance numbering) |

RODARE's number should NOT be read as "more valid than EMPS" - it measures
blob-level, not individual-carbide-level, displacement, and used a far more
aggressive downsize. The consistent conclusion across all three
measurements: **a source mask does not survive generation and must never
be inherited automatically.**

## 6. Whether synthetic data helps on a real held-out downstream task, at equal update budget

**INCONCLUSIVE on EMPS** (`downstream_experiment_run.ipynb`, Day 22): arm
(c) real+synthetic beat arm (a) real-only by +1.93 percentage points,
just under the pre-registered +2.0pp benefit margin (registered before the
run, in `configs/experiments/downstream_preregistration_v1.json`).

**More important than the verdict**: arm (a)'s own accuracy (53.22%) barely
exceeded the majority-class baseline (53.11%) - the classifier used (a
from-scratch 2-layer MLP, 500 gradient steps) had not meaningfully learned
the task at this scale. This run demonstrates the METHODOLOGY works
end-to-end (equal budget, group-level bootstrap, pre-registered thresholds
applied mechanically) - it does not yet give a statistically meaningful
answer to whether synthetic augmentation helps, because the classifier
itself needs to be capable of learning the task first.

**Not repeated on RODARE**: only 36 total field/hold-out pairs exist there,
too small for the same group-bootstrap methodology to carry any power.

---

## Reproduction gate

Per spec rule 7 ("another student should be able to rerun and explain the
layer before it is complete") and the Day 24 task ("someone else runs from
the documentation alone"): every notebook in `notebooks/pipeline/` was
re-executed from a completely fresh kernel, end to end, using only what is
committed to the repo (source code + `configs/` + the README's documented
folder layout for external data) - not relying on any leftover in-memory
state from earlier development. See the run log below for the actual
pass/fail per notebook.

**Result: 15/15 notebooks passed**, each executed in a brand-new kernel
process with no shared state between them:

```
approximate_denoiser.ipynb              OK
basic_evaluator.ipynb                   OK
coarse_to_fine_sampler.ipynb            OK
copying_diagnostic.ipynb                OK
data_pipeline_validation.ipynb          OK
denoiser_real_patches.ipynb             OK
downstream_experiment_run.ipynb         OK
downstream_experiment_setup.ipynb       OK
efficiency_profiling.ipynb              OK
exact_vs_fast_config.ipynb              OK
label_validity.ipynb                    OK
multiscale_patch_pipeline.ipynb         OK
rodare_arm.ipynb                        OK
single_scale_sampler_verification.ipynb OK
source_safe_split_and_loader.ipynb      OK
```

All 13 `src/` modules' own `__main__` self-tests also pass independently
(`guard.py`, `loader.py`, `furniture.py`, `pyramid.py`, `patches.py`,
`claims.py`, `augmentation.py`, `denoiser.py`, `sampler.py`,
`patch_classifier.py`, `metrics.py`, `copying.py`, `label_validity.py`).

## Where this project meets AMAT-3

Per the spec, two things must be aligned before either group's evaluation
numbers can be compared or combined:

1. **The statistical unit is the source group** - ours is the EMPS DOI
   group or the RODARE field, never a patch or a tile. Every interval this
   project reports states its `n_groups` explicitly (see `docs/findings.md`
   and every `results/tables/*.csv` file).
2. **A shared observer/metric definition** - the spec says to ask AMAT-3
   for their (already-debugged) frozen segmentation-observer rule rather
   than writing a fresh one, since they will have already found and fixed
   at least one defect in it. **This has not happened yet** - this project
   has no live channel to the AMAT-3 group as part of this solo work, so
   this is an explicit, stated action item for whoever presents this
   project's teach-back, not something resolved here.

**What to compare notes on, once contact is made**: this project's copying
diagnostic and AMAT-3's invention diagnostics are "the same question from
two sides" per the spec - what each diagnostic could NOT detect (this
project's stated blind spot: high-specificity, not high-sensitivity - see
point 3 above) is the most useful thing to exchange.
