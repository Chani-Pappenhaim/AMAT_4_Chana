# Multi-agent build plan: 7-layer training-free synthetic EM generator

## Context

The project spec lives at [`AMAT4_STUDENTS_Training_Free_SID_Layered_Tasks.md`](../משימות שלי/AMAT4_STUDENTS_Training_Free_SID_Layered_Tasks.md) and defines 7 sequential layers (L1–L7) that together build a training-free, single-image, patch-based diffusion generator for synthetic electron-microscopy images (the "Qiu method"), plus a trust-boundary evaluation of whether the synthetic images are actually usable.

The repo already has `src/`, `notebooks/pipeline/`, and a README claiming "layers 1–7 complete," plus some uncommitted in-progress files (`generate.py`, `src/models/generate.py`, a modified `sampler.py`, a new `min data set/` folder, `results/generated/`). This plan is a **full re-plan from the spec** rather than a review/completion of that existing state — it treats the current `src/` as prior art only, not a baseline to preserve or extend. The goal is a clean-architecture rebuild, developed by orchestrating five specialized agent roles (architecture, dev, ML, image-processing, SEM/EM-domain), with no duplicated logic between layers and no over-engineering beyond what the spec actually demands.

## Module architecture (`src/`)

One home per concern; every layer imports shared logic, never re-derives it.

```
src/
├── schemas.py            # SourceManifest row, GroupID, PatchBank record, GenerationConfig
│                          #   — plain dataclasses, single source of truth for L1..L7 shapes
├── data/                  # L1
│   ├── sources/{nist,emps,rodare}_loader.py
│   ├── grouping.py        # group-by-DOI/field — SHARED by L1 manifest, L7 bootstrap, L7 cross-group edges
│   ├── hidden_test_guard.py
│   ├── claims.py          # modality/claim-language enforcement — called at ingestion AND at report time
│   ├── furniture.py       # exclusion mask (border + extreme-intensity + long-line) — SHARED, L2 imports it
│   └── manifest.py        # integrates the above into L1_source_manifest.csv
├── pyramid/                # L2
│   ├── pyramid.py          # build_pyramid(), scale-factor based (handles EMPS's 218 geometries)
│   └── patch_bank.py       # extract/store/reconstruct — the ONLY patch code in the repo; L3–L6 query it
├── denoise/                 # L3 core + L6 acceleration variants
│   ├── similarity.py        # dispatch by config string ("exact"|"approximate") — plain function, no class hierarchy
│   ├── exact_backend.py      # L3 baseline, matches the toy case
│   ├── approximate_backend.py# L6, same input/output contract as exact_backend
│   └── denoiser.py            # denoise_patch(query, bank, backend=...) — ONE implementation for L4/L5/L6
├── sampler/                    # L4/L5
│   ├── schedule.py              # noise schedule/step-count config (L6 "fewer steps" = config change, not new code)
│   ├── single_scale.py
│   └── multiscale.py             # calls single_scale.py per pyramid level, does not duplicate the loop
├── efficiency/                    # L6 — config/orchestration only
│   ├── configs.py                  # ExactConfig / FastConfig presets
│   └── distribution_check.py        # same-seed exact-vs-fast comparison harness
├── evaluation/                       # L7
│   ├── metrics.py                     # one Evaluator over all 4 configs (real / single / multi / fast)
│   ├── copying.py                      # two-stage diagnostic; cross-group edges via data/grouping.py
│   ├── label_validity.py                # LV measurement, used at L5 gate and again per-arm in L7
│   ├── segmentation_metrics.py           # thin adapter stub for AMAT-3's frozen observer/metric rule (external dep, see below)
│   └── downstream_experiment.py           # 3-arm harness, equal-budget enforcement, bootstrap-over-groups
└── utils/
    ├── seeding.py, imageio.py, reporting.py   # small shared helpers, deliberately minimal
```

`tests/` mirrors `src/` 1:1 (currently empty — `.gitkeep` only, so this is from-scratch). `configs/experiments/` and `configs/models/` hold versioned JSON configs (existing repo convention) — figures/CSVs generated from code, never hand-edited.

## Agent roles → work packages, per layer

Roles: **ARCH** (interfaces + duplication/coupling review), **DEV** (glue code + notebooks), **ML** (denoiser/sampler/efficiency math), **IMG** (pyramid/patch/furniture/copying-diagnostic image processing), **DOM** (dataset quirks, modality rules, label validity, EMPS-vs-RODARE reporting split).

"Activating an agent" = a scoped subagent task naming exactly which files it owns, the interface contract from ARCH, the spec deliverable it must satisfy, and the test file it must pass — never touching files outside its ownership list.

- **L1 (data spine)**: ARCH fixes `schemas.py` + function signatures first (no implementation). Then fan out in parallel: DOM builds the 3 source loaders + grouping semantics; IMG builds `furniture.py` (tuned on a frozen sample, keeping failed policy versions); DOM builds `claims.py`. Sequential tail: DEV builds `hidden_test_guard.py` then `manifest.py` (needs finalized groups/splits). ARCH reviews for duplication before the gate.
- **L2 (pyramid + patches)**: depends on L1 gate. ARCH defines the `PatchBank` shape. IMG owns `pyramid.py` + `patch_bank.py` as one thread (extraction depends on the pyramid's output — don't split it). DOM spot-checks EMPS geometry handling. DEV wires the notebook.
- **L3 (closed-form denoiser)**: depends on L2 gate. ARCH pre-reviews the `similarity.py` dispatch contract so L6 can plug in later without touching `denoiser.py`. ML (mostly solo — this is the load-bearing math) builds `exact_backend.py`, `denoiser.py`, the hand-checkable 3×3 toy case, and the collapse-to-copy study. DEV wires the notebook. ARCH reviews.
- **L4 (single-scale sampler)**: depends on L3 gate (toy case passing). ML builds `schedule.py` + `single_scale.py`. DOM owns the NIST-before-EMPS checkpoint — hard sequential gate, not parallelizable. DEV wires the notebook.
- **L5 (coarse-to-fine + label validity)**: depends on L4 gate. Parallel: ML builds `multiscale.py`; DOM builds `label_validity.py` against an analytic control (independent of the sampler being finished). DEV integrates LV onto real multiscale output once both land, and establishes the EMPS(sparse)-vs-RODARE(dense) separate-reporting convention (reused in L7). ARCH confirms `multiscale.py` calls `single_scale.py` rather than forking it.
- **L6 (efficiency)**: depends on L5 gate + L3 toy case re-passing. ML implements accelerations in the spec's mandated order (fewer steps → smaller bank → chunked comparison → approximate/NN filtering → GPU only if profiling shows it's still needed), classifying each one as runtime-only vs distribution-changing **by running the same-seed exact-vs-fast test**, one acceleration at a time against a fixed prior baseline (inherently sequential). ARCH confirms "fast" variants are config flips, not forked copies.
- **L7 (evaluation + trust boundary)**: depends on L6 gate. Genuine parallelism: IMG builds `copying.py` (two-stage diagnostic); DEV builds `metrics.py` + `downstream_experiment.py`; DOM authors the pre-registration (margins/tolerances, written *before* running) and the EMPS/RODARE separate-reporting + "SEM earned here" framing. `segmentation_metrics.py` is flagged as an **external coordination item**: AMAT-3 owns the frozen observer/metric rule this project should reuse — ARCH designs the adapter interface now with a stub implementation so L7 isn't blocked; this needs contact with that team, not more code. ARCH does a final integration review before the gate.

## Sequencing

Layers are strictly sequential (L1→L7), each gated by its regression artifact — never move up while the current layer has an unexplained correctness failure. Real intra-layer parallelism exists in L1 (three sub-problems once the manifest schema is fixed), L5 (sampler vs. LV methodology), and L7 (copying vs. downstream harness vs. pre-registration). L2, L4, and L6 are inherently sequential internally (pyramid before patches; NIST before EMPS; one acceleration classified at a time against a fixed baseline).

## Overkill vs. load-bearing

**Keep as specified (explicit spec requirements, not optional rigor):** the hidden-test-guard + contaminated-record test, the L3 toy case as a permanent regression check, the NIST-before-EMPS gate, L5's LV measurement against an analytic control, L6's test-not-reasoning acceleration classification, L7's pre-registered margins/bootstrap-over-groups/never-pooled reporting, and `grouping.py` as one shared implementation (it's what makes "state group counts, never pool" enforceable).

**Trimmed as overkill for bootcamp scale:** no `SimilarityBackend` class hierarchy (a config-string-dispatched function is enough); no dedicated `src/domain/` DDD package (plain dataclasses in `schemas.py`); no full reporting framework (a few helper functions in `utils/reporting.py`); no speculative GPU path in L6 until profiling actually demands it; the 5 agent "roles" are task-scoped subagent prompts invoked per layer, not persistent services with APIs between them.

## Verification artifacts (one location each)

| Layer | Check | Location |
|---|---|---|
| L1 | Contaminated-record assertion fires | `tests/data/test_hidden_test_guard.py` |
| L1 | "SEM" token fails an EMPS-derived claim | `tests/data/test_claims.py` |
| L2 | Patch bank reconstructs the source image | `tests/pyramid/test_patch_reconstruction.py` |
| L3 | 3×3 hand-checkable toy case (permanent) | `tests/denoise/test_toy_case.py` — re-run against both backends at L6 |
| L4 | NIST fixture passes before EMPS is touched | `tests/sampler/test_nist_fixture.py` |
| L5 | LV measurement vs. analytic control (~26px) | `tests/evaluation/test_label_validity_analytic_control.py` |
| L6 | Exact-vs-fast same-seed distribution check | `tests/efficiency/test_distribution_preservation.py` |
| L7 | Two-stage copying diagnostic + cross-group edges | `tests/evaluation/test_copying_diagnostic.py` |
| L7 | Downstream 3-arm equal-budget + pre-registered margins | `tests/evaluation/test_downstream_budget.py` |

Each `notebooks/pipeline/` notebook calls into `src/` and its paired test rather than reimplementing checks inline, so the spec's "another student can rerun and explain this layer" rule holds without per-notebook drift.

## Execution note

This plan describes the target architecture and the agent-orchestration sequencing. Actual implementation should proceed layer by layer (L1 first), spawning the ARCH/DEV/ML/IMG/DOM subagents as scoped above, with a gate check (the layer's verification artifact passing) before starting the next layer.
