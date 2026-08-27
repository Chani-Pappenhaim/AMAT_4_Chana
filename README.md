# AMAT_4_Chana

```
Team: AMAT-4 (Synthetic SEM generation)
Student: Chana
Project: Training-free, single-image synthetic electron-microscopy generation
Current research question: Under limited real data, when and why do training-free
  synthetic microscopy images help - or harm - a downstream model on untouched
  real data?
Current status: Layers 1-6 complete (data spine, pyramid/patches, closed-form
  denoiser, single-scale sampler, coarse-to-fine multiscale sampler + label
  validity, efficiency/reproducibility). Layer 7 (evaluation + trust boundary)
  in progress: basic evaluator and copying diagnostic built and self-tested;
  downstream 3-arm experiment and the RODARE SEM-claim arm not yet run.
How to run: see notebooks/exploration for the initial EDA notebooks and
  notebooks/pipeline for every later layer's notebook (built from a paired
  build_*_notebook.py script in the same folder - regenerate a notebook by
  running its script, then execute it). Each is self-contained - installs
  its own dependencies at the top and reads from a local emps/ (EMPS) or
  AMAT/ (NIST, RODARE) folder (not committed - see .gitignore).
Main results: see docs/research_log.md and docs/findings.md.
```

## Datasets

Three datasets, split by role, per the project spec
(`AMAT4_STUDENTS_Training_Free_SID_Layered_Tasks.md`):

- **EMPS** (Electron Microscopy Particle Segmentation, Yildirim & Cole, 2021)
  - development + statistics dataset, Layers 5-7. 465 images, 322 independent
    DOI groups, MIT license. https://github.com/by256/emps
  - EMPS is mixed SEM and TEM with no per-image modality label - EMPS-derived
    claims say **"electron microscopy"**, never "SEM" (enforced by
    `src/data/claims.py`).
- **NIST** (`data.nist.gov/od/id/mds2-3838`) - correctness fixture, Layers
  2-4. One clean reference image with a known mask, used to verify the
  denoiser and sampler before touching real, more complex EMPS texture.
- **RODARE** (`rodare.hzdr.de/record/4124`) - the SEM claim arm, Layer 7
  only. 13 real SEM fields with real instance labels; only this arm may say
  "SEM" (card v6 sets `nm_claim_permitted = false` - pixels only).

## Repository structure

```
AMAT_4_Chana/
├── README.md
├── AMAT4_STUDENTS_Training_Free_SID_Layered_Tasks.md   full 7-layer spec
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── exploration/    early, pre-Layer-1 EDA notebooks
│   └── pipeline/       one notebook (+ build_*.py generator) per layer/day
├── src/
│   ├── data/           split, guard, furniture, claims, pyramid, patches, loader
│   ├── models/         denoiser, sampler
│   └── evaluation/      metrics, copying, label_validity
├── configs/
│   ├── experiments/    source_split_v1.json, exact_vs_fast_config_log.json
│   └── models/         pyramid_patch_config_v1.json
├── results/             tables/, figures/ - real artifacts written by notebooks
├── docs/                research_log.md, findings.md, limitations.md
└── tests/
```
