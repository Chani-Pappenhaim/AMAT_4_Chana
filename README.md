# AMAT_4_Chana

```
Team: AMAT-4 (Synthetic SEM generation)
Student: Chana
Project: Training-free, single-image synthetic electron-microscopy generation
Current research question: Under limited real data, when and why do training-free
  synthetic microscopy images help - or harm - a downstream model on untouched
  real data?
Current status: Week 2 (per general team plan v4.0, 18 Aug 2026). Completed
  Week 1 retrospective work on real EMPS data (image-formation baseline +
  artifact/noise diagnostics). Not yet started: source-safe split, generator_v0.
How to run: see notebooks/exploration for the initial EDA notebooks and
  notebooks/pipeline for the data-pipeline notebooks. Each is self-contained -
  installs its own dependencies at the top and downloads/reads from a local
  emps/ folder (not committed - see .gitignore).
Main results: see docs/research_log.md and docs/findings.md.
```

## Dataset

Working dataset: **EMPS** (Electron Microscopy Particle Segmentation, Yildirim &
Cole, 2021) - 465 real SEM images with pixel-level segmentation masks, MIT
license, from 322 distinct source papers. https://github.com/by256/emps

Per the general team plan: unverified local data with no acquisition/calibration
metadata is called **electron microscopy**, not SEM, until modality/calibration
evidence is available.

## Repository structure

```
AMAT_4_Chana/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/          daily/weekly work, one folder per week
├── src/                reusable code (data/models/evaluation/utils)
├── configs/            experiment settings (seed, split, model, etc.)
├── experiments/        one folder per experiment ID (EXP_NNN_name)
├── results/            tables, figures, summaries
├── docs/               research_log.md, findings.md, limitations.md
└── tests/
```
