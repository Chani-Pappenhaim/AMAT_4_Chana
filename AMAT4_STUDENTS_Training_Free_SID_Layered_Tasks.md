# AMAT-4 (Students) — Training-Free Single-Image Diffusion for Synthetic Electron-Microscopy Images

**Note on the title.** The method generates synthetic *electron-microscopy* images. It earns the word *SEM* only in Layer 7, and only on the RODARE arm. Layer 1 explains why, and the distinction is enforced by an automatic check, not by good intentions.

## Project goal

Build the AMAT-4 generator project from scratch as a sequence of working layers.

Each layer must leave behind something that the next layer can use.

Use this cycle throughout:

**Build fast with AI → understand → verify → experiment → break → compare → modify → reproduce → teach.**

---

## The data you will use — and why it is not what you would guess


### The rule that decides everything

For a single-image generator, **n is the number of independent source images**, not the number of pixels, crops or patches. You fit one generator per source image, so every source is one experimental unit.

That single fact rules out the obvious choice:

```text
EMPS      322 independent DOI groups   (465 images)
RODARE     13 usable real SEM fields
NIST        4 usable geometries
```


Your scarcity design needs k ∈ {1, 2, 4, 8} source images **with repeated draws**. At k = 1 the choice of source dominates the result, so you need many non-overlapping draws to say anything. With 4 geometries you get essentially none. With 322 groups you can run the curve properly and put a real interval on it.

So the datasets split by **role**, not by preference:

| Role | Dataset | Why |
|---|---|---|
| **Correctness fixture** (L2–L4) | **NIST**, set 1 clean reference | One image, known mask, controlled geometry. Perfect for verifying a denoiser and a sampler. Four geometries is plenty for a unit test and far too few for a statistic. |
| **Development + statistics** (L5–L7) | **EMPS** | 322 independent sources. The only source here that can carry a scarcity curve. |
| **The SEM claim** (L7) | **RODARE**, 13 fields | Real SEM, real instance labels, real acquisition artefacts. Small, but it is the claim arm. |

**Never CSIRO** (`csiro\`) — transmission microscopy, cannot support any SEM statement.
**Not KRISS** (`kriss_semdenoising\`) — reuse rights unresolved.

### EMPS — verified layout

folder `emps\EMPS\`

```text
images\        465 PNG, RGB-stored greyscale, variable size
segmaps\       465 PNG, uint16 instance labels
metadata.csv   columns: filename, doi, locator, regions
train.csv      366 image IDs, no header
test.csv        99 image IDs, no header
```

Measured facts you should not rediscover the hard way:

- **465 images come from 322 papers.** Up to 12 images share one DOI and are **not independent**.
- Author split: **366 train / 247 DOI groups**, **99 test / 75 DOI groups**, with zero image or DOI overlap.
- Files are stored as RGB but **R = G = B on all 465**. Convert to a single channel.
- **218 distinct canvas geometries**; only 29 images are exactly 512×512. Variable size is fine — the method fits per image.
- 11,535 instances in total, **median 8 per image**, range 1–497.
- **EMPS is mixed SEM and TEM, and the metadata has no per-image modality field.** The source paper includes both. Therefore EMPS licenses the phrase *"electron microscopy"* and never *"SEM"*.

### RODARE — verified layout

https://rodare.hzdr.de/record/4124

folder `rodare_4124_carbide_sem\` · CC BY 4.0 · read `DATASET_CARD.md` (v6, the only live card; v2–v5 are in `card_history\` and are superseded)

- **13 usable fields:** JFL 8 (NVision 40) + ANP-10 5 (Ultra 55).
- **ANP-3 is quarantined.** It shares an instrument serial with ANP-10, so it is not a third microscope.
- **Crop the 132-row databar:** 2048×1536 → 2048×1404. The `results\*_label.tif` files are already cropped; the raw TIFFs are not.
- **The two detector channels of one field are channel siblings, not two samples.**
- Group by field / material / instrument / session — never by tile.
- **Pixels only.** Card v6 sets `nm_claim_permitted = false`.

### NIST — verified layout

https://data.nist.gov/od/id/mds2-3838

folder `nist_detection_limits_sem\`

```text
raw\mask_sets\masks\set1_cex_noise_000_contrast_100.tiff        clean reference
raw\mask_sets\masks\mask_set1_cex_noise_000_contrast_100.tiff   its mask
raw\intensity_sets\set1\                                        567 degraded variants
```

Sets 1, 2, 3, 5 only. Set 4's mask is invalid. Set 6 is hidden. 512×512 `uint8`, masks `{0, 255}`, **no calibration — pixels only.**

---

## Overall schema

```text
Electron-microscopy sources  (EMPS dev · RODARE SEM claim · NIST fixture)
     │
     ▼
L1  Data spine + source selection
     eligibility / furniture / provenance / hidden test
     │
     ▼
L2  Multiscale image pyramid + patch datasets
     coarse → fine
     │
     ▼
L3  Closed-form patch denoiser
     noisy patch → weighted clean-patch estimate
     │
     ▼
L4  Single-scale diffusion sampler
     noise → iterative denoising → image
     │
     ▼
L5  Coarse-to-fine training-free SID
     multiscale generation + label validity
     │
     ▼
L6  Efficiency + reproducibility
     chunking / approximate search / step count
     │
     ▼
L7  Evaluation + trust boundary
     diversity / copying / structure / downstream utility
```

**Important:** every experiment must use the same source manifest, the same output contract, and the same evaluator.

Do not use hidden test data to build patch banks, choose parameters, tune thresholds, or select the final method.

Generated images are **synthetic outputs**. Do not automatically inherit the source mask after geometry-changing generation.

---

# Layer 1 — Data spine and valid sources

## Task

Create the data foundation for the generator.

Build one canonical manifest that every later layer uses. It must resolve four things.

### 1. Grouping

Group EMPS by **`doi`**, never by filename. Group RODARE by **field**. The manifest carries the group ID for every image, and every later split, draw and bootstrap uses it.

### 2. The hidden test set

EMPS `test.csv` — 99 images, 75 DOI groups — is evaluation-only and must never reach a generator, a patch bank, a threshold or a model-selection decision.

Write **one automatic assertion** that fails loudly if a test-set DOI appears anywhere in a generation lineage. Then deliberately feed it a contaminated record and show that it fires. An assertion that has never failed has not been tested.

### 3. Figure furniture — the thing most likely to embarrass you

EMPS images come from **published figures**. They carry burned-in scale bars, panel letters, axis labels and footers.

**A patch-based generator will happily reproduce a scale bar.** It has no idea the bar is not part of the specimen. This is the single most likely way your generated images end up unusable.

Build an exclusion mask per image and accept a patch only if it intersects **no** exclusion pixel. A workable policy combines:

- a fixed border fraction;
- masking of unlabelled extreme-intensity components above a minimum size;
- detection of long straight horizontal and vertical lines.

Tune it by **looking at the surviving patch support**, not just the raw mask, on a frozen sample of images you fix in advance. Then report how many images were affected and how much patch support each policy version left.

Keep every version that failed. The failures explain why the final policy has the shape it has.

### 4. Modality and claim language

EMPS has no per-image SEM/TEM label. Therefore:

- every EMPS-derived claim says **"electron microscopy"**;
- the word **"SEM"** appears only on the RODARE arm;
- write a check that **fails a release** if the token `SEM` appears in an EMPS-derived claim string.

Do not infer modality from how an image looks and then use that inference to support an SEM claim. That is circular.

## Investigate

- How many independent groups survive eligibility, and how many images does that correspond to?
- Which images are valid generator sources, and which are excluded and why?
- What must remain hidden from the generator?
- How much of each image survives furniture exclusion? Which images lose the most?
- What changes across images inside one DOI group — are they really the same specimen?
- Which claims are supported by the dataset metadata, and which are not?

## Deliver

- `L1_Data_Spine.ipynb`
- `L1_source_manifest.csv` — one row per image with group ID, split, eligibility, exclusion statistics
- representative image / segmap / exclusion-mask figures
- train/development/hidden-test policy
- **one automatic assertion preventing hidden-test use, plus proof it fires on a contaminated record**
- the furniture policy, its version history, and the count of affected images
- **the modality policy and the failing-claim check**
- short dataset/provenance note

## Layer gate

Every later layer loads the same valid sources from the manifest, hidden test data cannot accidentally enter generation, no accepted patch touches figure furniture, and no EMPS-derived claim can say "SEM".

---

# Layer 2 — Multiscale pyramid and patch datasets

## Task

Turn one source image into the internal dataset the method needs.

Create a multiscale image pyramid.

```text
full resolution
   ↓  ÷2
   ↓  ÷2
   ↓  ÷2
coarsest
```

Define the pyramid by **scale factors, not fixed pixel sizes** — EMPS has 218 distinct canvas geometries, so a hard-coded 512→256→128→64 ladder will not survive contact with the dataset.

At each scale:

1. extract overlapping patches;
2. store the patch bank;
3. record patch size, stride, scale and source ID;
4. verify that patches reconstruct the source image correctly.

Apply the Layer-1 exclusion mask **before** building the bank, and assert that no stored patch intersects an exclusion pixel.

Start with one source image and one simple configuration before expanding.

## Investigate

- How many patches exist at each scale?
- What does a fixed patch size represent at fine vs coarse resolution?
- Which scales capture texture and which capture larger structure?
- How does patch size affect memory and runtime?
- How much repetition already exists inside one image? (This is the assumption the whole method rests on. Measure it.)
- Does the chosen pyramid preserve the structures you care about?
- How much does furniture exclusion reduce the bank on the worst-affected images?

## Deliver

- `L2_Multiscale_Patches.ipynb`
- reusable `build_pyramid(...)`
- reusable `extract_patches(...)`
- patch-count table per scale
- patch reconstruction test
- assertion that no stored patch touches an exclusion pixel
- fine/medium/coarse patch visualizations
- frozen first pyramid configuration

## Layer gate

For one valid source, the code produces reproducible patch datasets at several scales and can reconstruct the original image from its extracted patches.

---

# Layer 3 — Closed-form patch denoiser

## Task

Implement the central training-free denoiser from the Qiu method.

For a noisy query patch:

```text
noisy query patch
       │
       ▼
compare against clean source patches
       │
       ▼
compute similarity weights
       │
       ▼
weighted average of source patches
       │
       ▼
denoised patch
```

Begin with a tiny toy example where the result can be checked **by hand** — a handful of 3×3 patches whose weighted average you can compute on paper.

Keep that toy case forever as a regression test. Every later optimisation in Layer 6 must still reproduce it exactly.

Then run the same denoiser on real patches. Do not optimize for speed yet.

## Investigate

- Why is the method training-free? Write the answer in your own words.
- Which source patches receive high weight?
- What changes when the noise level is high vs low?
- What happens if only the nearest patch is used? (This is the copying failure in its purest form — look at the output.)
- What happens if many patches receive similar weight?
- **Can the denoiser collapse toward copied source patches?** Find the parameter value where it does, and record it.
- Which parameter most strongly controls selectivity?

## Deliver

- `L3_Closed_Form_Denoiser.ipynb`
- reusable `denoise_patch(...)`
- one tiny hand-checkable example, retained as a regression test
- one brute-force verification
- real patch examples: noisy / denoised / closest source patches
- one selectivity/noise-level sweep, including the collapse-to-copy point
- short explanation of why no neural-network training is required

## Layer gate

The denoiser matches the expected toy/brute-force calculation and behaves sensibly on real patches.

---

# Layer 4 — Single-scale diffusion sampler

## Task

Use the closed-form patch denoiser inside an iterative diffusion-style sampler.

```text
random noise image
       │
       ▼
extract noisy patches
       │
       ▼
closed-form patch denoising
       │
       ▼
reconstruct image
       │
       ▼
next lower-noise step
       │
       ▼
repeat
       │
       ▼
synthetic image
```

Work at one image scale first. Use a small crop or reduced resolution for CPU execution.

Make seed, number of steps and noise schedule explicit and logged.

**Verify on the NIST fixture first.** One clean reference with a known mask and controlled geometry tells you immediately whether the sampler is producing structure or mush. Move to EMPS once it works.

## Investigate

- Does the image become progressively more source-like?
- How many sampling steps are actually needed?
- What happens with too few steps?
- What changes across random seeds?
- How deterministic is the sampler when stochasticity is reduced?
- Does single-scale generation preserve only local texture while losing global organization?

## Deliver

- `L4_Single_Scale_Sampler.ipynb`
- reusable `sample_single_scale(...)`
- at least 5 generated samples from different seeds
- intermediate images from several diffusion steps
- step-count comparison
- runtime per generated image, with a profile showing where the time goes
- failure gallery for the single-scale method

## Layer gate

A real source image can produce multiple reproducible synthetic images from random noise using the training-free denoiser.

---

# Layer 5 — Coarse-to-fine training-free SID

## Task

Implement the full multiscale idea.

```text
random noise
    │
    ▼
coarse-scale generation
    │
    ▼
upsample
    │
    ▼
medium-scale refinement
    │
    ▼
upsample
    │
    ▼
fine-scale refinement
    │
    ▼
synthetic image
```

Keep the single-scale implementation unchanged as a baseline. Use the same source image and seed when comparing.

## The label-validity question — new, and it decides Layer 7

If generation moves geometry, **the source mask no longer describes the generated image.** Inheriting it anyway silently fabricates ground truth.

Measure this rather than assuming it. On an analytic control where you know the true correspondence, the geometry displacement produced by this method family reached roughly **26 pixels** — more than enough to destroy label validity for small structures.

So define a label-validity check, run it, and report a number:

```text
LV = does the source mask still describe the generated geometry?
```

**Then note the regime difference, because it changes the answer completely:**

| | instances per image |
|---|---:|
| EMPS | **2–9** — sparse, isolated particles |
| RODARE | **~800** — dense, touching carbides |

A 26-pixel displacement barely matters for five isolated particles. It is catastrophic for eight hundred touching ones. **Report the two separately and never pool them.**

## Investigate

- What large-scale structures appear only after coarse-to-fine generation?
- Which scale contributes most to global morphology? Which to fine texture?
- Does multiscale generation reduce local repetition?
- Does it introduce new artifacts?
- How sensitive is the result to the number of scales?
- How large is the geometry displacement, and does label validity survive it?

## Deliver

- `L5_Coarse_to_Fine_SID.ipynb`
- reusable multiscale sampler
- single-scale vs multiscale comparison at fixed seed and source
- 3-scale / 4-scale ablation
- at least 5 multiscale synthetic outputs
- **label-validity measurement with a stated verdict**
- runtime and memory summary
- one figure showing coarse → medium → fine evolution

## Layer gate

The multiscale sampler produces traceable synthetic images, demonstrates a clear difference from the single-scale baseline, and states — with a measurement — whether the source mask may be transferred.

---

# Layer 6 — Efficiency and reproducibility

## Task

Make the method fast enough to experiment with repeatedly.

Do not change the scientific question yet.

Investigate accelerations in this order:

1. fewer diffusion steps;
2. reduced patch-bank size;
3. chunked patch comparison;
4. nearest-neighbour / approximate candidate filtering;
5. only later, GPU-specific acceleration if needed.

Keep one exact small configuration as the correctness reference, and keep the Layer-3 hand-checkable toy case passing throughout.

## The distinction that matters

Some accelerations change **only runtime**. Others change **the generated distribution**. Approximate nearest-neighbour search is the obvious suspect — it silently changes which source patches are reachable.

For every acceleration, classify it as one or the other **by testing**, not by reasoning about it. Generate the same seed under exact and fast configurations and compare the outputs directly.

## Investigate

- How much quality is lost when the number of steps is reduced?
- How much time is spent comparing patches? (Profile before optimizing.)
- How does patch-bank size affect runtime and result quality?
- Can approximate candidate search preserve results?
- Which acceleration changes only runtime, and which changes the distribution?
- What configuration gives the best quality/runtime trade-off?

## Deliver

- `L6_Efficiency.ipynb`
- exact-small reference configuration
- reduced/fast configuration
- runtime table
- memory table
- step-count sweep
- one approximation-vs-exact comparison at matched seed
- deterministic seed/config/output logging

## Layer gate

There is one reproducible reference implementation and one faster configuration whose quality loss is **measured rather than assumed**.

---

# Layer 7 — Evaluation and trust boundary

## Task

Evaluate whether the generated images are useful and where they should not be trusted.

Use the same evaluator for every generator configuration.

Compare:

```text
real source image
single-scale synthetic
multiscale synthetic
fast synthetic configuration
```

Do not judge success only by appearance.

## The copying diagnostic

Run it in two stages, because one stage alone either misses copies or drowns in false alarms:

1. **Candidates** — a fast perceptual-hash or descriptor pass over generated vs source content.
2. **Confirmation** — correlation or keypoint matching on the candidates only.

Report **cross-group** edges specifically. A generated image resembling its own source is expected; a generated image resembling a *different* group's source, or anything in the hidden test set, is a leak.

This is a high-specificity gate. It does not prove that all semantic similarity is absent — say so.

## The downstream experiment — where most projects like this overstate

Three arms:

```text
(a) real only
(b) real + classical augmentation
(c) real + your synthetic
```

Two rules decide whether this result means anything:

- **Equal optimizer update budget across all three arms.** If the synthetic arm gets more steps, the comparison is void. This is the single most common way augmentation results are overstated.
- **No generated sample's lineage may touch a held-out group.** Assert it in a validator, and test the validator against a deliberately contaminated record.

Specify arm (b) fully — every transform and every probability. An under-specified classical baseline is the second most common way these results are overstated, because the weak baseline does the work.

**Pre-register a benefit margin and a harm tolerance before running.** A benefit margin without a harm tolerance means synthetic data can only ever look good.

Evaluate on held-out groups never touched by any generator. Bootstrap over **groups**, never patches, and state the number of groups.

## Investigate

- Do synthetic images match real intensity statistics?
- Do they match gradient/edge statistics?
- Do they reproduce spatial/frequency structure?
- How diverse are different seeds?
- Are generated regions directly copied from the source? How much, and across groups or only within?
- Does good visual quality hide structural errors?
- Does the best-looking configuration also perform best quantitatively?
- Which structures or conditions repeatedly fail?
- Does synthetic augmentation improve a model on untouched real data — at equal update budget?

## The SEM claim

Everything above is **electron microscopy**. To say *SEM*, repeat the core comparison on **RODARE** — 13 fields, databar cropped, detector siblings paired, pixels only.

Expect it to be worse. RODARE is dense touching microstructure with ~800 instances per image, against EMPS's sparse 2–9. **If it is worse, that is a finding about dense microstructure, not a failure of the method** — and it is more interesting than a pooled average that hides it.

## Deliver

- `L7_Evaluation_and_Trust.ipynb`
- common evaluator used for all methods/configurations
- real-vs-synthetic histogram comparison
- gradient/edge comparison
- PSD or autocorrelation comparison
- **two-stage copying diagnostic, with cross-group edges reported separately**
- seed-diversity analysis
- failure gallery
- final configuration table
- **downstream three-arm result at equal update budget, with the pre-registered margin and harm tolerance**
- EMPS and RODARE reported separately, never pooled
- explicit trust-boundary statement

## Layer gate

The team can state, with evidence:

1. what the generator reproduces well;
2. what it fails to reproduce;
3. how much it copies, and whether any copying crosses groups;
4. which configuration is preferred and why;
5. whether the source mask may be transferred, and in which density regime;
6. whether synthetic data helps on a real held-out downstream task at equal update budget.

---

# Final project comparison

```text
L1  correct sources, grouped, furniture-excluded, modality-honest
        ↓
L2  multiscale patch datasets
        ↓
L3  verified closed-form denoiser
        ↓
L4  working single-scale generator
        ↓
L5  working multiscale SID + measured label validity
        ↓
L6  efficient reproducible implementation
        ↓
L7  measured utility + copying + trust boundary
        ↓
    RODARE arm → the only place the word SEM is earned
```

The project is successful even if the final result shows important limitations.

A valid conclusion may be:

> The training-free SID method reproduces local texture well but fails to preserve some larger-scale morphology.

or:

> Multiscale generation improves structural consistency but increases source-patch copying.

or:

> Synthetic augmentation improves downstream performance only in the low-data regime.

or:

> Label transfer is invalid beyond a measured displacement, so the method can generate images but not ground truth.

The result must come from the experiments, not from what we expected before building the pipeline.

---

# Where your project meets the other group

The AMAT-3 student group is testing when recovered detail stops being trustworthy. Two things must match between you:

1. **The statistical unit is the source group.** Yours is the DOI group or the field; theirs is the NIST set or the field. Never a patch.
2. **The observer and metric definitions.** Any segmentation you evaluate goes through the same frozen rule they build in their Layer 3 — including its polarity handling. Ask them for it before you write your own; they will have found at least one defect in it, and you want their repaired version, not a fresh copy of the bug.

Their invention diagnostics and your copying diagnostics are the same question asked from two sides: *is this detail supported by the evidence, or did the method supply it?* Compare notes on what your diagnostics could **not** detect. Those blind spots are the most useful thing either group will produce.

---

# Rules for all layers

1. Build the real project component first.
2. Use AI to accelerate implementation, not replace understanding.
3. Do not continue upward while the current layer has an unexplained correctness failure.
4. Keep the same source manifest and evaluator when comparing configurations.
5. A negative result is a result.
6. Preserve failure cases instead of tuning them away.
7. Another student should be able to rerun and explain the layer before it is complete.
8. **Group by DOI or field, never by filename or tile.** State the number of groups in every interval you report.
9. Figures and CSV results must be generated from code.
10. Every layer must leave behind an artifact that the next layer uses.
11. **EMPS is electron microscopy, never SEM.** Only the RODARE arm earns that word, and only in pixels.
12. **Hidden test data never touches a generator, a patch bank, a threshold, or a selection decision.** Assert it, and test the assertion with a contaminated record.
13. **A generated image is not labelled data** until label validity has been measured for that configuration.
14. **Equal optimizer update budget across every downstream arm.** No exceptions.
15. **Pre-register margins, tolerances and thresholds before running.** If one changes after a result is seen, say so in writing and rerun.

---

# Suggested pacing

| Week | Layer | Compute |
|---|---|---|
| 1 | L1 — data spine, grouping, furniture, modality | CPU |
| 2 | L2 — pyramid and patch banks | CPU |
| 3 | L3 — closed-form denoiser + hand check | CPU |
| 4 | L4 — single-scale sampler (verify on NIST) | CPU |
| 5 | L5 — coarse-to-fine + label validity | CPU / GPU |
| 6 | L6 — efficiency and reproducibility | CPU / GPU |
| 7 | L7 — evaluation, copying, downstream, RODARE | GPU |
| 8 | Write-up, reproduction, teach-back | — |

This method is dominated by patch search, not by matrix multiplication. Layers 1 to 4 run on CPU. Do not wait for a GPU to start.
