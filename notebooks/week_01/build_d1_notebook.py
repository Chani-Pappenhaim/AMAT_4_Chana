import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# ---------- 1. Goal ----------
md("""# AMAT-4 — Day 1: SEM Image Formation
## 1. Goal

Build the minimal, focused understanding of how an SEM image is formed — just enough
to reason about what a synthetic-SEM generator needs to reproduce — using the real
**EMPS** dataset as ground truth.

Scope is deliberately narrow (per the project brief): the electron-beam -> sample ->
detector -> pixel signal chain, SE vs. BSE at a conceptual level, and why edges look
bright. Electron-optics derivations, scattering equations, and quantum mechanics are
explicitly out of scope for this notebook.""")

# ---------- 2. Very short SEM image-formation explanation ----------
md("""## 2. Very short SEM image-formation explanation

- A focused electron beam scans the sample in a raster pattern (row by row).
- At each beam position, electrons hit the sample and knock out **secondary electrons (SE)**
  and reflect **backscattered electrons (BSE)**.
- A detector counts how many of these electrons arrive for that beam position.
- That count becomes the brightness of one pixel. Repeat for every raster position -> full image.

**Signal chain:** `electron beam -> sample -> emitted electrons -> detector -> pixel`

Consequence: SEM images are fundamentally a *count* of particles per pixel, not a
continuous light measurement like a camera. This is why Day 2 focuses on Poisson
(counting) statistics for the noise model.""")

# ---------- 3. Dataset examples ----------
md("""## 3. Dataset examples

Loading a handful of real SEM images from **EMPS** (Electron Microscopy Particle
Segmentation, Yildirim & Cole, 2021) — 465 real electron-microscopy images with
particle segmentation masks, MIT license. This is our ground-truth reference for
everything that follows.""")

code("""import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = os.path.join("emps", "images")
files = sorted(os.listdir(DATA_DIR))
print(f"Total EMPS images: {len(files)}")

rng = np.random.default_rng(0)
sample_files = rng.choice(files, size=6, replace=False)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for ax, fname in zip(axes.ravel(), sample_files):
    img = Image.open(os.path.join(DATA_DIR, fname)).convert("L")
    ax.imshow(img, cmap="gray")
    ax.set_title(fname, fontsize=9)
    ax.axis("off")
plt.suptitle("Sample real SEM images from EMPS")
plt.tight_layout()
plt.show()
""")

# ---------- 4. Image metadata / statistics ----------
md("""## 4. Image metadata / statistics

Basic stats across a sample of images: resolution, dtype, intensity range, mean, std.
This establishes the numeric baseline that any synthetic generator must land inside.""")

code("""stats_rows = []
sample_files2 = rng.choice(files, size=40, replace=False)
for fname in sample_files2:
    img = np.array(Image.open(os.path.join(DATA_DIR, fname)).convert("L"))
    stats_rows.append({
        "filename": fname,
        "height": img.shape[0],
        "width": img.shape[1],
        "min": int(img.min()),
        "max": int(img.max()),
        "mean": float(img.mean()),
        "std": float(img.std()),
    })

import pandas as pd
df = pd.DataFrame(stats_rows)
print(df.describe())
df.head()
""")

# ---------- 5. Histograms ----------
md("""## 5. Histograms

Intensity histograms for a few images — real SEM intensity distributions are not
uniform; they cluster around the substrate background level with a tail/peak for
brighter particle edges.""")

code("""fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, fname in zip(axes, sample_files[:3]):
    img = np.array(Image.open(os.path.join(DATA_DIR, fname)).convert("L"))
    ax.hist(img.ravel(), bins=64, color="steelblue")
    ax.set_title(f"Histogram: {fname}", fontsize=9)
    ax.set_xlabel("Intensity (0-255)")
    ax.set_ylabel("Pixel count")
plt.tight_layout()
plt.show()
""")

# ---------- 6. Zoomed regions ----------
md("""## 6. Zoomed regions

Cropping in on a small patch (e.g. a particle edge) to see pixel-level structure —
grain, edge brightness, local noise texture — that's invisible at full-image scale.""")

code("""fname = sample_files[0]
img = np.array(Image.open(os.path.join(DATA_DIR, fname)).convert("L"))
h, w = img.shape
cy, cx = h // 2, w // 2
crop_size = 80
crop = img[cy - crop_size:cy + crop_size, cx - crop_size:cx + crop_size]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img, cmap="gray")
axes[0].add_patch(plt.Rectangle((cx - crop_size, cy - crop_size), 2*crop_size, 2*crop_size,
                                  edgecolor="red", facecolor="none", linewidth=2))
axes[0].set_title(f"Full image: {fname}")
axes[0].axis("off")
axes[1].imshow(crop, cmap="gray")
axes[1].set_title("Zoomed crop (center region)")
axes[1].axis("off")
plt.tight_layout()
plt.show()
""")

# ---------- 7. SEM vs ordinary photograph ----------
md("""## 7. SEM vs. ordinary photograph

| Aspect | SEM image | Ordinary photograph |
|---|---|---|
| Signal carrier | Electrons (counted) | Photons (light intensity) |
| Color | Grayscale (single detector channel) | RGB (3 color channels) |
| Formation | Raster-scanned point by point | Captured all at once (sensor array) |
| Noise | Poisson (counting) — depends on dose | Mostly photon shot noise + read noise, higher SNR typically |
| Depth of field | Very large (near-parallel electron trajectories) | Limited by lens aperture |
| Illumination model | No real "lighting/shadow" — contrast comes from electron yield vs. surface angle | Illumination + surface reflectance (BRDF) |
| Scale | Nanometers to micrometers | Macroscopic (mm-m) |

The key generative implication: an SEM image generator has to reproduce a
**counting-noise, single-channel, raster-formed signal**, not a lit, multi-channel photo.""")

# ---------- 8. Questions / uncertainties ----------
md("""## 8. Questions / uncertainties

- What is the actual acquisition dose/scan-rate for the EMPS images? (Not stated in the
  dataset metadata — we cannot directly verify the Poisson noise level against known
  imaging parameters, only infer it from pixel statistics.)
- EMPS images come from many different source papers/instruments (see `doi` column in
  metadata.csv) — should we expect consistent noise statistics across the whole dataset,
  or per-source variation that needs to be modeled separately?
- Segmentation masks give us object shape ground truth, but not a "clean" noise-free
  reference image — how do we separate "real texture" from "detector noise" without one?
  (This maps directly to the ground-truth problem discussed earlier in the project.)
""")

nb["cells"] = cells

out_path = "AMAT4_D1_SEM_image_formation.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} with {len(cells)} cells")
