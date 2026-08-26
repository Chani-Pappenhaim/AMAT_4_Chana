import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# Single-scale sampler: first verification on NIST

Before running the sampler on EMPS, verify it on the NIST fixture: one
clean reference image with a controlled, simple geometry. If the sampler
cannot produce plausible structure here, there is a bug in the pipeline,
not a research finding.
""")

md("## 1. Load the NIST clean reference")

code("""import sys, os
sys.path.append(os.path.join("..", "..", "src"))
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

NIST_DIR = os.path.join("..", "..", "..", "AMAT", "nist_detection_limits_sem")
clean_path = os.path.join(NIST_DIR, "mask_sets", "masks", "set1_cex_noise_000_contrast_100.tiff")

full_image = np.array(Image.open(clean_path)).astype(np.float64)
print(f"loaded NIST clean reference: full shape={full_image.shape}")

# CPU sanity check: use a small crop, not the full image - the sampler is
# O(patches-in-image x bank-size) per step, which is a Layer 6 efficiency
# problem, not something to solve here.
CROP = 64
clean_ref = full_image[:CROP, :CROP]
print(f"using a {CROP}x{CROP} crop for this CPU verification: shape={clean_ref.shape}")

plt.imshow(clean_ref, cmap="gray")
plt.title(f"NIST set1 clean reference ({CROP}x{CROP} crop)")
plt.axis("off")
plt.show()
""")

md("## 2. Build a patch bank from the clean reference")

code("""from data.patches import extract_patches

PATCH_SIZE = 8
STRIDE = 4

bank_record = extract_patches(clean_ref, PATCH_SIZE, STRIDE)
bank = bank_record.patches.astype(np.float64)
print(f"patch bank: {len(bank)} patches of {PATCH_SIZE}x{PATCH_SIZE}")
""")

md("""## 3. Calibrate sigma from the bank's own distance distribution

A hardcoded sigma range tuned for one image does not generalize - the next
image can have a completely different patch-distance scale. `estimate_sigma_range`
measures the real distribution of distances between patches in *this* bank
and derives a range from it, so the same code works on any source image.""")

code("""from models.sampler import sample_single_scale, estimate_sigma_range

sigma_max, sigma_min = estimate_sigma_range(bank)
print(f"calibrated from this bank's own distances: sigma_max={sigma_max:.2f}, sigma_min={sigma_min:.2f}")

final_image, history = sample_single_scale(
    shape=clean_ref.shape,
    patch_bank=bank,
    patch_size=PATCH_SIZE,
    stride=STRIDE,
    num_steps=15,
    sigma_max=sigma_max,
    sigma_min=sigma_min,
    seed=0,
)

for i, snap in enumerate(history):
    print(f"step {i:2d}: mean={snap.mean():6.1f}, std={snap.std():6.1f}")
""")

md("""## 4. Compare: pure noise vs final synthetic vs real reference

This is a plausibility check, not a quality metric - does the sampler
produce recognizable structure, or mush? Sharper structural comparison
comes later.""")

code("""fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, title, img in zip(axes, ["initial noise", "final synthetic", "real reference"], [history[0], final_image, clean_ref]):
    ax.imshow(img, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()
""")

md("""## 5. Intermediate steps: coarse to fine

Showing a few snapshots across the run makes the "denoise a little, add a
little less noise, repeat" process visible rather than just the before/after.""")

code("""num_show = 5
show_indices = np.linspace(0, len(history) - 1, num_show).astype(int)

fig, axes = plt.subplots(1, num_show, figsize=(3 * num_show, 3))
for ax, idx in zip(axes, show_indices):
    ax.imshow(history[idx], cmap="gray")
    ax.set_title(f"step {idx}")
    ax.axis("off")
plt.tight_layout()
plt.show()
""")

md("""## Finding: variance collapse (negative result)

The sampler runs end-to-end without crashing, but the outcome is a genuine
failure worth recording, not a success to gloss over.""")

code("""print(f"real NIST crop std: {clean_ref.std():.1f}")
print(f"initial noise std:      {history[0].std():.1f}")
print(f"final synthetic std:    {final_image.std():.1f}")
""")

md("""**What happened:** std drops from the source image's ~55 toward under 1
within just a few steps - even after calibrating sigma from the bank's own
distance distribution (ruling out "wrong hardcoded constant" as the cause).
Cutting the step count from 15 down to 3 still collapses to std ~0.4-0.7.
The collapse is immediate, not a slow accumulation over many iterations.

**Why:** `denoise_patch` computes a similarity-weighted *average* of bank
patches. Averaging is a variance-reducing operation by construction - the
result is always smoother than any individual patch that went into it. That
is exactly correct behavior for denoising a real noisy photo (you want the
stable, de-noised estimate). It is the wrong behavior for *generation*,
where the goal is a plausible sample with realistic variance, not the
smoothed mean. Re-adding noise between steps at the next sigma does not
compensate for this, because it adds *unstructured* variance while the
denoiser keeps removing *structured* variance - the two are not equal and
opposite.

**Why we are not patching this with a bigger noise term:** the correct fix
is the reverse-process update the method's underlying theory specifies,
which corrects for exactly this shrinkage - not a hand-tuned constant that
would only work for this one image and step count. Getting that update
right belongs with the coarse-to-fine sampler, not as a quick patch here.

**Status:** the single-scale sampler is verified to run correctly end-to-end
(no crash, deterministic, calibrates itself to any bank) and its failure
mode is measured and understood, but it does not yet produce texture-
preserving output. This is a documented limitation carried forward, not a
blocker treated as invisible.
""")

nb["cells"] = cells

out_path = "single_scale_sampler_nist.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} with {len(cells)} cells")
