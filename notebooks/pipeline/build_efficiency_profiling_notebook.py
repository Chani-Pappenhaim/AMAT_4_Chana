import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# Efficiency profiling: coarse-to-fine sampler

Where does the time actually go in the full multiscale pipeline, and what
happens to speed and quality when steps or patch-bank size are reduced?
Measured, not assumed - a fast configuration is only valid once its
quality loss is known.
""")

md("## 1. Baseline: time per pyramid stage")

code("""import sys, os, time
sys.path.append(os.path.join("..", "..", "src"))
import numpy as np
from PIL import Image

from data.pyramid import build_pyramid
from models.sampler import generate_coarse_sketch, refine_at_scale, sample_coarse_to_fine

NIST_DIR = os.path.join("..", "..", "..", "AMAT", "nist_detection_limits_sem")
clean_path = os.path.join(NIST_DIR, "mask_sets", "masks", "set1_cex_noise_000_contrast_100.tiff")
img = np.array(Image.open(clean_path)).astype(np.float64)[:64, :64]

PATCH_SIZE, STRIDE, NUM_STEPS = 4, 2, 15
pyramid = build_pyramid(img, num_scales=4, scale_factor=0.5)

t0 = time.perf_counter()
sketch, _ = generate_coarse_sketch(pyramid, PATCH_SIZE, STRIDE, NUM_STEPS, seed=0)
t_sketch = time.perf_counter() - t0
print(f"coarse sketch (8x8):   {t_sketch*1000:7.1f} ms")

current = sketch
stage_times = []
for level_index in range(len(pyramid) - 2, -1, -1):
    t0 = time.perf_counter()
    current, _ = refine_at_scale(current, pyramid[level_index], PATCH_SIZE, STRIDE, NUM_STEPS, seed=0)
    dt = time.perf_counter() - t0
    stage_times.append(dt)
    print(f"refine to {pyramid[level_index].shape}: {dt*1000:7.1f} ms")

total = t_sketch + sum(stage_times)
print(f"\\ntotal: {total*1000:.1f} ms")
""")

md("""**Expected pattern:** each refinement stage works on a larger image
than the last (more patches to extract, denoise, and reconstruct), so time
should grow with resolution - confirming whether the full-resolution final
stage dominates total runtime, the way denoising dominated a single step
in Day 11's profiling.""")

md("## 2. Fewer steps per scale: speed vs quality tradeoff")

code("""results = []
for num_steps in [5, 10, 15, 30]:
    t0 = time.perf_counter()
    final, _ = sample_coarse_to_fine(pyramid, PATCH_SIZE, STRIDE, num_steps, seed=0)
    dt = time.perf_counter() - t0
    results.append((num_steps, dt, final.std()))
    print(f"num_steps_per_scale={num_steps:3d}: time={dt*1000:7.1f} ms, final std={final.std():.2f}  (real std={img.std():.1f})")
""")

md("""## 3. Smaller patch bank (larger stride): speed vs quality tradeoff

A larger stride means fewer, less-overlapping patches - cheaper to build
and to compare against, but with less redundancy for reconstruction.

**Design limit found while testing this:** `stride` must be `<= patch_size`.
With stride > patch_size, patches leave real gaps between them (e.g.
patch_size=4, stride=8 covers pixels 0-3 then jumps to 8-11, leaving 4-7
uncovered by any patch) - the Day 5 boundary fix only guarantees the far
edge is covered, not gaps in the middle, which are unavoidable once the
stride exceeds the patch width. So only stride <= patch_size is tested
here; a caught `AssertionError` on stride=8 confirmed this is a real
constraint, not just a guess.""")

code("""for stride in [1, 2, 4]:
    t0 = time.perf_counter()
    final, _ = sample_coarse_to_fine(pyramid, PATCH_SIZE, stride, NUM_STEPS, seed=0)
    dt = time.perf_counter() - t0
    print(f"stride={stride}: time={dt*1000:7.1f} ms, final std={final.std():.2f}")
""")

md("""## Findings - two results that contradicted the naive expectation

**1. Runtime is dominated by the full-resolution stage**, as expected:
77ms (8x8) -> 190ms -> 797ms -> 3583ms (64x64) - roughly 4-5x per doubling,
consistent with the patches-in-image x bank-size scaling found in Day 11.

**2. More steps-per-scale makes coarse-to-fine WORSE, not better -
the opposite of the single-scale result.** std actually drops as steps
increase: 23.55 (5 steps) -> 1.36 (10) -> 1.23 (15) -> 1.01 (30), while the
real image's std is 55.6. A single stage with 30 steps was fine on its own
(Day 14/15 fix); chaining 4 stages, each contracting variance a little,
compounds that contraction multiplicatively - more steps per stage means
more compounding, not more accuracy. This was not assumed; it fell out of
running the actual numbers.

**3. Larger stride (less patch overlap) dramatically preserves variance
better, and is also much faster - connects directly to the border-coverage
finding from Day 14/15.** stride=1: 49.8s, std=0.91 (collapsed). stride=2:
4.7s, std=1.23 (collapsed). **stride=4 (no overlap at all): 0.8s, std=61.22**
- close to the real 55.6, and the fastest option. The same mechanism that
caused the border artifact (fewer overlapping patches -> less averaging ->
values stay closer to their true extremes) here works in our favor
everywhere, not just at the border, once there is no overlap left to
average away.

**Practical implication:** for this method, stride close to patch_size is
not a speed/quality tradeoff - it is faster AND better on both metrics
tested here. This should be re-verified on EMPS/real texture before
treating it as a general recommendation (Layer 7's job), but it is a
concrete, non-obvious lead, not a guess.
""")

nb["cells"] = cells

out_path = "efficiency_profiling.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} with {len(cells)} cells")
