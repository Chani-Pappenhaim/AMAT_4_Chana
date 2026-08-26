import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# Approximate candidate filtering for denoise_patch

`denoise_patch` compares a query against every patch in the bank - Day 11
and Day 16 profiling showed this dominates runtime. `denoise_patch_approx`
prefilters by patch mean (cheap: one number per patch) before running the
expensive full-patch distance only on survivors. This measures how much
filtering actually helps on a real bank, and confirms it still matches the
exact result when given a generous tolerance.
""")

md("## 1. Build a real patch bank")

code("""import sys, os, time
sys.path.append(os.path.join("..", "..", "src"))
import numpy as np
from data.loader import DEV_SOURCE_IDS, load_generator_source
from data.patches import extract_patches
from models.denoiser import denoise_patch, denoise_patch_approx

img, sid = load_generator_source(DEV_SOURCE_IDS[0])
bank = extract_patches(img, patch_size=8, stride=4).patches.astype(np.float64)
print(f"source: {sid}, bank size: {len(bank)} patches")
""")

md("""## 2. Correctness: approx must match exact with a generous tolerance

Same query, same sigma, same bank - only the candidate-filtering tolerance
differs. A generous tolerance should keep every patch as a candidate and
reproduce the exact result exactly.""")

code("""query = bank[len(bank) // 2] + np.random.default_rng(0).normal(0, 10, size=bank[0].shape)
sigma = 30.0

exact_result, exact_weights = denoise_patch(query, bank, sigma)
approx_result, approx_weights, n_candidates = denoise_patch_approx(query, bank, sigma, mean_tolerance=1e6)

print(f"n_candidates with generous tolerance: {n_candidates} / {len(bank)}")
print(f"exact and approx match: {np.allclose(exact_result, approx_result)}")
""")

md("## 3. How much does a realistic tolerance actually filter, and how much time does it save?")

code("""for tolerance in [1e6, 100, 50, 20, 10]:
    t0 = time.perf_counter()
    for _ in range(20):
        result, weights, n_candidates = denoise_patch_approx(query, bank, sigma, mean_tolerance=tolerance)
    dt = (time.perf_counter() - t0) / 20

    exact_result, _ = denoise_patch(query, bank, sigma)
    max_pixel_diff = np.abs(result - exact_result).max()

    print(f"tolerance={tolerance:7.1f}: candidates={n_candidates:4d}/{len(bank)}, "
          f"time={dt*1000:.3f}ms, max_pixel_diff_vs_exact={max_pixel_diff:.4f}")
""")

md("""## Findings

- With a generous tolerance, `denoise_patch_approx` reproduces
  `denoise_patch` exactly (candidate filtering removes nothing, so it is
  the same computation) - the approximation only kicks in once the
  tolerance is tight enough to actually drop patches.
- As tolerance tightens, candidate count drops and so does the per-call
  time - but `max_pixel_diff_vs_exact` must be read alongside that speedup,
  not assumed to stay near zero. This is the runtime-vs-distribution
  distinction the project's rules require checking, at the single-patch
  level; Day 18 checks it at the full-image level (same seed, exact vs
  fast sampler configuration).
""")

nb["cells"] = cells

out_path = "approximate_denoiser.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} with {len(cells)} cells")
