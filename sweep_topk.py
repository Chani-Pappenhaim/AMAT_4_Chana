"""Empirical topk sweep, run on OUR OWN images through OUR OWN pipeline -
not a borrowed tuned value from any other project.

topk restricts the softmax-weighted patch average to the k nearest bank
patches. Too small (topk=1) and the denoiser stops averaging and starts
copying a single real patch verbatim; too large (topk=full bank) and
far-away, barely-relevant patches keep contributing enough combined weight
to blur the result. The right value depends on how our own patch banks are
distributed - it is not a universal constant, so it has to be measured on
our own data, not copied from another project's tuned number.

Two gates, mirroring the "don't just optimise one number" logic used
elsewhere in this project:
  - copy_fraction: fraction of the synthetic image's patches that are
    near-exact copies of a single source patch (nearest-neighbor distance
    below a tight threshold). High copy_fraction = memorization, not
    synthesis.
  - diversity: mean pairwise pixel-RMSE between multiple independent seeds
    of the same source. Near-zero diversity = the sampler collapsed to one
    deterministic output regardless of its random seed.
A topk is "viable" only if it keeps copy_fraction low AND diversity above a
floor; among viable values we prefer the one closest to the full-bank
softmax's own output (least distortion of the exact Bayes-optimal weighting
while still cutting off the far-away blur contributors).

Usage:
  python sweep_topk.py "min data set"
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from data.furniture import build_exclusion_mask  # noqa: E402
from data.patches import extract_patches  # noqa: E402
from data.pyramid import build_pyramid  # noqa: E402
from models.generate import load_grayscale  # noqa: E402
from models.sampler import sample_coarse_to_fine  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
COPY_DISTANCE_THRESHOLD = 3.0   # per-patch Euclidean distance (0..255 scale) below which a patch counts as "copied"
COPY_FRACTION_LIMIT = 0.05      # same gate the borrowed-methodology reference used
DIVERSITY_FLOOR = 2.0           # mean pairwise pixel-RMSE below this = collapsed to one output regardless of seed


def prepare_source(path, downscale, num_scales):
    image = load_grayscale(path)
    mask = build_exclusion_mask(image)
    processed = image.copy()
    if mask.any() and not mask.all():
        processed[mask] = np.median(image[~mask])
    if downscale != 1.0:
        h, w = processed.shape
        processed = np.array(
            Image.fromarray(processed).resize((max(1, int(w * downscale)), max(1, int(h * downscale))), Image.BILINEAR)
        )
    return build_pyramid(processed, num_scales=num_scales, scale_factor=0.5)


def copy_fraction(synthetic, source_bank, patch_size, stride, threshold=COPY_DISTANCE_THRESHOLD):
    """Fraction of the synthetic image's own patches whose nearest neighbor
    in the REAL source bank (same resolution) is closer than `threshold` -
    i.e. patches that are essentially a verbatim copy of one real patch,
    not a blend of several. Distances computed as one batched matrix op
    (not a Python loop per patch) - this is a pure speed choice, identical
    result either way.
    """
    synth_patches = extract_patches(synthetic, patch_size, stride).patches.reshape(-1, patch_size * patch_size)
    bank_flat = source_bank.reshape(len(source_bank), -1)
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b, batched via a single matmul
    sq_dist = (
        np.sum(synth_patches ** 2, axis=1, keepdims=True)
        + np.sum(bank_flat ** 2, axis=1)[None, :]
        - 2 * synth_patches @ bank_flat.T
    )
    nearest = np.sqrt(np.maximum(sq_dist.min(axis=1), 0))
    return float((nearest < threshold).mean())


def diversity(images):
    """Mean pairwise pixel-RMSE across independent-seed runs of the same
    source - near 0 means every seed produced (near-)the same image, i.e.
    the sampler is not actually exploring different syntheses.
    """
    pairs = list(itertools.combinations(images, 2))
    rmses = [np.sqrt(np.mean((a - b) ** 2)) for a, b in pairs]
    return float(np.mean(rmses))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir")
    parser.add_argument("--downscale", type=float, default=0.2)
    parser.add_argument("--num-scales", type=int, default=3)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--num-steps-per-scale", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--topk-grid", type=int, nargs="+", default=[2, 4, 8, 16, 32, 99999])  # 99999 stands in for "full bank" (skipped-threshold sentinel below)
    parser.add_argument("--max-images", type=int, default=2, help="cap how many source images this sweep runs on (CPU cost)")
    args = parser.parse_args()

    input_dir = args.input_dir if os.path.isabs(args.input_dir) else os.path.join(ROOT, args.input_dir)
    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".tif", ".tiff", ".jpg", ".jpeg")))[: args.max_images]
    print(f"sweeping topk={args.topk_grid} on {len(files)} source(s): {files}\n", flush=True)

    rows = []
    for fname in files:
        path = os.path.join(input_dir, fname)
        pyramid = prepare_source(path, args.downscale, args.num_scales)
        finest_bank = extract_patches(pyramid[0], args.patch_size, args.stride).patches.astype(np.float64)
        print(f"{fname}: shape={pyramid[0].shape}, finest bank size={len(finest_bank)}")

        for topk in args.topk_grid:
            # a topk at/above the finest level's own bank size is really
            # asking for the full-bank baseline (no restriction at all) -
            # run it as topk=None instead of silently restricting to
            # "min(topk, bank size)", which would just be a confusing
            # relabeling of the full-bank case under a misleading topk value.
            effective_topk = None if topk >= len(finest_bank) else topk
            label = "full" if effective_topk is None else str(topk)
            samples = []
            for seed in args.seeds:
                synthetic, _ = sample_coarse_to_fine(
                    pyramid, args.patch_size, args.stride, args.num_steps_per_scale, seed,
                    topk=effective_topk, window_sigma=0.25, ddim=True,
                )
                samples.append(synthetic)

            cf = float(np.mean([copy_fraction(s, finest_bank, args.patch_size, args.stride) for s in samples]))
            div = diversity(samples)
            row = {"file": fname, "topk": effective_topk, "copy_fraction": cf, "diversity": div,
                   "mean_std": float(np.mean([s.std() for s in samples]))}
            rows.append(row)
            print(f"  topk={label:>4}: copy_fraction={cf:.3f} diversity={div:.2f} mean_std={row['mean_std']:.2f}", flush=True)

    out_path = os.path.join(ROOT, "results", "topk_sweep.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nraw sweep results -> {out_path}")

    viable = [r for r in rows if r["copy_fraction"] < COPY_FRACTION_LIMIT and r["diversity"] > DIVERSITY_FLOOR]
    print(f"\n{'=' * 60}\nviable topk values (copy_fraction < {COPY_FRACTION_LIMIT}, diversity > {DIVERSITY_FLOOR}):")
    if not viable:
        print("  NONE - every topk either copies too much or collapsed diversity across seeds.")
        print("  closest by copy_fraction:")
        for r in sorted(rows, key=lambda r: r["copy_fraction"])[:5]:
            print(f"    {r}")
    else:
        by_topk = {}
        for r in viable:
            by_topk.setdefault(r["topk"], []).append(r)
        summary = sorted(
            ((topk, float(np.mean([r["copy_fraction"] for r in rs])), float(np.mean([r["diversity"] for r in rs])))
             for topk, rs in by_topk.items()),
            key=lambda t: (t[0] is None, t[0] if t[0] is not None else 0),
        )
        for topk, cf, div in summary:
            print(f"  topk={('full' if topk is None else topk):>4}: mean copy_fraction={cf:.3f}, mean diversity={div:.2f}")
        finite = [t for t in summary if t[0] is not None]
        chosen = min(finite, key=lambda t: t[1]) if finite else summary[0]
        print(f"\nCHOSEN topk = {chosen[0]} (lowest copy_fraction among viable, non-full-bank values)")


if __name__ == "__main__":
    main()
