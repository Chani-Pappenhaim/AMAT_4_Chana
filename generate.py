"""CLI entry point: run the training-free generator on every image in a
folder, one independent generator per source image (see
src/models/generate.py). Saves each synthetic output plus a real-vs-synthetic
comparison grid under results/generated/<input-folder-name>/.

Usage:
  python generate.py "min data set"
  python generate.py "min data set" --downscale 0.25 --seed 0
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from models.generate import generate_from_image, load_grayscale  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", help="folder of source images (relative to the repo root, or absolute)")
    parser.add_argument("--downscale", type=float, default=0.25, help="resize factor applied before generation (CPU feasibility)")
    parser.add_argument("--num-scales", type=int, default=6, help="default 6, paired with --scale-factor 0.75 (gentler per-level jump than the old 3-level/0.5 pyramid - confirmed via real-image comparison to recover more distinct particle shapes instead of blotchy texture, e.g. 802e607c7c.png)")
    parser.add_argument("--scale-factor", type=float, default=0.75, help="pyramid downscale ratio between levels (default 0.75, gentler than the old 0.5); needs more --num-scales to reach the same coarsest level")
    parser.add_argument("--patch-size", type=int, default=4, help="the frozen v1 config's patch_size=8 is tuned for full-res EMPS, not a heavily downscaled crop - default here is smaller to match the working resolution")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--num-steps-per-scale", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-furniture-mask", action="store_true", help="skip scale-bar/panel-letter exclusion")
    parser.add_argument("--topk", type=int, default=8, help="restrict softmax weighting to k nearest bank patches (default 8, chosen via sweep_topk.py on our own images)")
    parser.add_argument("--window-sigma", type=float, default=0.25, help="Gaussian reconstruction window relative sigma (default 0.25); pass 0 to disable and use uniform averaging")
    parser.add_argument("--no-ddim", action="store_true", help="use the original Langevin-style sampler instead of the deterministic DDIM update")
    parser.add_argument("--eta", type=float, default=0.0, help="stochastic blend for the ddim step (0=deterministic, matches this project's prior behavior; the original Qiu paper uses a nonzero eta)")
    parser.add_argument("--laplacian-blend", action="store_true", help="compose each refined scale as upsampled-structure + this-level's-own-high-frequency-detail instead of letting each scale's full multi-step result stand on its own (see refine_at_scale)")
    parser.add_argument("--no-jitter", dest="jitter", action="store_false", help="disable the jittered patch grid and fall back to a fixed grid every step (default: jitter on - redrawing the grid's phase each denoising step spreads any single-step patch-seam disagreement across different pixels instead of reinforcing it at the same fixed grid lines every step; validated across all 5 real min-dataset images with no regression on any of them, and a real reduction in blockiness on the two images that showed it worst, e.g. 391ef00939.png and 802e607c7c.png - see results/generated/min data set/grid_artifact_fix/)")
    parser.set_defaults(jitter=True)
    parser.add_argument("--coarse-patch-size", type=int, default=None, help="patch size used ONLY for the coarsest-level sketch, in place of --patch-size (default: none, use --patch-size everywhere like before). A small patch_size cannot see a whole particle at the tiny coarsest resolution, so it can only reproduce local blur instead of discrete particle shapes/spacing - see sample_coarse_to_fine's coarse_patch_size docstring. Finer levels keep --patch-size unchanged, so this does not reintroduce the blocking regression a uniformly-larger patch size caused (see docs/findings.md).")
    parser.add_argument("--coarse-stride", type=int, default=None, help="stride paired with --coarse-patch-size (default: none, use --stride everywhere like before)")
    parser.add_argument("--nn-alpha", type=float, default=1.0, help="switch patch estimation from the softmax weighted average to HARD nearest-neighbour selection with GPNN's normalized distance, using this as alpha_rel (default: 1.0). Averaging several real patches whose particle edges sit in different places produces a patch with no sharp edge at all, which was the main measured cause of real particles dissolving into smooth blobs; hard selection returns exact real patches instead. PROMOTED TO DEFAULT after an alpha sweep (0.1/0.5/1.0/2.0/1e12) scored on all 5 real min-dataset images: alpha_rel=1.0 gave the best completeness and best edge_ratio (0.66->0.92, vs 1.38 overshoot at 0.1 and 0.66 unchanged at plain-NN 1e12) of every value tried, with no regression on any image - see results/generated/min data set/grid_artifact_fix/04_nn10_*.png and docs/findings.md. Pass a large value (e.g. 1e12) to disable the normalization and get plain nearest neighbour, or --nn-alpha=0 to restore the pre-existing softmax weighted average exactly (0 is otherwise an invalid alpha_rel, so it is special-cased as the escape hatch here). See denoiser.select_patches_nn.")
    parser.add_argument("--robust-norm", type=float, default=None, help="replace the plain weighted MEAN used to stitch overlapping patches back together with the robust IRLS aggregation of Kwatra et al. 2005, using this exponent r (try 0.8; must be in (0,2); default: none = keep the mean). This is a THIRD averaging stage, independent of --nn-alpha and --histogram-match: even when every selected patch is an exact sharp real patch, stride < patch-size means each output pixel is the mean of several overlapping patches, and averaging patches that disagree about where an edge sits turns that edge into a ramp. r < 2 downweights the disagreeing patches so the result converges to a mode instead of a mean. NOT promoted to default: measured no further gain once --nn-alpha/--histogram-match are on (see docs/findings.md), so it is left opt-in rather than adding a knob with no demonstrated benefit. See data.patches.reconstruct_from_patches.")
    parser.add_argument("--no-histogram-match", dest="histogram_match", action="store_false", help="disable histogram matching and fall back to only the sampler's own mean/std matching (default: histogram matching ON). The variance-preserving renormalization matches only the first two moments of the real image's intensity distribution - which a real electron-microscopy field's asymmetric, bimodal histogram (sparse dark particles on a bright support; measured 18-39%% particle area, skew -1.3 to -4.3 across our 5 real images) is not determined by. The sampler was satisfying those two moments with a balanced ~50/50 texture instead, which is what made separate particles merge into connected masses (measured particle-area-fraction error 0.373 with mean/std matching alone). PROMOTED TO DEFAULT after full-histogram matching (Heeger & Bergen 1995) cut that area-fraction error to 0.002 across all 5 real images with no regression, and also improved fidelity (0.44->0.32) and completeness (0.42->0.37) - the biggest single measured gain of any fix tried. See sampler.match_histogram and docs/findings.md.")
    parser.set_defaults(histogram_match=True)
    args = parser.parse_args()
    if args.nn_alpha == 0:
        args.nn_alpha = None  # explicit escape hatch back to the pre-existing softmax weighted average

    input_dir = args.input_dir if os.path.isabs(args.input_dir) else os.path.join(ROOT, args.input_dir)
    folder_name = os.path.basename(os.path.normpath(input_dir))
    out_dir = os.path.join(ROOT, "results", "generated", folder_name)
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".tif", ".tiff", ".jpg", ".jpeg")))
    if not files:
        raise SystemExit(f"no image files found in {input_dir}")
    print(f"found {len(files)} source image(s) in {input_dir}")

    fig, axes = plt.subplots(2, len(files), figsize=(3.2 * len(files), 6.4), squeeze=False)

    for col, fname in enumerate(files):
        path = os.path.join(input_dir, fname)
        image = load_grayscale(path)

        processed, synthetic, mask = generate_from_image(
            image,
            downscale=args.downscale,
            num_scales=args.num_scales,
            scale_factor=args.scale_factor,
            patch_size=args.patch_size,
            stride=args.stride,
            num_steps_per_scale=args.num_steps_per_scale,
            seed=args.seed,
            mask_furniture=not args.no_furniture_mask,
            topk=args.topk,
            window_sigma=(None if args.window_sigma == 0 else args.window_sigma),
            ddim=not args.no_ddim,
            eta=args.eta,
            laplacian_blend=args.laplacian_blend,
            jitter=args.jitter,
            coarse_patch_size=args.coarse_patch_size,
            coarse_stride=args.coarse_stride,
            nn_alpha=args.nn_alpha,
            robust_norm=args.robust_norm,
            histogram_match=args.histogram_match,
        )

        stem = os.path.splitext(fname)[0]
        out_path = os.path.join(out_dir, f"synthetic_{stem}.png")
        Image.fromarray(np.clip(synthetic, 0, 255).astype(np.uint8)).save(out_path)

        axes[0, col].imshow(processed, cmap="gray")
        axes[0, col].set_title(f"real (processed)\n{fname}", fontsize=8)
        axes[0, col].axis("off")
        axes[1, col].imshow(synthetic, cmap="gray")
        axes[1, col].set_title(f"synthetic\nstd={synthetic.std():.1f} (real std={processed.std():.1f})", fontsize=8)
        axes[1, col].axis("off")

        print(f"  {fname}: shape={processed.shape}, furniture excluded={mask.mean():.1%}, "
              f"real std={processed.std():.2f}, synthetic std={synthetic.std():.2f} -> {out_path}")

    plt.tight_layout()
    grid_path = os.path.join(out_dir, "real_vs_synthetic_grid.png")
    plt.savefig(grid_path, dpi=150)
    print(f"\nsaved comparison grid -> {grid_path}")


if __name__ == "__main__":
    main()
