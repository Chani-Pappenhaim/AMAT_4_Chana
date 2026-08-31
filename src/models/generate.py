"""Single entry point for turning any electron-microscopy image into a
training-free synthetic image: furniture exclusion -> pyramid -> coarse-to-
fine sampler. One call per image, since each generator is fit to exactly one
source (see the project spec's opening rule: n is source images, not
crops/patches - this function does not combine multiple images).
"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.furniture import build_exclusion_mask  # noqa: E402
from data.pyramid import build_pyramid  # noqa: E402
from models.sampler import sample_coarse_to_fine  # noqa: E402

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "models", "pyramid_patch_config_v1.json")
with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    _FROZEN_CONFIG = json.load(f)


def load_grayscale(path):
    """PIL's own luminance conversion - matches data.loader.load_generator_source,
    and stays correct even for an image that is not already R=G=B."""
    return np.array(Image.open(path).convert("L")).astype(np.float64)


def generate_from_image(
    image,
    downscale=1.0,
    num_scales=_FROZEN_CONFIG["pyramid"]["num_scales"],
    patch_size=_FROZEN_CONFIG["patches"]["patch_size"],
    stride=_FROZEN_CONFIG["patches"]["stride"],
    num_steps_per_scale=15,
    seed=0,
    mask_furniture=True,
    topk=8,
    window_sigma=0.25,
    ddim=True,
    scale_factor=0.75,
    eta=0.0,
    laplacian_blend=False,
    jitter=False,
    coarse_patch_size=None,
    coarse_stride=None,
    nn_alpha=None,
    robust_norm=None,
    histogram_match=False,
):
    """image: single-channel array, any electron-microscopy source.

    downscale < 1.0 trades resolution for CPU runtime - the same tradeoff the
    project's own EMPS/RODARE notebooks make (see docs/limitations.md for why
    an aggressive downscale can itself distort the measured result).

    topk=8, window_sigma=0.25, ddim=True are this project's own defaults for
    the real generator entry point - topk-restricted softmax weighting,
    Gaussian-centered reconstruction, and the deterministic DDIM update, in
    place of full-bank softmax + uniform averaging + Langevin-style noise
    injection. topk=8 was chosen after an empirical sweep on our own images
    (sweep_topk.py, results/topk_sweep.json) found copy_fraction stayed at
    0.0 across every topk tested INCLUDING topk=1 - this pipeline's own
    multi-step, multi-scale, overlapping-window reconstruction already
    prevents literal patch copying from surviving to the final image
    structurally, so the sweep's copying/diversity gates (borrowed from a
    different project's single-step diagnostic) could not discriminate a
    "best" topk here. topk=8 is a moderate, non-extreme choice made on that
    basis - trimming the long tail of far, blur-contributing patches from
    the softmax without any measured downside - not a sharp empirical
    optimum. Pass ddim=False to fall back to the original Langevin-style
    sampler (topk/window_sigma still apply to it too, unless left None).

    Returns (processed_real, synthetic, furniture_mask) - `processed_real` is
    what the pyramid was actually built from (furniture-neutralized, resized),
    so callers can compare like-for-like rather than against the raw input.
    """
    image = np.asarray(image, dtype=np.float64)

    mask = build_exclusion_mask(image) if mask_furniture else np.zeros(image.shape, dtype=bool)
    processed = image.copy()
    if mask.any() and not mask.all():
        processed[mask] = np.median(image[~mask])  # neutralize furniture before it can enter any patch bank

    if downscale != 1.0:
        h, w = processed.shape
        processed = np.array(
            Image.fromarray(processed).resize((max(1, int(w * downscale)), max(1, int(h * downscale))), Image.BILINEAR)
        )

    pyramid = build_pyramid(processed, num_scales=num_scales, scale_factor=scale_factor)
    synthetic, _ = sample_coarse_to_fine(
        pyramid, patch_size, stride, num_steps_per_scale, seed=seed,
        topk=topk, window_sigma=window_sigma, ddim=ddim,
        eta=eta, laplacian_blend=laplacian_blend, jitter=jitter,
        coarse_patch_size=coarse_patch_size, coarse_stride=coarse_stride,
        nn_alpha=nn_alpha, robust_norm=robust_norm, histogram_match=histogram_match,
    )
    return processed, synthetic, mask


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # a toy source with real repeating structure, so a working generator
    # should visibly recover blocky texture rather than staying pure noise
    toy = np.zeros((64, 64))
    toy[::4, :] = 200.0
    toy[:, ::4] = 200.0
    toy += rng.normal(0, 5, size=toy.shape)

    processed, synthetic, mask = generate_from_image(
        toy, downscale=1.0, num_scales=3, patch_size=4, stride=2, num_steps_per_scale=10, seed=0, mask_furniture=False,
    )
    assert synthetic.shape == processed.shape
    assert not np.array_equal(synthetic, processed), "generator must produce something other than a copy of the input"
    print(f"toy check passed: real std={processed.std():.2f}, synthetic std={synthetic.std():.2f}")

    # furniture masking sanity: a synthetic caption footer must not survive into `processed`
    with_footer = toy.copy()
    with_footer[60:64, :] = 250.0
    processed_footer, _, footer_mask = generate_from_image(
        with_footer, downscale=1.0, num_scales=2, patch_size=4, stride=2, num_steps_per_scale=5, seed=0,
    )
    assert footer_mask[62, 10], "the synthetic footer must be detected"
    assert abs(processed_footer[62, 10] - np.median(with_footer[~footer_mask])) < 1e-6, "footer pixels must be neutralized"
    print("furniture-masking check passed: footer pixels neutralized before pyramid construction")
