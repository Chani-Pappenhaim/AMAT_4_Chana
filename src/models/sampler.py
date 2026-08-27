"""Single-scale iterative sampler: turns pure noise into a synthetic image
by repeatedly denoising overlapping patches against a real patch bank, with
a shrinking noise level at each step.
"""
import os
import sys

import numpy as np
from scipy.ndimage import zoom

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.patches import extract_patches, reconstruct_from_patches  # noqa: E402
from models.denoiser import denoise_patch, denoise_patch_approx  # noqa: E402


def estimate_sigma_range(patch_bank, num_samples=2000, seed=0):
    """Derive a sensible (sigma_max, sigma_min) from the bank's own
    patch-to-patch distance distribution, instead of a hardcoded constant
    that only happens to fit one particular image.
    """
    rng = np.random.default_rng(seed)
    flat = patch_bank.reshape(len(patch_bank), -1)
    idx_a = rng.integers(0, len(flat), num_samples)
    idx_b = rng.integers(0, len(flat), num_samples)
    distances = np.sqrt(np.sum((flat[idx_a] - flat[idx_b]) ** 2, axis=1))

    sigma_max = np.percentile(distances, 50)  # "typical" separation: permissive but not everything-looks-alike
    sigma_min = np.percentile(distances, 5)   # near the closest-neighbor scale: selective

    # A bank that is mostly one flat value (e.g. a synthetic mostly-black
    # control image, or a near-uniform coarsest pyramid level) can have so
    # many identical patches that even the MEDIAN distance is 0 - not just
    # the 5th percentile. Fall back to the bank's own pixel value range,
    # which is never 0 unless the whole bank is a single constant.
    if sigma_max <= 1e-6:
        value_range = patch_bank.max() - patch_bank.min()
        sigma_max = value_range if value_range > 1e-6 else 1.0

    # a small or near-uniform bank (e.g. the coarsest pyramid level) can have
    # many identical/near-identical patches, making the 5th percentile land
    # on exact duplicates - distance 0. A geometric sigma schedule cannot
    # include 0, and sigma=0 would divide by zero in denoise_patch, so floor
    # sigma_min to a small fraction of sigma_max instead of letting it hit 0.
    sigma_min = max(sigma_min, sigma_max * 0.01)
    return sigma_max, sigma_min


def sample_single_scale(shape, patch_bank, patch_size, stride, num_steps, sigma_max, sigma_min, seed, step_fraction=0.5, init=None, mean_tolerance=None):
    """step_fraction controls how far each step moves toward the denoised
    mean (Langevin-style partial step) instead of jumping fully onto it -
    a full jump (step_fraction=1) collapses variance almost immediately,
    since averaging is inherently variance-reducing.

    init: start from this image instead of pure noise - used for
    refinement, where the starting point already has coarse structure
    from a previous scale and only needs detail added, not a fresh start.

    mean_tolerance: None runs the exact denoiser (denoise_patch) as before.
    A number switches every patch call to the approximate, mean-prefiltered
    denoiser (denoise_patch_approx) with that tolerance - lets a caller
    build a "fast" configuration without duplicating this whole function.
    """
    assert stride <= patch_size, (
        f"stride ({stride}) must be <= patch_size ({patch_size}) - a larger "
        "stride leaves real gaps between patches that no pixel covers"
    )
    rng = np.random.default_rng(seed)
    sigmas = np.geomspace(sigma_max, sigma_min, num_steps)

    x = init.copy() if init is not None else rng.normal(loc=128.0, scale=sigma_max, size=shape)
    history = [x.copy()]

    # Border pixels are covered by far fewer overlapping patches than
    # interior pixels (a corner gets 1 patch, an edge gets 2, the interior
    # gets 4 with patch_size=4/stride=2) - fewer patches averaged together
    # means less smoothing, so extreme (denoised) values cluster at the
    # border. Padding by reflection before each step gives every real
    # pixel the same interior-level coverage; cropping after reconstruction
    # removes the padding again, so the returned shape is unchanged.
    pad = patch_size

    for step, sigma in enumerate(sigmas):
        x_padded = np.pad(x, pad, mode="reflect")
        record = extract_patches(x_padded, patch_size, stride)
        if mean_tolerance is None:
            denoised_patches = np.stack([
                denoise_patch(patch, patch_bank, sigma)[0] for patch in record.patches
            ])
        else:
            denoised_patches = np.stack([
                denoise_patch_approx(patch, patch_bank, sigma, mean_tolerance)[0] for patch in record.patches
            ])
        record.patches = denoised_patches
        denoised_padded = reconstruct_from_patches(record, x_padded.shape)
        denoised = denoised_padded[pad:pad + shape[0], pad:pad + shape[1]]

        # partial Langevin-style step toward the mean, plus noise scaled
        # to the CURRENT sigma (not the shrinking next_sigma) - this is
        # what keeps the process from collapsing to the mean image
        x = x + step_fraction * (denoised - x)
        is_last_step = step == len(sigmas) - 1
        if not is_last_step:
            x = x + rng.normal(0, sigma, size=shape)

        history.append(x.copy())

    return x, history


def generate_coarse_sketch(pyramid, patch_size, stride, num_steps, seed, step_fraction=0.5, mean_tolerance=None):
    """Run the single-scale sampler on only the coarsest pyramid level.

    At that tiny resolution, a patch covers a large fraction of the whole
    image, so "local" and "global" are nearly the same thing - which is
    exactly why the single-scale method can lay out large-scale structure
    here, even though it cannot at full resolution.
    """
    coarsest_level = pyramid[-1]
    bank = extract_patches(coarsest_level, patch_size, stride).patches.astype(np.float64)
    sigma_max, sigma_min = estimate_sigma_range(bank)

    sketch, history = sample_single_scale(
        shape=coarsest_level.shape,
        patch_bank=bank,
        patch_size=patch_size,
        stride=stride,
        num_steps=num_steps,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        seed=seed,
        step_fraction=step_fraction,
        mean_tolerance=mean_tolerance,
    )
    return sketch, history


def refine_at_scale(current, target_level, patch_size, stride, num_steps, seed, step_fraction=0.5, noise_scale=0.3, mean_tolerance=None):
    """Upsample `current` (a coarser-scale result) to `target_level`'s
    resolution, then add detail by denoising against a bank built from the
    REAL image at that resolution - not by generating from scratch.
    """
    zoom_factors = (target_level.shape[0] / current.shape[0], target_level.shape[1] / current.shape[1])
    upsampled = zoom(current, zoom_factors, order=1)

    bank = extract_patches(target_level, patch_size, stride).patches.astype(np.float64)
    sigma_max, sigma_min = estimate_sigma_range(bank)

    rng = np.random.default_rng(seed)
    # only a little fresh noise - most of the structure should already be
    # right from the upsampled coarser result; this just gives the
    # refinement room to add detail rather than only smoothing the upsample
    noisy_init = upsampled + rng.normal(0, sigma_max * noise_scale, size=target_level.shape)

    refined, history = sample_single_scale(
        shape=target_level.shape,
        patch_bank=bank,
        patch_size=patch_size,
        stride=stride,
        num_steps=num_steps,
        sigma_max=sigma_max * noise_scale,
        sigma_min=sigma_min,
        seed=seed,
        step_fraction=step_fraction,
        init=noisy_init,
        mean_tolerance=mean_tolerance,
    )
    return refined, history


def sample_coarse_to_fine(pyramid, patch_size, stride, num_steps_per_scale, seed, step_fraction=0.5, mean_tolerance=None):
    """Full coarse-to-fine generation: lay out global structure at the
    coarsest scale, then add detail one pyramid level at a time, using each
    level's own real patches - see generate_coarse_sketch and
    refine_at_scale for what happens at each stage.

    mean_tolerance: forwarded to every sample_single_scale call - None for
    the exact reference configuration, a number to build a "fast"
    configuration using the approximate denoiser at every stage.
    """
    sketch, _ = generate_coarse_sketch(pyramid, patch_size, stride, num_steps_per_scale, seed, step_fraction, mean_tolerance=mean_tolerance)
    stages = [sketch]

    current = sketch
    for level_index in range(len(pyramid) - 2, -1, -1):
        # pyramid[-1] is coarsest (already used); walk back toward
        # pyramid[0], the full resolution, one level at a time
        target_level = pyramid[level_index]
        current, _ = refine_at_scale(current, target_level, patch_size, stride, num_steps_per_scale, seed, step_fraction, mean_tolerance=mean_tolerance)
        stages.append(current)

    return current, stages


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # a small synthetic "source image" with real repeating structure -
    # a checkerboard, so a working sampler should visibly recover blocky
    # structure instead of staying pure noise
    block = np.zeros((16, 16))
    block[::4, :] = 200.0
    block[:, ::4] = 200.0
    source_bank_record = extract_patches(block, patch_size=4, stride=2)
    bank = source_bank_record.patches.astype(np.float64)

    final_image, history = sample_single_scale(
        shape=(16, 16),
        patch_bank=bank,
        patch_size=4,
        stride=2,
        num_steps=10,
        sigma_max=50.0,
        sigma_min=1.0,
        seed=0,
    )

    print(f"history has {len(history)} snapshots (initial noise + {len(history)-1} steps)")
    print(f"initial std: {history[0].std():.2f}, final std: {final_image.std():.2f}")

    assert final_image.shape == (16, 16)
    assert not np.array_equal(history[0], final_image), "sampler must change the image, not return the input noise unchanged"
    print("sanity check passed: sampler transformed pure noise into a different image")
