"""Single-scale iterative sampler: turns pure noise into a synthetic image
by repeatedly denoising overlapping patches against a real patch bank, with
a shrinking noise level at each step.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.patches import extract_patches, reconstruct_from_patches  # noqa: E402
from models.denoiser import denoise_patch  # noqa: E402


def sample_single_scale(shape, patch_bank, patch_size, stride, num_steps, sigma_max, sigma_min, seed):
    rng = np.random.default_rng(seed)
    sigmas = np.geomspace(sigma_max, sigma_min, num_steps)

    x = rng.normal(loc=128.0, scale=sigma_max, size=shape)
    history = [x.copy()]

    for step, sigma in enumerate(sigmas):
        record = extract_patches(x, patch_size, stride)
        denoised_patches = np.stack([
            denoise_patch(patch, patch_bank, sigma)[0] for patch in record.patches
        ])
        record.patches = denoised_patches
        x = reconstruct_from_patches(record, shape)

        is_last_step = step == len(sigmas) - 1
        if not is_last_step:
            next_sigma = sigmas[step + 1]
            x = x + rng.normal(0, next_sigma, size=shape)

        history.append(x.copy())

    return x, history


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
