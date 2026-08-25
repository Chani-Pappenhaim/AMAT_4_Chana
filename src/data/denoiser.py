"""Closed-form patch denoiser (Qiu et al.) - no neural-network training.

Given a noisy query patch and a finite bank of clean patches (from the same
source image), the Bayes-optimal denoised estimate under a Gaussian-kernel
prior is a similarity-weighted average of the bank: patches closer to the
query get more weight, patches far from it get almost none.
"""
import numpy as np


def denoise_patch(noisy_patch, patch_bank, sigma):
    noisy_flat = noisy_patch.reshape(-1)
    bank_flat = patch_bank.reshape(patch_bank.shape[0], -1)

    sq_dist = np.sum((bank_flat - noisy_flat) ** 2, axis=1)
    exponent = -sq_dist / (2 * sigma ** 2)
    exponent -= exponent.max()  # numerical stability: avoid all-zero underflow
    weights = np.exp(exponent)
    weights /= weights.sum()

    denoised_flat = weights @ bank_flat
    return denoised_flat.reshape(noisy_patch.shape), weights


if __name__ == "__main__":
    # Hand-checkable toy case: three 1x2 "patches" (small enough for pen-and-paper).
    # Bank: [0, 0], [10, 10], [10, 10] - query close to the second/third patch.
    bank = np.array([[0.0, 0.0], [10.0, 10.0], [10.0, 10.0]])
    query = np.array([9.0, 9.0])
    sigma = 1.0

    # By hand: sq_dist = [162, 2, 2] -> exp(-81), exp(-1), exp(-1)
    # exp(-81) is ~0, so weight ~0/2/2 normalized -> [0, 0.5, 0.5]
    # denoised = 0*[0,0] + 0.5*[10,10] + 0.5*[10,10] = [10, 10]
    denoised, weights = denoise_patch(query, bank, sigma)
    print(f"weights: {weights}")
    print(f"denoised: {denoised}")

    assert np.allclose(weights, [0.0, 0.5, 0.5], atol=1e-30)
    assert np.allclose(denoised, [10.0, 10.0])
    print("hand-checked toy case passed")

    # Brute-force verification: recompute with a plain Python loop, no numpy vectorization.
    def brute_force(query, bank, sigma):
        raw_weights = []
        for patch in bank:
            d2 = sum((a - b) ** 2 for a, b in zip(query, patch))
            raw_weights.append(np.exp(-d2 / (2 * sigma ** 2)))
        total = sum(raw_weights)
        norm_weights = [w / total for w in raw_weights]
        result = [0.0] * len(query)
        for w, patch in zip(norm_weights, bank):
            for i, v in enumerate(patch):
                result[i] += w * v
        return result, norm_weights

    bf_denoised, bf_weights = brute_force(query, bank, sigma)
    assert np.allclose(bf_denoised, denoised)
    assert np.allclose(bf_weights, weights)
    print("brute-force verification passed")

    # Collapse-to-copy check: as sigma shrinks, the weight concentrates on
    # the single nearest patch - proves the "copying" failure mode exists
    # and shows where it kicks in.
    for small_sigma in [1.0, 0.1, 0.01]:
        _, w = denoise_patch(query, bank, small_sigma)
        print(f"sigma={small_sigma}: weights={w}, max weight={w.max():.4f}")
