"""Closed-form patch denoiser (Qiu et al.) - no neural-network training.

Given a noisy query patch and a finite bank of clean patches (from the same
source image), the Bayes-optimal denoised estimate under a Gaussian-kernel
prior is a similarity-weighted average of the bank: patches closer to the
query get more weight, patches far from it get almost none.

Why training-free: a neural denoiser learns its weights through many
gradient-descent steps over a large dataset, which takes hours. Here the
"model" is just the patch bank itself - a finite set of real patches - and
the weighted-average formula is the exact, direct solution for that model,
computed once with no optimization loop and no learned parameters.
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


def denoise_patch_approx(noisy_patch, patch_bank, sigma, mean_tolerance):
    """Same result as denoise_patch, computed cheaper: prefilter the bank to
    patches whose mean is close to the query's mean (one number per patch -
    cheap) before running the expensive full-patch distance on survivors
    only. An approximation, not free - see the efficiency notebook for
    whether it changes only runtime or also the output distribution.
    """
    noisy_flat = noisy_patch.reshape(-1)
    bank_flat = patch_bank.reshape(patch_bank.shape[0], -1)

    bank_means = bank_flat.mean(axis=1)
    query_mean = noisy_flat.mean()
    candidate_mask = np.abs(bank_means - query_mean) <= mean_tolerance

    if not candidate_mask.any():
        # mean_tolerance too strict for this query - fall back to the full
        # bank rather than silently returning garbage from an empty set
        candidate_mask[:] = True

    return denoise_patch(noisy_patch, bank_flat[candidate_mask].reshape(-1, *patch_bank.shape[1:]), sigma) + (candidate_mask.sum(),)


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

    # Collapse-to-copy check: needs a bank with one UNIQUE nearest patch -
    # the earlier bank had two tied-equidistant patches, which can never
    # show single-patch collapse no matter how small sigma gets.
    asym_bank = np.array([[0.0, 0.0], [8.0, 8.0], [9.5, 9.5], [20.0, 20.0]])
    asym_query = np.array([9.0, 9.0])  # patch index 2 is uniquely closest

    print("\ncollapse-to-copy sweep (asymmetric bank, one unique nearest patch):")
    for small_sigma in [5.0, 1.0, 0.5, 0.1, 0.01]:
        _, w = denoise_patch(asym_query, asym_bank, small_sigma)
        print(f"sigma={small_sigma:>4}: weights={np.round(w, 4)}, max weight={w.max():.4f}")
    # As sigma shrinks, weight on index 2 (the unique nearest patch) climbs
    # toward 1.0 and the others toward 0 - the denoised output becomes an
    # exact copy of that one patch instead of a blend. This is the failure
    # mode the project must measure, not just avoid by accident: a
    # generator run at too-small sigma is not synthesizing, it is copying.

    # denoise_patch_approx: with a generous tolerance, must match denoise_patch exactly
    exact_denoised, exact_weights = denoise_patch(asym_query, asym_bank, sigma=1.0)
    approx_denoised, approx_weights, n_candidates = denoise_patch_approx(asym_query, asym_bank, sigma=1.0, mean_tolerance=1000.0)
    assert np.allclose(exact_denoised, approx_denoised)
    assert n_candidates == len(asym_bank)
    print(f"\napprox with generous tolerance matches exact: {exact_denoised} == {approx_denoised}")

    # a mean_tolerance so strict nothing survives must fall back to the full bank, not crash
    _, _, n_fallback = denoise_patch_approx(asym_query, asym_bank, sigma=1.0, mean_tolerance=-1.0)
    assert n_fallback == len(asym_bank)
    print("empty-candidate-set fallback to full bank: passed")
