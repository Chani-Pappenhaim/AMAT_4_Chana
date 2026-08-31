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


def denoise_patch(noisy_patch, patch_bank, sigma, topk=None):
    """topk restricts the weighted average to the k nearest bank patches
    instead of the full bank. Weight decays exponentially in squared
    distance, so a far-away patch contributes an amount that is technically
    nonzero but immeasurably small on its own - the problem is that summed
    over thousands of such patches, that residual mass still blurs the
    result. topk=None (default) is the exact full-bank sum, unchanged from
    before this parameter existed.
    """
    noisy_flat = noisy_patch.reshape(-1)
    bank_flat = patch_bank.reshape(patch_bank.shape[0], -1)

    sq_dist = np.sum((bank_flat - noisy_flat) ** 2, axis=1)
    if topk is not None and topk < len(bank_flat):
        idx = np.argpartition(sq_dist, topk - 1)[:topk]
        sq_dist = sq_dist[idx]
        bank_flat = bank_flat[idx]
    exponent = -sq_dist / (2 * sigma ** 2)
    exponent -= exponent.max()  # numerical stability: avoid all-zero underflow
    weights = np.exp(exponent)
    weights /= weights.sum()

    denoised_flat = weights @ bank_flat
    return denoised_flat.reshape(noisy_patch.shape), weights


def select_patches_nn(query_patches, patch_bank, alpha_rel=0.5, device=None):
    """Hard nearest-neighbour patch selection with GPNN's normalized
    distance - an alternative to the softmax weighted average above.

    Why this exists: the weighted average is the Bayes-optimal DENOISER, but
    this project is not denoising a real noisy photo, it is SYNTHESIZING an
    image. Averaging k real patches that each contain a particle edge in a
    slightly different place produces a patch containing no sharp edge at
    all - so repeated over many steps and many overlapping patches, real
    discrete particles dissolve into smooth blobs. That is the measured
    failure on this project's own images (see docs/findings.md). Granot et
    al. 2022 ("Drop the GAN: In Defense of Patch Nearest Neighbors as Single
    Image Generative Models", CVPR) make exactly this argument for the
    single-image generative setting this project is in, and their reference
    implementation selects with a hard argmin, never an average.

    The normalized distance is the second half of their method:

        norm_dist[q, k] = dist[q, k] / (min_over_q'(dist[q', k]) + alpha)

    Each bank patch's column is divided by the best match THAT patch
    achieves against any query. So a bank patch some query already matches
    closely (small denominator) gets its distance inflated and becomes less
    attractive to every other query, while a bank patch that nothing matches
    well (large denominator) gets deflated and becomes more attractive. The
    net effect is a bias toward using rare/hard-to-match source patches -
    a COMPLETENESS pressure (more of the source's real structure appears
    somewhere in the output), traded against coherence.

    MEASURED, not assumed: on this project's own real images, lowering
    alpha_rel does NOT increase the number of distinct bank patches used -
    it decreases it (4725 queries against a 4725-patch bank: 2416 distinct
    patches at alpha_rel=1e12, down to 1531 at alpha_rel=0.01, with max
    single-patch reuse rising 14 -> 80). An earlier version of this
    docstring claimed the opposite ("stops a few generic patches being
    selected everywhere"); that was wrong and the probe above disproved it.
    Whether the completeness bias is worth the coherence cost is therefore
    an empirical question per image set, which is why alpha_rel is a swept
    parameter here and not a hardcoded constant.

    alpha_rel scales alpha relative to the data's OWN typical best-match
    distance (alpha = alpha_rel * mean_k(min_q dist)), rather than a fixed
    constant. The reference implementation's constant assumes images
    normalized to [0, 1]; this project's images are 0..255 with a per-image
    std spanning 12..58, so a borrowed constant would mean a wildly
    different amount of normalization per image. Large alpha_rel ->
    normalization effectively off (plain nearest neighbour).

    Returns (selected_patches, nn_indices).
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    patch_shape = query_patches.shape[1:]
    n_queries = query_patches.shape[0]
    n_bank = patch_bank.shape[0]

    queries = torch.as_tensor(np.asarray(query_patches), dtype=torch.float64, device=device).reshape(n_queries, -1)
    bank = torch.as_tensor(np.asarray(patch_bank), dtype=torch.float64, device=device).reshape(n_bank, -1)

    dist = torch.cdist(queries, bank, p=2)  # (n_queries, n_bank)

    per_key_best = dist.min(dim=0, keepdim=True).values  # (1, n_bank)
    alpha = alpha_rel * float(per_key_best.mean())
    if alpha <= 0:
        alpha = 1e-8
    norm_dist = dist / (per_key_best + alpha)

    nn_idx = norm_dist.argmin(dim=1)
    selected = bank[nn_idx].reshape(n_queries, *patch_shape)
    return selected.cpu().numpy(), nn_idx.cpu().numpy()


def denoise_patches_batch(query_patches, patch_bank, sigma, device=None, topk=None):
    """Same math as calling denoise_patch once per query patch (exact Bayes-
    optimal weighted average, no approximation) - computed as one batched
    tensor operation instead of a Python loop over queries.

    This exists because the sampler calls denoise_patch once per patch per
    step (thousands of calls per image), and a Python-level loop pays
    per-call overhead that dwarfs the actual math - batching removes that
    overhead on CPU, and additionally lets the whole batch run on a GPU when
    one is available (Colab, a CUDA machine), since a GPU's advantage is
    doing many independent numeric ops - like the sq-distance between every
    query and every bank patch - in parallel rather than one at a time.

    device=None auto-selects "cuda" if available, else "cpu" - so the same
    call is correct and simply faster wherever a GPU exists, no code change
    needed to move from a laptop to a GPU runtime.

    topk: same meaning as in denoise_patch - restrict each query's weighted
    average to its k nearest bank patches. None (default) is the exact
    full-bank behavior, unchanged.
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    patch_shape = query_patches.shape[1:]
    n_queries = query_patches.shape[0]
    n_bank = patch_bank.shape[0]

    queries = torch.as_tensor(np.asarray(query_patches), dtype=torch.float64, device=device).reshape(n_queries, -1)
    bank = torch.as_tensor(np.asarray(patch_bank), dtype=torch.float64, device=device).reshape(n_bank, -1)

    sq_dist = torch.cdist(queries, bank, p=2) ** 2  # (n_queries, n_bank)

    if topk is not None and topk < n_bank:
        sq_dist, idx = torch.topk(sq_dist, k=topk, dim=1, largest=False)  # (n_queries, topk)
        candidate_bank = bank[idx]  # (n_queries, topk, dim) - a distinct candidate set per query
        exponent = -sq_dist / (2 * sigma ** 2)
        exponent = exponent - exponent.max(dim=1, keepdim=True).values
        weights = torch.exp(exponent)
        weights = weights / weights.sum(dim=1, keepdim=True)
        denoised = torch.einsum("nk,nkd->nd", weights, candidate_bank)
        denoised = denoised.reshape(n_queries, *patch_shape).cpu().numpy()
        return denoised, weights.cpu().numpy()

    exponent = -sq_dist / (2 * sigma ** 2)
    exponent = exponent - exponent.max(dim=1, keepdim=True).values  # numerical stability, per query
    weights = torch.exp(exponent)
    weights = weights / weights.sum(dim=1, keepdim=True)

    denoised = weights @ bank
    denoised = denoised.reshape(n_queries, *patch_shape).cpu().numpy()
    return denoised, weights.cpu().numpy()


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

    # denoise_patches_batch (GPU-capable): must match denoise_patch exactly,
    # patch by patch - batching is a performance change only, never a
    # numerical one. Uses real-sized patches (not the 1x2 toy case) since
    # that is the shape the sampler actually calls this with.
    rng = np.random.default_rng(0)
    real_bank = rng.random((30, 4, 4)) * 255.0
    real_queries = rng.random((7, 4, 4)) * 255.0
    for test_sigma in [50.0, 5.0, 0.5]:
        expected = np.stack([denoise_patch(q, real_bank, test_sigma)[0] for q in real_queries])
        batched, _ = denoise_patches_batch(real_queries, real_bank, test_sigma, device="cpu")
        assert np.allclose(expected, batched, atol=1e-8), (
            f"batched denoiser diverged from the per-patch loop at sigma={test_sigma}"
        )
    print("denoise_patches_batch matches the per-patch loop exactly, across sigma scales: passed")

    # topk (both single-query and batched) must match a brute-force
    # nearest-k selection, and topk=1 must reproduce the exact pure-copy
    # failure mode the collapse-to-copy sweep above demonstrates.
    for test_topk in [1, 5, 15]:
        expected_topk = np.stack([denoise_patch(q, real_bank, 5.0, topk=test_topk)[0] for q in real_queries])
        batched_topk, _ = denoise_patches_batch(real_queries, real_bank, 5.0, device="cpu", topk=test_topk)
        assert np.allclose(expected_topk, batched_topk, atol=1e-8), (
            f"batched topk={test_topk} diverged from the per-patch loop"
        )
    topk1_denoised, _ = denoise_patch(asym_query, asym_bank, sigma=1.0, topk=1)
    nearest_patch = asym_bank[np.argmin(np.sum((asym_bank - asym_query) ** 2, axis=1))]
    assert np.allclose(topk1_denoised, nearest_patch), "topk=1 must exactly copy the single nearest bank patch"
    print("topk check passed: single-query and batched paths agree, topk=1 reproduces pure copying")

    # select_patches_nn: every returned patch must be an EXACT copy of some
    # bank patch (that is the whole point - no averaging, so no blur), and
    # with a huge alpha_rel (normalization switched off) it must reduce to
    # plain nearest neighbour, which is independently checkable by argmin.
    selected, idx = select_patches_nn(real_queries, real_bank, alpha_rel=1e12, device="cpu")
    plain_nn = np.argmin(
        ((real_queries.reshape(len(real_queries), 1, -1) - real_bank.reshape(1, len(real_bank), -1)) ** 2).sum(-1),
        axis=1,
    )
    assert np.array_equal(idx, plain_nn), "alpha_rel -> infinity must reduce to plain nearest neighbour"
    for i, j in enumerate(idx):
        assert np.array_equal(selected[i], real_bank[j]), "selected patch must be an exact bank patch, not a blend"
    print("select_patches_nn check passed: exact bank patches returned, huge alpha reduces to plain NN")

    # The normalization must actually take effect (a finite alpha_rel must
    # change which patches get picked vs. plain NN) and must specifically
    # promote a HARD-TO-MATCH bank patch - that is its definitional
    # behaviour (see the docstring): dividing by each key's own best match
    # deflates keys nothing matches well. Asserting the mechanism itself,
    # rather than a downstream "more variety" side effect - an earlier
    # version of this test asserted increased patch variety and FAILED,
    # because on real data the normalization concentrates selection rather
    # than spreading it. The metric that actually decides whether this
    # helps is the real-image fidelity/completeness sweep, not this test.
    outlier_bank = np.concatenate([
        rng.random((20, 4, 4)) * 10.0 + 100.0,        # a tight cluster, easy to match
        np.full((1, 4, 4), 240.0),                    # one far-away patch nothing matches well
    ])
    near_cluster_queries = rng.random((40, 4, 4)) * 10.0 + 100.0
    _, idx_plain = select_patches_nn(near_cluster_queries, outlier_bank, alpha_rel=1e12, device="cpu")
    _, idx_norm = select_patches_nn(near_cluster_queries, outlier_bank, alpha_rel=0.01, device="cpu")
    outlier_index = len(outlier_bank) - 1
    assert (idx_plain == outlier_index).sum() == 0, "plain NN should never pick the far-away outlier here"
    assert (idx_norm == outlier_index).sum() > 0, (
        "strong normalization must promote the hard-to-match outlier patch - that is its defining effect"
    )
    assert not np.array_equal(idx_plain, idx_norm), "a finite alpha_rel must change the selection"
    print(
        f"normalized-distance check passed: outlier picked {(idx_plain == outlier_index).sum()}x by plain NN, "
        f"{(idx_norm == outlier_index).sum()}x with normalization"
    )
