"""Copying/memorization diagnostic: does the generator's output contain
patches that are near-exact copies of real patches, rather than genuine
variations?

Two-stage design (cheap filter, then expensive confirmation) - the same
performance pattern as denoise_patch_approx (Day 17), reused here for a
different purpose: catching copying, not speeding up denoising.

Stage 1 (average-hash): a coarse, cheap descriptor. Cheap enough to compare
every synthetic patch against every reference patch.
Stage 2 (normalized cross-correlation): expensive but exact, run only on the
small number of stage-1 candidates. NCC close to 1.0 means the two patches
are (near-)identical up to brightness/contrast, not just similarly-shaped.

Cross-group reporting is the point of this module, not an afterthought: a
high copying rate against the SAME source image the generator was run on is
expected (the denoiser is built from that image's own patches by design). A
high copying rate against a DIFFERENT group's image is leakage - it would
mean the generator reproduced patches it was never given for this run.
"""
import numpy as np
from scipy.ndimage import zoom


def average_hash(patch, hash_size=8):
    """Resize the patch to a tiny (hash_size x hash_size) thumbnail, then
    threshold every pixel against the thumbnail's own mean -> a bit vector.
    Two patches that look alike at a glance (same rough light/dark layout)
    get the same or a very close bit vector, even if their exact pixel
    values differ slightly - which is exactly what makes this cheap enough
    to run on every pair.
    """
    zoom_factors = (hash_size / patch.shape[0], hash_size / patch.shape[1])
    small = zoom(patch, zoom_factors, order=1)
    return small > small.mean()


def hamming_distance(hash_a, hash_b):
    """Number of bits that differ between two hashes - 0 means identical
    thumbnails, hash_size**2 means every bit flipped (total opposite).
    """
    return int(np.sum(hash_a != hash_b))


def normalized_cross_correlation(patch_a, patch_b):
    """1.0 = identical up to a brightness/contrast scaling, 0.0 = unrelated,
    -1.0 = exact inverse. Both patches are mean-centered first so that two
    patches which are the same SHAPE of variation but at different overall
    brightness still score close to 1.0 - copying means "same content",
    not "same absolute pixel values".
    """
    a = patch_a - patch_a.mean()
    b = patch_b - patch_b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    if denom < 1e-12:
        # both patches are (near-)flat: variance is ~0, so "correlation" is
        # mathematically undefined here, not 1.0. Two DIFFERENT flat patches
        # (e.g. one all-black, one all-white) would otherwise be reported as
        # perfectly correlated just because neither has any texture to
        # compare - a false "copying confirmed" on two patches that don't
        # even share a pixel value.
        return 0.0
    return float((a * b).sum() / denom)


def find_candidates(synthetic_patches, reference_patches, hash_size=8, max_hamming=5):
    """Stage 1: cheap filter. Returns (i, j, hamming_distance) triples where
    synthetic_patches[i] and reference_patches[j] look alike enough to be
    worth the expensive check in stage 2.
    """
    ref_hashes = [average_hash(p, hash_size) for p in reference_patches]
    candidates = []
    for i, sp in enumerate(synthetic_patches):
        sh = average_hash(sp, hash_size)
        for j, rh in enumerate(ref_hashes):
            d = hamming_distance(sh, rh)
            if d <= max_hamming:
                candidates.append((i, j, d))
    return candidates


def confirm_candidates(synthetic_patches, reference_patches, candidates, ncc_threshold=0.95):
    """Stage 2: expensive but exact. Re-checks only the stage-1 candidates
    with normalized cross-correlation, and keeps only the ones that pass
    ncc_threshold - a real confirmation of copying, not a hash coincidence.
    """
    confirmed = []
    for i, j, d in candidates:
        ncc = normalized_cross_correlation(synthetic_patches[i], reference_patches[j])
        if ncc >= ncc_threshold:
            confirmed.append((i, j, d, ncc))
    return confirmed


def copying_rate(synthetic_patches, reference_patches, hash_size=8, max_hamming=5, ncc_threshold=0.95):
    """Fraction of synthetic patches that have at least one confirmed
    near-duplicate in reference_patches. Runs both stages; returns the rate
    plus the full confirmed list for inspection.
    """
    candidates = find_candidates(synthetic_patches, reference_patches, hash_size, max_hamming)
    confirmed = confirm_candidates(synthetic_patches, reference_patches, candidates, ncc_threshold)
    confirmed_synthetic_indices = {i for i, j, d, ncc in confirmed}
    rate = len(confirmed_synthetic_indices) / len(synthetic_patches)
    return rate, confirmed


def cross_group_copying_report(synthetic_patches, own_group_patches, other_group_patches, **kwargs):
    """The actual deliverable of this module: two copying rates side by
    side, computed the same way, so they can be compared directly.

    own_group_patches: real patches from the SAME image/group the generator
    was run on for this sample - a high rate here is expected and fine.
    other_group_patches: real patches from a DIFFERENT group (different DOI)
    that had nothing to do with generating this sample - any confirmed
    match here is leakage, not expected behavior.
    """
    own_rate, own_confirmed = copying_rate(synthetic_patches, own_group_patches, **kwargs)
    other_rate, other_confirmed = copying_rate(synthetic_patches, other_group_patches, **kwargs)
    return {
        "own_group_rate": own_rate,
        "other_group_rate": other_rate,
        "own_confirmed": own_confirmed,
        "other_confirmed": other_confirmed,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # an exact copy of a reference patch must be confirmed: hash distance 0,
    # NCC == 1.0 (it IS itself)
    real_patch = rng.random((8, 8)) * 200
    identical_copy = real_patch.copy()
    rate, confirmed = copying_rate([identical_copy], [real_patch])
    assert rate == 1.0 and len(confirmed) == 1
    print("exact-copy check passed: identical patch confirmed as copying")

    # a totally unrelated patch (different random draw) must NOT be
    # confirmed - proves the pipeline doesn't just rubber-stamp everything
    unrelated_patch = rng.random((8, 8)) * 200
    rate, confirmed = copying_rate([unrelated_patch], [real_patch])
    assert rate == 0.0
    print("unrelated-patch check passed: no false positive")

    # two DIFFERENT flat patches (one near-black, one near-white): the
    # average-hash of a flat patch is meaningless (every pixel equals the
    # mean, so "> mean" is all False regardless of the actual value) - so
    # stage 1 would flag this pair as a hash match. Stage 2's NCC must
    # correctly reject it (denom ~0 -> return 0.0), not confirm a "copy"
    # between two patches that don't even share a pixel value.
    flat_dark = np.full((8, 8), 10.0)
    flat_bright = np.full((8, 8), 245.0)
    candidates = find_candidates([flat_dark], [flat_bright], max_hamming=64)
    assert len(candidates) == 1, "flat patches should hash-collide (that's the edge case)"
    confirmed = confirm_candidates([flat_dark], [flat_bright], candidates)
    assert len(confirmed) == 0, "NCC must reject the flat/flat hash-collision, not confirm it"
    print("flat-patch hash-collision check passed: stage 2 correctly rejects the false candidate")

    # cross-group report: synthetic patches deliberately built as exact
    # copies of "own group" patches, and unrelated to "other group" patches
    own_group = [rng.random((8, 8)) * 200 for _ in range(5)]
    other_group = [rng.random((8, 8)) * 200 for _ in range(5)]
    synthetic = [p.copy() for p in own_group]  # deliberately: exact copies
    report = cross_group_copying_report(synthetic, own_group, other_group)
    assert report["own_group_rate"] == 1.0, "synthetic patches ARE copies of own_group - must be caught"
    assert report["other_group_rate"] == 0.0, "own_group and other_group are independent random draws - no cross-group match expected"
    print(f"cross-group report check passed: own_group_rate={report['own_group_rate']:.2f}, other_group_rate={report['other_group_rate']:.2f}")
