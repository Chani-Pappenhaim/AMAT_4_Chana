"""Patch extraction and reconstruction for one pyramid level. A patch is
dropped if it intersects any excluded (figure-furniture) pixel, so the
patch bank never contains a scale bar, caption, or panel letter.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class PatchRecord:
    patches: np.ndarray
    positions: list
    patch_size: int
    stride: int


def extract_patches(image, patch_size, stride, exclusion_mask=None):
    image = np.asarray(image)
    h, w = image.shape
    assert patch_size <= h and patch_size <= w

    # guarantee full coverage even when stride doesn't evenly divide
    # (h - patch_size) / (w - patch_size) - always include the exact
    # bottom/right-aligned position, not just stride-spaced ones
    row_starts = list(range(0, h - patch_size + 1, stride))
    if row_starts[-1] != h - patch_size:
        row_starts.append(h - patch_size)
    col_starts = list(range(0, w - patch_size + 1, stride))
    if col_starts[-1] != w - patch_size:
        col_starts.append(w - patch_size)

    patches, positions = [], []
    for row in row_starts:
        for col in col_starts:
            if exclusion_mask is not None:
                window = exclusion_mask[row:row + patch_size, col:col + patch_size]
                if window.any():
                    continue
            patches.append(image[row:row + patch_size, col:col + patch_size])
            positions.append((row, col))

    patches = np.stack(patches) if patches else np.empty((0, patch_size, patch_size))
    return PatchRecord(patches=patches, positions=positions, patch_size=patch_size, stride=stride)


def gaussian_window(patch_size, relative_sigma=0.25):
    """A patch_size x patch_size weight matrix peaked at the patch's center
    and falling off toward its edges (separable Gaussian outer product).

    Used by reconstruct_from_patches to weight each patch's contribution by
    how central a pixel is within it, instead of every pixel in every patch
    counting equally. Uniform averaging lets a pixel's value be dragged
    equally by patches where it sits near a noisy, low-confidence edge - a
    center-weighted average trusts each patch most where it is most
    reliable, which reduces the per-step blur that compounds over the
    sampler's many steps.

    relative_sigma is the Gaussian's std as a fraction of patch_size, so the
    same relative shape applies regardless of patch size.
    """
    coords = np.arange(patch_size) - (patch_size - 1) / 2.0
    sigma = relative_sigma * patch_size
    kernel_1d = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    window = np.outer(kernel_1d, kernel_1d)
    return window / window.max()  # peak weight 1.0, matching the old uniform weight of 1.0


def reconstruct_from_patches(record, image_shape, window=None, robust_norm=None, robust_iters=3):
    """window=None (default): uniform averaging, unchanged from before this
    parameter existed. A patch_size x patch_size weight matrix (e.g. from
    gaussian_window) instead weights each patch's contribution by that
    matrix - every position accumulates window * patch and window into
    accum/counts respectively, so the divide-by-counts step still yields a
    proper weighted average, not just a differently-scaled sum.

    robust_norm=None (default): the plain weighted MEAN above, byte-identical
    to this function's behavior before the parameter existed.

    robust_norm=r (0 < r < 2) instead minimizes sum_p ||x_p - z_p||^r over the
    overlapping patches by iteratively reweighted least squares, which is the
    aggregation step of Kwatra et al. 2005 ("Texture Optimization for
    Example-based Synthesis", r=0.8). WHY THIS MATTERS HERE: with stride <
    patch_size every output pixel is covered by several patches, and the
    r=2 mean is pulled to the midpoint of ALL of them. When those patches
    disagree - which is exactly what happens at a particle edge, where some
    overlapping patches place the edge a pixel or two over - the midpoint of
    "edge here" and "edge there" is a ramp, i.e. no edge at all. This is a
    SECOND, independent blurring stage on top of the denoiser's own softmax
    averaging (see denoiser.select_patches_nn): even when every contributing
    patch is an exact, perfectly sharp real patch, averaging them re-blurs
    the result. r < 2 downweights the patches that disagree most, so the
    aggregate converges toward the dominant cluster (a mode) rather than the
    mean of all clusters, and an edge survives as an edge.

    The IRLS weight is per-patch (one scalar per patch, from its current
    residual), matching Kwatra's formulation - not per-pixel - so a patch
    that agrees with its neighbourhood keeps full influence everywhere it
    covers, and one that disagrees loses influence everywhere it covers.
    """
    h, w = image_shape
    weight = 1.0 if window is None else window

    def blend(patch_weights):
        accum = np.zeros((h, w), dtype=np.float64)
        counts = np.zeros((h, w), dtype=np.float64)
        for (row, col), patch, pw in zip(record.positions, record.patches, patch_weights):
            accum[row:row + record.patch_size, col:col + record.patch_size] += patch * weight * pw
            counts[row:row + record.patch_size, col:col + record.patch_size] += weight * pw
        assert np.all(counts > 0), "uncovered pixels - stride too large or exclusion too aggressive"
        return accum / counts

    ones = np.ones(len(record.positions), dtype=np.float64)
    result = blend(ones)
    if robust_norm is None:
        return result

    assert 0 < robust_norm < 2, "robust_norm must be in (0, 2); 2 is the plain mean, use None for that"
    for _ in range(robust_iters):
        # residual of each patch against the current estimate, in per-pixel
        # RMS units so the exponent's scale does not depend on patch_size
        residuals = np.array([
            np.sqrt(np.mean((result[row:row + record.patch_size, col:col + record.patch_size] - patch) ** 2))
            for (row, col), patch in zip(record.positions, record.patches)
        ])
        # floor keeps an exactly-matching patch from getting infinite weight
        floor = max(1e-6, 1e-3 * float(np.median(residuals)))
        result = blend(np.maximum(residuals, floor) ** (robust_norm - 2.0))
    return result


def assert_no_excluded_patch(record, exclusion_mask):
    for row, col in record.positions:
        window = exclusion_mask[row:row + record.patch_size, col:col + record.patch_size]
        assert not window.any(), f"patch at ({row},{col}) touches excluded pixels"


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    toy = rng.random((16, 16))

    record = extract_patches(toy, patch_size=4, stride=2)
    print(f"extracted {len(record.positions)} patches of size {record.patch_size} at stride {record.stride}")

    rebuilt = reconstruct_from_patches(record, toy.shape)
    assert np.allclose(toy, rebuilt, atol=1e-10)
    print("reconstruction check passed: rebuilt image matches source exactly")

    mask = np.zeros_like(toy, dtype=bool)
    mask[0:3, 0:3] = True
    filtered = extract_patches(toy, patch_size=4, stride=2, exclusion_mask=mask)
    assert_no_excluded_patch(filtered, mask)
    assert len(filtered.positions) < len(record.positions)
    print(f"exclusion check passed: {len(record.positions)} -> {len(filtered.positions)} patches after masking")

    # gaussian_window: symmetric, peaked at 1.0 in the center, smaller at the edges
    win = gaussian_window(5, relative_sigma=0.25)
    assert win.shape == (5, 5)
    assert np.isclose(win[2, 2], 1.0), "center weight must be the peak (1.0)"
    assert win[2, 2] > win[0, 0], "center must outweigh a corner"
    assert np.allclose(win, win.T), "a symmetric kernel must be symmetric"
    print("gaussian_window check passed: peaked at center, symmetric")

    # windowed reconstruction on a constant image must still recover it
    # exactly (weighted average of identical values is that same value,
    # whatever the weights are) - this is the windowed analogue of the
    # exact round-trip check above.
    const_img = np.full((16, 16), 7.0)
    const_record = extract_patches(const_img, patch_size=4, stride=2)
    rebuilt_windowed = reconstruct_from_patches(const_record, const_img.shape, window=gaussian_window(4))
    assert np.allclose(const_img, rebuilt_windowed, atol=1e-10)
    print("windowed reconstruction check passed: constant image round-trips exactly")

    # robust_norm=None must stay byte-identical to the pre-parameter behavior
    assert np.array_equal(
        reconstruct_from_patches(record, toy.shape),
        reconstruct_from_patches(record, toy.shape, robust_norm=None),
    ), "robust_norm=None must not change anything"
    print("robust aggregation backward-compat check passed: default is byte-identical")

    # a consistent (non-conflicting) patch set must round-trip under IRLS too:
    # if every overlapping patch already agrees, the mode IS the mean.
    robust_consistent = reconstruct_from_patches(record, toy.shape, robust_norm=0.8)
    assert np.allclose(toy, robust_consistent, atol=1e-8), "IRLS must not disturb an already-consistent reconstruction"
    print("robust aggregation consistency check passed: agreeing patches round-trip exactly")

    # the actual mechanism: a step edge whose overlapping patches DISAGREE
    # about where the edge sits. The mean ramps it; IRLS must keep it steeper.
    edge = np.zeros((16, 16))
    edge[:, 8:] = 100.0
    edge_record = extract_patches(edge, patch_size=4, stride=2)
    shifted = edge_record.patches.copy()
    for i, (row, col) in enumerate(edge_record.positions):
        if i % 2 == 0:  # half the patches vote for an edge one pixel to the left
            shifted[i] = np.roll(edge_record.patches[i], -1, axis=1)
    conflicted = PatchRecord(shifted, edge_record.positions, edge_record.patch_size, edge_record.stride)
    mean_edge = reconstruct_from_patches(conflicted, edge.shape)
    irls_edge = reconstruct_from_patches(conflicted, edge.shape, robust_norm=0.8)
    mean_slope = np.abs(np.diff(mean_edge[8])).max()
    irls_slope = np.abs(np.diff(irls_edge[8])).max()
    assert irls_slope > mean_slope, f"IRLS must sharpen a contested edge ({irls_slope:.1f} vs mean {mean_slope:.1f})"
    print(f"robust aggregation edge check passed: contested edge slope {mean_slope:.1f} (mean) -> {irls_slope:.1f} (IRLS r=0.8)")
