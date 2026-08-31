"""Multiscale image pyramid, defined by scale factors (not fixed pixel sizes) -
EMPS has 218 distinct canvas geometries, so a hard-coded 512->256->128->64
ladder does not survive contact with the dataset.

Blur before decimating at every level - undersampled detail does not vanish,
it reappears disguised as coarse structure that was never there (aliasing).
"""
import numpy as np
from scipy.ndimage import gaussian_filter, zoom


def build_pyramid(image, num_scales, scale_factor=0.5):
    assert 0 < scale_factor < 1
    assert num_scales >= 1

    image = np.asarray(image, dtype=np.float64)
    pyramid = [image]
    sigma = (1.0 / scale_factor) / 2.0

    current = image
    for _ in range(num_scales - 1):
        blurred = gaussian_filter(current, sigma=sigma)
        current = zoom(blurred, scale_factor, order=1)
        pyramid.append(current)

    return pyramid


def detail_sigmas(pyramid, sigma_max=None):
    """Per-level noise budget, measured directly from how much genuinely
    new detail each level adds over an upsample of the level coarser than
    it - std(actual_level - upsample(next_coarser_level)) - instead of a
    generic percentile-of-bank-distances heuristic that has no relation to
    what that specific level actually contains.

    build_pyramid orders finest-first (pyramid[0] = full resolution,
    pyramid[-1] = coarsest) - the opposite of a coarse-first convention -
    so this walks from pyramid[-1] (which starts from pure noise, hence
    gets sigma_max) back toward pyramid[0], matching that same order in
    its return value: sigmas[i] is the budget for pyramid[i].

    sigma_max=None (default) derives the coarsest level's budget from the
    pyramid's own data (that level's pixel std) instead of a fixed
    constant - this project's images live in 0..255, not 0..1, so a
    borrowed constant tuned for a different intensity range would be
    meaningless here.
    """
    if sigma_max is None:
        sigma_max = float(pyramid[-1].std())

    n = len(pyramid)
    sigmas = [None] * n
    sigmas[-1] = sigma_max
    for level in range(n - 2, -1, -1):
        target_shape = pyramid[level].shape
        coarser = pyramid[level + 1]
        zoom_factors = (target_shape[0] / coarser.shape[0], target_shape[1] / coarser.shape[1])
        upsampled = zoom(coarser, zoom_factors, order=1)
        sigmas[level] = float((pyramid[level] - upsampled).std())

    return sigmas


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    toy = rng.random((64, 64))

    levels = build_pyramid(toy, num_scales=4, scale_factor=0.5)
    for level, arr in enumerate(levels):
        print(f"level {level}: shape={arr.shape}")

    assert [lvl.shape for lvl in levels] == [(64, 64), (32, 32), (16, 16), (8, 8)]

    # anti-alias proof: a pattern at the Nyquist limit must differ between
    # naive strided decimation and blur-then-decimate
    high_freq = np.zeros((64, 64))
    high_freq[::2, :] = 1.0
    naive = high_freq[::2, ::2]
    blurred_then_decimated = build_pyramid(high_freq, num_scales=2, scale_factor=0.5)[1]
    assert not np.allclose(naive, blurred_then_decimated)
    print("anti-alias check passed: blur-then-decimate differs from naive strided decimation")

    # detail_sigmas: finest-first order preserved, coarsest level pinned to
    # sigma_max, every finer level's budget must be a real nonnegative
    # measured std (not a fixed geometric decay unrelated to the pyramid's
    # actual content).
    budgets = detail_sigmas(levels, sigma_max=10.0)
    assert len(budgets) == len(levels)
    assert budgets[-1] == 10.0, "coarsest level (pyramid[-1]) must be pinned to sigma_max, since it starts from pure noise"
    assert all(b >= 0 for b in budgets), "a std can never be negative"
    print(f"detail_sigmas check passed: {[round(b, 3) for b in budgets]}")

    # sigma_max=None must derive a real, positive value from the pyramid's
    # own coarsest-level pixel std, not silently fall back to 0 or crash.
    auto_budgets = detail_sigmas(levels)
    assert auto_budgets[-1] == float(levels[-1].std())
    assert auto_budgets[-1] > 0
    print(f"detail_sigmas auto sigma_max check passed: {auto_budgets[-1]:.4f}")
