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
