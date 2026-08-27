"""Basic image-level metrics for comparing a synthetic image against a real
one, or comparing several synthetic images against each other.

None of these decide correctness on their own - they are the raw
measurements later layers (copying diagnostics, downstream experiments)
build on.
"""
import numpy as np


def intensity_histogram(image, bins=256, value_range=None):
    """Distribution of pixel brightness values. Two images with similar
    overall tone/contrast produce similar histograms, regardless of where
    in the image that brightness occurs.
    """
    if value_range is None:
        value_range = (image.min(), image.max())
    counts, edges = np.histogram(image, bins=bins, range=value_range)
    return counts / counts.sum(), edges


def gradient_magnitude_histogram(image, bins=256):
    """Distribution of local edge strength. np.gradient returns the
    per-pixel finite-difference slope along each axis; combining the two
    axes (Pythagorean sum) gives one edge-strength number per pixel,
    independent of edge direction.
    """
    gy, gx = np.gradient(image)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    max_magnitude = magnitude.max()
    if max_magnitude <= 1e-12:
        # a perfectly flat image has zero gradient everywhere; a (0, 0)
        # np.histogram range would silently widen to (-0.5, 0.5) and place
        # the mass off bin 0, misreporting "no texture" as "some texture"
        counts = np.zeros(bins)
        counts[0] = 1.0
        return counts, np.linspace(0, 1, bins + 1)

    counts, edges = np.histogram(magnitude, bins=bins, range=(0, max_magnitude))
    return counts / counts.sum(), edges


def power_spectral_density(image):
    """Radially-averaged power spectrum: how much texture energy exists at
    each spatial frequency, independent of where in the image it occurs.

    np.fft.fft2 decomposes the image into 2D sine/cosine waves; the squared
    magnitude of each wave's coefficient is its "power". Averaging over all
    directions at each distance from the zero-frequency center collapses
    the 2D spectrum to a 1D profile (frequency -> average power).

    The mean is subtracted first: the zero-frequency bin's power is
    proportional to the square of the image's total brightness, which
    would otherwise dominate the whole spectrum and hide the actual
    texture - overall brightness is not texture.
    """
    image = image - image.mean()
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    power = np.abs(spectrum) ** 2

    h, w = image.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)

    max_radius = radius.max()
    radial_power = np.zeros(max_radius + 1)
    for r in range(max_radius + 1):
        mask = radius == r
        if mask.any():
            radial_power[r] = power[mask].mean()
    return radial_power


def autocorrelation(image):
    """How similar the image is to a shifted copy of itself, at every
    shift. Computed via the Wiener-Khinchin theorem: autocorrelation is the
    inverse FFT of the power spectrum, which is far cheaper than directly
    computing every pairwise shifted overlap.
    """
    image = image - image.mean()
    spectrum = np.fft.fft2(image)
    power = spectrum * np.conj(spectrum)
    corr = np.fft.ifft2(power).real
    corr = np.fft.fftshift(corr)
    return corr / corr.max()


def diversity_across_seeds(images):
    """Mean pairwise pixel difference across a set of same-source,
    different-seed generations. A generator that always produces nearly
    the same output regardless of seed (mode collapse) scores near 0 here,
    even if any single output looks individually plausible.
    """
    n = len(images)
    assert n >= 2, "diversity requires at least 2 images to compare"

    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(np.abs(images[i] - images[j]).mean())
    return np.mean(diffs)


if __name__ == "__main__":
    # a flat image has no texture: gradients ~0, PSD concentrated at the
    # zero-frequency center, autocorrelation ~1 everywhere (self-similar
    # under any shift)
    flat = np.full((32, 32), 100.0)
    grad_counts, _ = gradient_magnitude_histogram(flat)
    assert grad_counts[0] > 0.99, "flat image should have almost all gradients near 0"
    print("flat image gradient check passed")

    # a checkerboard has maximum-frequency texture: PSD power should be
    # concentrated away from the zero-frequency center, not at it
    checker = np.zeros((32, 32))
    checker[::2, ::2] = 200.0
    checker[1::2, 1::2] = 200.0
    psd = power_spectral_density(checker)
    assert np.argmax(psd) > 0, "checkerboard's peak power should not be at zero frequency"
    print("checkerboard PSD check passed")

    # two identical images have zero diversity; two very different images
    # (flat vs checkerboard) have high diversity
    same_diversity = diversity_across_seeds([flat, flat.copy()])
    assert same_diversity == 0.0
    different_diversity = diversity_across_seeds([flat, checker])
    assert different_diversity > same_diversity
    print(f"diversity check passed: identical={same_diversity:.2f}, different={different_diversity:.2f}")

    # intensity histogram of a flat image: all mass in one bin
    hist, _ = intensity_histogram(flat, bins=10, value_range=(0, 200))
    assert hist.max() == 1.0, "flat image's histogram should have all mass in a single bin"
    print("intensity histogram check passed")

    # autocorrelation at zero shift (the center, after fftshift) must be
    # the maximum - an image is always most similar to itself unshifted
    corr = autocorrelation(checker)
    center = tuple(s // 2 for s in corr.shape)
    assert corr[center] == corr.max()
    print("autocorrelation zero-shift-is-peak check passed")
