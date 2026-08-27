"""Real-data label validity: does a source instance mask still describe
where that instance ended up in a GENERATED image?

Day 14-15 answered this on an analytic control (one blob at a known pixel
location) because that case has a known correspondence to track through the
pipeline. A real EMPS image with several particle instances has no such
known correspondence - the generator recombines real patches stochastically,
it does not apply a traceable geometric transform to each instance. So the
measurement method has to be different: local template matching (normalized
cross-correlation) around each instance's original position, not "follow the
known transform".

A LOCAL, not global, search window is used deliberately: coarse-to-fine
generation anchors large-scale structure near its original position (the
Day 12 finding - a bright blob appeared in the correct corner). A global
search would let an unrelated coincidental match anywhere in the image win,
which would measure "is there a similar-looking patch somewhere", not "did
this instance stay near where it was".
"""
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evaluation.copying import normalized_cross_correlation  # noqa: E402


def instance_centroids(segmap, patch_size=16):
    """One (row, col) centroid per non-background instance label, skipping
    instances too close to the border to crop a full patch_size template
    around - a padded/cropped template would bias the correlation search
    toward the crop edge, not the actual specimen texture.
    """
    labels = np.unique(segmap)
    labels = labels[labels != 0]
    h, w = segmap.shape
    half = patch_size // 2

    centroids = []
    for label in labels:
        cy, cx = ndimage.center_of_mass(segmap == label)
        cy, cx = int(round(cy)), int(round(cx))
        if half <= cy < h - half and half <= cx < w - half:
            centroids.append((cy, cx))
    return centroids


def measure_instance_displacement(real_image, segmap, synthetic_image, patch_size=16, search_radius=40):
    """For each real instance: crop a template centered on its centroid from
    the REAL image, then search a local window of the SYNTHETIC image
    (same coordinates, +/- search_radius) for the best-matching location by
    normalized cross-correlation. Returns one (displacement_px, best_score)
    pair per instance - best_score close to 1.0 means a confident match was
    actually found; a low best_score means no good correspondence exists at
    all, which is itself part of the label-validity finding, not noise to
    discard.
    """
    half = patch_size // 2
    h, w = real_image.shape
    results = []

    for cy, cx in instance_centroids(segmap, patch_size=patch_size):
        template = real_image[cy - half:cy + half, cx - half:cx + half]

        y0, y1 = max(half, cy - search_radius), min(h - half, cy + search_radius)
        x0, x1 = max(half, cx - search_radius), min(w - half, cx + search_radius)

        best_score, best_pos = -np.inf, (cy, cx)
        for sy in range(y0, y1 + 1):
            for sx in range(x0, x1 + 1):
                candidate = synthetic_image[sy - half:sy + half, sx - half:sx + half]
                score = normalized_cross_correlation(template, candidate)
                if score > best_score:
                    best_score, best_pos = score, (sy, sx)

        displacement = float(np.sqrt((best_pos[0] - cy) ** 2 + (best_pos[1] - cx) ** 2))
        results.append((displacement, float(best_score)))

    return results


def nearest_neighbor_pseudo_label(synthetic_patch, bank_patches, bank_labels):
    """A synthetic patch has no real mask of its own - Day 14/15 measured
    that the source mask cannot be transferred by position (17-27px
    displacement). This assigns a PSEUDO-label instead, by content: find the
    real bank patch closest in pixel space (L2 distance) to this synthetic
    patch, and inherit ITS label. This is not the naive "reuse the source
    mask at the same coordinates" that was already shown invalid - it is a
    content-based nearest-neighbor label, and should be reported as an
    approximation, not ground truth.
    """
    flat_bank = bank_patches.reshape(len(bank_patches), -1).astype(np.float64)
    flat_query = synthetic_patch.ravel().astype(np.float64)
    distances = np.sum((flat_bank - flat_query) ** 2, axis=1)
    return bank_labels[np.argmin(distances)]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    image = rng.random((80, 80)) * 200

    # two instances, well inside the border for patch_size=16
    segmap = np.zeros((80, 80), dtype=np.uint16)
    segmap[15:20, 15:20] = 1
    segmap[55:62, 40:48] = 2

    centroids = instance_centroids(segmap, patch_size=16)
    assert len(centroids) == 2, f"expected 2 in-bounds instances, got {len(centroids)}"
    print(f"instance_centroids check passed: found {centroids}")

    # identical image as its own "synthetic" output: the best match for
    # every instance must be at displacement 0, score ~1.0 (it IS itself)
    results = measure_instance_displacement(image, segmap, image, patch_size=16, search_radius=20)
    for displacement, score in results:
        assert displacement == 0.0, f"self-match displacement should be 0, got {displacement}"
        assert score > 0.999, f"self-match score should be ~1.0, got {score}"
    print(f"self-match check passed: displacement=0 for all instances, scores={[round(s, 4) for _, s in results]}")

    # a synthetic image that is the real image shifted by a KNOWN offset:
    # the measured displacement must recover that offset (within search_radius)
    dy, dx = 6, -4
    shifted = np.roll(np.roll(image, dy, axis=0), dx, axis=1)
    results = measure_instance_displacement(image, segmap, shifted, patch_size=16, search_radius=20)
    expected = np.sqrt(dy ** 2 + dx ** 2)
    for displacement, score in results:
        assert abs(displacement - expected) < 1e-6, f"expected displacement {expected:.2f}, got {displacement:.2f}"
        assert score > 0.999
    print(f"known-shift recovery check passed: expected {expected:.2f}px, measured {[round(d, 2) for d, _ in results]}")

    # nearest_neighbor_pseudo_label: a query patch identical to bank[2] must
    # get label[2], even with several other bank entries to choose from
    bank_patches = np.stack([rng.random((8, 8)) * 200 for _ in range(5)])
    bank_labels = np.array([0, 0, 1, 0, 1])
    query = bank_patches[2].copy()
    label = nearest_neighbor_pseudo_label(query, bank_patches, bank_labels)
    assert label == 1, f"query identical to bank[2] (label=1) should get label 1, got {label}"
    print("nearest_neighbor_pseudo_label check passed: exact match returns the matched patch's own label")
