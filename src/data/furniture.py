"""Figure-furniture exclusion masks: scale bars, panel letters, burned-in
captions/footers that EMPS images carry from their source publication figures.
A patch-based generator has no notion these aren't specimen texture and will
happily reproduce them - see docs/findings.md.

Pure functions only: image array in, boolean mask out. No dependency on
loader/split/guard, so this stays usable on any image from any source.
"""
import numpy as np
from scipy import ndimage


def border_mask(image_shape, border_fraction=0.03):
    h, w = image_shape
    bh, bw = max(1, int(round(h * border_fraction))), max(1, int(round(w * border_fraction)))
    mask = np.zeros(image_shape, dtype=bool)
    mask[:bh, :] = mask[-bh:, :] = True
    mask[:, :bw] = mask[:, -bw:] = True
    return mask


def extreme_intensity_mask(image, low_percentile=1.0, high_percentile=99.0, min_size=20):
    """Flags unlabelled components more extreme than the image's own
    intensity range - a footer/caption box is usually far darker or
    brighter than any real specimen texture. `min_size` drops small
    speckle so genuine noise pixels aren't swept in.
    """
    lo, hi = np.percentile(image, [low_percentile, high_percentile])
    extreme = (image <= lo) | (image >= hi)

    labels, n = ndimage.label(extreme)
    sizes = ndimage.sum(extreme, labels, index=np.arange(1, n + 1))
    keep_labels = np.arange(1, n + 1)[sizes >= min_size]
    return np.isin(labels, keep_labels)


def long_line_mask(image, min_length_fraction=0.5, flatness_threshold=2.0):
    """Flags rows/columns that are almost constant-intensity over a long
    span - the signature of a straight ruled line (scale bar, panel border)
    rather than specimen texture, which varies locally almost everywhere.
    """
    h, w = image.shape
    mask = np.zeros_like(image, dtype=bool)

    row_std = image.std(axis=1)
    flat_rows = np.where(row_std < flatness_threshold)[0]
    if len(flat_rows) and w >= min_length_fraction * w:
        mask[flat_rows, :] = True

    col_std = image.std(axis=0)
    flat_cols = np.where(col_std < flatness_threshold)[0]
    if len(flat_cols):
        mask[:, flat_cols] = True

    return mask


def build_exclusion_mask(image, border_fraction=0.03, min_component_size=20, flatness_threshold=2.0):
    return (
        border_mask(image.shape, border_fraction)
        | extreme_intensity_mask(image, min_size=min_component_size)
        | long_line_mask(image, flatness_threshold=flatness_threshold)
    )


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    toy = rng.normal(loc=128, scale=15, size=(100, 100)).clip(0, 255)

    border = border_mask(toy.shape, border_fraction=0.05)
    assert border[0, 0] and border[-1, -1] and not border[50, 50]
    print(f"border_mask: {border.sum()} / {border.size} pixels flagged")

    # a synthetic caption footer: a bright solid block, far outside the specimen range
    with_footer = toy.copy()
    with_footer[90:100, :] = 250
    extreme = extreme_intensity_mask(with_footer, min_size=20)
    assert extreme[95, 50], "the synthetic footer block must be flagged"
    assert not extreme[50, 50], "normal texture must not be flagged"
    print(f"extreme_intensity_mask: caught the synthetic footer block")

    # a synthetic scale bar: one dead-flat row
    with_bar = toy.copy()
    with_bar[70, :] = 200
    lines = long_line_mask(with_bar, flatness_threshold=2.0)
    assert lines[70, :].all(), "the synthetic scale-bar row must be fully flagged"
    assert not lines[50, :].any(), "normal texture rows must not be flagged"
    print(f"long_line_mask: caught the synthetic scale-bar row")

    combined = build_exclusion_mask(with_footer)
    print(f"build_exclusion_mask: {combined.sum() / combined.size:.1%} of pixels excluded on synthetic test image")
