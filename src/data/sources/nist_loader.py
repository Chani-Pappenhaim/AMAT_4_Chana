"""NIST raw-file-to-manifest-row adapter for the detection-limits SEM
dataset. Unlike EMPS/RODARE, NIST is not a texture source for the
generator - it is a correctness fixture for Layers 2-4 (verifying the
denoiser/sampler against a known clean image + mask before touching real
EMPS texture). Only set 1's clean reference is meant to be used by default;
sets 2/3/5 are usable if needed, set 4 is invalid, and set 6 is a hidden
fixture that must never be silently loaded (same hidden-test principle as
EMPS's TEST split).
"""
import glob
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from schemas import SourceManifestRow  # noqa: E402
from data.grouping import group_id_for_nist  # noqa: E402

NIST_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "AMAT", "nist_detection_limits_sem")

# Sets with a usable clean reference + mask for correctness-fixture purposes.
USABLE_SETS = ["set1", "set2", "set3", "set5"]
# set4 is invalid - excluded per the spec (not a usable geometry).
INVALID_SETS = ["set4"]
# set6 is a hidden fixture, never to be loaded by default - same
# hidden-test principle as EMPS's TEST split.
HIDDEN_SETS = ["set6"]


def clean_reference_row(set_id, nist_root=NIST_ROOT):
    """Build one SourceManifestRow for set_id's clean reference image and
    its ground-truth mask. Raises ValueError for an invalid set.
    """
    if set_id in INVALID_SETS:
        raise ValueError(f"NIST {set_id} is marked invalid - no usable clean reference")

    masks_dir = os.path.join(nist_root, "mask_sets", "masks")
    image_path = os.path.join(masks_dir, f"{set_id}_cex_noise_000_contrast_100.tiff")
    label_path = os.path.join(masks_dir, f"mask_{set_id}_cex_noise_000_contrast_100.tiff")

    with Image.open(image_path) as img:
        width, height = img.size

    # NIST's only role is a unit-test fixture, not a statistic, so the
    # split policy here is deliberately trivial: hidden sets are "test",
    # everything else is "train" (used as a correctness fixture, never
    # evaluated against).
    split = "test" if set_id in HIDDEN_SETS else "train"

    # Instance count computed as the number of distinct nonzero label
    # values in the mask - cheap for these small binary/few-class masks
    # and gives a meaningful non-zero instance_count instead of a bare 0.
    mask_arr = np.array(Image.open(label_path))
    instance_count = int(len(np.unique(mask_arr[mask_arr != 0])))

    return SourceManifestRow(
        source_id=f"nist_{set_id}_clean",
        dataset="nist",
        group_id=group_id_for_nist(set_id),
        split=split,
        image_path=image_path,
        label_path=label_path,
        modality_claim="electron microscopy",
        height=height,
        width=width,
        instance_count=instance_count,
        furniture_fraction=0.0,
    )


def load_nist_image(set_id, nist_root=NIST_ROOT):
    """Load set_id's clean reference image as a numpy array."""
    path = os.path.join(nist_root, "mask_sets", "masks", f"{set_id}_cex_noise_000_contrast_100.tiff")
    return np.array(Image.open(path))


def load_nist_mask(set_id, nist_root=NIST_ROOT):
    """Load set_id's ground-truth mask as a numpy array."""
    path = os.path.join(nist_root, "mask_sets", "masks", f"mask_{set_id}_cex_noise_000_contrast_100.tiff")
    return np.array(Image.open(path))


def iter_intensity_variants(set_id, nist_root=NIST_ROOT):
    """Yield (variant_path, numpy_array) pairs for every degraded
    noise/contrast variant under intensity_sets/set_id/ - for later-layer
    robustness experiments, not needed for the L2-L4 fixture itself.
    """
    variants_dir = os.path.join(nist_root, "intensity_sets", set_id)
    for variant_path in sorted(glob.glob(os.path.join(variants_dir, "*.tiff"))):
        yield variant_path, np.array(Image.open(variant_path))


if __name__ == "__main__":
    row = clean_reference_row("set1")
    img = load_nist_image("set1")
    mask = load_nist_mask("set1")

    assert img.shape == mask.shape, (
        f"NIST set1 clean image shape {img.shape} does not match mask shape {mask.shape}"
    )
    assert (row.height, row.width) == img.shape, (
        f"manifest row shape (height={row.height}, width={row.width}) does not match loaded image {img.shape}"
    )

    try:
        clean_reference_row("set4")
        raise SystemExit("ERROR: set4 (invalid) was NOT rejected!")
    except ValueError as e:
        print(f"correctly rejected invalid set4: {e}")

    print(
        f"loaded NIST set1: image shape={img.shape}, mask shape={mask.shape}, "
        f"group_id={row.group_id!r}, split={row.split!r}, instance_count={row.instance_count}"
    )
