"""Patch extraction and reconstruction for one pyramid level. A patch is
dropped if it intersects any excluded (figure-furniture) pixel - Layer 2
must apply the Layer-1 exclusion mask before building the bank.
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

    patches, positions = [], []
    for row in range(0, h - patch_size + 1, stride):
        for col in range(0, w - patch_size + 1, stride):
            if exclusion_mask is not None:
                window = exclusion_mask[row:row + patch_size, col:col + patch_size]
                if window.any():
                    continue
            patches.append(image[row:row + patch_size, col:col + patch_size])
            positions.append((row, col))

    patches = np.stack(patches) if patches else np.empty((0, patch_size, patch_size))
    return PatchRecord(patches=patches, positions=positions, patch_size=patch_size, stride=stride)


def reconstruct_from_patches(record, image_shape):
    h, w = image_shape
    accum = np.zeros((h, w), dtype=np.float64)
    counts = np.zeros((h, w), dtype=np.float64)
    for (row, col), patch in zip(record.positions, record.patches):
        accum[row:row + record.patch_size, col:col + record.patch_size] += patch
        counts[row:row + record.patch_size, col:col + record.patch_size] += 1.0

    assert np.all(counts > 0), "uncovered pixels - stride too large or exclusion too aggressive"
    return accum / counts


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
