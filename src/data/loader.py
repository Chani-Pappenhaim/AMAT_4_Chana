"""Generator-facing data loader. The generator must obtain its input image
through load_generator_source() - never a hardcoded filename - so the
TEST-source rejection below is the only path into the generator.
"""
import json
import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from guard import assert_lineage_is_test_safe, TestLeakageError  # noqa: E402

_SPLIT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "experiments", "source_split_v1.json")
_EMPS_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "AMAT", "amat4-week1", "emps", "images")

with open(_SPLIT_PATH, "r", encoding="utf-8") as f:
    _SPLIT = json.load(f)

TRAIN_IDS = _SPLIT["train_ids"]
VALIDATION_IDS = _SPLIT["validation_ids"]
TEST_IDS = set(_SPLIT["test_ids"])

# Curated TRAIN images selected for generator development (see notebook for how/why).
DEV_SOURCE_IDS = ["00655d9628", "0144266d21", "01ac659240"]


def load_generator_source(source_id):
    """Load one image by id for the generator. Refuses any TEST-split id."""
    assert_lineage_is_test_safe([source_id])
    assert source_id in TRAIN_IDS or source_id in VALIDATION_IDS, f"Unknown source_id: '{source_id}'"

    path = os.path.join(_EMPS_IMAGES_DIR, source_id + ".png")
    img = np.array(Image.open(path).convert("L"))
    return img, source_id


if __name__ == "__main__":
    for sid in DEV_SOURCE_IDS:
        img, source_id = load_generator_source(sid)
        assert source_id not in TEST_IDS
        print(f"loaded {source_id}: shape={img.shape}")

    try:
        load_generator_source(next(iter(TEST_IDS)))
        raise SystemExit("ERROR: TEST source was not rejected!")
    except TestLeakageError as e:
        print(f"Correctly rejected a TEST source: {e}")
