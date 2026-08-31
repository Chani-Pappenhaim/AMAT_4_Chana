"""EMPS raw-file-to-manifest-row adapter: reads the EMPS dataset's own
metadata.csv and image files and turns each row into a SourceManifestRow,
the common shape every dataset loader in src/data/sources/ must produce for
manifest.py to combine. Split assignment is not this module's concern - the
split field is left as a placeholder here.
"""
import ast
import csv
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from schemas import SourceManifestRow  # noqa: E402
from data.grouping import group_id_for_emps  # noqa: E402

EMPS_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "AMAT", "amat4-week1", "emps")


def iter_emps_rows(emps_root=EMPS_ROOT):
    """Read metadata.csv under emps_root and yield one SourceManifestRow per
    image, grouped by DOI (never by filename or locator).
    """
    metadata_path = os.path.join(emps_root, "metadata.csv")
    images_dir = os.path.join(emps_root, "images")

    with open(metadata_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            filename = row["filename"]
            source_id = os.path.splitext(filename)[0]
            image_path = os.path.join(images_dir, filename)

            regions = ast.literal_eval(row["regions"]) if row["regions"] else []
            instance_count = len(regions)

            with Image.open(image_path) as img:
                width, height = img.size

            yield SourceManifestRow(
                source_id=source_id,
                dataset="emps",
                group_id=group_id_for_emps(row["doi"]),
                split="unassigned",
                image_path=image_path,
                label_path=None,
                modality_claim="electron microscopy",
                height=height,
                width=width,
                instance_count=instance_count,
                furniture_fraction=0.0,
            )


def load_emps_image(source_id, emps_root=EMPS_ROOT):
    """Load one EMPS image by id as a grayscale numpy array."""
    path = os.path.join(emps_root, "images", source_id + ".png")
    return np.array(Image.open(path).convert("L"))


if __name__ == "__main__":
    rows = list(iter_emps_rows())
    assert len(rows) >= 400, f"expected >= 400 EMPS rows, got {len(rows)}"

    group_ids = {row.group_id for row in rows}
    assert len(group_ids) >= 300, f"expected >= 300 DOI groups, got {len(group_ids)}"

    sample_row = rows[0]
    img = load_emps_image(sample_row.source_id)
    assert img.shape == (sample_row.height, sample_row.width), (
        f"loaded image shape {img.shape} does not match manifest row "
        f"(height={sample_row.height}, width={sample_row.width})"
    )

    print(f"read {len(rows)} EMPS rows across {len(group_ids)} DOI groups")
    print(
        f"sanity-checked source_id={sample_row.source_id!r}: "
        f"loaded shape={img.shape}, instance_count={sample_row.instance_count}"
    )
