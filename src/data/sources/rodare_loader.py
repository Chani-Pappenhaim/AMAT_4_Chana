"""RODARE raw-file-to-manifest-row adapter. RODARE (record 4124) is real,
verified SEM with real instance-adjacent labels - the only dataset in this
project allowed to claim "SEM" (`modality_claim="SEM"`), never a physical
nanometer/calibration claim (`nm_claim_permitted=false` in the spec; there
is no schema field for that yet, so this loader simply never sets one).

Two corrections baked into this loader, both already worked out the hard
way in `notebooks/pipeline/build_rodare_arm_notebook.py`:

1. **`preprocessed/hold-out/` is quarantined, not a second field pool.**
   Inspecting the raw archive shows `hold-out/` (`WD6mm_31`..`WD6mm_40`) is
   built entirely from instrument `ANP_3`, which the spec explicitly
   quarantines - "ANP-3 shares an instrument serial with ANP-10, so it is
   not a third microscope." It is not legitimate extra data despite
   superficially looking like more usable fields; this loader never reads
   `hold-out/` at all.

2. **Two detector channels of one field are siblings, not two samples.**
   The spec states "13 usable fields" (JFL x8 + ANP-10 x5), but
   `preprocessed/images/` holds 26 `.tif` files. Pairwise pixel
   correlation within each filename block (see investigation below) shows
   this is because each field was scanned by two detector channels, saved
   as two adjacent-numbered files - not 26 independent fields. Grouping by
   field (never by channel) is required so `group_id_for_rodare()` and the
   split-safety invariant in `grouping.py` do not accidentally treat one
   field's two channels as two independent groups.

Investigation (actual numbers from the files on disk, not assumed):
pairwise Pearson correlation of raw pixel values was computed for every
pair of images within each of the three filename blocks
(`WD6mm_21`..`WD6mm_28`, `WD6mm_47`/`WD6mm_48`, `WD_06mm_001`..`_016`).
Every block resolved cleanly into adjacent pairs with correlation ~0.48-0.84
(clearly above the near-zero correlation seen between any cross-pair
combination), and no other pairing gave a comparably high score:
(21,22) (23,24) (25,26) (27,28) - 4 fields; (47,48) - 1 field;
(001,002) (003,004) (005,006) (007,008) (009,010) (011,012) (013,014)
(015,016) - 8 fields. That is 4 + 1 + 8 = 13 fields, exactly matching the
spec's "13 usable fields" - so the channel-sibling pairing below is used
directly rather than the one-field-per-file fallback.
"""
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from schemas import SourceManifestRow  # noqa: E402
from data.grouping import group_id_for_rodare  # noqa: E402

RODARE_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "AMAT", "rodare_4124_carbide_sem", "cloud", "preprocessed"
)

# ANP_3 shares an instrument serial with ANP-10 and is not a third
# microscope (spec rule) - it lives under preprocessed/hold-out/
# (WD6mm_31..WD6mm_40) and must never be read by this loader.
QUARANTINED_INSTRUMENT = "ANP_3"

# Field grouping derived from the pairwise-correlation investigation above:
# each pair is the two detector channels of one physical field, so both
# members of a pair share one group id (the first member's name).
_FIELD_PAIRS = [
    ("WD6mm_21", "WD6mm_22"),
    ("WD6mm_23", "WD6mm_24"),
    ("WD6mm_25", "WD6mm_26"),
    ("WD6mm_27", "WD6mm_28"),
    ("WD6mm_47", "WD6mm_48"),
    ("WD_06mm_001", "WD_06mm_002"),
    ("WD_06mm_003", "WD_06mm_004"),
    ("WD_06mm_005", "WD_06mm_006"),
    ("WD_06mm_007", "WD_06mm_008"),
    ("WD_06mm_009", "WD_06mm_010"),
    ("WD_06mm_011", "WD_06mm_012"),
    ("WD_06mm_013", "WD_06mm_014"),
    ("WD_06mm_015", "WD_06mm_016"),
]

# source_id -> field id (the pair's first member), for iter_rodare_rows()
# and load_rodare_*() to look up without re-deriving the pairing.
_FIELD_OF = {name: pair[0] for pair in _FIELD_PAIRS for name in pair}

USABLE_SOURCE_IDS = sorted(_FIELD_OF.keys())


def field_id_for_source(source_id):
    """Look up the field id a usable RODARE source id belongs to."""
    if source_id not in _FIELD_OF:
        raise ValueError(f"Unknown or quarantined RODARE source_id: {source_id!r}")
    return _FIELD_OF[source_id]


def iter_rodare_rows(rodare_root=RODARE_ROOT):
    """Yield one SourceManifestRow per usable image in preprocessed/images/
    (never preprocessed/hold-out/, which is quarantined ANP_3 data).
    """
    images_dir = os.path.join(rodare_root, "images")
    labels_dir = os.path.join(rodare_root, "labels")

    for source_id in USABLE_SOURCE_IDS:
        image_path = os.path.join(images_dir, source_id + ".tif")
        label_path = os.path.join(labels_dir, source_id + "_label.png")

        with Image.open(image_path) as img:
            width, height = img.size

        # Instance count as connected components of the binary label mask.
        # RODARE's label is foreground/background only, not per-instance,
        # so a true carbide count would need instance segmentation; a
        # connected-component count is the closest cheap proxy and is the
        # same choice already made in build_rodare_arm_notebook.py (Day 23)
        # - it undercounts touching carbides but needs no extra model.
        label_arr = np.array(Image.open(label_path))
        _, instance_count = ndimage.label(label_arr > 0)

        yield SourceManifestRow(
            source_id=source_id,
            dataset="rodare",
            group_id=group_id_for_rodare(field_id_for_source(source_id)),
            # RODARE is the Layer-7-only, SEM-claim arm: the spec treats its
            # entire usable pool as held for the final downstream/trust
            # evaluation, never as generator patch-bank material, so every
            # row is "test" rather than split into train/validation like
            # EMPS/NIST.
            split="test",
            image_path=image_path,
            label_path=label_path,
            modality_claim="SEM",
            height=height,
            width=width,
            instance_count=int(instance_count),
            furniture_fraction=0.0,
        )


def load_rodare_image(source_id, rodare_root=RODARE_ROOT):
    """Load one RODARE field-channel image by id as a numpy array."""
    field_id_for_source(source_id)  # raises on unknown/quarantined ids
    path = os.path.join(rodare_root, "images", source_id + ".tif")
    return np.array(Image.open(path))


def load_rodare_label(source_id, rodare_root=RODARE_ROOT):
    """Load one RODARE field-channel's binary label mask as a numpy array."""
    field_id_for_source(source_id)  # raises on unknown/quarantined ids
    path = os.path.join(rodare_root, "labels", source_id + "_label.png")
    return np.array(Image.open(path))


if __name__ == "__main__":
    rows = list(iter_rodare_rows())
    assert len(rows) == 26, f"expected 26 usable RODARE rows, got {len(rows)}"

    for row in rows:
        assert "hold-out" not in row.image_path and "hold-out" not in row.label_path, (
            f"row {row.source_id!r} touches hold-out/: {row.image_path}"
        )
        assert QUARANTINED_INSTRUMENT not in row.source_id, (
            f"row {row.source_id!r} touches quarantined instrument {QUARANTINED_INSTRUMENT}"
        )

    field_groups = {}
    for row in rows:
        field_groups.setdefault(row.group_id, []).append(row.source_id)
    for group_id, members in field_groups.items():
        assert len(members) >= 1, f"field group {group_id!r} has no members"

    sample_row = rows[0]
    img = load_rodare_image(sample_row.source_id)
    label = load_rodare_label(sample_row.source_id)
    assert img.shape == label.shape == (sample_row.height, sample_row.width), (
        f"loaded image shape {img.shape} / label shape {label.shape} does not match "
        f"manifest row (height={sample_row.height}, width={sample_row.width})"
    )

    print(f"read {len(rows)} usable RODARE rows across {len(field_groups)} field groups")
    print(f"field groups: { {g: len(m) for g, m in field_groups.items()} }")
    print(
        f"sanity-checked source_id={sample_row.source_id!r}: "
        f"loaded image shape={img.shape}, label shape={label.shape}, "
        f"instance_count={sample_row.instance_count}, modality_claim={sample_row.modality_claim!r}"
    )
