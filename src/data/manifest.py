"""Layer 1 deliverable: integrates the three dataset loaders
(data/sources/{emps,nist,rodare}_loader.py) with grouping, furniture, and
claims into one manifest - configs/experiments/L1_source_manifest_v1.csv,
the artifact every later layer reads instead of touching raw dataset files
directly. This is the sequential tail of L1: it needs the loaders' finalized
rows and EMPS's existing DOI-safe split (split.py) before it can run.
"""
import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict, replace

sys.path.insert(0, os.path.dirname(__file__))
from claims import assert_emps_claim_is_valid  # noqa: E402
from furniture import build_exclusion_mask  # noqa: E402
from grouping import assert_no_group_split_across_sets, build_group_of  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data.sources.emps_loader import iter_emps_rows, load_emps_image  # noqa: E402
from src.data.sources.nist_loader import USABLE_SETS, clean_reference_row, load_nist_image  # noqa: E402
from src.data.sources.rodare_loader import iter_rodare_rows, load_rodare_image  # noqa: E402

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
_EMPS_SPLIT_PATH = os.path.join(_REPO_ROOT, "configs", "experiments", "source_split_v1.json")
MANIFEST_PATH = os.path.join(_REPO_ROOT, "configs", "experiments", "L1_source_manifest_v1.csv")


def _load_emps_split_of_id():
    with open(_EMPS_SPLIT_PATH, "r", encoding="utf-8") as f:
        split = json.load(f)
    split_of_id = {}
    for source_id in split["train_ids"]:
        split_of_id[source_id] = "train"
    for source_id in split["validation_ids"]:
        split_of_id[source_id] = "validation"
    for source_id in split["test_ids"]:
        split_of_id[source_id] = "test"
    return split_of_id


def _furniture_fraction(image):
    return float(build_exclusion_mask(image).mean())


def build_manifest():
    """Read all three source loaders, assign the EMPS split (NIST/RODARE
    already assign their own split - fixture vs. Layer-7-only respectively),
    measure furniture fraction, enforce the modality-claim rule, and check
    the DOI/field-safe split invariant before writing the CSV.
    """
    split_of_id = _load_emps_split_of_id()
    rows = []

    for row in iter_emps_rows():
        assert_emps_claim_is_valid(row.modality_claim)
        image = load_emps_image(row.source_id)
        rows.append(replace(
            row,
            split=split_of_id.get(row.source_id, "unassigned"),
            furniture_fraction=_furniture_fraction(image),
        ))

    for set_id in USABLE_SETS:
        row = clean_reference_row(set_id)
        image = load_nist_image(set_id)
        rows.append(replace(row, furniture_fraction=_furniture_fraction(image)))

    for row in iter_rodare_rows():
        image = load_rodare_image(row.source_id)
        rows.append(replace(row, furniture_fraction=_furniture_fraction(image)))

    _assert_groups_are_split_safe(rows)

    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys())
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    return rows


def _assert_groups_are_split_safe(rows):
    """No group (DOI, field, or NIST set) may appear under more than one
    split label - the same invariant split.py already enforces for EMPS,
    checked here across all three datasets combined.
    """
    ids_by_split = defaultdict(list)
    for row in rows:
        ids_by_split[row.split].append(row.source_id)
    group_of = build_group_of(rows)
    assert_no_group_split_across_sets(group_of, **ids_by_split)


if __name__ == "__main__":
    rows = build_manifest()

    by_dataset = defaultdict(list)
    for row in rows:
        by_dataset[row.dataset].append(row)

    assert len(by_dataset["emps"]) >= 400, f"expected >= 400 EMPS rows, got {len(by_dataset['emps'])}"
    assert len(by_dataset["nist"]) == len(USABLE_SETS), (
        f"expected {len(USABLE_SETS)} NIST rows, got {len(by_dataset['nist'])}"
    )
    assert len(by_dataset["rodare"]) == 26, f"expected 26 RODARE rows, got {len(by_dataset['rodare'])}"
    assert all(row.modality_claim != "SEM" for row in by_dataset["emps"] + by_dataset["nist"]), (
        "an EMPS or NIST row claimed 'SEM' - only RODARE may"
    )
    assert os.path.exists(MANIFEST_PATH), f"manifest CSV was not written to {MANIFEST_PATH}"

    for dataset, dataset_rows in sorted(by_dataset.items()):
        n_groups = len({row.group_id for row in dataset_rows})
        print(f"{dataset}: {len(dataset_rows)} rows across {n_groups} groups")
    print(f"wrote {len(rows)} rows to {MANIFEST_PATH}")
