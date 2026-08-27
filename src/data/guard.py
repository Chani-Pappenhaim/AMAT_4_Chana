"""Lineage guard against hidden-test leakage - generalizes loader.py's
single-image TEST check to any list of source ids/DOIs. The void rule:
if a lineage touches TEST, the run is void.
"""
import json
import os

_SPLIT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "experiments", "source_split_v1.json")

with open(_SPLIT_PATH, "r", encoding="utf-8") as f:
    _SPLIT = json.load(f)

TEST_IDS = set(_SPLIT["test_ids"])
TEST_SOURCES = set(_SPLIT["test_sources"])  # DOIs
VALIDATION_IDS = set(_SPLIT["validation_ids"])
VALIDATION_SOURCES = set(_SPLIT["validation_sources"])  # DOIs
_GROUP_OF = _SPLIT["group_of"]  # image_id -> DOI


class TestLeakageError(AssertionError):
    """A generation lineage touched hidden TEST data. The run is void."""


class HoldoutLeakageError(AssertionError):
    """A generation lineage touched a held-out (evaluation-only) group. The
    run is void - used for the Layer 7 downstream experiment's held-out
    evaluation groups, which is a DIFFERENT boundary than hidden TEST: the
    downstream experiment's held-out set is the VALIDATION split, reserved
    so no generator or arm ever touches the groups it is evaluated on.
    """


def _lineage_contamination(lineage_ids, forbidden_ids, forbidden_sources):
    return [
        entry for entry in lineage_ids
        if entry in forbidden_ids or entry in forbidden_sources or _GROUP_OF.get(entry) in forbidden_sources
    ]


def assert_lineage_is_test_safe(lineage_ids):
    """Raise TestLeakageError if any id/DOI in lineage_ids belongs to TEST."""
    contaminated = _lineage_contamination(lineage_ids, TEST_IDS, TEST_SOURCES)
    if contaminated:
        raise TestLeakageError(
            f"generation lineage touches hidden TEST data: {contaminated}. "
            "The run is void - not flagged, void."
        )


def assert_lineage_avoids_holdout(lineage_ids):
    """Raise HoldoutLeakageError if any id/DOI in lineage_ids belongs to the
    downstream experiment's held-out (VALIDATION) groups. Used by the Day 21
    downstream-experiment setup: no generated sample used in the synthetic
    training arm may come from a group the experiment will evaluate on.
    """
    contaminated = _lineage_contamination(lineage_ids, VALIDATION_IDS, VALIDATION_SOURCES)
    if contaminated:
        raise HoldoutLeakageError(
            f"generation lineage touches a held-out evaluation group: {contaminated}. "
            "The run is void - not flagged, void."
        )


if __name__ == "__main__":
    clean_lineage = _SPLIT["train_ids"][:3]
    assert_lineage_is_test_safe(clean_lineage)
    print(f"clean lineage passed: {clean_lineage}")

    # deliberately contaminated record: a real TEST image id mixed into an
    # otherwise-clean lineage - proves the assertion actually fires
    contaminated_id = sorted(TEST_IDS)[0]  # deterministic - set iteration order is not stable across runs
    contaminated_lineage = _SPLIT["train_ids"][:2] + [contaminated_id]
    try:
        assert_lineage_is_test_safe(contaminated_lineage)
        raise SystemExit("ERROR: contaminated lineage (TEST image id) was NOT rejected!")
    except TestLeakageError as e:
        print(f"correctly rejected a lineage contaminated by a TEST image id: {e}")

    # deliberately contaminated record: a bare TEST DOI (no image id form)
    contaminated_doi = sorted(TEST_SOURCES)[0]  # deterministic - set iteration order is not stable across runs
    try:
        assert_lineage_is_test_safe([contaminated_doi])
        raise SystemExit("ERROR: contaminated lineage (TEST DOI) was NOT rejected!")
    except TestLeakageError as e:
        print(f"correctly rejected a lineage contaminated by a bare TEST DOI: {e}")

    # holdout (VALIDATION) guard - a separate boundary from TEST, used by
    # the downstream experiment so no generator ever touches its own
    # evaluation groups
    clean_for_holdout = _SPLIT["train_ids"][:3]
    assert_lineage_avoids_holdout(clean_for_holdout)
    print(f"clean-of-holdout lineage passed: {clean_for_holdout}")

    contaminated_validation_id = sorted(VALIDATION_IDS)[0]
    try:
        assert_lineage_avoids_holdout(_SPLIT["train_ids"][:2] + [contaminated_validation_id])
        raise SystemExit("ERROR: contaminated lineage (VALIDATION image id) was NOT rejected!")
    except HoldoutLeakageError as e:
        print(f"correctly rejected a lineage contaminated by a held-out VALIDATION image id: {e}")
