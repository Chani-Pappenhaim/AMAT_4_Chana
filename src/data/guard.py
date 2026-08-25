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
_GROUP_OF = _SPLIT["group_of"]  # image_id -> DOI


class TestLeakageError(AssertionError):
    """A generation lineage touched hidden TEST data. The run is void."""


def assert_lineage_is_test_safe(lineage_ids):
    """Raise TestLeakageError if any id/DOI in lineage_ids belongs to TEST."""
    contaminated = [
        entry for entry in lineage_ids
        if entry in TEST_IDS or entry in TEST_SOURCES or _GROUP_OF.get(entry) in TEST_SOURCES
    ]
    if contaminated:
        raise TestLeakageError(
            f"generation lineage touches hidden TEST data: {contaminated}. "
            "The run is void - not flagged, void."
        )


if __name__ == "__main__":
    clean_lineage = _SPLIT["train_ids"][:3]
    assert_lineage_is_test_safe(clean_lineage)
    print(f"clean lineage passed: {clean_lineage}")

    # deliberately contaminated record: a real TEST image id mixed into an
    # otherwise-clean lineage - proves the assertion actually fires
    contaminated_id = next(iter(TEST_IDS))
    contaminated_lineage = _SPLIT["train_ids"][:2] + [contaminated_id]
    try:
        assert_lineage_is_test_safe(contaminated_lineage)
        raise SystemExit("ERROR: contaminated lineage (TEST image id) was NOT rejected!")
    except TestLeakageError as e:
        print(f"correctly rejected a lineage contaminated by a TEST image id: {e}")

    # deliberately contaminated record: a bare TEST DOI (no image id form)
    contaminated_doi = next(iter(TEST_SOURCES))
    try:
        assert_lineage_is_test_safe([contaminated_doi])
        raise SystemExit("ERROR: contaminated lineage (TEST DOI) was NOT rejected!")
    except TestLeakageError as e:
        print(f"correctly rejected a lineage contaminated by a bare TEST DOI: {e}")
