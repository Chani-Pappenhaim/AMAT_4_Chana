"""L1 verification artifact: a lineage that touches hidden TEST data (or,
separately, the downstream-experiment holdout) must be rejected - proven
against a deliberately contaminated record, not assumed. Pytest conversion
of data/hidden_test_guard.py's __main__ block.
"""
import pytest

from data.hidden_test_guard import (
    HoldoutLeakageError,
    TestLeakageError,
    TEST_IDS,
    TEST_SOURCES,
    VALIDATION_IDS,
    _SPLIT,
    assert_lineage_avoids_holdout,
    assert_lineage_is_test_safe,
)


def test_clean_lineage_passes():
    assert_lineage_is_test_safe(_SPLIT["train_ids"][:3])


def test_lineage_contaminated_by_test_image_id_is_rejected():
    contaminated_id = sorted(TEST_IDS)[0]
    with pytest.raises(TestLeakageError):
        assert_lineage_is_test_safe(_SPLIT["train_ids"][:2] + [contaminated_id])


def test_lineage_contaminated_by_bare_test_doi_is_rejected():
    contaminated_doi = sorted(TEST_SOURCES)[0]
    with pytest.raises(TestLeakageError):
        assert_lineage_is_test_safe([contaminated_doi])


def test_clean_lineage_passes_holdout_check():
    assert_lineage_avoids_holdout(_SPLIT["train_ids"][:3])


def test_lineage_contaminated_by_validation_id_is_rejected_by_holdout_check():
    contaminated_id = sorted(VALIDATION_IDS)[0]
    with pytest.raises(HoldoutLeakageError):
        assert_lineage_avoids_holdout(_SPLIT["train_ids"][:2] + [contaminated_id])
