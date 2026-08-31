"""L1 verification artifact: an EMPS-derived claim using the word 'SEM'
must be rejected. Pytest conversion of data/claims.py's __main__ block.
"""
import pytest

from data.claims import ModalityClaimError, assert_emps_claim_is_valid


def test_valid_electron_microscopy_claim_is_accepted():
    assert_emps_claim_is_valid("The generator reproduces electron microscopy texture statistics well.")


def test_sem_token_in_emps_claim_is_rejected():
    with pytest.raises(ModalityClaimError):
        assert_emps_claim_is_valid("The generator reproduces SEM texture statistics well.")


def test_semiconductor_is_not_a_false_positive():
    """Word-boundary check: 'SEMICONDUCTOR' contains the letters SEM but is not the SEM token."""
    assert_emps_claim_is_valid("The SEMICONDUCTOR sample shows clear grain boundaries.")
