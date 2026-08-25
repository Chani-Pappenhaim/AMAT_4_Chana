"""Modality-claim guard: EMPS is unlabelled SEM+TEM, so any EMPS-derived
claim string must say "electron microscopy", never "SEM". Only the RODARE
arm (real, verified SEM) earns that word.
"""
import re

_SEM_TOKEN = re.compile(r"\bSEM\b")


class ModalityClaimError(AssertionError):
    """An EMPS-derived claim used the word 'SEM', which EMPS cannot support."""


def assert_emps_claim_is_valid(claim_text):
    if _SEM_TOKEN.search(claim_text):
        raise ModalityClaimError(
            f"EMPS-derived claim uses 'SEM', which EMPS (mixed, unlabelled SEM+TEM) "
            f"cannot support - say 'electron microscopy' instead: {claim_text!r}"
        )


if __name__ == "__main__":
    valid = "The generator reproduces electron microscopy texture statistics well."
    assert_emps_claim_is_valid(valid)
    print(f"correctly accepted a valid claim: {valid!r}")

    # deliberately contaminated claim - proves the check actually fires
    contaminated = "The generator reproduces SEM texture statistics well."
    try:
        assert_emps_claim_is_valid(contaminated)
        raise SystemExit("ERROR: a claim using 'SEM' was NOT rejected!")
    except ModalityClaimError as e:
        print(f"correctly rejected an EMPS claim using 'SEM': {e}")

    # word-boundary check: "SEMICONDUCTOR" contains the letters SEM but isn't the SEM token
    not_a_false_positive = "The SEMICONDUCTOR sample shows clear grain boundaries."
    assert_emps_claim_is_valid(not_a_false_positive)
    print(f"correctly did not false-positive on: {not_a_false_positive!r}")
