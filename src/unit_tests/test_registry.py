import pytest

from src.phase1.operators import (
    ALL_OPERATORS,
    get_operators,
    AOR,
    SD,
    BOR,
    BSOD
)


def test_get_operators_success():
    """Verify that valid operator codes return instantiated objects of the correct type."""
    ops = get_operators(["AOR", "SD", "BOR"])
    
    assert len(ops) == 3
    assert isinstance(ops[0], AOR)
    assert isinstance(ops[1], SD)
    assert isinstance(ops[2], BOR)


def test_get_operators_empty():
    """Verify that an empty list returns an empty list without error."""
    ops = get_operators([])
    assert len(ops) == 0


def test_get_operators_invalid():
    """Verify that an invalid operator code raises a precise ValueError."""
    with pytest.raises(ValueError, match="Unknown operator code: FAKE_OP"):
        get_operators(["AOR", "FAKE_OP"])


def test_all_operators_registered_correctly():
    """Spot check that the ALL_OPERATORS map points to the correct uninstantiated classes."""
    assert "AOR" in ALL_OPERATORS
    assert "SD" in ALL_OPERATORS
    assert "BSOD" in ALL_OPERATORS
    
    # Assert identity mapping
    assert ALL_OPERATORS["AOR"] is AOR
    assert ALL_OPERATORS["SD"] is SD
    assert ALL_OPERATORS["BSOD"] is BSOD


def test_operator_instantiation_independence():
    """Verify that multiple requests for the same operator return independent instances."""
    ops_1 = get_operators(["AOR"])
    ops_2 = get_operators(["AOR"])
    
    assert isinstance(ops_1[0], AOR)
    assert isinstance(ops_2[0], AOR)
    assert ops_1[0] is not ops_2[0]  # They must be separate instances