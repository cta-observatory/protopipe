"""Unit tests for protopipe.benchmarks.utils"""
import pytest

from protopipe.benchmarks.utils import raise_, string_to_boolean


def test_raise_():
    """Check that raise_ raises any exception."""
    with pytest.raises(Exception):
        raise_(ValueError)


def test_string_to_boolean():
    """Check that string_to_boolean deal with all cases."""

    test_values = [True, False, "True", "False"]
    result = string_to_boolean(test_values)
    assert result == [True, False, True, False]

    with pytest.raises(ValueError) as e:
        test_value = ["meh"]
        string_to_boolean(test_value)
        assert str(e.value) == "meh is not a valid boolean."
