"""Test the dict_lookup module."""

import pytest

from delaynet.utils.dict_lookup import dict_lookup


@pytest.mark.parametrize(
    "lookup, inverted_lookup",
    [
        ({"a": 1, "b": 2, "c": 1}, {1: ["a", "c"], 2: ["b"]}),
        ({"a": 1, "b": 2, "c": 1, "d": 2}, {1: ["a", "c"], 2: ["b", "d"]}),
        ({}, {}),
    ],
)
def test_dict_lookup(lookup, inverted_lookup):
    """Test the dict_lookup function."""
    assert dict_lookup(lookup) == inverted_lookup


@pytest.mark.parametrize("lookup", [123, "invalid", None])
def test_dict_lookup_with_type_error(lookup):
    """Test the dict_lookup function with invalid lookup."""
    with pytest.raises(TypeError):
        dict_lookup(lookup)
