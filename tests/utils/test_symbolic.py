"""Test the symbolic module."""

import pytest
from numpy import array

from delaynet.utils.symbolic import check_symbolic_pairwise, to_symbolic


@pytest.mark.parametrize(
    "array1, array2, max_symbols",
    [
        ([1, 2, 3], [4, 5, 6], 6),
        ([1, 1, 1], [1, 1, 1], 1),
        ([1, 1, 1], [1, 1, 1], None),
        ([1, 2, 3], [4, 5, 6], None),
        ([-1, -2, -3], [1, 2, 3], 6),
    ],
)
def test_check_symbolic_pairwise(array1, array2, max_symbols):
    """Test check_symbolic_pairwise by design."""
    check_symbolic_pairwise(array(array1), array(array2), max_symbols)


@pytest.mark.parametrize(
    "array1, array2, max_symbols, match",
    [
        ([1, 2, 3], [4, 5, 6], 5, "have more than 5 unique symbols"),
        ([1, 1, 1], [1, 1, 1], 0, "max_symbols must be greater than 0."),
        (
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            None,
            "Input arrays cannot be of float type.",
        ),
        ([-1, -2, -3], [1, 2, 3], 5, "have more than 5 unique symbols"),
        ([1, "a", 3], [4, 5, 6], None, "Input arrays must be of integer type."),
    ],
)
def test_check_symbolic_pairwise_raises(array1, array2, max_symbols, match):
    """Test check_symbolic_pairwise with invalid input."""
    with pytest.raises(ValueError, match=match):
        check_symbolic_pairwise(array(array1), array(array2), max_symbols)


@pytest.mark.parametrize(
    "array_in, max_symbols, expected",
    [
        ([1.1, 2.2, 3.3], 3, [0, 1, 2]),
        ([1.1, 2.2, 3.5], 3, [0, 1, 2]),
        ([1.1, 2.2, 3.6], 3, [0, 1, 2]),
        ([1.1, 2.9, 3.0], 3, [0, 2, 2]),
        ([1.1, 2.2, 3.3], 2, [0, 1, 1]),
        ([1.1, 2.1, 3.1], 2, [0, 0, 1]),
        ([1.1, 2.2, 3.3], None, [1, 2, 3]),
        ([1.1, 1.9, 34.0], None, [1, 2, 34]),
        ([1.1, 2.2, 3.3], 1, [0, 0, 0]),
    ],
)
def test_to_symbolic(array_in, max_symbols, expected):
    """Test to_symbolic by design."""
    result = to_symbolic(array(array_in), max_symbols)
    assert result.dtype == int
    assert result.tolist() == expected


@pytest.mark.parametrize(
    "array_in, max_symbols, match",
    [
        ([1.1, 2.2, 3.3], 0, "max_symbols must be greater than 0."),
        ([-1, -2, -3], 3, "Input array must be of float type to convert."),
        (["a", "b", "c"], None, "Input array must be of float type to convert."),
    ],
)
def test_to_symbolic_raises(array_in, max_symbols, match):
    """Test to_symbolic with invalid input."""
    with pytest.raises(ValueError, match=match):
        to_symbolic(array(array_in), max_symbols)
