"""Test the symbolic module."""

import pytest
from numpy import array

from delaynet.utils.symbolic import check_symbolic_pairwise


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
