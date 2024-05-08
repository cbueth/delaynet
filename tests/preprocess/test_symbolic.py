"""Test the symbolic module."""

from operator import lt, le

import pytest
from numpy import array, median, issubdtype, integer

from delaynet.preprocess.symbolic import (
    check_symbolic_pairwise,
    quantize,
    round_to_int,
    binarize,
    quantilize,
    to_symbolic,
)


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
        ([1.1, 2.2, 3.3], 1, [0, 0, 0]),
        ([1, 2, 3], 2, [0, 0, 1]),
        ([1, 2, 3, 4], 2, [0, 0, 1, 1]),
        ([1, 2, 3, 4, 5], 2, [0, 0, 0, 1, 1]),
    ],
)
def test_quantize(array_in, max_symbols, expected):
    """Test quantize by design."""
    result = quantize(array(array_in), max_symbols)
    assert issubdtype(result.dtype, integer)
    assert result.tolist() == expected


@pytest.mark.parametrize(
    "array_in, max_symbols, match",
    [
        ([1.1, 2.2, 3.3], 0, "max_symbols must be a positive integer."),
        ([1.1, 2.2, 3.3], -1, "max_symbols must be a positive integer."),
        ([1, 2, 3], None, "max_symbols must be a positive integer."),
        (["a", "b", "c"], 1, "Input array must be of numeric type to convert."),
        ([1, "a", 3], 1, "Input array must be of numeric type to convert."),
        ([True, False, True], 1, "Input array must be of numeric type to convert."),
        (["a", "b", "c"], None, "max_symbols must be a positive integer."),
        ([1, "a", 3], None, "max_symbols must be a positive integer."),
        ([True, False, True], None, "max_symbols must be a positive integer."),
    ],
)
def test_quantize_raises(array_in, max_symbols, match):
    """Test quantize with invalid input.
    Not uint, int, or float, or max_symbols <= 0.
    """
    with pytest.raises(ValueError, match=match):
        quantize(array(array_in), max_symbols)


@pytest.mark.parametrize(
    "array_in, expected",
    [
        ([1.1, 2.2, 3.3], [1, 2, 3]),
        ([1.1, 2.2, 3.5], [1, 2, 4]),
        ([1.1, 2.2, 3.6], [1, 2, 4]),
        ([1, 2, 3], [1, 2, 3]),
        ([-1.1, -2.2, -3.3], [-1, -2, -3]),
        ([-1.1, -2.5, -3.6], [-1, -2, -4]),
    ],
)
def test_round_to_int(array_in, expected):
    """Test round_to_int by design."""
    result = round_to_int(array(array_in))
    assert issubdtype(result.dtype, integer)
    assert result.tolist() == expected


@pytest.mark.parametrize(
    "array_in, threshold, operator, expected",
    [
        ([1.1, 2.2, 3.3], 2.1, None, [0, 1, 1]),
        ([1.1, 2.2, 3.3], 2.2, None, [0, 0, 1]),
        ([1.1, 2.2, 3.3], 2.3, None, [0, 0, 1]),
        ([1, 2, 3], 2, None, [0, 0, 1]),
        ([-1, -2, -3], -2, None, [1, 0, 0]),
        ([-1, -2, -3], -1, None, [0, 0, 0]),
        ([-1, -2, -3], -3, None, [1, 1, 0]),
        ([1, 2, 3], lambda x: x.mean(), None, [0, 0, 1]),
        ([1, 2, 3], median, None, [0, 0, 1]),
        ([1, 2, 3], lambda x: x.max(), None, [0, 0, 0]),
        ([1, 2, 3], lambda x: x.min(), None, [0, 1, 1]),
        ([1.1, 2.2, 3.3], 2.1, lt, [1, 0, 0]),  # operator less than
        ([1.1, 2.2, 3.3], 2.2, lt, [1, 0, 0]),
        ([1.1, 2.2, 3.3], 2.3, lt, [1, 1, 0]),
        ([1.1, 2.2, 3.3], 2.2, le, [1, 1, 0]),  # operator less than or equal
        ([1, 2, 3, 4], 2.5, lt, [1, 1, 0, 0]),  # mixed int array and float threshold
    ],
)
def test_binarize(array_in, threshold, operator, expected):
    """Test binarize by design."""
    kwargs = {} if operator is None else {"operator": operator}
    result = binarize(array(array_in), threshold, **kwargs)
    assert issubdtype(result.dtype, integer)
    assert result.tolist() == expected


@pytest.mark.parametrize(
    "array_in, num_quantiles, expected",
    [
        ([1], 1, [1]),
        ([1, 2], 2, [1, 2]),
        ([1, 2, 3], 2, [1, 2, 2]),
        ([1, 2, 3, 4], 2, [1, 1, 2, 2]),
        ([1.0, 2.0, 3.0, 4.0], 2, [1, 1, 2, 2]),
        ([1, 2, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7], 3, [1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3]),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 4, [1, 1, 1, 2, 2, 3, 3, 4, 4, 4]),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5, [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 6, [1, 1, 2, 3, 3, 4, 5, 5, 6, 6]),
    ],
)
def test_quantilize(array_in, num_quantiles, expected):
    """Test quantilize by design."""
    result = quantilize(array(array_in), num_quantiles)
    assert issubdtype(result.dtype, integer)
    assert result.tolist() == expected


@pytest.mark.parametrize(
    "array_in, num_quantiles, match",
    [
        ([1], 0, "num_quantiles must be a positive integer."),
        ([1], -1, "num_quantiles must be a positive integer."),
        ([1], None, "num_quantiles must be a positive integer."),
        (["a", "b", "c"], 1, "Input array must be of numeric type to convert."),
        ([1, "a", 3], 1, "Input array must be of numeric type to convert."),
        ([True, False, True], 1, "Input array must be of numeric type to convert."),
        (["a", "b", "c"], None, "num_quantiles must be a positive integer."),
        ([1, "a", 3], None, "num_quantiles must be a positive integer."),
        ([True, False, True], None, "num_quantiles must be a positive integer."),
    ],
)
def test_quantilize_raises(array_in, num_quantiles, match):
    """Test quantilize with invalid input.
    Not uint, int, or float, or num_quantiles <= 0.
    """
    with pytest.raises(ValueError, match=match):
        quantilize(array(array_in), num_quantiles)


@pytest.mark.parametrize(
    "method, kwargs, expected",
    [
        ["quantize", {"max_symbols": 2}, [0, 1, 1]],
        ["round_to_int", {}, [1, 2, 3]],
        ["binarize", {"threshold": 2.1}, [0, 1, 1]],
        ["quantilize", {"num_quantiles": 5}, [1, 3, 5]],
    ],
)
def test_to_symbolic(method, kwargs, expected):
    """Test to_symbolic by design."""
    array_in = array([1.1, 2.2, 3.3])
    result = to_symbolic(array_in, method, **kwargs)
    assert issubdtype(result.dtype, integer)
    assert result.tolist() == expected
