"""Test the connectivity decorator."""

# pylint: disable=unexpected-keyword-arg
import pytest
from numpy import array, sum as np_sum

from delaynet.decorators import connectivity


def test_connectivity_decorator_simple():
    """Test the connectivity decorator by designing a simple connectivity metric."""

    @connectivity
    def simple_connectivity(ts1, ts2):
        """Return the sum of the two time series."""
        return np_sum(ts1) + np_sum(ts2)

    assert simple_connectivity(array([1.0, 2.0, 3.0]), array([4.0, 5.0, 6.0])) == 21.0


@pytest.mark.parametrize(
    "mult, expected",
    [
        (1, 21.0),
        (2, 42.0),
        (5, 105.0),
    ],
)
def test_connectivity_decorator_kwargs(mult, expected):
    """Test the connectivity decorator by designing a simple connectivity metric with
    kwargs."""

    @connectivity
    def simple_connectivity(ts1, ts2, mult=1):
        """Return the sum of the two time series."""
        return mult * (np_sum(ts1) + np_sum(ts2))

    assert (
        simple_connectivity(array([1.0, 2.0, 3.0]), array([4.0, 5.0, 6.0]), mult=mult)
        == expected
    )


def test_connectivity_decorator_kwargs_unknown():
    """Test the connectivity decorator by designing a simple connectivity metric with
    unknown kwargs."""

    @connectivity
    def simple_connectivity(ts1, ts2, mult=1):
        """Return the sum of the two time series."""
        return mult * (np_sum(ts1) + np_sum(ts2))

    with pytest.raises(TypeError, match="got an unexpected keyword argument 'b'"):
        simple_connectivity(array([1.0, 2.0, 3.0]), array([4.0, 5.0, 6.0]), b=2)


def test_connectivity_decorator_kwargs_unknown_ignored():
    """Test the connectivity decorator by designing a simple connectivity metric with
    unknown kwargs and kwarg checker off."""

    @connectivity
    def simple_connectivity(ts1, ts2, mult=1):
        """Return the sum of the two time series."""
        return mult * (np_sum(ts1) + np_sum(ts2))

    assert (
        simple_connectivity(
            array([1.0, 2.0, 3.0]), array([4.0, 5.0, 6.0]), check_kwargs=False, b=2
        )
        == 21.0
    )


@pytest.mark.parametrize(
    "array1, array2, check_symbolic, expected, ",
    [
        ([1, 2, 3], [4, 5, 6], True, 21.0),
        ([1, 2, 3], [4, 5, 6], 6, 21.0),
        ([1, 2, 3], [4, 5, 6], 0, 21.0),  # 0 is treated as False
        ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], False, 21.0),
    ],
)
def test_connectivity_decorator_symbolic(array1, array2, check_symbolic, expected):
    """Test the connectivity decorator by designing a simple symbolic connectivity
    metric."""

    @connectivity(check_symbolic=check_symbolic)
    def simple_connectivity(ts1, ts2):
        """Return the sum of the two time series."""
        return float(np_sum(ts1) + np_sum(ts2))

    assert simple_connectivity(array(array1), array(array2)) == expected


@pytest.mark.parametrize(
    "array1, array2, check_symbolic, match",
    [
        ([1, 2, 3], [4, 5, 6], 5, "have more than 5 unique symbols"),
        (
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            None,
            "Input arrays cannot be of float type.",
        ),
        ([-1, -2, -3], [1, 2, 3], 5, "have more than 5 unique symbols"),
        ([1, "a", 3], [4, 5, 6], None, "Input arrays must be of integer type."),
        (
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            None,
            "Input arrays cannot be of float type.",
        ),
    ],
)
def test_connectivity_decorator_symbolic_raises(array1, array2, check_symbolic, match):
    """Test the connectivity decorator with invalid input."""

    @connectivity(check_symbolic=check_symbolic)
    def simple_connectivity(ts1, ts2):
        """Return the sum of the two time series."""
        return float(np_sum(ts1) + np_sum(ts2))

    with pytest.raises(ValueError, match=match):
        simple_connectivity(array(array1), array(array2))


@pytest.mark.parametrize("entropy_like", [True, False])
def test_connectivity_decorator_entropy_like(entropy_like):
    """Test the connectivity decorator by designing a simple entropy-like connectivity
    metric."""

    @connectivity(entropy_like=entropy_like)
    def simple_connectivity(ts1, ts2):
        """Return the sum of the two time series."""
        return float(np_sum(ts1) + np_sum(ts2))

    assert simple_connectivity.is_entropy_like == entropy_like
