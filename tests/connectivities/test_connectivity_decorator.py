"""Test the connectivity decorator."""

import pytest
from numpy import array, sum as np_sum

from delaynet.decorators import connectivity


def test_connectivity_decorator_simple():
    """Test the connectivity decorator by designing a simple connectivity metric."""

    @connectivity
    def simple_connectivity(ts1, ts2, lag_steps):
        """Return the sum of the two time series."""
        return np_sum(ts1) + np_sum(ts2), lag_steps[0]

    assert (
        simple_connectivity(array([1.0, 2.0, 3.0]), array([4.0, 5.0, 6.0]), lag_steps=5)
        == 21.0,
        1,
    )


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
    def simple_connectivity(ts1, ts2, mult=1, lag_steps=None):
        """Return the sum of the two time series."""
        return mult * (np_sum(ts1) + np_sum(ts2)), lag_steps[0]

    assert simple_connectivity(
        array([1.0, 2.0, 3.0]), array([4.0, 5.0, 6.0]), mult=mult, lag_steps=4
    ) == (expected, 1)


def test_connectivity_decorator_kwargs_unknown():
    """Test the connectivity decorator by designing a simple connectivity metric with
    unknown kwargs."""

    @connectivity
    def simple_connectivity(ts1, ts2, mult=1, lag_steps=None):
        """Return the sum of the two time series."""
        return mult * (np_sum(ts1) + np_sum(ts2)), lag_steps[0]

    with pytest.raises(TypeError, match="got an unexpected keyword argument 'b'"):
        simple_connectivity(
            array([1.0, 2.0, 3.0]), array([4.0, 5.0, 6.0]), b=2, lag_steps=5
        )


def test_connectivity_decorator_kwargs_unknown_ignored():
    """Test the connectivity decorator by designing a simple connectivity metric with
    unknown kwargs and kwarg checker off."""

    @connectivity
    def simple_connectivity(ts1, ts2, mult=1, lag_steps=None):
        """Return the sum of the two time series."""
        return mult * (np_sum(ts1) + np_sum(ts2)), lag_steps[0]

    assert simple_connectivity(
        array([1.0, 2.0, 3.0]),
        array([4.0, 5.0, 6.0]),
        check_kwargs=False,
        b=2,
        lag_steps=[4, 5, 6],
    ) == (21.0, 4)
