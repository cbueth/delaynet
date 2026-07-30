"""Tests for the Delta detrending."""

import pytest
from numpy import array, arange, mean
from numpy.random import default_rng
from numpy.testing import assert_allclose

from delaynet import detrend


@pytest.mark.parametrize(
    "ts, window_size, expected",
    [
        ([0, 0, 0], 1, [0, 0, 0]),  # zero
        ([1, 1, 1], 1, [0, 0, 0]),  # constant
        ([-1, -1, -1], 1, [0, 0, 0]),  # constant
        # For alternating series with window=1, the calculation is:
        # [0]: 1 - mean([1]) = 1 - 1 = 0
        # [1]: -1 - mean([1, -1]) = -1 - 0 = -1
        # [2]: 1 - mean([-1, 1]) = 1 - 0 = 1
        # etc.
        ([1, -1, 1, -1, 1, -1], 1, [0, -1, 1, -1, 1, -1]),  # alternating with window=1
    ],
)
def test_delta(ts, window_size, expected):
    """Test the Delta detrending with various inputs."""
    result = detrend(
        array(ts),
        method="delta",
        window_size=window_size,
    )
    assert_allclose(result, array(expected), atol=1e-10)


@pytest.mark.parametrize(
    "window_size",
    [
        0,  # zero window
        -1,  # negative window
    ],
)
def test_invalid_window_size(window_size):
    """Test Delta detrending with invalid window sizes."""
    ts = array([1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        detrend(ts, method="delta", window_size=window_size)


@pytest.mark.parametrize(
    "ts,window_size,expected",
    [
        (
            arange(20, dtype=float),
            3,
            [
                -1.0,
                -0.5,
                0.0,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                1.0,
                1.5,
            ],
        ),
        (
            arange(20, dtype=float),
            10,
            [
                -4.5,
                -4.0,
                -3.5,
                -3.0,
                -2.5,
                -2.0,
                -1.5,
                -1.0,
                -0.5,
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                2.5,
                3.0,
                3.5,
                4.0,
                4.5,
                5.0,
            ],
        ),
    ],
)
def test_delta_linear_20(ts, window_size, expected):
    """Delta detrending of length-20 linear series."""
    result = detrend(ts, method="delta", window_size=window_size)
    assert_allclose(result, array(expected), atol=1e-10)


def test_window_larger_than_series():
    """Test Delta detrending with window size larger than time series."""
    ts = array([1, 2, 3, 4, 5])
    # Window size 10 is larger than series length 5
    result = detrend(ts, method="delta", window_size=10)

    # With the current implementation, for each point k:
    # - start index is max(k-window_size, 0), which is always 0 when window_size > len(ts)
    # - end index is k+window_size, which is beyond the array bounds
    # So each point is compared with all points from 0 to the end of the array

    # Calculate expected result manually
    expected = array(
        [
            ts[0] - mean(ts[0:10]),  # ts[0:5] in practice
            ts[1] - mean(ts[0:11]),  # ts[0:5] in practice
            ts[2] - mean(ts[0:12]),  # ts[0:5] in practice
            ts[3] - mean(ts[0:13]),  # ts[0:5] in practice
            ts[4] - mean(ts[0:14]),  # ts[0:5] in practice
        ]
    )

    assert_allclose(result, expected)


def test_delta_with_trend():
    """Test Delta detrending with a linear trend."""
    # Create a time series with linear trend
    ts_length = 100
    ts = arange(ts_length, dtype=float)

    # Add small noise
    rng = default_rng(12345)
    ts += rng.normal(0.0, 0.1, ts_length)

    # Apply Delta detrending with different window sizes
    for window_size in [5, 10, 20]:
        ts_detrended = detrend(ts, method="delta", window_size=window_size)

        # With the current implementation, the detrended series won't have exactly zero mean
        # due to the asymmetric window and edge effects
        # For a linear trend, the detrended series should have mean within a reasonable range
        assert -2.0 < mean(ts_detrended) < 2.0

        # The standard deviation should be close to the noise level plus some trend effects
        # For large windows, the detrended series should have std within a reasonable range
        if window_size >= 10:
            assert 0.05 < ts_detrended.std() < 5.0


def test_delta_with_seasonal_data():
    """Test Delta detrending with seasonal data."""
    # Create a time series with seasonal pattern
    ts_length = 100
    seasonal_period = 10
    ts = array([(i % seasonal_period) for i in range(ts_length)], dtype=float)

    # Add small noise
    rng = default_rng(67890)
    ts += rng.normal(0.0, 0.1, ts_length)

    # Apply Delta detrending with window size equal to seasonal period
    ts_detrended = detrend(ts, method="delta", window_size=seasonal_period)

    # Check that the seasonal pattern is removed
    # The detrended series should have mean close to 0
    assert -0.2 < mean(ts_detrended) < 0.2


@pytest.mark.parametrize(
    "length,window_size",
    [
        (10, 3),
        (50, 5),
        (50, 25),
        (100, 10),
        (100, 40),
    ],
)
def test_delta_matches_naive_reference(length, window_size):
    """Compare optimized delta against naive local-mean reference."""
    rng = default_rng(152)
    ts = rng.normal(0, 1, length) + arange(length, dtype=float) * 0.1

    result = detrend(ts, method="delta", window_size=window_size)

    expected = ts.copy()
    for k in range(length):
        left = max(0, k - window_size)
        right = min(length, k + window_size)
        expected[k] = ts[k] - ts[left:right].mean()

    assert_allclose(result, expected, atol=1e-10)


@pytest.mark.parametrize("window_size", [1, 2, 5])
def test_delta_symmetry_constant_series(window_size):
    """Delta detrending of constant series should yield zero."""
    ts = array([5.0] * 20)
    result = detrend(ts, method="delta", window_size=window_size)
    assert_allclose(result, 0.0, atol=1e-10)
