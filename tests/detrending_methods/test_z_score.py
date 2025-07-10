"""Tests for the Z-Score detrending."""

import pytest
from numpy import array, arange, std, mean, zeros
from numpy.ma import mod
from numpy.random import default_rng
from numpy.testing import assert_allclose

from delaynet import detrend, logging


@pytest.mark.parametrize(
    "ts, periodicity, max_periods, expected",
    [
        ([0, 0, 0], 1, -1, [0, 0, 0]),  # zero
        ([1, 1, 1], 1, -1, [0, 0, 0]),  # constant
        ([-1, -1, -1], 1, -1, [0, 0, 0]),  # constant
        ([1, -1, 1, -1, 1, -1], 1, -1, [1, -1, 1, -1, 1, -1]),  # alternating
        ([1, -1, 1, -1, 1, -1], 1, -1, [1, -1, 1, -1, 1, -1]),  # alternating
        ([1, -1, 1, -1, 1, -1], 1, -1, [1, -1, 1, -1, 1, -1]),  # alternating
        ([1, 0, 0, 1, 0, 0, 1, 0, 0], 3, 1, [0, 0, 0, 0, 0, 0, 0, 0, 0]),  # periodic
        ([4, 0, 0, 4, 0, 0, 4, 0, 0], 3, 1, [0, 0, 0, 0, 0, 0, 0, 0, 0]),  # periodic
        (
            [4, 0, 0, 4, 0, 0, 4, 0, 0],
            1,
            -1,
            [
                1.41421356,
                -0.70710678,
                -0.70710678,
                1.41421356,
                -0.70710678,
                -0.70710678,
                1.41421356,
                -0.70710678,
                -0.70710678,
            ],
        ),  # periodic
        (
            [9, 0, 0, 9, 0, 0, 9, 0, 0],
            1,
            -1,
            [
                1.41421356,
                -0.70710678,
                -0.70710678,
                1.41421356,
                -0.70710678,
                -0.70710678,
                1.41421356,
                -0.70710678,
                -0.70710678,
            ],
        ),  # periodic
        ([-1, 0, -1, 0, -1, 0], 2, 1, [0, 0, 0, 0, 0, 0]),  # periodic, negative
    ],
)
def test_z_score(ts, periodicity, max_periods, expected):
    """Test the Z-Score detrending by design."""
    result = detrend(
        array(ts),
        method="z_score",
        periodicity=periodicity,
        max_periods=max_periods,
    )
    assert_allclose(result, array(expected))


@pytest.mark.parametrize(
    "param, val",
    [
        ("periodicity", 0),  # non-positive
        ("periodicity", -1),  # non-positive
        ("periodicity", 2.0),  # float
        ("max_periods", -2),  # non-positive, nor -1
        ("max_periods", 5.0),  # float
    ],
)
def test_faulty_kwargs(time_series, param, val):
    """Test the Z-Score detrending with faulty kwargs."""
    with pytest.raises(ValueError):
        detrend(time_series, "z_score", **{param: val})


@pytest.mark.parametrize(
    "ts_len, period, max_periods, max_periods_larger",
    [
        (10, 1, 1, False),  # max_p*period+1 = 2 < 10
        (10, 1, 8, False),  # max_p*period+1 = 9 < 10
        (10, 1, 9, True),  # max_p*period+1 = 10 !< 10
        (10, 1, 10, True),  # max_p*period+1 = 11 !< 10
        (10, 2, 1, False),  # max_p*period+1 = 5 < 10
        (10, 2, 4, False),  # max_p*period+1 = 9 < 10
        (10, 2, 5, True),  # max_p*period+1 = 11 !< 10
    ],
)
def test_all_period_detection(ts_len, period, max_periods, max_periods_larger, caplog):
    """Test the Z-Score detrending detection that
    max_periods is larger than available periods."""
    time_series = array(range(ts_len))
    detrend(time_series, "z_score", periodicity=period, max_periods=max_periods)
    logging.getLogger().setLevel(logging.DEBUG)
    assert (
        "is larger than or equal to the available periods" in caplog.text
    ) == max_periods_larger


@pytest.mark.parametrize(
    "ts_len, period, raises",
    [
        (10, 2, False),  # 2*period+1 = 5 <= 10
        (10, 4, False),  # 2*period+1 = 9 <= 10
        (10, 5, True),  # 2*period+1 = 11 !<= 10
        (10, 6, True),  # 2*period+1 = 13 !<= 10
        (3, 1, False),  # 2*period+1 = 3 <= 3
        (3, 2, True),  # 2*period+1 = 5 !<= 3
    ],
)
def test_periodicity_too_large(ts_len, period, raises):
    """Test the Z-Score detrending with periodicity too large."""
    time_series = array(range(ts_len))
    if raises:
        with pytest.raises(ValueError):
            detrend(time_series, "z_score", periodicity=period)
    else:
        detrend(time_series, "z_score", periodicity=period)

def test_z_score_with_periodicity():
    """Test Z-Score detrending with periodicity on a sawtooth wave."""
    # Create a sawtooth wave with period 24
    periodicity = 24
    ts_length = 240
    ts = array(mod(arange(ts_length), periodicity), dtype=float)
    # Add small noise
    rng = default_rng(25689)
    ts += rng.normal(0., 0.01, ts_length)

    # Apply Z-Score detrending with periodicity=24
    ts_detrended = detrend(ts, method="z_score", periodicity=periodicity)

    # Check that the detrended time series has the expected properties
    # Each phase should have a mean close to 0 and a standard deviation close to 1
    for phase in range(periodicity):
        phase_indices = arange(phase, ts_length, periodicity)
        phase_values = ts_detrended[phase_indices]
        # The mean should be close to 0 (but not exactly 0 due to the leave-one-out approach)
        assert -0.2 < mean(phase_values) < 0.2
        # The standard deviation should be close to 1 (but not exactly 1 due to the leave-one-out approach)
        assert 0.8 < std(phase_values) < 1.6


def test_z_score_with_zero_std():
    """Test Z-Score detrending with zero standard deviation."""
    # Create a time series with constant values at each phase
    periodicity = 3
    ts_length = 9
    ts = array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=float)

    # Apply Z-Score detrending with periodicity=3
    ts_detrended = detrend(ts, method="z_score", periodicity=periodicity)

    # Check that the detrended time series is all zeros
    # When the standard deviation is 0, the detrended value should be 0
    assert_allclose(ts_detrended, zeros(ts_length))