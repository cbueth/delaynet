"""Tests for the Z-Score norm."""

import pytest
from numpy import array, array_equal

from delaynet import normalise, logging


@pytest.mark.parametrize(
    "ts, periodicity, max_periods, expected",
    [
        ([0, 0, 0], 1, 1, [0, 0, 0]),  # zero
        ([1, 1, 1], 1, 1, [1, 1, 1]),  # constant
        ([-1, -1, -1], 1, 1, [-1, -1, -1]),  # constant
        ([1, -1, 1, -1, 1, -1], 1, 1, [1, -1, 1, -1, 1, -1]),  # alternating
        ([1, -1, 1, -1, 1, -1], 1, 2, [1, -1, 1, -1, 1, -1]),  # alternating
        ([1, -1, 1, -1, 1, -1], 1, -1, [1, -1, 1, -1, 1, -1]),  # alternating
        ([1, 0, 0, 1, 0, 0, 1, 0, 0], 3, 1, [1, 0, 0, 1, 0, 0, 1, 0, 0]),  # periodic
        ([4, 0, 0, 4, 0, 0, 4, 0, 0], 3, 1, [4, 0, 0, 4, 0, 0, 4, 0, 0]),  # periodic
        ([4, 0, 0, 4, 0, 0, 4, 0, 0], 1, 1, [4, -1, -1, 4, -1, -1, 4, -1, 0]),
        ([-1, 0, -1, 0, -1, 0], 2, 1, [-1, 0, -1, 0, -1, 0]),  # periodic, negative
    ],
)
def test_z_score(ts, periodicity, max_periods, expected):
    """Test the Z-Score norm by design."""
    result = normalise(
        array(ts),
        norm="z_score",
        periodicity=periodicity,
        max_periods=max_periods,
    )
    assert array_equal(result, array(expected))


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
    """Test the Z-Score norm with faulty kwargs."""
    with pytest.raises(ValueError):
        normalise(time_series, "z_score", **{param: val})


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
    """Test the Z-Score norm detection that
    max_periods is larger than available periods."""
    time_series = array(range(ts_len))
    normalise(time_series, "z_score", periodicity=period, max_periods=max_periods)
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
    """Test the Z-Score norm with periodicity too large."""
    time_series = array(range(ts_len))
    if raises:
        with pytest.raises(ValueError):
            normalise(time_series, "z_score", periodicity=period)
    else:
        normalise(time_series, "z_score", periodicity=period)
