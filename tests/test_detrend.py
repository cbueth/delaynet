"""Tests for the detrending function."""

import pytest
from numpy import array, array_equal, ndarray
from delaynet.detrending import detrend


def test_detrend_with_string_metric(time_series):
    """Test detrending with the string method.
    All detrends are programmatically tested in detrending_methods/test_all_detrends.py.
    """
    if time_series.ndim > 1:
        # For 2D arrays, test with axis=0 and axis=1
        result_axis0 = detrend(time_series, method="id", axis=0)
        result_axis1 = detrend(time_series, method="id", axis=1)
        assert array_equal(result_axis0, time_series)
        assert array_equal(result_axis1, time_series)
    else:
        assert array_equal(detrend(time_series, method="id"), time_series)


@pytest.mark.parametrize(
    "detrending_function",
    [
        lambda ts: ts,
        lambda ts, a=1: ts * a,
        lambda ts, a=1, b=0: ts * a + b,
    ],
)
def test_detrend_with_valid_detrend(time_series, detrending_function):
    """Test detrending and pass detrending_function as function."""
    if time_series.ndim > 1:
        # For 2D arrays, test with axis=0
        result = detrend(time_series, detrending_function, axis=0)
        assert isinstance(result, ndarray)
    else:
        result = detrend(time_series, detrending_function)
        assert isinstance(result, ndarray)


# check that when passing a detrending method, the decorator detrending_function
# is applied
@pytest.mark.parametrize(
    "invalid_detrend",
    [
        # Callable
        lambda ts: "invalid",  # wrong output type
        # Unknown string
        "invalid",
        # Not-Callable
        123,
        None,
    ],
)
def test_detrend_with_invalid_detrend_type(time_series, invalid_detrend):
    """Test detrending and pass invalid detrending function type."""
    with pytest.raises(ValueError):
        detrend(time_series, method=invalid_detrend)


def test_detrend_kwargs_unknown(time_series):
    """Test detrending with unknown keyword argument."""
    with pytest.raises(TypeError, match="got an unexpected keyword argument 'b'"):
        if time_series.ndim > 1:
            detrend(time_series, method="id", axis=0, b=2)
        else:
            detrend(time_series, method="id", b=2)


def test_detrend_ts_positional_only(time_series):
    """Test detrending with time series as keyword argument."""
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'ts'"):
        detrend(ts=time_series, method="id")
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'ts'"):
        detrend(method="id", ts=time_series)


@pytest.mark.parametrize(
    "invalid_time_series",
    [
        123,  # not an ndarray
        None,  # not an ndarray
        array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),  # 3D
    ],
)
def test_detrend_invalid_time_series(invalid_time_series):
    """Test detrending with invalid time series."""
    if hasattr(invalid_time_series, "ndim") and invalid_time_series.ndim > 1:
        # For 3D arrays, we expect ValueError about missing axis parameter
        with pytest.raises(ValueError, match="axis.*kwarg must be specified"):
            detrend(invalid_time_series, method="id")
    else:
        # For non-ndarray types, we expect TypeError
        with pytest.raises(TypeError):
            detrend(invalid_time_series, method="id")


@pytest.mark.parametrize(
    "empty_time_series_array",
    [
        array([]),  # 1D empty
        array([[]]),  # 2D empty
        array([[], []]),  # 2D empty
    ],
)
def test_detrend_empty_time_series(empty_time_series_array):
    """Test detrending with empty time series."""
    with pytest.raises(ValueError):
        detrend(empty_time_series_array, method="id")
