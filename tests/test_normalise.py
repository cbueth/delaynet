"""Tests for the normalisation function."""
import pytest
from numpy import array, array_equal, ndarray
from delaynet.normalisation import normalise


def test_normalise_with_string_metric(time_series):
    assert array_equal(normalise(time_series, norm="id"), time_series)


@pytest.mark.parametrize(
    "norm",
    [
        lambda ts: ts,
        lambda ts, a=1: ts * a,
        lambda ts, a=1, b=0: ts * a + b,
    ],
)
def test_normalise_with_valid_norm(time_series, norm):
    result = normalise(time_series, norm)
    assert isinstance(result, ndarray)


# check that when passing a norm, the decorator norm is applied
@pytest.mark.parametrize(
    "invalid_norm",
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
def test_normalise_with_invalid_norm_type(time_series, invalid_norm):
    with pytest.raises(ValueError):
        normalise(time_series, norm=invalid_norm)


def test_normalise_kwargs_unknown(time_series):
    with pytest.raises(TypeError, match="got an unexpected keyword argument 'b'"):
        normalise(time_series, norm="id", b=2)


@pytest.mark.parametrize(
    "invalid_time_series",
    [
        123,  # not an ndarray
        None,  # not an ndarray
        array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),  # 3D
        array([]),  # 1D empty
        array([[]]),  # 2D empty
    ],
)
def test_normalise_invalid_time_series(invalid_time_series):
    with pytest.raises(TypeError):
        normalise(invalid_time_series, norm="id")


@pytest.mark.parametrize(
    "empty_time_series_array",
    [
        array([]),  # 1D empty
        array([[]]),  # 2D empty
        array([[], []]),  # 2D empty
    ],
)
def test_normalise_empty_time_series(empty_time_series_array):
    with pytest.raises(ValueError):
        normalise(empty_time_series_array, norm="id")
