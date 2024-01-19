"""Tests for the normalisation function."""
import pytest
from numpy import array, array_equal, ndarray
from delaynet.normalisation import normalise


@pytest.fixture
def time_series():
    return array([1, 2, 3, 4, 5])


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
