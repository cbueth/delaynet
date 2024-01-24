"""Tests for the connectivity function."""
import pytest
from numpy import array, corrcoef
from delaynet.connectivity import connectivity


@pytest.fixture
def time_series():
    ts1 = array([1, 2, 3, 4, 5])
    ts2 = array([5, 4, 3, 2, 1])
    return ts1, ts2


def test_connectivity_with_string_metric(time_series):
    ts1, ts2 = time_series
    result = connectivity(ts1, ts2, "gc")
    assert isinstance(result, (float, tuple))


@pytest.mark.parametrize(
    "metric",
    [
        lambda ts1, ts2: corrcoef(ts1, ts2)[0, 1],
        lambda ts1, ts2: 1.0,
        lambda ts1, ts2: (1.0, 1),
    ],
)
def test_connectivity_with_valid_metric(time_series, metric):
    ts1, ts2 = time_series
    result = connectivity(ts1, ts2, metric)
    assert isinstance(result, (float, tuple))


@pytest.mark.parametrize(
    "invalid_metric",
    [
        # Callable
        lambda ts1, ts2: "invalid",  # valid shape, wrong type
        lambda ts1, ts2: 123,  # valid shape, wrong type
        lambda ts1, ts2: (123, 123),  # valid shape, wrong type
        lambda ts1, ts2: (1, 1, 1),  # invalid shape
        lambda ts1, ts2: [1, 1],  # valid shape, wrong type
        lambda ts1, ts2: [0.1, 0.1],  # valid shape, wrong type
        lambda ts1, ts2: [1],  # invalid shape
        # Unknown string
        "invalid",
        # Not-Callable
        123,
        None,
    ],
)
def test_connectivity_with_invalid_metric(time_series, invalid_metric):
    ts1, ts2 = time_series

    with pytest.raises(ValueError):
        connectivity(ts1, ts2, invalid_metric)
