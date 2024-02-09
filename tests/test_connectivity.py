"""Tests for the connectivity function."""

import pytest
from numpy import corrcoef
from delaynet.connectivity import connectivity


def test_connectivity_with_string_metric(two_time_series):
    """Test connectivity with string metric.
    All metrics are programmatically tested in
    connectivities/test_all_connectivities.py.
    """
    ts1, ts2 = two_time_series
    result = connectivity(ts1, ts2, "lc")
    assert isinstance(result, (float, tuple))


@pytest.mark.parametrize(
    "metric",
    [
        lambda ts1, ts2: corrcoef(ts1, ts2)[0, 1],
        lambda ts1, ts2: 1.0,
        lambda ts1, ts2: (1.0, 1),
    ],
)
def test_connectivity_with_valid_metric(two_time_series, metric):
    """Test connectivity and pass metric as function."""
    ts1, ts2 = two_time_series
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
def test_connectivity_with_invalid_metric(two_time_series, invalid_metric):
    """Test connectivity and pass invalid metric."""
    ts1, ts2 = two_time_series

    with pytest.raises(ValueError):
        connectivity(ts1, ts2, invalid_metric)
