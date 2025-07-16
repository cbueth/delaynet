"""Tests for the connectivity function."""

import pytest
import io
import sys
from numpy import corrcoef
from delaynet.connectivity import connectivity, show_connectivity_metrics


def test_connectivity_with_string_metric(two_time_series):
    """Test connectivity with string metric.
    All metrics are programmatically tested in
    connectivities/test_all_connectivities.py.
    """
    ts1, ts2 = two_time_series
    result = connectivity(ts1, ts2, "lc", lag_steps=5)
    assert isinstance(result, tuple)
    assert len(result) == 2


@pytest.mark.parametrize(
    "metric",
    [
        lambda ts1, ts2, lag_steps=None: (corrcoef(ts1, ts2)[0, 1], 2),
        lambda ts1, ts2, lag_steps=None: (1.0, 1),
    ],
)
def test_connectivity_with_valid_metric(two_time_series, metric):
    """Test connectivity and pass metric as function."""
    ts1, ts2 = two_time_series
    result = connectivity(ts1, ts2, metric, lag_steps=5)
    assert isinstance(result, tuple)
    assert len(result) == 2


@pytest.mark.parametrize(
    "invalid_metric",
    [
        # Callable
        lambda ts1, ts2, lag_steps=None: "invalid",  # invalid shape, wrong type
        lambda ts1, ts2, lag_steps=None: 123,  # invalid shape, wrong type
        lambda ts1, ts2, lag_steps=None: 1.0,  # invalid shape, wrong type
        lambda ts1, ts2, lag_steps=None: (123, 123),  # valid shape, wrong type
        lambda ts1, ts2, lag_steps=None: (123, 1.0),  # valid shape, wrong type
        lambda ts1, ts2, lag_steps=None: (1, 1, 1),  # invalid shape
        lambda ts1, ts2, lag_steps=None: [1, 1],  # valid shape, wrong type
        lambda ts1, ts2, lag_steps=None: [0.1, 0.1],  # valid shape, wrong type
        lambda ts1, ts2, lag_steps=None: [1],  # invalid shape
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
        connectivity(ts1, ts2, invalid_metric, lag_steps=5)


def test_connectivity_ts_positional_only(two_time_series):
    """Test connectivity with time series as keyword arguments."""
    ts1, ts2 = two_time_series

    with pytest.raises(
        TypeError, match="missing 1 required positional argument: 'ts2'"
    ):
        connectivity(ts1, ts2=ts2, metric="lc", lag_steps=5)
    with pytest.raises(
        TypeError, match="missing 1 required positional argument: 'ts2'"
    ):
        connectivity(ts2, ts1=ts1, metric="lc", lag_steps=5)
    with pytest.raises(
        TypeError, match="missing 2 required positional arguments: 'ts1' and 'ts2'"
    ):
        connectivity(metric="lc", lag_steps=5)
    with pytest.raises(
        TypeError, match="missing 2 required positional arguments: 'ts1' and 'ts2'"
    ):
        connectivity(metric="lc", ts1=ts1, ts2=ts2, lag_steps=5)


def test_show_connectivity_metrics():
    """Test the show_connectivity_metrics function."""
    # Capture stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Call the function
    show_connectivity_metrics()

    # Restore stdout
    sys.stdout = sys.__stdout__

    # Check the output
    output = captured_output.getvalue()
    assert "Available connectivity metrics:" in output
    assert "Metric:" in output
    assert "Aliases:" in output

    # Check for some common metrics
    assert "linear_correlation" in output or "lc" in output
    assert "mutual_information" in output or "mi" in output
