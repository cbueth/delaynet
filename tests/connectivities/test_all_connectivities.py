"""Programmatic tests for all connectivities in connectivities module."""

import pytest

from delaynet import connectivity
from delaynet.connectivities import __all_connectivity_metrics_names__


def test_all_connectivities(connectivity_metric, two_fmri_time_series):
    """Test all connectivity metrics with two fMRI time series."""
    ts1, ts2 = two_fmri_time_series
    kwargs = {}
    if connectivity_metric.is_entropy_like:
        kwargs["symbolic_bins"] = 50
    result = connectivity(ts1, ts2, metric=connectivity_metric.__name__, **kwargs)
    assert isinstance(result, (float, tuple))


@pytest.mark.parametrize("metric_str", __all_connectivity_metrics_names__.keys())
def test_all_conn_querying(metric_str, two_random_time_series):
    """Test querying all connectivity metrics."""
    ts1, ts2 = two_random_time_series
    kwargs = {}
    if __all_connectivity_metrics_names__[metric_str].is_entropy_like:
        kwargs["symbolic_bins"] = 50
    result = connectivity(ts1, ts2, metric=metric_str, **kwargs)
    assert isinstance(result, (float, tuple))
