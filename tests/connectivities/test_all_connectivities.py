"""Programmatic tests for all connectivities in connectivities module."""

from delaynet import connectivity


def test_all_connectivities(connectivity_metric_kwargs, two_fmri_time_series):
    """Test all connectivity metrics with two fMRI time series."""
    connectivity_metric, kwargs = connectivity_metric_kwargs
    ts1, ts2 = two_fmri_time_series
    result = connectivity(ts1, ts2, metric=connectivity_metric, **kwargs)
    assert isinstance(result, (float, tuple))


def test_all_conn_querying(connectivity_metric_shorthand, two_random_time_series):
    """Test querying all connectivity metrics."""
    connectivity_metric, kwargs = connectivity_metric_shorthand
    ts1, ts2 = two_random_time_series
    result = connectivity(ts1, ts2, metric=connectivity_metric, **kwargs)
    assert isinstance(result, (float, tuple))
