"""Module for test fixtures available for all test files."""

import pytest
from numpy import array

from delaynet.preparation.data_generator import gen_delayed_causal_network, gen_fmri
from delaynet.connectivities import __all_connectivity_metrics_names_simple__
from delaynet.detrending_methods import __all_detrending__
from delaynet.network_analysis.metrics import (
    link_density,
    reciprocity,
    betweenness_centrality,
    global_efficiency,
    transitivity,
    eigenvector_centrality,
    isolated_nodes_inbound,
    isolated_nodes_outbound,
)


CONN_METRICS = {
    name: {
        "shorthands": value,
        "additional_kwargs": [{}],
    }
    for name, value in __all_connectivity_metrics_names_simple__.items()
}
CONN_METRICS["mutual_information"]["additional_kwargs"] = [
    {"approach": "discrete"},
    {"approach": "metric"},
    {"approach": "kernel", "bandwidth": 0.3, "kernel": "box"},
    {"approach": "renyi", "alpha": 1.0},
]
CONN_METRICS["transfer_entropy"]["additional_kwargs"] = [
    {"approach": "discrete"},
    {"approach": "metric"},
]
# list of all shorthands, only testing the first set of kwargs
CONN_METRICS_SHORTHANDS = [
    (shorthand, metric["additional_kwargs"][0])
    for metric in CONN_METRICS.values()
    for shorthand in metric["shorthands"]
]
# list of all kwargs, only testing the first shorthand
CONN_METRICS_KWARGS = [
    (metric["shorthands"][0], kwargs)
    for metric in CONN_METRICS.values()
    for kwargs in metric["additional_kwargs"]
]

# ******************************************************************************
# Dynamic methods
# ******************************************************************************


@pytest.fixture(
    scope="session",
    params=CONN_METRICS_SHORTHANDS,
    ids=[shorthand for shorthand, _ in CONN_METRICS_SHORTHANDS],
)
def connectivity_metric_shorthand(request):
    """Return a connectivity metric shorthand and its additional kwargs.
    All shorthands are tested with the first set of kwargs."""
    return request.param


@pytest.fixture(
    scope="session",
    params=CONN_METRICS_KWARGS,
    ids=[f"{shorthand}{i}" for i, (shorthand, _) in enumerate(CONN_METRICS_KWARGS)],
)
def connectivity_metric_kwargs(request):
    """Return a connectivity metric shorthand and its additional kwargs.
    All kwargs are tested with the first shorthand."""
    return request.param


@pytest.fixture(
    scope="session",
    params=__all_detrending__,
)
def detrending_function(request):
    """Return a detrending function."""
    return request.param


# ******************************************************************************
# Dynamic metrics for network analysis
# ******************************************************************************

# List all network analysis metrics with their output kind (scalar vs vector)
# - scalar: returns a single number (float/int) when not normalised
# - vector: returns a 1-D numpy array
NETWORK_METRICS_AND_KIND = [
    (link_density, "scalar"),
    (reciprocity, "scalar"),
    (transitivity, "scalar"),
    (global_efficiency, "scalar"),
    (isolated_nodes_inbound, "scalar"),
    (isolated_nodes_outbound, "scalar"),
    (betweenness_centrality, "vector"),
    (eigenvector_centrality, "vector"),
]


@pytest.fixture(
    scope="session",
    params=NETWORK_METRICS_AND_KIND,
    ids=[f"{fn.__name__}:{kind}" for fn, kind in NETWORK_METRICS_AND_KIND],
)
def network_metric_and_kind(request):
    """Provide each network metric together with its output kind.

    Returns a tuple (fn, kind) where fn is the metric function and kind is
    either "scalar" or "vector" indicating the raw (non-normalised) output type.
    """
    return request.param


# ******************************************************************************
# Random data
# ******************************************************************************


@pytest.fixture(scope="session")
def two_random_time_series():
    """Return two random time series. Fixed seed."""
    _, _, ts = gen_delayed_causal_network(
        ts_len=1000, n_nodes=2, l_dens=0.5, wm_min_max=(0.5, 1.5), rng=0
    )
    return ts[0, :], ts[1, :]


@pytest.fixture(scope="session")
def random_time_series(two_random_time_series):
    """Return random time series."""
    return two_random_time_series[0]


@pytest.fixture(scope="session")
def two_fmri_time_series():
    """Return two random fMRI time series. Fixed seed."""
    lin_coupl = 1.0
    coupling = 1.0
    ts = gen_fmri(
        ts_len=10000,
        downsampling_factor=10,
        time_resolution=0.2,
        coupling_strength=coupling,
        noise_initial_sd=1.0,
        noise_final_sd=0.05,
        rng=0,
    )
    ts[2:, 1] += lin_coupl * coupling * ts[:-2, 0]
    return ts[:, 0], ts[:, 1]


@pytest.fixture(scope="session")
def fmri_time_series(two_fmri_time_series):
    """Return fMRI time series."""
    return two_fmri_time_series[0]


# ******************************************************************************
# Static data
# ******************************************************************************


@pytest.fixture(
    scope="module",
    params=[
        array([1, 2, 3, 4, 5]),
        array([[1, 2, 3], [4, 5, 6]]),
        array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    ],
)
def time_series(request):
    """Return a time series of different shapes. 1D and 2D."""
    return request.param


@pytest.fixture(scope="module")
def two_time_series():
    """Return two time series."""
    ts1 = array([1, 2, 3, 4, 5, 6, 7, 8])
    ts2 = array([8, 7, 6, 5, 4, 3, 2, 1])
    return ts1, ts2
