"""Module for test fixtures available for all test files."""

import pytest
from numpy import array

from delaynet.preparation.data_generator import gen_rand_data, gen_frmi
from delaynet.connectivities import __all_connectivity_metrics__
from delaynet.norms import __all_norms__


# ******************************************************************************
# Dynamic methods
# ******************************************************************************


@pytest.fixture(
    scope="session",
    params=__all_connectivity_metrics__,
)
def connectivity_metric(request):
    """Return a connectivity metric function."""
    return request.param


@pytest.fixture(
    scope="session",
    params=__all_norms__,
)
def norm(request):
    """Return a norm function."""
    return request.param


# ******************************************************************************
# Random data
# ******************************************************************************


@pytest.fixture(scope="session")
def two_random_time_series():
    """Return two random time series."""
    _, _, ts = gen_rand_data(ts_len=1000, n_nodes=2, l_dens=0.5, wm_min_max=(0.5, 1.5))
    return ts[:, 0], ts[:, 1]


@pytest.fixture(scope="session")
def random_time_series(two_random_time_series):
    """Return random time series."""
    return two_random_time_series[0]


@pytest.fixture(scope="session")
def two_fmri_time_series():
    """Return two random fMRI time series."""
    lin_coupl = 1.0
    coupling = 1.0
    ts = gen_frmi(
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
    ts1 = array([1, 2, 3, 4, 5])
    ts2 = array([5, 4, 3, 2, 1])
    return ts1, ts2
