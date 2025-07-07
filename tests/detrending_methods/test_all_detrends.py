"""Programmatic tests for all detrending methods in the detrending_methods module."""

import pytest
from numpy import ndarray

from delaynet import detrend
from delaynet.detrending_methods import __all_detrending_names__


def test_all_detrending_methods(detrending_function, fmri_time_series):
    """Test all detrending methods with fMRI time series."""
    result = detrend(fmri_time_series, method=detrending_function.__name__)
    assert isinstance(result, ndarray)


@pytest.mark.parametrize("detrend_str", __all_detrending_names__.keys())
def test_all_detrend_querying(detrend_str, random_time_series):
    """Test querying all detrending methods with random time series."""
    result = detrend(random_time_series, method=detrend_str)
    assert isinstance(result, ndarray)
