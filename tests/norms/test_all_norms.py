"""Programmatic tests for all norms in norms module."""

import pytest
from numpy import ndarray

from delaynet import normalise
from delaynet.norms import __all_norms_names__


def test_all_norms(norm, fmri_time_series):
    """Test all norms with fMRI time series."""
    result = normalise(fmri_time_series, norm=norm.__name__)
    assert isinstance(result, ndarray)


@pytest.mark.parametrize("norm_str", __all_norms_names__.keys())
def test_all_norm_querying(norm_str, random_time_series):
    """Test querying all norms."""
    result = normalise(random_time_series, norm=norm_str)
    assert isinstance(result, ndarray)
