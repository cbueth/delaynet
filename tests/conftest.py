"""Module for test fixtures available for all test files."""

import pytest
from numpy import array

from delaynet.preparation.data_generator import gen_rand_data


@pytest.fixture(scope="session", params=range(5))
def random_data():
    """Return random data, five timeseries of 1000 samples each."""
    return gen_rand_data(5, 1000, 0.5, (0.5, 1.5), 0)


@pytest.fixture(
    scope="module",
    params=[
        array([1, 2, 3, 4, 5]),
        array([[1, 2, 3], [4, 5, 6]]),
        array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    ],
)
def time_series(request):
    return request.param
