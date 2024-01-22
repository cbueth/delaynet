"""Module for test fixtures available for all test files."""

import pytest

from delaynet.preparation.data_generator import gen_rand_data


@pytest.fixture(scope="session", params=range(5))
def random_data():
    """Return random data, five timeseries of 1000 samples each."""
    return gen_rand_data(5, 1000, 0.5, (0.5, 1.5), 0)
