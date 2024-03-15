"""Test example data generator for preparation module."""

import pytest
from numpy import array_equal
from numpy.random import default_rng

from delaynet.preparation.data_generator import gen_rand_data


@pytest.mark.parametrize("n_nodes", [1, 2, 10])
@pytest.mark.parametrize("ts_len", [1, 2, 10])
@pytest.mark.parametrize("l_dens", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("wm_min_max", [(0.0, 1.0), (0.5, 1.5), (1.0, 2.0)])
@pytest.mark.parametrize("rng", [None, 0, default_rng(0)])
def test_gen_rand_data(n_nodes, ts_len, l_dens, wm_min_max: tuple[float, float], rng):
    """Test the gen_rand_data function."""
    am, wm, ts = gen_rand_data(ts_len, n_nodes, l_dens, wm_min_max, rng)
    assert am.shape == (n_nodes, n_nodes)
    assert wm.shape == (n_nodes, n_nodes)
    assert ts.shape == (n_nodes, ts_len)
    assert am.dtype == bool
    assert wm.dtype == float
    assert ts.dtype == float
    assert am.sum() <= n_nodes**2 - n_nodes
    assert 0 <= wm.sum() <= (n_nodes**2 - n_nodes) * wm_min_max[1]
    assert ts.sum() >= 0.0


@pytest.mark.parametrize(
    "rng, is_stable", [(None, False), (0, True), (default_rng(0), False)]
)
def test_gen_rand_data_stability(rng, is_stable):
    """Test if the gen_rand_data function is stable.
    This is, if the same random seed is used, the output should be the same.
    But using the same random generator twice should not produce the same output.
    """
    n_nodes = 10
    ts_len = 10
    l_dens = 0.5
    wm_min_max = (0.5, 1.5)
    am1, wm1, ts1 = gen_rand_data(ts_len, n_nodes, l_dens, wm_min_max, rng)
    am2, wm2, ts2 = gen_rand_data(ts_len, n_nodes, l_dens, wm_min_max, rng)
    assert array_equal(am1, am2) == is_stable
    assert array_equal(wm1, wm2) == is_stable
    assert array_equal(ts1, ts2) == is_stable


@pytest.mark.parametrize(
    "n_nodes, ts_len, l_dens, wm_min_max, rng, error",
    [
        (0, 1, 0.0, (0.0, 1.0), None, ValueError),  # n_nodes is not a positive integer
        (1.0, 1, 0.0, (0.0, 1.0), None, ValueError),  # n_nodes is not an integer
        (1, 0, 0.0, (0.0, 1.0), None, ValueError),  # ts_len is not a positive integer
        (1, 1.0, 0.0, (0.0, 1.0), None, ValueError),  # ts_len is not an integer
        (1, 1, -0.1, (0.0, 1.0), None, ValueError),  # l_dens is not in [0, 1]
        (1, 1, 1.1, (0.0, 1.0), None, ValueError),  # l_dens is not in [0, 1]
        (1, 1, 0.0, (1.0, 0.0), None, ValueError),  # wm_min_max[0] > wm_min_max[1]
        (1, 1, 0.0, (0.0, 1.0), "invalid", TypeError),
        # rng is not `None`, an `int` or a `numpy.random.Generator`
    ],
)
def test_gen_rand_data_invalid_inputs(n_nodes, ts_len, l_dens, wm_min_max, rng, error):
    """Test the gen_rand_data function with invalid inputs."""
    with pytest.raises(error):
        gen_rand_data(ts_len, n_nodes, l_dens, wm_min_max, rng)
