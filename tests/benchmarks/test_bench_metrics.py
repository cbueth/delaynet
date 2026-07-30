"""Isolated single-pair metric benchmarks (no pair-loop overhead)."""

import numpy as np
import pytest

from delaynet.connectivities.continuous_ordinal_patterns import random_patterns
from delaynet.connectivities.granger import gt_multi_lag
from delaynet.connectivities.gravity import gravity
from delaynet.connectivities.linear_correlation import linear_correlation
from delaynet.connectivities.rank_correlation import rank_correlation
from delaynet.connectivities.mutual_information import mutual_information
from delaynet.connectivities.transfer_entropy import transfer_entropy
from delaynet.preparation.data_generator import gen_delayed_causal_network

from .conftest import LAG_STEPS

_ts = np.ascontiguousarray(
    gen_delayed_causal_network(ts_len=200, n_nodes=2, l_dens=0.5, rng=0)[2].T
)
TS1, TS2 = _ts[:, 0], _ts[:, 1]

_disc = np.random.default_rng(0).integers(0, 4, size=(200, 2))
DTS1, DTS2 = _disc[:, 0], _disc[:, 1]

N_TESTS = 2


@pytest.mark.benchmark(group="metrics")
@pytest.mark.parametrize(
    "metric_func,kwargs",
    [
        (linear_correlation, {}),
        (rank_correlation, {}),
    ],
    ids=["lc", "rc"],
)
def test_continuous_metric(benchmark, metric_func, kwargs):
    benchmark(metric_func, TS1, TS2, lag_steps=LAG_STEPS, **kwargs)


@pytest.mark.benchmark(group="metrics")
def test_gravity(benchmark):
    benchmark(gravity, TS1, TS2, lag_steps=LAG_STEPS, n_tests=N_TESTS)


@pytest.mark.benchmark(group="metrics")
def test_granger_f_test(benchmark):
    ts = gen_delayed_causal_network(ts_len=50, n_nodes=5, l_dens=0.3, rng=0)[2].T
    ts1, ts2 = ts[:, 0], ts[:, 1]
    benchmark(gt_multi_lag, ts1, ts2, lag_steps=LAG_STEPS)


@pytest.mark.benchmark(group="metrics")
def test_ordinal_patterns(benchmark):
    ts = gen_delayed_causal_network(ts_len=50, n_nodes=5, l_dens=0.3, rng=0)[2].T
    ts1, ts2 = ts[:, 0], ts[:, 1]
    random_patterns(ts1, ts2, p_size=3, num_rnd_patterns=2, lag_steps=1)
    benchmark(random_patterns, ts1, ts2, p_size=3, num_rnd_patterns=2, lag_steps=1)


@pytest.mark.benchmark(group="metrics")
@pytest.mark.parametrize(
    "metric_func",
    [mutual_information, transfer_entropy],
    ids=["mi", "te"],
)
def test_discrete_metric(benchmark, metric_func):
    benchmark(
        metric_func,
        DTS1,
        DTS2,
        approach="discrete",
        lag_steps=LAG_STEPS,
        n_tests=N_TESTS,
    )
