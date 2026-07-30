"""find_optimal_lag wrapper overhead, measured with a lightweight metric."""

import numpy as np
import pytest

from delaynet.preparation.data_generator import gen_delayed_causal_network
from delaynet.utils.lag_steps import find_optimal_lag

from .conftest import LAG_STEPS

_ts = np.ascontiguousarray(
    gen_delayed_causal_network(ts_len=500, n_nodes=2, l_dens=0.5, rng=0)[2].T
)
TS1, TS2 = _ts[:, 0], _ts[:, 1]


def _l2_metric(ts1, ts2, lag):
    return float(np.sum((ts1[:-lag] - ts2[lag:]) ** 2))


@pytest.mark.benchmark(group="utils")
def test_find_optimal_lag(benchmark):
    benchmark(find_optimal_lag, _l2_metric, TS1, TS2, list(range(1, LAG_STEPS + 1)))
