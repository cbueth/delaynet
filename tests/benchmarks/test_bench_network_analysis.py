"""Network analysis metric benchmarks: global efficiency, transitivity, etc."""

import numpy as np
import pytest

from delaynet.network_analysis import global_efficiency, transitivity, link_density

N_NODES = 50
DENSITY = 0.3

_rng = np.random.default_rng(152)
_adj = (_rng.uniform(0, 1, (N_NODES, N_NODES)) < DENSITY).astype(float)
np.fill_diagonal(_adj, 0)

# ensure at least some structure: a directed chain
for i in range(N_NODES - 1):
    _adj[i, i + 1] = 1.0

# symmetric version for undirected metrics
_adj_sym = (_adj + _adj.T) > 0
_adj_sym = _adj_sym.astype(float)


@pytest.mark.benchmark(group="network-analysis")
@pytest.mark.parametrize(
    "metric_func, kwargs, data",
    [
        (global_efficiency, {"directed": True}, _adj),
        (global_efficiency, {"directed": False}, _adj_sym),
        (transitivity, {}, _adj_sym),
        (link_density, {"directed": True}, _adj),
    ],
    ids=[
        "global_efficiency-dir",
        "global_efficiency-undir",
        "transitivity",
        "link_density",
    ],
)
def test_network_metric(benchmark, metric_func, kwargs, data):
    benchmark(metric_func, data, **kwargs)
