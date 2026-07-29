"""Shared fixtures and constants for benchmarks."""

import numpy as np
import pytest

from delaynet.preparation.data_generator import gen_delayed_causal_network

TIME_POINTS = 200
LAG_STEPS = 5
NODE_SIZES = [10, 20, 40, 50]
SMALL_NODES = [10, 20]

CONTINUOUS_METRICS = ["lc", "rc", "gc", "gv", "cop"]


def gen_continuous(n_nodes: int, rng: int = 19425) -> np.ndarray:
    _, _, ts = gen_delayed_causal_network(
        ts_len=TIME_POINTS, n_nodes=n_nodes, l_dens=0.3, rng=rng
    )
    return np.ascontiguousarray(ts.T)


@pytest.fixture(scope="module", params=NODE_SIZES, ids=lambda n: f"nodes{n}")
def continuous_data(request) -> np.ndarray:
    return gen_continuous(request.param)


@pytest.fixture(scope="module", params=SMALL_NODES, ids=lambda n: f"nodes{n}")
def continuous_data_small(request) -> np.ndarray:
    return gen_continuous(request.param)


@pytest.fixture(scope="module")
def continuous_data_20() -> np.ndarray:
    return gen_continuous(20)


def pytest_collection_modifyitems(config, items):
    if not config.getoption("codspeed", False):
        try:
            from pytest_codspeed.plugin import has_benchmark_fixture

            msg = pytest.mark.skip(reason="use --codspeed to run benchmarks")
            for item in items:
                if has_benchmark_fixture(item):
                    item.add_marker(msg)
        except ImportError:
            pass
