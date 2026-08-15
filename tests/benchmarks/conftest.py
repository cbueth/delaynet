"""Shared fixtures and constants for benchmarks.

Benchmark tests require the ``benchmark`` fixture provided by the
``pytest-codspeed`` plugin. When the plugin is absent (e.g. downstream
packaging like conda-forge running ``pytest tests/``), the tests must be
skipped instead of failing on a missing fixture or an unknown mark.
"""

import numpy as np
import pytest

from delaynet.preparation.data_generator import gen_delayed_causal_network

TIME_POINTS = 200
LAG_STEPS = 3
NODE_SIZES = [5, 10]

CONTINUOUS_METRICS = ["lc", "rc", "gv"]

_BENCHMARK_FIXTURES = {"benchmark", "codspeed_benchmark"}


def gen_continuous(n_nodes: int, rng: int = 19425) -> np.ndarray:
    _, _, ts = gen_delayed_causal_network(
        ts_len=TIME_POINTS, n_nodes=n_nodes, l_dens=0.3, rng=rng
    )
    return np.ascontiguousarray(ts.T)


@pytest.fixture(scope="module", params=NODE_SIZES, ids=lambda n: f"nodes{n}")
def continuous_data(request) -> np.ndarray:
    return gen_continuous(request.param)


def pytest_configure(config):
    """Register the benchmark markers so they are not "unknown" without pytest-codspeed.

    Without this, pytest raises a ``PytestUnknownMarkWarning`` for
    ``@pytest.mark.benchmark`` which is escalated to an error by the
    ``filterwarnings = ["error"]`` setting in ``pyproject.toml``.
    """
    existing = set()
    for entry in config.getini("markers"):
        name = entry.split(":", 1)[0].strip() if isinstance(entry, str) else entry[0]
        existing.add(name)
    for marker in sorted(_BENCHMARK_FIXTURES):
        if marker not in existing:
            config.addinivalue_line(
                "markers",
                f"{marker}: micro-benchmark; run with --codspeed",
            )


def _is_benchmark_item(item: pytest.Item) -> bool:
    return bool(_BENCHMARK_FIXTURES.intersection(getattr(item, "fixturenames", [])))


def pytest_collection_modifyitems(config, items):
    """Skip benchmark tests unless explicitly run with ``--codspeed``.

    Works with or without the ``pytest-codspeed`` plugin installed: benchmark
    tests are identified by the ``benchmark`` fixture they request, so they are
    skipped even when the plugin (and thus the fixture) is unavailable.
    """
    if not config.getoption("codspeed", False):
        msg = pytest.mark.skip(reason="use --codspeed to run benchmarks")
        for item in items:
            if _is_benchmark_item(item):
                item.add_marker(msg)
