"""Detrending method benchmarks: delta, identity, z-score, second-difference."""

import numpy as np
import pytest

from delaynet import detrend

TS_LEN = 1000

_rng = np.random.default_rng(2154)
_ts = _rng.normal(0, 1, TS_LEN)
_ts_wave = np.sin(np.linspace(0, 4 * np.pi, TS_LEN))


@pytest.mark.benchmark(group="detrend")
@pytest.mark.parametrize(
    "method,kwargs,data",
    [
        ("delta", {"window_size": 10}, _ts),
        ("delta", {"window_size": 100}, _ts),
        ("identity", {}, _ts),
        ("z_score", {"periodicity": 1}, _ts),
        ("z_score", {"periodicity": 10, "max_periods": 5}, _ts),
        ("second_difference", {}, _ts_wave),
    ],
    ids=[
        "delta-w10",
        "delta-w100",
        "identity",
        "z_score-p1",
        "z_score-p10",
        "second_difference",
    ],
)
def test_detrend_method(benchmark, method, kwargs, data):
    benchmark(detrend, data, method=method, **kwargs)
