"""reconstruct_network benchmarks: full pipeline across metrics and sizes."""

import pytest

from delaynet.network_reconstruction import reconstruct_network

from .conftest import LAG_STEPS, SMALL_NODES

_FAST_RECON_METRICS = ["lc", "rc"]
_SLOW_RECON_METRICS = ["gc", "gv"]
LAG_STEPS_SLOW = 3


@pytest.mark.benchmark(group="reconstruct-network")
@pytest.mark.parametrize("metric", _FAST_RECON_METRICS)
def test_reconstruct_fast(benchmark, continuous_data, metric):
    benchmark(reconstruct_network, continuous_data, metric, lag_steps=LAG_STEPS)


@pytest.mark.benchmark(group="reconstruct-network")
@pytest.mark.parametrize("metric", _SLOW_RECON_METRICS)
def test_reconstruct_slow(benchmark, continuous_data_small, metric):
    benchmark(
        reconstruct_network, continuous_data_small, metric, lag_steps=LAG_STEPS_SLOW
    )


@pytest.mark.benchmark(group="parallel-scaling")
@pytest.mark.parametrize("workers", [1, 4], ids=["seq", "par4"])
def test_reconstruct_workers(benchmark, continuous_data_20, workers):
    benchmark(
        reconstruct_network,
        continuous_data_20,
        "lc",
        lag_steps=LAG_STEPS,
        workers=workers,
    )
