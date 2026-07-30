"""reconstruct_network benchmarks: full pipeline across metrics and sizes."""

import pytest

from delaynet.network_reconstruction import reconstruct_network

from .conftest import LAG_STEPS, gen_continuous

_RECON_METRICS = ["lc", "rc", "gc", "gv"]


@pytest.mark.benchmark(group="reconstruct-network")
@pytest.mark.parametrize("metric", _RECON_METRICS)
def test_reconstruct(benchmark, continuous_data, metric):
    benchmark(reconstruct_network, continuous_data, metric, lag_steps=LAG_STEPS)


@pytest.mark.benchmark(group="parallel-scaling")
@pytest.mark.parametrize("workers", [1, 2], ids=["seq", "par2"])
def test_reconstruct_workers(benchmark, workers):
    data = gen_continuous(5)
    benchmark(
        reconstruct_network,
        data,
        "lc",
        lag_steps=LAG_STEPS,
        workers=workers,
    )
