"""Small benchmark for ER-based normalisation overhead in network metrics.

This script compares the runtime of raw metrics vs the normalised (z-score) variant
implemented via the normalise-against-random decorator. It reports, for each metric:

- time_metric:     time to compute the metric once on the input graph
- time_normalised: time to compute the metric once with normalise=True and n_random
- 28*time_metric + overhead ≈ time_normalised (when n_random=28)

Usage:
    python examples/benchmark_normalisation.py --n 100 --m 400 --n-random 28 --repeats 3 --seed 123

Notes:
- Normalisation is binary-only and requires a zero diagonal (no self-loops), consistent
  with the package's constraints.
- igraph's randomness is seeded via Python's random module for reproducibility.
"""

from __future__ import annotations

import argparse
import time
import random as _py_random
from typing import Callable, Any

import numpy as np

import delaynet as dn
from delaynet.network_analysis._normalisation import _random_directed_gnm_igraph


def build_binary_graph(n: int, m: int, seed: int | None) -> np.ndarray:
    if seed is not None:
        _py_random.seed(seed)
    A = _random_directed_gnm_igraph(n, m)
    # Ensure strictly binary int and zero diagonal (generator already does this)
    A = (A != 0).astype(int)
    np.fill_diagonal(A, 0)
    return A


def time_call(fn: Callable[..., Any], *args, repeats: int = 1, **kwargs) -> float:
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(*args, **kwargs)
    t1 = time.perf_counter()
    return (t1 - t0) / repeats


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ER normalisation overhead for delaynet metrics"
    )
    parser.add_argument("--n", type=int, default=80, help="Number of nodes")
    parser.add_argument("--m", type=int, default=320, help="Number of directed edges")
    parser.add_argument(
        "--n-random",
        type=int,
        default=28,
        dest="n_random",
        help="Number of ER realisations for normalisation",
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="Number of repeats for timing each call"
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed for ER generation"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="*",
        default=[
            "link_density",
            "reciprocity",
            "transitivity",
            "global_efficiency",
            "betweenness_centrality",
            "eigenvector_centrality",
        ],
        help="Subset of metrics to benchmark",
    )

    args = parser.parse_args()

    # Generate one input graph to keep comparisons fair
    A = build_binary_graph(args.n, args.m, args.seed)

    # Map metric names to callables (from delaynet.network_analysis)
    net = dn.network_analysis
    metric_map: dict[str, Callable[..., Any]] = {
        "link_density": net.link_density,
        "reciprocity": net.reciprocity,
        "transitivity": net.transitivity,
        "global_efficiency": net.global_efficiency,
        "betweenness_centrality": net.betweenness_centrality,
        "eigenvector_centrality": net.eigenvector_centrality,
    }

    print(
        f"Benchmarking on G(n={args.n}, m={args.m}) with n_random={args.n_random}, repeats={args.repeats}, seed={args.seed}"
    )
    print("-" * 88)
    header = f"{'metric':28s}  {'time_metric [s]':>16s}  {'time_normalised [s]':>20s}  {'overhead [s]':>14s}  {'check':>10s}"
    print(header)
    print("-" * 88)

    for name in args.metrics:
        if name not in metric_map:
            print(f"Skipping unknown metric: {name}")
            continue
        fn = metric_map[name]

        # Time raw metric
        # Important: For betweenness_centrality we keep normalize=True (scaling) as its own
        # parameter and orthogonal to z-score normalisation; this call uses default behaviour
        t_raw = time_call(fn, A, repeats=args.repeats)

        # Time normalised metric with n_random realisations
        t_norm = time_call(
            fn,
            A,
            repeats=args.repeats,
            normalise=True,
            n_random=args.n_random,
            random_seed=args.seed,
        )

        # Compute overhead: time spent beyond n_random * time_metric
        overhead = t_norm - args.n_random * t_raw

        # Simple qualitative check string
        approx = (
            "≈" if t_norm >= 0 and (abs(overhead) < max(1e-6, 0.2 * t_norm)) else "~"
        )

        print(
            f"{name:28s}  {t_raw:16.6f}  {t_norm:20.6f}  {overhead:14.6f}  {approx:>10s}"
        )

    print("-" * 88)
    print(
        "Interpretation: time_normalised should be roughly n_random * time_metric + overhead from ER generation (igraph graph construction + adjacency).\n"
        "Overhead depends on metric cost vs graph-generation cost and may dominate for cheap metrics like link_density."
    )


if __name__ == "__main__":
    main()
