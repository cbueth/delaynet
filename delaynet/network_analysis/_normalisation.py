"""Normalisation utilities for network metrics.

This module provides a decorator to compute z-score normalisation of network
metrics against an ensemble of directed Erdos–Rényi graphs with the same
number of nodes and links (G(n, m)), generated on-the-fly with igraph.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray
from igraph import Graph
import random as _py_random
import warnings


def _is_square_matrix(weight_matrix: NDArray[np.number | np.bool_]) -> bool:
    return weight_matrix.ndim == 2 and weight_matrix.shape[0] == weight_matrix.shape[1]


def _is_binary_matrix(weight_matrix: NDArray[np.number | np.bool_]) -> bool:
    """Return True if all entries are in {0, 1} (including 0.0/1.0 or booleans)."""
    try:
        uniq = np.unique(weight_matrix.astype(float))
    except Exception:
        return False
    return np.all(np.isin(uniq, (0.0, 1.0)))


def _diagonal_is_zero(weight_matrix: NDArray[np.number | np.bool_]) -> bool:
    """Check whether the diagonal entries are all zero (no self-loops)."""
    if not _is_square_matrix(weight_matrix):
        return False
    return not np.any(np.diag(weight_matrix) != 0)


def _random_directed_gnm_igraph(n: int, m: int) -> NDArray[np.int_]:
    """Sample a directed G(n, m) adjacency with no self-loops using igraph.

    Uses igraph.Graph.Erdos_Renyi with directed=True and loops=False and returns a
    binary adjacency matrix of shape (n, n) with exactly m ones off-diagonal.
    """
    g = Graph.Erdos_Renyi(n=n, m=m, directed=True, loops=False)
    return np.array(g.get_adjacency().data, dtype=int)


# Ellipsis is used in defaults to detect whether users explicitly passed keyword arguments
def normalise_against_random(metric_fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator adding z-score normalisation vs. directed ER G(n,m) ensemble.

    Adds keyword-only parameters to the wrapped metric function:
    - normalise: bool | None = None
    - n_random: int = 20
    - random_seed: int | None = None

    Behaviour:
    - If normalise is False or None: the metric runs unchanged; if the user
      provides normalisation parameters (n_random or random_seed), a ValueError is
      raised explaining these are only valid with normalise=True.
    - If normalise is True: the input matrix must be strictly binary with a zero
      diagonal (no self-loops). The decorator computes the metric on the given
      binary matrix and on an ensemble of directed ER graphs with the same n and m,
      then returns the element-wise z-score (x_true - mu_null) / sigma_null.
      If sigma_null == 0 at any position, the returned z-score there is 0.0.

    Notes:
    - Randomness for igraph is controlled by the standard library's 'random' module.
      If a random_seed is provided, random.seed(seed) is called once before the
      sampling loop to ensure reproducibility.
    - This decorator does not support weighted normalisation; pass a binary matrix
      if you wish to enable normalisation. Weighted metrics remain available when
      normalise is False/None.
    """

    @wraps(metric_fn)
    def wrapper(
        weight_matrix: NDArray[np.number | np.bool_],
        *args: Any,
        normalise: bool | None = None,
        n_random=...,
        random_seed=...,
        **kwargs: Any,
    ):
        # Pass-through behaviour (no normalisation)
        if normalise is None or normalise is False:
            if (n_random is not ...) or (random_seed is not ...):
                raise ValueError(
                    "Normalisation parameters 'n_random' and 'random_seed' can only be used "
                    "when normalise=True. Remove them or set normalise=True to enable "
                    "normalisation."
                )
            return metric_fn(weight_matrix, *args, **kwargs)

        # From here normalise must be True
        if normalise is not True:
            raise ValueError(
                "Parameter 'normalise' must be either True, False, or None."
            )

        # Validate matrix shape first
        if not _is_square_matrix(weight_matrix):
            raise ValueError(
                f"weight_matrix must be square, got shape {weight_matrix.shape}"
            )

        # Enforce binary-only normalisation
        if not _is_binary_matrix(weight_matrix):
            raise ValueError(
                "Normalisation against random equivalent networks requires a binary adjacency "
                "matrix (values strictly in {0,1}). Weighted normalisation is not supported. "
                "To normalise, first binarise your graph (e.g., via statistical pruning) and "
                "call the metric with normalise=True. If you need weighted metrics, call the "
                "metric with normalise=False (or omit 'normalise') to retain weighted behaviour."
            )

        # Enforce no self-loops (diagonal must be zero) before computing n and m
        if not _diagonal_is_zero(weight_matrix):
            raise ValueError(
                "Normalisation assumes no self-loops: the diagonal of the adjacency matrix "
                "must be zero before computing the null-model (G(n,m)). Please remove or "
                "zero self-loops and try again."
            )

        n = weight_matrix.shape[0]
        m = int(
            weight_matrix.sum()
        )  # matrix is binary with zero diagonal by this point

        # Resolve parameters
        n_rand_val = 20 if (n_random is ...) else int(n_random)
        seed_val = None if (random_seed is ...) else int(random_seed)

        # Seed igraph's RNG via Python's random module if requested
        if seed_val is not None:
            _py_random.seed(seed_val)

        # Compute the true metric value on the provided (binary) matrix
        x_true = metric_fn(weight_matrix, *args, **kwargs)

        # Warn users when normalising link density: null ensemble preserves m → σ=0
        if getattr(metric_fn, "__name__", "").lower() == "link_density":
            warnings.warn(
                "Normalising link density against G(n,m) is typically not meaningful: "
                "the null model preserves the number of links, so the null distribution "
                "is degenerate (σ=0) and the z-score is undefined (NaN).",
                UserWarning,
                stacklevel=2,
            )

        # Sample ensemble and compute metric values
        samples = []
        for _ in range(n_rand_val):
            R = _random_directed_gnm_igraph(n, m)
            x_r = metric_fn(R, *args, **kwargs)
            samples.append(np.asarray(x_r))

        samples_arr = np.stack(samples, axis=0)  # shape: (n_random, ...)
        mu = samples_arr.mean(axis=0)
        sigma = samples_arr.std(axis=0, ddof=0)

        x_true_arr = np.asarray(x_true, dtype=float)
        # Start with NaNs; fill only where sigma>0
        z = np.full_like(x_true_arr, np.nan, dtype=float)
        mask = sigma > 0
        z[mask] = (x_true_arr[mask] - mu[mask]) / sigma[mask]
        # Where sigma == 0, z remains NaN by design

        # Preserve output type/shape parity with raw metric outputs
        # - If the raw metric returned a scalar (Python float or NumPy scalar), return a Python float
        # - Otherwise, return an ndarray matching the raw output shape
        if np.isscalar(x_true) or np.asarray(x_true).shape == ():
            return float(z)
        return z

    return wrapper
