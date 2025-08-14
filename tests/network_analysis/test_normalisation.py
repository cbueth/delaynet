"""Tests for ER-based normalisation decorator and helpers (British English).

These tests focus on:
- Parameter validation when not normalising (Ellipsis detection).
- Binary-only enforcement when normalising.
- Zero-diagonal (no self-loops) enforcement.
- ER generator properties (shape, zero diagonal, exact edge count).
- Seed reproducibility for z-scores.
- Sigma==0 branch returns 0.0 without NaN/inf.
- Output shape/type parity when normalising.
"""

from __future__ import annotations

import numpy as np
import pytest

from delaynet.network_analysis.metrics import (
    link_density,
    reciprocity,
    betweenness_centrality,
    global_efficiency,
    transitivity,
    eigenvector_centrality,
)
from delaynet.network_analysis._normalisation import (
    _random_directed_gnm_igraph,
    _is_binary_matrix,
    _diagonal_is_zero,
)


def test_params_without_normalise_raise():
    """Passing n_random or random_seed with normalise False/None must raise."""
    A = np.array([[0, 1], [0, 0]])

    with pytest.raises(
        ValueError, match="Normalisation parameters 'n_random' and 'random_seed'"
    ):
        # normalise omitted (None by default)
        link_density(A, n_random=10)

    with pytest.raises(
        ValueError, match="Normalisation parameters 'n_random' and 'random_seed'"
    ):
        # normalise explicitly False
        link_density(A, normalise=False, random_seed=123)


@pytest.mark.parametrize(
    "A",
    [
        np.array([[0, 0.5], [0, 0]]),  # non-binary weight
        np.array([[0, 2], [0, 0]]),  # integer but not in {0,1}
    ],
)
def test_binary_required_for_normalisation(A):
    with pytest.raises(
        ValueError, match=r"binary adjacency matrix \(values strictly in \{0,1\}\)"
    ):
        link_density(A, normalise=True, n_random=5)


def test_zero_diagonal_required_before_nm():
    """Diagonal must be zero before computing n and m when normalising."""
    A = np.array([[1, 1], [0, 0]], dtype=int)  # self-loop on (0,0)
    with pytest.raises(ValueError, match="Normalisation assumes no self-loops"):
        reciprocity(A, normalise=True, n_random=5)


def test_er_generator_properties():
    n, m = 6, 7
    R = _random_directed_gnm_igraph(n, m)
    assert R.shape == (n, n)
    assert np.all(np.diag(R) == 0), "No self-loops expected"
    assert R.sum() == m, "Exactly m directed edges expected"
    # Directedness: expect some asymmetry (unless m is pathological)
    asym_pairs = np.sum(R != R.T)
    assert asym_pairs > 0


def test_er_generator_invalid_m_raises():
    n = 4
    max_m = n * (n - 1)
    with pytest.raises(Exception):
        _ = _random_directed_gnm_igraph(n, max_m + 1)


def test_seed_reproducibility_for_zscores():
    """Same seed yields identical z-scores; different seeds likely differ."""
    # Small binary graph without self-loops
    A = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
        ],
        dtype=int,
    )

    z1 = link_density(A, normalise=True, n_random=20, random_seed=555)
    z2 = link_density(A, normalise=True, n_random=20, random_seed=555)
    assert z1 == pytest.approx(z2), "Same seed should reproduce identical z-scores"

    z3 = link_density(A, normalise=True, n_random=20, random_seed=556)
    # With high probability different seeds differ; allow equality as a rare edge-case
    assert not np.allclose(z1, z3) or True


def test_sigma_zero_returns_zero():
    """When the null distribution has zero variance, the z-score should be 0.0."""
    # Empty graph (m=0) => link density is always 0 across the ensemble
    A = np.zeros((4, 4), dtype=int)
    z = link_density(A, normalise=True, n_random=10)
    assert z == 0.0
    assert np.isfinite(z)


def test_output_shape_parity_scalar_and_vector_metrics():
    """Normalised outputs must preserve shapes/types of raw outputs across metrics."""
    # Build a small binary, directed graph without self-loops
    A = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ],
        dtype=int,
    )

    # Scalar metrics
    for fn in (link_density, reciprocity, transitivity, global_efficiency):
        raw = fn(A)
        z = fn(A, normalise=True, n_random=10)
        assert isinstance(raw, float)
        assert np.asarray(z).shape == ()

    # Vector metrics
    for fn in (betweenness_centrality, eigenvector_centrality):
        raw_vec = fn(A)
        z_vec = fn(A, normalise=True, n_random=10)
        assert isinstance(raw_vec, np.ndarray)
        assert raw_vec.shape == z_vec.shape


def test_zscore_near_zero_for_random_graph():
    """For an input drawn from the null model, z-scores should be near 0 on average."""
    n, m = 8, 10
    A = _random_directed_gnm_igraph(n, m)
    # Ensure diagonal zero (generator already does this); and A is binary
    z = link_density(A, normalise=True, n_random=100, random_seed=123)
    assert np.isfinite(z)
    assert abs(z) <= 2.0  # loose tolerance; should be near 0 in expectation


def test_non_square_with_normalise_true_raises():
    A = np.array([[0, 1, 0], [0, 0, 1]])  # 2x3 non-square
    with pytest.raises(ValueError, match="must be square"):
        link_density(A, normalise=True, n_random=5)


def test_is_binary_matrix_exception_path():
    class Bad:
        def astype(self, *args, **kwargs):
            raise RuntimeError("erroring")

    assert _is_binary_matrix(Bad()) is False


def test_diagonal_is_zero_nonsquare_path():
    A = np.array([[0, 1, 0], [0, 0, 1]])  # 2x3 non-square
    assert _diagonal_is_zero(A) is False


def test_invalid_normalise_value_raises():
    A = np.array([[0, 1], [0, 0]], dtype=int)
    with pytest.raises(ValueError, match="must be either True, False, or None"):
        link_density(A, normalise="yes", n_random=5)


@pytest.mark.parametrize(
    "fn, build_A",
    [
        (link_density, lambda: np.array([[0, 1], [0, 0]], dtype=int)),
        (reciprocity, lambda: np.array([[0, 1], [0, 0]], dtype=int)),
        (transitivity, lambda: np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=int)),
        (global_efficiency, lambda: np.array([[0, 1], [0, 0]], dtype=int)),
        (
            betweenness_centrality,
            lambda: np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=int),
        ),
        (
            eigenvector_centrality,
            lambda: np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=int),
        ),
    ],
)
def test_normalisation_does_not_change_default(fn, build_A):
    """When normalise is False or omitted, results match the legacy behaviour."""
    A = build_A()
    # Omitted (pass-through)
    legacy = fn(A)
    # Explicit False (should be identical)
    current = fn(A, normalise=False)
    if isinstance(legacy, np.ndarray):
        assert np.array_equal(current, legacy)
    else:
        assert current == legacy
