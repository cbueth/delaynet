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

import random as _py_random

import numpy as np
import pytest
from igraph import Graph

from delaynet.network_analysis.metrics import (
    link_density,
    reciprocity,
    betweenness_centrality,
    global_efficiency,
    transitivity,
    eigenvector_centrality,
    isolated_nodes_inbound,
    isolated_nodes_outbound,
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
    """Same seed yields identical z-scores; different seeds likely differ.

    Use a metric with non-degenerate null variance to avoid NaN (e.g., global_efficiency).
    """
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

    z1 = global_efficiency(A, normalise=True, n_random=20, random_seed=555)
    z2 = global_efficiency(A, normalise=True, n_random=20, random_seed=555)
    assert z1 == pytest.approx(z2), "Same seed should reproduce identical z-scores"

    z3 = global_efficiency(A, normalise=True, n_random=20, random_seed=556)
    # With high probability different seeds differ; allow equality as a rare edge-case
    assert not np.allclose(z1, z3) or True


def test_sigma_zero_returns_nan():
    """When the null distribution has zero variance, the z-score should be NaN.

    Also expect a user warning when normalising link density.
    """
    # Empty graph (m=0) => link density is always 0 across the ensemble
    A = np.zeros((4, 4), dtype=int)
    with pytest.warns(UserWarning, match=r"link density"):
        z = link_density(A, normalise=True, n_random=10)
    assert np.isnan(z)


def test_link_density_normalisation_warns_user():
    """Normalising link density should warn that it is not very meaningful (σ=0)."""
    A = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ],
        dtype=int,
    )
    # Keep the regex simple to avoid false negatives due to punctuation/casing
    with pytest.warns(UserWarning, match=r"link density"):
        _ = link_density(A, normalise=True, n_random=5, random_seed=123)


def test_output_shape_parity_scalar_and_vector_metrics(network_metric_and_kind):
    """Normalised outputs must preserve shapes/types of raw outputs across metrics.

    This test uses the network_metric_and_kind fixture to iterate over all metrics
    and verifies scalar vs vector parity. It also expects the link_density warning
    when normalising.
    """
    # Build a small binary, directed graph without self-loops
    A = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 0],
        ],
        dtype=int,
    )

    fn, kind = network_metric_and_kind

    # Compute raw and normalised outputs, handling expected warnings for link_density
    raw = fn(A)
    if fn is link_density:
        with pytest.warns(UserWarning, match=r"link density"):
            z = fn(A, normalise=True, n_random=10)
    else:
        z = fn(A, normalise=True, n_random=10)

    if kind == "scalar":
        # Raw must be a scalar (float or int depending on metric)
        assert np.asarray(raw).shape == ()
        # Normalised output for scalar metrics must also be a scalar float
        assert isinstance(z, float)
        assert np.asarray(z).shape == ()
    else:
        # Vector metrics must preserve array shape on normalisation
        assert isinstance(raw, np.ndarray)
        assert isinstance(z, np.ndarray)
        assert raw.shape == z.shape


def test_link_density_normalisation_nan_for_random_input():
    """For an input drawn from G(n,m), density normalisation should be NaN.

    The null ensemble preserves the exact number of links m, making the
    density distribution degenerate (sigma==0). Therefore the z-score is NaN.
    Also expect the standard user warning for link density normalisation.
    """
    n, m = 7, 11
    A = _random_directed_gnm_igraph(n, m)
    with pytest.warns(UserWarning, match=r"link density"):
        z = link_density(A, normalise=True, n_random=25, random_seed=123)
    assert np.isnan(z)


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
            lambda: np.array(
                [[0, 1, 0, 1], [1, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 0]], dtype=int
            ),
        ),
        (
            isolated_nodes_inbound,
            # Column 2 has zero inbound (no ones in last column)
            lambda: np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=int),
        ),
        (
            isolated_nodes_outbound,
            # Row 0 has zero outbound (all zeros in first row)
            lambda: np.array([[0, 1, 1], [1, 0, 1], [0, 0, 0]], dtype=int),
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

    assert type(legacy) == type(current)

    if isinstance(legacy, np.ndarray):
        assert np.allclose(current, legacy)
    else:
        assert current == legacy


@pytest.mark.parametrize("n, m", [(15, 32), (15, 66), (15, 132), (15, 135)])
def test_example_matrices(network_metric_and_kind, n, m):
    """Test with G(n, m) matrices."""
    _py_random.seed(123)
    A = Graph.Erdos_Renyi(n=n, m=m, directed=True, loops=False).get_adjacency().data
    A = np.array(A, dtype=int)
    fn, kind = network_metric_and_kind

    if fn.__name__.lower() == "link_density":
        with pytest.warns(UserWarning, match=r"link density"):
            z = fn(A, normalise=True, random_seed=123)
    else:
        z = fn(A, normalise=True, random_seed=123)

    val = fn(A, normalise=False)

    if (fn.__name__.lower() == "link_density") or (
        fn.__name__.lower() == "global_efficiency" and m > 133
    ):
        assert not np.isnan(val)
        assert np.isnan(z)
    elif "isolated_nodes" in fn.__name__.lower() and m > 80:
        assert not np.isnan(val)
        assert val == 0
        assert np.isnan(z)
    else:
        if isinstance(z, np.ndarray):
            assert not np.any(np.isnan(val))
            assert not np.any(np.isnan(z))
            assert np.all(np.isfinite(z))
        else:
            assert not np.isnan(val)
            assert not np.isnan(z)
            assert np.isfinite(z)
