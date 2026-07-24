"""Parametrized correctness tests for fast Granger F-test implementations.

Every fast variant must produce (near-)identical p-values to the
statsmodels reference across a wide range of time-series lengths,
lag values, and conditioning regimes.
"""

import numpy as np
import pytest

from delaynet.connectivities._granger_fast import (
    F_TEST_IMPLS,
    _build_lag_design_matrix,
    _build_restriction_matrix,
    gt_single_lag_fast,
    gt_multi_lag_fast,
)
from delaynet.connectivities.granger import gt_single_lag

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
# Well-conditioned random data:    machine epsilon
# Near-collinear / ill-conditioned: 1e-10
# Collinear:                       1e-8  (singular X'X triggers fallback path)

APPROX_TIGHT = 1e-13
APPROX_NEAR_SINGULAR = 1e-8

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _ref_pvalue(ts1, ts2, lag_step):
    """Reference p-value via the statsmodels-based gt_single_lag."""
    return gt_single_lag(ts1, ts2, lag_step)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def causal_ts(rng):
    """ts2 depends on ts1 with a known 3-step causal lag."""
    T = 200
    ts1 = rng.normal(0, 1, T)
    noise = rng.normal(0, 0.05, T)
    ts2 = np.zeros(T)
    ts2[3:] = 0.8 * ts1[:-3]
    ts2 += noise
    return ts1, ts2


# ---------------------------------------------------------------------------
# Parametrized correctness: all methods ≡ statsmodels reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "T, L, method",
    [
        pytest.param(T, L, m, id=f"T={T}_L={L}__{m}")
        for (T, L) in [
            # small-T / minimum-lag edge cases
            (10, 1),
            (15, 2),
            (15, 3),
            (25, 5),
            # standard sizes covering grid
            (20, 1),
            (20, 3),
            (20, 5),
            (50, 1),
            (50, 5),
            (50, 10),
            (100, 1),
            (100, 5),
            (100, 10),
            (100, ((100 - 2) // 3)),  # max identifiable lag
            (200, 1),
            (200, 10),
            (200, 21),
            (500, 1),
            (500, 10),
            (500, 21),
        ]
        for m in sorted(F_TEST_IMPLS.keys())
    ],
)
def test_pvalue_matches_statsmodels(rng, T, L, method):
    """Every fast F-test variant matches the statsmodels reference to <1e-13.

    Covers small T (10–25), standard sizes (50, 100, 200, 500),
    minimal lag (1), and maximum identifiable lag.
    """
    ts1 = rng.normal(0, 1, T)
    ts2 = rng.normal(0, 1, T)
    p_ref = _ref_pvalue(ts1, ts2, L)
    p_fast = gt_single_lag_fast(ts1, ts2, L, method)
    assert p_fast == pytest.approx(
        p_ref, abs=APPROX_TIGHT
    ), f"method={method} T={T} L={L}: fast={p_fast:.15e} ref={p_ref:.15e}"


# ---------------------------------------------------------------------------
# Causal-data: correct-lag detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", list(F_TEST_IMPLS.keys()))
def test_detects_causal_lag(causal_ts, method):
    """Known causal lag 3: fast multi-lag returns lag=3 with low p-value."""
    ts1, ts2 = causal_ts
    lag_steps = list(range(1, 11))
    p_val, best_lag = gt_multi_lag_fast(ts1, ts2, lag_steps=lag_steps, method=method)
    assert best_lag == 3, f"method={method}: got lag={best_lag}, expected 3"
    assert p_val < 0.001, f"method={method}: p={p_val:.3e}, expected <0.001"


@pytest.mark.parametrize("method", list(F_TEST_IMPLS.keys()))
def test_causal_pvalues_match_reference(causal_ts, method):
    """For causal data: per-lag p-value matches statsmodels to <1e-14."""
    ts1, ts2 = causal_ts
    for L in [1, 2, 3, 4, 5, 7, 10]:
        p_ref = _ref_pvalue(ts1, ts2, L)
        p_fast = gt_single_lag_fast(ts1, ts2, L, method)
        assert p_fast == pytest.approx(
            p_ref, abs=APPROX_TIGHT
        ), f"method={method} L={L}: fast={p_fast:.15e} ref={p_ref:.15e}"


# ---------------------------------------------------------------------------
# Multi-lag: optimal lag agrees with per-lag reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", list(F_TEST_IMPLS.keys()))
def test_multi_lag_fast_optimal_lag_matches_reference(causal_ts, method):
    """gt_multi_lag_fast always picks the same optimal lag as per-lag statsmodels."""
    ts1, ts2 = causal_ts
    lag_steps = list(range(1, 11))

    # Reference: per-lag via statsmodels
    ref_pvals = [_ref_pvalue(ts1, ts2, L) for L in lag_steps]
    ref_best = lag_steps[int(np.argmin(ref_pvals))]

    # Fast (per-lag loop - same sample sizes, identical approach)
    _, fast_best = gt_multi_lag_fast(ts1, ts2, lag_steps=lag_steps, method=method)

    assert (
        fast_best == ref_best
    ), f"method={method}: fast={fast_best}, ref per-lag={ref_best}"


# ---------------------------------------------------------------------------
# Parity: "gc" via connectivity dispatch ≡ old gt_multi_lag
# ---------------------------------------------------------------------------


def test_gc_dispatch_parity(rng):
    """connectivity(ts1, ts2, 'gc', ...) produces identical results as the
    old statsmodels-based gt_multi_lag."""
    from delaynet.connectivity import connectivity
    from delaynet.connectivities.granger import gt_multi_lag as gt_multi_lag_ref

    ts1 = rng.normal(0, 1, 200)
    ts2 = rng.normal(0, 1, 200)

    for lag_steps_val in [3, 5, 10, 21]:
        # Reference: old gt_multi_lag (statsmodels)
        p_ref, lag_ref = gt_multi_lag_ref(
            ts1, ts2, lag_steps=list(range(1, lag_steps_val + 1))
        )

        # Via connectivity dispatch -> gt_multi_lag_fast
        p_gc, lag_gc = connectivity(ts1, ts2, "gc", lag_steps=lag_steps_val)

        assert p_gc == pytest.approx(
            p_ref, abs=APPROX_TIGHT
        ), f"lag_steps={lag_steps_val}: gc_p={p_gc:.15e} ref_p={p_ref:.15e}"
        assert (
            lag_gc == lag_ref
        ), f"lag_steps={lag_steps_val}: gc_lag={lag_gc} ref_lag={lag_ref}"


# ---------------------------------------------------------------------------
# Edge case: near-collinear data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "eps, L, method",
    [
        pytest.param(eps, L, m, id=f"eps={eps}_L={L}__{m}")
        for eps in [1e-1, 1e-2, 1e-3]
        for L in [1, 3, 5]
        for m in sorted(F_TEST_IMPLS.keys())
    ],
)
def test_near_collinear_matches(rng, eps, L, method):
    """When ts1 ≈ ts2 (eps noise), p-values still match within <1e-8."""
    T = 200
    base = rng.normal(0, 1, T)
    ts1 = base.astype(np.float64)
    ts2 = base + eps * rng.normal(0, 1, T).astype(np.float64)

    try:
        p_ref = _ref_pvalue(ts1, ts2, L)
    except Exception:
        pytest.skip("statsmodels failed on near-collinear data")
        return

    p_fast = gt_single_lag_fast(ts1, ts2, L, method)
    assert p_fast == pytest.approx(p_ref, abs=APPROX_NEAR_SINGULAR), (
        f"method={method} eps={eps} L={L}: "
        f"fast={p_fast:.12e} ref={p_ref:.12e} diff={abs(p_fast - p_ref):.3e}"
    )


# ---------------------------------------------------------------------------
# Edge case: constant / degenerate time series
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", list(F_TEST_IMPLS.keys()))
def test_constant_ts_no_crash(method):
    """Constant time series should not crash (both may return extreme p-values)."""
    ts1 = np.ones(100, dtype=np.float64)
    ts2 = np.ones(100, dtype=np.float64)

    try:
        _ref_pvalue(ts1, ts2, 3)
    except Exception:
        pytest.skip("statsmodels itself rejects constant input")

    try:
        p_val = gt_single_lag_fast(ts1, ts2, 3, method)
        assert 0.0 <= p_val <= 1.0
    except Exception as exc:
        pytest.fail(f"method={method} crashed on constant input: {exc}")


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", list(F_TEST_IMPLS.keys()))
def test_p_value_in_valid_range(rng, method):
    """All p-values are valid probabilities ∈ [0, 1] for typical inputs."""
    ts1 = rng.normal(0, 1, 200)
    ts2 = rng.normal(0, 1, 200)
    for L in [1, 5, 10, 21]:
        p_val = gt_single_lag_fast(ts1, ts2, L, method)
        assert 0.0 <= p_val <= 1.0, f"method={method} L={L}: p={p_val}"

    _, best_lag = gt_multi_lag_fast(
        ts1, ts2, lag_steps=list(range(1, 11)), method=method
    )
    assert best_lag >= 1


@pytest.mark.parametrize("method", list(F_TEST_IMPLS.keys()))
def test_deterministic(rng, method):
    """Two calls with identical input produce identical output (bitwise)."""
    ts1 = rng.normal(0, 1, 100)
    ts2 = rng.normal(0, 1, 100)
    p1 = gt_single_lag_fast(ts1, ts2, 5, method)
    p2 = gt_single_lag_fast(ts1, ts2, 5, method)
    assert p1 == p2


# ---------------------------------------------------------------------------
# Design matrix helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("L", [1, 3, 5, 10, 21])
def test_design_matrix_shape(L):
    rng = np.random.default_rng(0)
    ts1 = rng.normal(0, 1, 200)
    ts2 = rng.normal(0, 1, 200)
    y, X = _build_lag_design_matrix(ts1, ts2, L)
    assert X.shape == (200 - L, 2 * L + 1)
    assert y.shape == (200 - L,)


@pytest.mark.parametrize("L", [1, 5, 10, 21])
def test_design_matrix_full_rank(rng, L):
    """Random data produces full-rank design matrices."""
    ts1 = rng.normal(0, 1, 500)
    ts2 = rng.normal(0, 1, 500)
    _, X = _build_lag_design_matrix(ts1, ts2, L)
    rank = np.linalg.matrix_rank(X)
    assert rank == X.shape[1], f"L={L}: rank={rank} < k={X.shape[1]}"


# ---------------------------------------------------------------------------
# Restriction matrix
# ---------------------------------------------------------------------------


def test_restriction_matrix():
    for L in [1, 3, 5, 10]:
        R = _build_restriction_matrix(L)
        assert R.shape == (L, 2 * L + 1)
        # AR columns (first L): all zeros
        assert np.all(R[:, :L] == 0)
        # Cross columns (next L): identity
        assert np.all(R[:, L : 2 * L] == np.eye(L))
        # Constant column (last): zero
        assert np.all(R[:, 2 * L] == 0)


# ---------------------------------------------------------------------------
# @connectivity decorator integration
# ---------------------------------------------------------------------------


def test_multi_lag_fast_via_connectivity_dispatch(rng):
    """gt_multi_lag_fast works through the connectivity() dispatch function."""
    from delaynet.connectivity import connectivity

    ts1 = rng.normal(0, 1, 200)
    ts2 = rng.normal(0, 1, 200)

    # Direct call
    p_direct, lag_direct = gt_multi_lag_fast(ts1, ts2, lag_steps=list(range(1, 8)))

    # Via connectivity dispatch (int -> converts to list via decorator)
    p_disp, lag_disp = connectivity(ts1, ts2, gt_multi_lag_fast, lag_steps=7)

    assert p_direct == pytest.approx(p_disp, abs=APPROX_TIGHT)
    assert lag_direct == lag_disp


def test_multi_lag_fast_via_connectivity_with_method(rng):
    """Passing method kwarg through connectivity dispatch works."""
    from delaynet.connectivity import connectivity

    ts1 = rng.normal(0, 1, 200)
    ts2 = rng.normal(0, 1, 200)

    for method in ["normal_eqn", "qr", "cholesky"]:
        _, lag_disp = connectivity(
            ts1, ts2, gt_multi_lag_fast, lag_steps=5, method=method
        )
        _, lag_direct = gt_multi_lag_fast(
            ts1, ts2, lag_steps=[1, 2, 3, 4, 5], method=method
        )
        assert lag_disp == lag_direct


@pytest.mark.parametrize("method", list(F_TEST_IMPLS.keys()))
def test_stronger_causal_signal_lower_pvalue(rng, method):
    """Stronger causal signal (higher coupling) -> lower p-value.

    Numerical monotonicity: F-statistic increases with stronger signal
    """
    T = 200
    ts1 = rng.normal(0, 1, T)
    lag_step = 3
    pvals = []
    for coupling in [0.0, 0.2, 0.5, 0.8]:
        noise = 0.05 * rng.normal(0, 1, T)
        ts2 = np.zeros(T)
        ts2[lag_step:] = coupling * ts1[:-lag_step]
        ts2 += noise
        pvals.append(gt_single_lag_fast(ts1, ts2, lag_step, method))

    # p-values should decrease as coupling increases
    for i in range(len(pvals) - 1):
        assert (
            pvals[i] >= pvals[i + 1] - APPROX_NEAR_SINGULAR
        ), f"method={method}: p[{i}]={pvals[i]:.3e} ≥ p[{i + 1}]={pvals[i + 1]:.3e}"


@pytest.mark.parametrize("method", list(F_TEST_IMPLS.keys()))
def test_deterministic(rng, method):
    """Two calls with identical input produce identical output."""
    ts1 = rng.normal(0, 1, 100)
    ts2 = rng.normal(0, 1, 100)
    p1 = gt_single_lag_fast(ts1, ts2, 5, method)
    p2 = gt_single_lag_fast(ts1, ts2, 5, method)
    assert p1 == p2
