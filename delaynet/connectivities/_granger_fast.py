"""
Fast Granger causality F-test implementations avoiding statsmodels overhead.

The statsmodels ``OLS.fit()`` + ``f_test()`` path carries significant per-call
overhead from class instantiation, data validation, and result-object tree
construction.  For 51,450 regressions (2,450 pairs × 21 lags) this overhead
dominates.  This module provides drop-in replacements that compute the same
p-value using only numpy/scipy linear algebra, achieving an ~8× speedup
with identical output.

Exposed API
-----------
``gt_single_lag_fast``   – per-lag Granger F-test (replaces ``gt_single_lag``)
``gt_multi_lag_fast``    – multi-lag (replaces ``gt_multi_lag``; wired as ``"gc"``)
``F_TEST_IMPLS``         – dict mapping method names to low-level F-test callables
``run_all_benchmarks``   – convenience benchmark runner
"""

from __future__ import annotations

import time as _time

import numpy as np
from numpy.typing import NDArray as _NDArray
from scipy.linalg import lstsq as _sp_lstsq
from scipy.linalg import solve_triangular as _sp_solve_triangular
from scipy.stats import f as _f_dist

from ..decorators import connectivity as _connectivity


# ---------------------------------------------------------------------------
# Low-level OLS + F-test building blocks
# ---------------------------------------------------------------------------


def _ols_f_test_normal_eqn(  # noqa: D103
    y: _NDArray[np.float64],
    X: _NDArray[np.float64],
    R: _NDArray[np.float64],
) -> float:
    n, k = X.shape
    q = R.shape[0]

    XTX = X.T @ X
    try:
        beta = np.linalg.solve(XTX, X.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]

    resid = y - X @ beta
    rss = float(resid @ resid)
    dof = n - k
    sigma2 = rss / dof

    try:
        XTX_inv = np.linalg.inv(XTX)
    except np.linalg.LinAlgError:
        XTX_inv = np.linalg.pinv(XTX)

    Rb = R @ beta
    R_vcov_R = R @ XTX_inv @ R.T
    R_vcov_R *= sigma2

    try:
        inner = np.linalg.solve(R_vcov_R, Rb)
    except np.linalg.LinAlgError:
        inner = np.linalg.lstsq(R_vcov_R, Rb, rcond=None)[0]

    f_stat = float(Rb @ inner) / q
    return float(_f_dist.sf(f_stat, q, dof))


def _ols_f_test_qr(  # noqa: D103
    y: _NDArray[np.float64],
    X: _NDArray[np.float64],
    R: _NDArray[np.float64],
) -> float:
    n, k = X.shape
    q = R.shape[0]

    Q, R_qr = np.linalg.qr(X)
    QTy = Q.T @ y
    beta = _sp_solve_triangular(R_qr[:k], QTy[:k])

    resid = y - X @ beta
    rss = float(resid @ resid)
    dof = n - k
    sigma2 = rss / dof

    R_inv = _sp_solve_triangular(R_qr[:k], np.eye(k), lower=False)
    XTX_inv = R_inv @ R_inv.T

    Rb = R @ beta
    R_vcov_R = R @ XTX_inv @ R.T
    R_vcov_R *= sigma2

    try:
        inner = np.linalg.solve(R_vcov_R, Rb)
    except np.linalg.LinAlgError:
        inner = np.linalg.lstsq(R_vcov_R, Rb, rcond=None)[0]

    f_stat = float(Rb @ inner) / q
    return float(_f_dist.sf(f_stat, q, dof))


def _ols_f_test_cholesky(
    y: _NDArray[np.float64],
    X: _NDArray[np.float64],
    R: _NDArray[np.float64],
) -> float:
    n, k = X.shape
    q = R.shape[0]

    XTX = X.T @ X
    try:
        L = np.linalg.cholesky(XTX)
    except np.linalg.LinAlgError:
        return _ols_f_test_normal_eqn(y, X, R)

    z = _sp_solve_triangular(L, X.T @ y, lower=True)
    beta = _sp_solve_triangular(L.T, z, lower=False)

    resid = y - X @ beta
    rss = float(resid @ resid)
    dof = n - k
    sigma2 = rss / dof

    L_inv = _sp_solve_triangular(L, np.eye(k), lower=True)
    XTX_inv = L_inv.T @ L_inv

    Rb = R @ beta
    R_vcov_R = R @ XTX_inv @ R.T
    R_vcov_R *= sigma2

    try:
        inner = np.linalg.solve(R_vcov_R, Rb)
    except np.linalg.LinAlgError:
        inner = np.linalg.lstsq(R_vcov_R, Rb, rcond=None)[0]

    f_stat = float(Rb @ inner) / q
    return float(_f_dist.sf(f_stat, q, dof))


def _ols_f_test_scipy_lstsq(
    y: _NDArray[np.float64],
    X: _NDArray[np.float64],
    R: _NDArray[np.float64],
) -> float:
    n, k = X.shape
    q = R.shape[0]

    beta, _, _, _ = _sp_lstsq(X, y)

    resid = y - X @ beta
    rss = float(resid @ resid)
    dof = n - k
    sigma2 = rss / dof

    XTX = X.T @ X
    try:
        XTX_inv = np.linalg.inv(XTX)
    except np.linalg.LinAlgError:
        XTX_inv = np.linalg.pinv(XTX)

    Rb = R @ beta
    R_vcov_R = R @ XTX_inv @ R.T
    R_vcov_R *= sigma2

    try:
        inner = np.linalg.solve(R_vcov_R, Rb)
    except np.linalg.LinAlgError:
        inner = np.linalg.lstsq(R_vcov_R, Rb, rcond=None)[0]

    f_stat = float(Rb @ inner) / q
    return float(_f_dist.sf(f_stat, q, dof))


def _ols_f_test_two_regression(
    y: _NDArray[np.float64],
    X: _NDArray[np.float64],
    R: _NDArray[np.float64],
) -> float:
    """F-test via unrestricted-vs-restricted RSS comparison (QR, no inversion)."""
    n, k = X.shape
    q = R.shape[0]

    Q, R_qr = np.linalg.qr(X)
    QTy = Q.T @ y
    beta_full = _sp_solve_triangular(R_qr[:k], QTy[:k])
    rss_full = float(np.sum((y - X @ beta_full) ** 2))

    restricted_mask = np.all(np.abs(R) < 1e-12, axis=0)
    X_r = X[:, restricted_mask]
    Q_r, R_qr_r = np.linalg.qr(X_r)
    k_r = X_r.shape[1]
    QTy_r = Q_r.T @ y
    beta_r = _sp_solve_triangular(R_qr_r[:k_r], QTy_r[:k_r])
    rss_r = float(np.sum((y - X_r @ beta_r) ** 2))

    dof = n - k
    f_stat = ((rss_r - rss_full) / q) / (rss_full / dof)
    return float(_f_dist.sf(f_stat, q, dof))


def _ols_f_test_two_regression_chol(
    y: _NDArray[np.float64],
    X: _NDArray[np.float64],
    R: _NDArray[np.float64],
) -> float:
    """F-test via unrestricted-vs-restricted RSS comparison (Cholesky, no inversion)."""
    n, k = X.shape
    q = R.shape[0]

    XTX = X.T @ X
    L = np.linalg.cholesky(XTX)
    z = _sp_solve_triangular(L, X.T @ y, lower=True)
    beta_full = _sp_solve_triangular(L.T, z, lower=False)
    rss_full = float(np.sum((y - X @ beta_full) ** 2))

    restricted_mask = np.all(np.abs(R) < 1e-12, axis=0)
    X_r = X[:, restricted_mask]
    XTX_r = X_r.T @ X_r
    L_r = np.linalg.cholesky(XTX_r)
    z_r = _sp_solve_triangular(L_r, X_r.T @ y, lower=True)
    beta_r = _sp_solve_triangular(L_r.T, z_r, lower=False)
    rss_r = float(np.sum((y - X_r @ beta_r) ** 2))

    dof = n - k
    f_stat = ((rss_r - rss_full) / q) / (rss_full / dof)
    return float(_f_dist.sf(f_stat, q, dof))


# Public registry of F-test implementations
F_TEST_IMPLS: dict[str, callable] = {
    "normal_eqn": _ols_f_test_normal_eqn,
    "qr": _ols_f_test_qr,
    "cholesky": _ols_f_test_cholesky,
    "scipy_lstsq": _ols_f_test_scipy_lstsq,
    "two_regression": _ols_f_test_two_regression,
    "two_regression_chol": _ols_f_test_two_regression_chol,
}

# Keep internal alias for backwards compatibility
_F_TEST_IMPLS = F_TEST_IMPLS


# ---------------------------------------------------------------------------
# Design matrix helpers
# ---------------------------------------------------------------------------


def _build_lag_design_matrix(
    ts1: _NDArray[np.float64],
    ts2: _NDArray[np.float64],
    lag_step: int,
) -> tuple[_NDArray[np.float64], _NDArray[np.float64]]:
    """Build (y, X) for a single lag step using trim="both".

    Matches the reference statsmodels implementation exactly.
    """
    from statsmodels.tools.tools import add_constant
    from statsmodels.tsa.tsatools import lagmat2ds

    full_ts = np.column_stack([ts2, ts1])
    dta = lagmat2ds(full_ts, lag_step, trim="both", dropex=1)
    dtajoint = add_constant(dta[:, 1:], prepend=False)

    y = np.ascontiguousarray(dta[:, 0], dtype=np.float64)
    X = np.ascontiguousarray(dtajoint, dtype=np.float64)
    return y, X


def _build_restriction_matrix(lag_step: int) -> _NDArray[np.float64]:
    """Build R matrix testing H0: all cross-lag coefficients = 0."""
    return np.column_stack(
        (
            np.zeros((lag_step, lag_step)),
            np.eye(lag_step, lag_step),
            np.zeros((lag_step, 1)),
        )
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def gt_single_lag_fast(
    ts1: _NDArray[np.float64],
    ts2: _NDArray[np.float64],
    lag_step: int,
    method: str = "normal_eqn",
) -> float:
    """Granger causality F-test p-value for a fixed lag — fast version.

    Drop-in replacement for :func:`delaynet.connectivities.granger.gt_single_lag`.
    Produces identical p-values while avoiding statsmodels overhead.

    Parameters
    ----------
    ts1 : (T,) array
        Candidate cause time series.
    ts2 : (T,) array
        Effect time series.
    lag_step : int
        Number of lags to include.
    method : str, optional
        F-test implementation. One of ``"normal_eqn"`` (default, fastest),
        ``"qr"``, ``"cholesky"``, ``"scipy_lstsq"``, ``"two_regression"``,
        ``"two_regression_chol"``.

    Returns
    -------
    p_value : float
    """
    y, X = _build_lag_design_matrix(ts1, ts2, lag_step)
    R = _build_restriction_matrix(lag_step)
    return F_TEST_IMPLS[method](y, X, R)


@_connectivity
def gt_multi_lag_fast(
    ts1: _NDArray[np.float64],
    ts2: _NDArray[np.float64],
    *,
    lag_steps: list[int] | None = None,
    method: str = "normal_eqn",
    **kwargs,
) -> tuple[float, int]:
    """Multi-lag Granger causality — fast drop-in replacement for ``gt_multi_lag``.

    Evaluates each lag in ``lag_steps`` via :func:`gt_single_lag_fast`
    (identical p-values to statsmodels) and returns the combination
    with the smallest p-value.

    Registered as ``"gc"`` / ``"granger causality"`` via the connectivity
    dispatch in :mod:`delaynet.connectivities`.

    Parameters
    ----------
    ts1 : (T,) array
        Candidate cause time series.
    ts2 : (T,) array
        Effect time series.
    lag_steps : list[int]
        Lag values to evaluate (the ``@connectivity`` decorator converts
        ``int`` to ``list[int]`` automatically).
    method : str, optional
        F-test implementation.  Default ``"normal_eqn"`` (fastest).

    Returns
    -------
    best_p_value : float
    best_lag : int
    """
    _ = kwargs  # consumed by @connectivity decorator
    p_values = [gt_single_lag_fast(ts1, ts2, lag, method) for lag in lag_steps]
    idx_best = int(np.argmin(p_values))
    return p_values[idx_best], lag_steps[idx_best]


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def _bench_single(
    y: _NDArray[np.float64],
    X: _NDArray[np.float64],
    R: _NDArray[np.float64],
    impl: callable,
    n_repeats: int,
) -> dict:
    """Time a single F-test implementation."""
    _ = impl(y, X, R)  # warmup
    times = []
    for _ in range(n_repeats):
        t0 = _time.perf_counter()
        _ = impl(y, X, R)
        times.append(_time.perf_counter() - t0)
    times_ms = 1000.0 * np.array(times)
    return {
        "mean_ms": float(np.mean(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "p_value": float(impl(y, X, R)),
    }


def _bench_statsmodels(
    y: _NDArray[np.float64],
    X: _NDArray[np.float64],
    R: _NDArray[np.float64],
    n_repeats: int,
) -> dict:
    """Time the reference statsmodels OLS + f_test."""
    from statsmodels.regression.linear_model import OLS

    res = OLS(y, X).fit()
    _ = res.f_test(R)  # warmup
    times = []
    for _ in range(n_repeats):
        t0 = _time.perf_counter()
        res = OLS(y, X).fit()
        ftres = res.f_test(R)
        pv = float(np.squeeze(ftres.pvalue)[()])
        times.append(_time.perf_counter() - t0)

    times_ms = 1000.0 * np.array(times)
    return {
        "mean_ms": float(np.mean(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "p_value": float(pv),
    }


def run_all_benchmarks(
    ts_len: int = 200,
    n_lags: int = 21,
    n_repeats: int = 200,
    seed: int = 42,
) -> None:
    """Print timing comparison of all F-test implementations vs statsmodels."""
    rng = np.random.default_rng(seed)
    ts1 = rng.normal(0, 1, ts_len)
    ts2 = rng.normal(0, 1, ts_len)

    y, X = _build_lag_design_matrix(ts1, ts2, n_lags)
    R = _build_restriction_matrix(n_lags)

    print(f"Benchmark: T={ts_len}, L={n_lags}, repeats={n_repeats}")
    print(f"  design matrix shape: {X.shape}")
    print()

    results = {}

    res_sm = _bench_statsmodels(y, X, R, n_repeats)
    results["statsmodels"] = res_sm
    print(
        f"  {'statsmodels':<22s}  {res_sm['mean_ms']:9.3f} ms"
        f"  ± {res_sm['std_ms']:.3f}  p={res_sm['p_value']:.6e}"
    )

    for name in F_TEST_IMPLS:
        res = _bench_single(y, X, R, F_TEST_IMPLS[name], n_repeats)
        results[name] = res
        speedup = res_sm["mean_ms"] / res["mean_ms"]
        p_match = abs(res["p_value"] - res_sm["p_value"]) < 1e-12
        status = "✓" if p_match else "✗"
        print(
            f"  {name:<22s}  {res['mean_ms']:9.3f} ms"
            f"  ± {res['std_ms']:.3f}"
            f"  p={res['p_value']:.6e}  {status}  ({speedup:.1f}x)"
        )

    print()
    best = min(results, key=lambda k: results[k]["mean_ms"])
    print(
        f"  Fastest: {best} ({results[best]['mean_ms']:.3f} ms, "
        f"{results['statsmodels']['mean_ms'] / results[best]['mean_ms']:.1f}x"
        f" vs statsmodels)"
    )


if __name__ == "__main__":
    run_all_benchmarks()
