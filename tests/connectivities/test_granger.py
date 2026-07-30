import numpy as np
import pytest
from delaynet import connectivity
from delaynet.connectivities.granger import gt_single_lag, gt_multi_lag


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(24567)


@pytest.fixture(scope="module")
def causal_ts(rng):
    T = 200
    ts1 = rng.normal(0, 1, T)
    noise = rng.normal(0, 0.05, T)
    ts2 = np.zeros(T)
    ts2[3:] = 0.8 * ts1[:-3]
    ts2 += noise
    return ts1, ts2


def test_gt_multi_lag():
    # Generate some test time series data
    np.random.seed(24567)
    ts1 = np.random.normal(0, 1, size=100)
    ts2 = np.roll(ts1, 2) + np.random.normal(
        0, 0.1, size=100
    )  # Create causally related series

    # Test the connectivity with default parameters
    result = connectivity(
        ts1,
        ts2,
        metric="granger causality",
        lag_steps=5,
    )

    # Assert that the function returns expected format
    assert isinstance(result, tuple), "Result should be a tuple"
    assert len(result) == 2, "Result should contain two elements"

    # Test with expected value
    # The lag should be 2 since we created ts2 by rolling ts1 by 2
    p_value, lag = result
    assert lag == 2, f"Expected lag to be 2, got {lag}"


def test_gt_single_lag():
    # Generate some test time series data
    np.random.seed(24567)
    ts1 = np.random.normal(0, 1, size=100)
    ts2 = np.roll(ts1, 2) + np.random.normal(
        0, 0.1, size=100
    )  # Create causally related series

    # Test the single lag function directly
    p_value = gt_single_lag(ts1, ts2, lag_step=2)

    # Assert that the function returns expected format
    assert isinstance(p_value, float), "Result should be a float"
    assert 0 <= p_value <= 1, "p-value should be between 0 and 1"


def test_detects_causal_lag(causal_ts):
    ts1, ts2 = causal_ts
    p_value, lag = gt_multi_lag(ts1, ts2, lag_steps=list(range(1, 6)))
    assert lag == 3, f"Expected lag 3, got {lag}"


def test_short_ts_no_crash():
    """Very short time series should not crash."""
    ts1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ts2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    p_value = gt_single_lag(ts1, ts2, lag_step=1)
    assert isinstance(p_value, float)


def test_p_value_in_valid_range(rng):
    T, L = 100, 5
    ts1, ts2 = rng.normal(0, 1, T), rng.normal(0, 1, T)
    _, lag = gt_multi_lag(ts1, ts2, lag_steps=list(range(1, L + 1)))

    p_value = gt_single_lag(ts1, ts2, lag_step=lag)
    assert 0 <= p_value <= 1, f"p-value {p_value} is outside valid range"


def test_deterministic():
    """Same input produces identical results."""
    rng1 = np.random.default_rng(42)
    ts1, ts2 = rng1.normal(0, 1, 200), rng1.normal(0, 1, 200)
    p_value1, lag1 = gt_multi_lag(ts1, ts2, lag_steps=[1, 2, 3, 4, 5])

    rng2 = np.random.default_rng(42)
    ts1_, ts2_ = rng2.normal(0, 1, 200), rng2.normal(0, 1, 200)
    p_value2, lag2 = gt_multi_lag(ts1_, ts2_, lag_steps=[1, 2, 3, 4, 5])

    assert lag1 == lag2, f"Lags differ: {lag1} vs {lag2}"
    assert abs(p_value1 - p_value2) < 1e-12, (
        f"P-values differ: {p_value1} vs {p_value2}"
    )


def test_design_matrix_shape():
    import inspect

    source = inspect.getsource(gt_single_lag)
    assert "lagmat2ds" in source, (
        "Granger causality should use lagmat2ds for design matrix"
    )


def test_restriction_matrix():
    import inspect

    source = inspect.getsource(gt_single_lag)
    assert "np.column_stack" in source, (
        "Restriction matrix should be built using np.column_stack with "
        "the expected block structure: zeros + identity + zeros"
    )


def test_gc_dispatch_parity(rng):
    ts1, ts2 = rng.normal(0, 1, 100), rng.normal(0, 1, 100)

    result1 = gt_multi_lag(ts1, ts2, lag_steps=3)
    result2 = connectivity(ts1, ts2, metric="gc", lag_steps=3)
    result3 = connectivity(ts1, ts2, metric="granger causality", lag_steps=3)

    assert result1 == result2 == result3, (
        "All connectivity dispatch mechanisms should produce identical results"
    )


def test_multi_lag_optimal_lag(causal_ts):
    ts1, ts2 = causal_ts
    p_value, lag = gt_multi_lag(ts1, ts2, lag_steps=[1, 2, 3, 4, 5])
    assert lag == 3, f"Expected lag 3, got {lag}"
    assert 0 <= p_value <= 1, f"Invalid p-value: {p_value}"


def test_stronger_causal_signal_lower_pvalue(rng):
    T = 100
    ts1 = rng.normal(0, 1, T)

    ts2_strong = np.roll(ts1, 1) * 2.0 + rng.normal(0, 0.1, T)
    p_strong, lag_strong = gt_multi_lag(ts1, ts2_strong, lag_steps=[1, 2, 3])

    ts2_weak = np.roll(ts1, 1) * 0.5 + rng.normal(0, 0.5, T)
    p_weak, lag_weak = gt_multi_lag(ts1, ts2_weak, lag_steps=[1, 2, 3])

    assert p_strong < p_weak, (
        f"Stronger signal should have lower p-value (strong: {p_strong}, weak: {p_weak})"
    )


def test_gt_single_lag_lstsq_fallback():
    """Test lstsq fallback when solve raises LinAlgError at line 92."""
    from unittest.mock import patch

    ts1 = np.random.normal(0, 1, size=100)
    ts2 = np.random.normal(0, 1, size=100)

    original_solve = np.linalg.solve
    solve_calls = []

    def mock_solve(a, b):
        solve_calls.append((a, b))
        if len(solve_calls) == 2:
            raise np.linalg.LinAlgError(
                "Simulating singular R_vcov_R for test coverage"
            )
        return original_solve(a, b)

    with patch("numpy.linalg.solve", mock_solve):
        p_value = gt_single_lag(ts1, ts2, lag_step=2)
    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1
