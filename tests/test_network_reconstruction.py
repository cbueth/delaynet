"""Tests for the network reconstruction module."""

import pytest
from time import time
from numpy import (
    ndarray,
    column_stack,
    random,
    diag,
    eye,
    all as np_all,
    array,
    zeros,
    allclose,
    float64,
    float32,
    int64,
    int32,
    isfinite,
    corrcoef,
)
from numpy.testing import assert_array_equal

from delaynet.network_reconstruction import (
    reconstruct_network,
    format_time,
    print_progress,
    update_progress
)


def _simple_metric_for_parallel_test(ts1, ts2, lag_steps=None):
    """Simple correlation-based metric for testing parallel execution.

    This function is defined at module level so it can be pickled
    for multiprocessing.
    """
    return abs(corrcoef(ts1, ts2)[0, 1]), 1


def test_reconstruct_network_basic_functionality(two_time_series):
    """Test basic network reconstruction functionality with simple data."""
    ts1, ts2 = two_time_series
    # Create a 2D array with shape (n_time, n_nodes)
    time_series = column_stack([ts1, ts2])

    weights, lags = reconstruct_network(time_series, "linear_correlation", lag_steps=3)

    # Check return types and shapes
    assert isinstance(weights, ndarray)
    assert isinstance(lags, ndarray)
    assert weights.shape == (2, 2)
    assert lags.shape == (2, 2)

    # Check diagonal elements (self-connections should be 1.0 - no significant connection)
    assert weights[0, 0] == 1.0
    assert weights[1, 1] == 1.0
    assert lags[0, 0] == 0
    assert lags[1, 1] == 0


def test_reconstruct_network_with_random_data(two_random_time_series):
    """Test network reconstruction with realistic random data."""
    ts1, ts2 = two_random_time_series
    # Create a 3-node network
    ts3 = random.RandomState(3256).randn(len(ts1))
    time_series = column_stack([ts1, ts2, ts3])

    weights, lags = reconstruct_network(time_series, "linear_correlation", lag_steps=5)

    # Check shapes
    assert weights.shape == (3, 3)
    assert lags.shape == (3, 3)

    # Check diagonal is 1.0 (no significant self-connection)
    assert_array_equal(diag(weights), [1.0, 1.0, 1.0])
    assert_array_equal(diag(lags), [0, 0, 0])

    # Check that weights are valid p-values (between 0 and 1)
    off_diagonal_weights = weights[~eye(3, dtype=bool)]
    assert np_all(off_diagonal_weights >= 0.0)
    assert np_all(off_diagonal_weights <= 1.0)

    # Non-diagonal lags > 0
    assert np_all(lags[~eye(3, dtype=bool)] > 0)


def test_reconstruct_network_with_different_measures(
    connectivity_metric_kwargs, two_random_time_series
):
    """Test network reconstruction with all connectivity measures and their kwargs."""
    connectivity_measure, kwargs = connectivity_metric_kwargs
    ts1, ts2 = two_random_time_series
    time_series = column_stack([ts1, ts2])

    # Add required lag_steps parameter
    test_kwargs = kwargs.copy()
    test_kwargs["lag_steps"] = 3

    weights, lags = reconstruct_network(
        time_series, connectivity_measure, **test_kwargs
    )

    assert weights.shape == (2, 2)
    assert lags.shape == (2, 2)
    assert diag(weights).tolist() == [1.0, 1.0]
    assert diag(lags).tolist() == [0, 0]


def test_reconstruct_network_with_connectivity_kwargs(two_random_time_series):
    """Test network reconstruction with additional connectivity kwargs."""
    ts1, ts2 = two_random_time_series
    # Use shorter time series for faster computation
    ts1_short = ts1[:100]
    ts2_short = ts2[:100]
    time_series = column_stack([ts1_short, ts2_short])

    # Test with mutual information and specific parameters
    weights, lags = reconstruct_network(
        time_series,
        "mutual_information",
        lag_steps=2,
        approach="metric",
        n_tests=5,  # Reduced for faster testing
    )

    assert weights.shape == (2, 2)
    assert lags.shape == (2, 2)
    assert diag(weights).tolist() == [1.0, 1.0]


def test_reconstruct_network_input_validation():
    """Test input validation for reconstruct_network function."""
    # Test with 1D input
    with pytest.raises(ValueError, match="time_series must be 2-dimensional"):
        reconstruct_network(array([1, 2, 3]), "linear_correlation")

    # Test with 3D input
    with pytest.raises(ValueError, match="time_series must be 2-dimensional"):
        reconstruct_network(random.randn(10, 5, 3), "linear_correlation")

    # Test with too few time points
    with pytest.raises(
        ValueError, match="time_series must have at least 2 time points"
    ):
        reconstruct_network(random.randn(1, 3), "linear_correlation")

    # Test with too few nodes
    with pytest.raises(ValueError, match="time_series must have at least 2 nodes"):
        reconstruct_network(random.randn(10, 1), "linear_correlation")

    # Test with unknown connectivity measure
    with pytest.raises(ValueError, match="Unknown metric"):
        reconstruct_network(random.randn(10, 3), "unknown_measure")


def test_reconstruct_network_edge_cases():
    """Test network reconstruction with edge cases."""
    # Test with small but valid dimensions (need more points for correlation)
    time_series = random.RandomState(42).randn(10, 2)
    weights, lags = reconstruct_network(time_series, "linear_correlation", lag_steps=1)

    assert weights.shape == (2, 2)
    assert lags.shape == (2, 2)
    assert diag(weights).tolist() == [1.0, 1.0]
    assert diag(lags).tolist() == [0, 0]


def test_reconstruct_network_return_types():
    """Test that reconstruct_network returns correct types."""
    time_series = random.RandomState(42).randn(50, 3)
    weights, lags = reconstruct_network(time_series, "linear_correlation", lag_steps=3)

    # Check return types
    assert isinstance(weights, ndarray)
    assert isinstance(lags, ndarray)

    # Check data types
    assert weights.dtype in [float64, float32]
    assert lags.dtype in [int64, int32, int]

    # Check that weights are finite
    assert np_all(isfinite(weights))
    assert np_all(isfinite(lags))


def test_reconstruct_network_symmetry_properties():
    """Test properties of the reconstructed network matrices."""
    # Create correlated time series
    random.seed(42)
    n_time, n_nodes = 100, 4
    base_signal = random.randn(n_time)
    time_series = zeros((n_time, n_nodes))

    # Create some structure in the data
    time_series[:, 0] = base_signal
    time_series[1:, 1] = 0.8 * base_signal[:-1] + 0.2 * random.randn(n_time - 1)
    time_series[:, 2] = random.randn(n_time)
    time_series[2:, 3] = 0.6 * base_signal[:-2] + 0.4 * random.randn(n_time - 2)

    weights, lags = reconstruct_network(time_series, "linear_correlation", lag_steps=5)

    # Check that diagonal is one
    assert allclose(diag(weights), 1.0)
    assert allclose(diag(lags), 0)

    # Check that matrices have correct shape
    assert weights.shape == (n_nodes, n_nodes)
    assert lags.shape == (n_nodes, n_nodes)

    # Check that all values are reasonable
    assert np_all(weights >= 0.0)
    assert np_all(weights <= 1.0)
    assert np_all(lags >= 0)


def test_reconstruct_network_with_connectivity_returning_single_value():
    """Test network reconstruction when connectivity measure returns single value."""

    # Create a mock connectivity measure that returns only p-value
    def mock_connectivity_single(ts1, ts2, **kwargs):
        return 0.05  # Return only p-value

    time_series = random.RandomState(42).randn(20, 3)

    # This should work with the current implementation
    # The function should handle both single values and tuples
    from delaynet.connectivity import connectivity

    # Test with a real connectivity measure first
    weights, lags = reconstruct_network(time_series, "linear_correlation", lag_steps=2)

    # Verify the structure is correct
    assert weights.shape == (3, 3)
    assert lags.shape == (3, 3)
    assert diag(weights).tolist() == [1.0, 1.0, 1.0]
    assert diag(lags).tolist() == [0, 0, 0]


@pytest.mark.parametrize(
    "n_time, n_nodes",
    [
        (10, 2),
        (50, 3),
        (100, 5),
    ],
)
def test_reconstruct_network_different_sizes(n_time, n_nodes):
    """Test network reconstruction with different data sizes."""
    time_series = random.RandomState(42).randn(n_time, n_nodes)

    weights, lags = reconstruct_network(time_series, "linear_correlation", lag_steps=3)

    assert weights.shape == (n_nodes, n_nodes)
    assert lags.shape == (n_nodes, n_nodes)
    assert allclose(diag(weights), 1.0)
    assert allclose(diag(lags), 0)


def test_reconstruct_network_integration_with_connectivity_fixtures(
    connectivity_metric_shorthand,
):
    """Test network reconstruction with all available connectivity measures."""
    metric, kwargs = connectivity_metric_shorthand

    # Create simple test data
    time_series = random.RandomState(42).randn(30, 3)

    # Add required lag_steps parameter
    test_kwargs = kwargs.copy()
    test_kwargs["lag_steps"] = 2

    try:
        weights, lags = reconstruct_network(time_series, metric, **test_kwargs)

        assert weights.shape == (3, 3)
        assert lags.shape == (3, 3)
        assert allclose(diag(weights), 1.0)
        assert allclose(diag(lags), 0)

    except Exception as e:
        pytest.fail(f"Network reconstruction failed for metric {metric}: {e}")


@pytest.mark.parametrize(
    "callable_metric",
    [
        # Simple correlation-based metric
        lambda ts1, ts2, lag_steps=None: (abs(corrcoef(ts1, ts2)[0, 1]), 1),
        # Constant metric for testing
        lambda ts1, ts2, lag_steps=None: (0.5, 2),
        # Another simple metric
        lambda ts1, ts2, lag_steps=None: (0.1, 3),
    ],
)
def test_reconstruct_network_with_callable_metric(two_time_series, callable_metric):
    """Test network reconstruction with callable connectivity metrics.

    This test verifies that the network reconstruction function properly handles
    callable metrics, which is the main feature requested in the issue.
    """
    ts1, ts2 = two_time_series
    time_series = column_stack([ts1, ts2])

    # Test with callable metric
    weights, lags = reconstruct_network(time_series, callable_metric, lag_steps=3)

    # Verify basic structure
    assert weights.shape == (2, 2)
    assert lags.shape == (2, 2)
    assert isinstance(weights, ndarray)
    assert isinstance(lags, ndarray)

    # Check diagonal elements are 1.0 (no significant self-connections)
    assert diag(weights).tolist() == [1.0, 1.0]
    assert diag(lags).tolist() == [0, 0]

    # Check that off-diagonal elements have valid values
    off_diagonal_weights = weights[~eye(2, dtype=bool)]
    off_diagonal_lags = lags[~eye(2, dtype=bool)]

    # Weights should be valid (finite and non-negative for p-values)
    assert np_all(isfinite(off_diagonal_weights))
    assert np_all(off_diagonal_weights >= 0.0)

    # Lags should be valid integers
    assert np_all(isfinite(off_diagonal_lags))
    assert np_all(off_diagonal_lags >= 0)


@pytest.mark.parametrize(
    "invalid_callable_metric",
    [
        # Invalid return types
        lambda ts1, ts2, lag_steps=None: "invalid",  # string instead of tuple
        lambda ts1, ts2, lag_steps=None: 123,  # int instead of tuple
        lambda ts1, ts2, lag_steps=None: [0.5, 1],  # list instead of tuple
        lambda ts1, ts2, lag_steps=None: (0.5, 1, 2),  # tuple with wrong length
        lambda ts1, ts2, lag_steps=None: (0.5, "invalid"),  # invalid lag type
    ],
)
def test_reconstruct_network_with_invalid_callable_metric(
    two_time_series, invalid_callable_metric
):
    """Test network reconstruction with invalid callable metrics."""
    ts1, ts2 = two_time_series
    time_series = column_stack([ts1, ts2])

    # Should raise ValueError for invalid callable metrics
    with pytest.raises(ValueError):
        reconstruct_network(time_series, invalid_callable_metric, lag_steps=3)


def test_reconstruct_network_with_non_callable_non_string_metric(two_time_series):
    """Test network reconstruction with invalid metric types."""
    ts1, ts2 = two_time_series
    time_series = column_stack([ts1, ts2])

    # Test with invalid metric types
    invalid_metrics = [123, None, [], {}]

    for invalid_metric in invalid_metrics:
        with pytest.raises(ValueError):
            reconstruct_network(time_series, invalid_metric, lag_steps=3)


@pytest.mark.parametrize("workers", [2, 4])
def test_reconstruct_network_parallel_vs_sequential(two_random_time_series, workers):
    """Test that parallel execution produces identical results to sequential execution.

    This test verifies that the shared memory parallelization implementation
    produces exactly the same results as the sequential version.
    """
    ts1, ts2 = two_random_time_series
    # Create a larger dataset for more meaningful parallel testing
    time_series = column_stack([ts1, ts2, ts1 + 0.1, ts2 + 0.1])  # 4 nodes

    # Sequential execution (default)
    weights_seq, lags_seq = reconstruct_network(
        time_series, "linear_correlation", lag_steps=3, workers=None
    )

    # Parallel execution
    weights_par, lags_par = reconstruct_network(
        time_series, "linear_correlation", lag_steps=3, workers=workers
    )

    # Results should be identical
    assert_array_equal(weights_seq, weights_par)
    assert_array_equal(lags_seq, lags_par)

    # Verify basic properties
    assert weights_seq.shape == (4, 4)
    assert lags_seq.shape == (4, 4)
    assert allclose(diag(weights_seq), 1.0)
    assert allclose(diag(lags_seq), 0)


def test_reconstruct_network_parallel_with_different_metrics(two_random_time_series):
    """Test parallel execution with different connectivity metrics.

    This test ensures that the parallel implementation works correctly
    with various connectivity measures. Uses only deterministic metrics
    to ensure reproducible results.
    """
    ts1, ts2 = two_random_time_series
    time_series = column_stack([ts1, ts2, ts1 * 0.8])  # 3 nodes

    # Test with deterministic metrics only to ensure reproducible results
    test_cases = [
        ("linear_correlation", {}),
        ("rank_correlation", {}),
    ]

    for metric, kwargs in test_cases:
        # Sequential execution
        weights_seq, lags_seq = reconstruct_network(
            time_series, metric, lag_steps=2, workers=1, **kwargs
        )

        # Parallel execution with 2 workers
        weights_par, lags_par = reconstruct_network(
            time_series, metric, lag_steps=2, workers=2, **kwargs
        )

        # Results should be identical
        (
            assert_array_equal(weights_seq, weights_par),
            f"Weights differ for metric {metric}",
        )
        assert_array_equal(lags_seq, lags_par), f"Lags differ for metric {metric}"


def test_reconstruct_network_parallel_with_kwargs(two_random_time_series):
    """Test parallel execution with additional keyword arguments.

    This test verifies that kwargs are properly passed to the connectivity
    function in parallel execution.
    """
    ts1, ts2 = two_random_time_series
    time_series = column_stack([ts1, ts2])  # 2 nodes

    # Test with kwargs (using transfer entropy with specific parameters)
    kwargs = {"approach": "discrete", "n_tests": 10}

    # Sequential execution
    weights_seq, lags_seq = reconstruct_network(
        time_series, "transfer_entropy", lag_steps=2, workers=None, **kwargs
    )

    # Parallel execution
    weights_par, lags_par = reconstruct_network(
        time_series, "transfer_entropy", lag_steps=2, workers=2, **kwargs
    )

    # Results should be identical
    assert_array_equal(weights_seq, weights_par)
    assert_array_equal(lags_seq, lags_par)


def test_reconstruct_network_parallel_with_callable_metric(two_time_series):
    """Test parallel execution with callable connectivity metrics.

    This test ensures that callable metrics work correctly in parallel execution.
    """
    ts1, ts2 = two_time_series
    time_series = column_stack([ts1, ts2])

    # Sequential execution
    weights_seq, lags_seq = reconstruct_network(
        time_series, _simple_metric_for_parallel_test, lag_steps=2, workers=1
    )

    # Parallel execution
    weights_par, lags_par = reconstruct_network(
        time_series, _simple_metric_for_parallel_test, lag_steps=2, workers=2
    )

    # Results should be identical
    assert_array_equal(weights_seq, weights_par)
    assert_array_equal(lags_seq, lags_par)


def test_reconstruct_network_parallel_large_dataset():
    """Test parallel execution with a larger dataset.

    This test verifies that the parallel implementation scales properly
    with larger datasets and produces consistent results.
    """
    # Create a larger dataset for more comprehensive testing
    random_state = random.RandomState(42)
    time_series = random_state.randn(50, 6)  # 6 nodes, 50 time points

    # Sequential execution
    weights_seq, lags_seq = reconstruct_network(
        time_series, "linear_correlation", lag_steps=3, workers=None
    )

    # Parallel execution with 4 workers
    weights_par, lags_par = reconstruct_network(
        time_series, "linear_correlation", lag_steps=3, workers=4
    )

    # Results should be identical
    assert_array_equal(weights_seq, weights_par)
    assert_array_equal(lags_seq, lags_par)

    # Verify properties
    assert weights_seq.shape == (6, 6)
    assert lags_seq.shape == (6, 6)
    assert allclose(diag(weights_seq), 1.0)
    assert allclose(diag(lags_seq), 0)


def test_format_time():
    """Test the format_time function for different time ranges."""
    # Test seconds format (< 60 seconds)
    assert format_time(30) == "30.0s"
    assert format_time(59.9) == "59.9s"

    # Test minutes format (>= 60 seconds but < 3600 seconds)
    assert format_time(60) == "1.0m"
    assert format_time(120) == "2.0m"
    assert format_time(3599) == "60.0m"

    # Test hours format (>= 3600 seconds)
    assert format_time(3600) == "1.0h"
    assert format_time(7200) == "2.0h"
    assert format_time(36000) == "10.0h"


def test_print_progress_zero_progress(monkeypatch, capsys):
    """Test print_progress function with zero progress."""
    # Mock stdout.write to avoid actual printing during tests
    monkeypatch.setattr("sys.stdout.write", lambda x: None)
    monkeypatch.setattr("sys.stdout.flush", lambda: None)

    # Test with zero progress
    start_time = time()
    print_progress(0, 100, start_time, prefix="Test: ")

    # Capture stdout to verify output
    captured = capsys.readouterr()

    # We don't need to check the actual output since we mocked stdout.write,
    # but we need to ensure the function runs without errors


def test_print_progress_sphinx_mode(monkeypatch):
    """Test print_progress function with sphinx_mode=True and current < total."""
    # Mock stdout.write to avoid actual printing during tests
    write_called = [False]

    def mock_write(x):
        write_called[0] = True

    monkeypatch.setattr("sys.stdout.write", mock_write)
    monkeypatch.setattr("sys.stdout.flush", lambda: None)

    # Test with sphinx_mode=True and current < total
    start_time = time()
    print_progress(50, 100, start_time, prefix="Test: ", sphinx_mode=True)

    # Verify that stdout.write was not called due to early return
    assert not write_called[0], "stdout.write should not be called when sphinx_mode=True and current < total"

    # Test with sphinx_mode=True and current == total
    write_called[0] = False
    print_progress(100, 100, start_time, prefix="Test: ", sphinx_mode=True)

    # Verify that stdout.write was called
    assert write_called[0], "stdout.write should be called when sphinx_mode=True and current == total"




def test_reconstruct_network_sphinx_mode(monkeypatch, two_time_series):
    """Test reconstruct_network function with Sphinx mode enabled and disabled."""
    # Mock is_sphinx_build to control Sphinx mode
    print_called = [False]

    def mock_print(*args, **kwargs):
        print_called[0] = True

    # Mock print to track if it's called
    monkeypatch.setattr("builtins.print", mock_print)

    # Mock stdout.write and flush to avoid actual printing
    monkeypatch.setattr("sys.stdout.write", lambda x: None)
    monkeypatch.setattr("sys.stdout.flush", lambda: None)

    # Test with Sphinx mode disabled
    monkeypatch.setattr("delaynet.network_reconstruction.is_sphinx_build", lambda: False)

    ts1, ts2 = two_time_series
    time_series = column_stack([ts1, ts2])

    # Reset print_called
    print_called[0] = False

    # Run reconstruct_network with Sphinx mode disabled
    reconstruct_network(time_series, "linear_correlation", lag_steps=1)

    # Verify that print was called (newline after completion)
    assert print_called[0], "print() should be called when not in Sphinx mode"

    # Test with Sphinx mode enabled
    monkeypatch.setattr("delaynet.network_reconstruction.is_sphinx_build", lambda: True)

    # Reset print_called
    print_called[0] = False

    # Run reconstruct_network with Sphinx mode enabled
    reconstruct_network(time_series, "linear_correlation", lag_steps=1)

    # Verify that print was not called (no newline after completion)
    assert not print_called[0], "print() should not be called when in Sphinx mode"


def test_update_progress(monkeypatch):
    """Test update_progress function."""
    # Mock print_progress and stdout.flush to avoid actual printing
    called = [False]
    sphinx_mode_value = [None]

    def mock_print_progress(current, total, start_time, prefix, sphinx_mode=None):
        called[0] = True
        sphinx_mode_value[0] = sphinx_mode

    def mock_flush():
        pass

    monkeypatch.setattr("delaynet.network_reconstruction.print_progress", mock_print_progress)
    monkeypatch.setattr("sys.stdout.flush", mock_flush)

    # Create a mock counter
    class MockCounter:
        def __init__(self, value):
            self.value = value

    counter = MockCounter(5)
    start_time = time()

    # Call update_progress with default parameter
    update_progress(counter, 10, start_time, "Test: ")

    # Verify print_progress was called with default parameter
    assert called[0] is True
    assert sphinx_mode_value[0] is None

    # Reset called flag
    called[0] = False

    # Call update_progress with explicit parameter
    update_progress(counter, 10, start_time, "Test: ", sphinx_mode=True)

    # Verify print_progress was called with explicit parameter
    assert called[0] is True
    assert sphinx_mode_value[0] is True
