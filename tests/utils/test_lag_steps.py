"""Tests for the lag_steps module."""

import pytest
import numpy as np
from delaynet.utils.lag_steps import assure_lag_list, find_optimal_lag


def test_assure_lag_list_with_integer():
    """Test assure_lag_list with a positive integer."""
    result = assure_lag_list(5)
    assert isinstance(result, list)
    assert result == [1, 2, 3, 4, 5]


def test_assure_lag_list_with_list():
    """Test assure_lag_list with a list of integers."""
    input_list = [1, 3, 5, 7]
    result = assure_lag_list(input_list)
    assert result == input_list
    assert result is input_list  # Should return the same list object


def test_assure_lag_list_with_negative_integer():
    """Test assure_lag_list with a negative integer."""
    with pytest.raises(ValueError, match="must be a positive integer or list of such"):
        assure_lag_list(-5)


def test_assure_lag_list_with_zero():
    """Test assure_lag_list with zero."""
    with pytest.raises(ValueError, match="must be a positive integer or list of such"):
        assure_lag_list(0)


def test_assure_lag_list_with_non_integer_list():
    """Test assure_lag_list with a list containing non-integers."""
    with pytest.raises(ValueError, match="must be an integer or a list of integers"):
        assure_lag_list([1, 2, "3", 4])


def test_assure_lag_list_with_non_list_non_integer():
    """Test assure_lag_list with a value that is neither a list nor an integer."""
    with pytest.raises(ValueError, match="must be an integer or a list of integers"):
        assure_lag_list("not a list or integer")


def test_assure_lag_list_with_negative_integers_in_list(caplog):
    """Test assure_lag_list with a list containing negative integers."""
    result = assure_lag_list([1, -2, 3])
    assert result == [1, -2, 3]
    assert "Some elements in `lag_steps` are non-positive" in caplog.text


def test_assure_lag_list_with_zero_in_list(caplog):
    """Test assure_lag_list with a list containing zero."""
    result = assure_lag_list([1, 0, 3])
    assert result == [1, 0, 3]
    assert "Some elements in `lag_steps` are non-positive" in caplog.text


def test_find_optimal_lag_with_min_operator():
    """Test find_optimal_lag with the min operator."""

    # Create a simple metric function that returns the lag value as the metric
    def metric_func(ts1, ts2, lag, **kwargs):
        return lag

    ts1 = np.array([1, 2, 3, 4, 5])
    ts2 = np.array([1, 2, 3, 4, 5])
    lag_steps = [1, 2, 3, 4, 5]

    # The min operator should select the smallest lag value (1)
    value, lag = find_optimal_lag(metric_func, ts1, ts2, lag_steps)
    assert value == 1
    assert lag == 1


def test_find_optimal_lag_with_max_operator():
    """Test find_optimal_lag with the max operator."""

    # Create a simple metric function that returns the lag value as the metric
    def metric_func(ts1, ts2, lag, **kwargs):
        return lag

    ts1 = np.array([1, 2, 3, 4, 5])
    ts2 = np.array([1, 2, 3, 4, 5])
    lag_steps = [1, 2, 3, 4, 5]

    # The max operator should select the largest lag value (5)
    value, lag = find_optimal_lag(metric_func, ts1, ts2, lag_steps, op=max)
    assert value == 5
    assert lag == 5


def test_find_optimal_lag_with_custom_operator():
    """Test find_optimal_lag with a custom operator."""

    # Create a simple metric function that returns the lag value as the metric
    def metric_func(ts1, ts2, lag, **kwargs):
        return lag

    # Custom operator that selects the middle value
    def middle_op(indices, key):
        return len(indices) // 2

    ts1 = np.array([1, 2, 3, 4, 5])
    ts2 = np.array([1, 2, 3, 4, 5])
    lag_steps = [1, 2, 3, 4, 5]

    # The middle operator should select the middle lag value (3)
    value, lag = find_optimal_lag(metric_func, ts1, ts2, lag_steps, op=middle_op)
    assert value == 3
    assert lag == 3


def test_find_optimal_lag_with_kwargs():
    """Test find_optimal_lag with additional keyword arguments."""

    # Create a metric function that uses additional kwargs
    def metric_func(ts1, ts2, lag, **kwargs):
        multiplier = kwargs.get("multiplier", 1)
        return lag * multiplier

    ts1 = np.array([1, 2, 3, 4, 5])
    ts2 = np.array([1, 2, 3, 4, 5])
    lag_steps = [1, 2, 3, 4, 5]

    # With multiplier=2, the values should be [2, 4, 6, 8, 10]
    # The min operator should still select the smallest value (2)
    value, lag = find_optimal_lag(metric_func, ts1, ts2, lag_steps, multiplier=2)
    assert value == 2
    assert lag == 1


def test_find_optimal_lag_with_custom_metric():
    """Test find_optimal_lag with a custom metric function."""

    # Create a custom metric function that returns a specific pattern
    def custom_metric(ts1, ts2, lag, **kwargs):
        # Return a pattern where lag=3 gives the minimum value
        if lag == 3:
            return 0.1
        return 0.5

    ts1 = np.array([1, 2, 3, 4, 5])
    ts2 = np.array([1, 2, 3, 4, 5])
    lag_steps = [1, 2, 3, 4, 5]

    # The min operator should select lag=3 which gives the minimum value (0.1)
    value, lag = find_optimal_lag(custom_metric, ts1, ts2, lag_steps)
    assert value == 0.1
    assert lag == 3
