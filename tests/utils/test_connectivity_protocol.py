"""Tests for the Connectivity protocol in lag_steps.py."""

import pytest
import numpy as np
from delaynet.utils.lag_steps import Connectivity, find_optimal_lag


def test_connectivity_protocol():
    """Test the Connectivity protocol by creating a function that adheres to it."""

    # Define a function that follows the Connectivity protocol
    def example_connectivity(
        ts1: np.ndarray, ts2: np.ndarray, lag: int, *args, **kwargs
    ) -> float:
        """Example connectivity function that follows the protocol."""
        return float(lag)  # Just return the lag as a float for testing

    # Create an instance of the protocol
    connectivity_func: Connectivity = example_connectivity

    # Use the function through the protocol
    result = connectivity_func(
        np.array([1, 2, 3]), np.array([4, 5, 6]), 2, extra_arg="test"
    )

    # Verify the result
    assert result == 2.0
    assert isinstance(result, float)


def test_connectivity_protocol_with_find_optimal_lag():
    """Test the Connectivity protocol with find_optimal_lag function.

    This test ensures that the Connectivity protocol is properly used
    in the find_optimal_lag function, which directly uses the __call__
    method of the protocol.
    """

    # Define a function that follows the Connectivity protocol
    def example_connectivity(
        ts1: np.ndarray, ts2: np.ndarray, lag: int, *args, **kwargs
    ) -> float:
        """Example connectivity function that follows the protocol."""
        # Use the lag parameter to determine the return value
        if lag == 3:
            return 0.1  # Minimum value for lag=3
        return 0.5

    # Create test data
    ts1 = np.array([1, 2, 3, 4, 5])
    ts2 = np.array([1, 2, 3, 4, 5])
    lag_steps = [1, 2, 3, 4, 5]

    # Use find_optimal_lag with our connectivity function
    value, lag = find_optimal_lag(example_connectivity, ts1, ts2, lag_steps)

    # Verify that find_optimal_lag correctly identified the optimal lag
    assert value == 0.1
    assert lag == 3

    # Test with additional kwargs
    def example_connectivity_with_kwargs(
        ts1: np.ndarray, ts2: np.ndarray, lag: int, **kwargs
    ) -> float:
        """Example connectivity function that uses kwargs."""
        multiplier = kwargs.get("multiplier", 1)
        return lag * multiplier

    # Use find_optimal_lag with kwargs
    value, lag = find_optimal_lag(
        example_connectivity_with_kwargs, ts1, ts2, lag_steps, multiplier=0.1
    )

    # Verify that kwargs were properly passed through the protocol
    assert value == 0.1  # 1 * 0.1
    assert lag == 1
