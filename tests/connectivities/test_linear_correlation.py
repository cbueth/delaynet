import numpy as np
import pytest
from delaynet import connectivity


def test_linear_correlation():
    # Generate some test time series data
    np.random.seed(24567)
    ts1 = np.random.normal(0, 1, size=100)
    ts2 = np.roll(ts1, 2) + np.random.normal(0, 0.1, size=100)  # Create causally related series

    # Test the connectivity with default parameters
    result = connectivity(
        ts1,
        ts2,
        metric="linear_correlation",
        lag_steps=5,
    )

    # Assert that the function returns expected format
    assert isinstance(result, tuple), "Result should be a tuple"
    assert len(result) == 2, "Result should contain two elements"
    
    # Test with expected value
    # The lag should be 2 since we created ts2 by rolling ts1 by 2
    p_value, lag = result
    assert lag == 2, f"Expected lag to be 2, got {lag}"