"""Tests for error conditions in the connectivity decorator."""

import pytest
import numpy as np
from delaynet.decorators import connectivity


def test_connectivity_decorator_missing_lag_steps_param():
    """Test that the decorator raises an error if the function doesn't accept lag_steps.
    
    This test targets line 49 in decorators.py.
    """
    # Define a function that doesn't accept lag_steps
    with pytest.raises(TypeError, match="does not accept 'lag_steps' as a keyword argument"):
        @connectivity
        def invalid_connectivity(ts1, ts2):
            """This function doesn't accept lag_steps and should raise an error."""
            return 0.0, 0


def test_connectivity_decorator_non_ndarray_inputs():
    """Test that the decorator raises an error if inputs are not ndarrays.
    
    This test targets line 91 in decorators.py.
    """
    @connectivity
    def simple_connectivity(ts1, ts2, lag_steps=None):
        """Simple connectivity function."""
        return 0.0, lag_steps[0]
    
    # Both inputs are not ndarrays
    with pytest.raises(TypeError, match="must be of type ndarray"):
        simple_connectivity([1, 2, 3], [4, 5, 6], lag_steps=5)


def test_connectivity_decorator_different_shape_inputs():
    """Test that the decorator raises an error if inputs have different shapes.
    
    This test targets line 97 in decorators.py.
    """
    @connectivity
    def simple_connectivity(ts1, ts2, lag_steps=None):
        """Simple connectivity function."""
        return 0.0, lag_steps[0]
    
    # Inputs have different shapes
    with pytest.raises(ValueError, match="must have the same shape"):
        simple_connectivity(
            np.array([1, 2, 3]), 
            np.array([4, 5, 6, 7]), 
            lag_steps=5
        )


def test_connectivity_decorator_missing_lag_steps_arg():
    """Test that the decorator raises an error if lag_steps is not provided.
    
    This test targets line 106 in decorators.py.
    """
    @connectivity
    def simple_connectivity(ts1, ts2, lag_steps=None):
        """Simple connectivity function."""
        return 0.0, 1
    
    # lag_steps is not provided
    with pytest.raises(ValueError, match="must be passed to the connectivity function"):
        simple_connectivity(np.array([1, 2, 3]), np.array([4, 5, 6]))