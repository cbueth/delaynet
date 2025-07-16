"""Test the evaluation module."""

import pytest
import numpy as np
from delaynet.evaluation import roc_auc_rank_c


def test_roc_auc_rank_c_valid_input():
    """Test the roc_auc_rank_c function with valid input."""
    # Create sample data
    n_nodes = 5
    orig_net = np.zeros((n_nodes, n_nodes))
    orig_net[0, 1] = orig_net[1, 2] = orig_net[2, 3] = 1  # Some connections

    weight_mat = np.zeros((n_nodes, n_nodes))
    weight_mat[0, 1] = 0.8
    weight_mat[1, 2] = 0.6
    weight_mat[2, 3] = 0.7

    rec_net = np.zeros((n_nodes, n_nodes))
    rec_net[0, 1] = 0.75
    rec_net[1, 2] = 0.65
    rec_net[2, 3] = 0.55
    rec_net[3, 4] = 0.45  # False positive

    # Call the function
    roc, auc, rank_c = roc_auc_rank_c(orig_net, weight_mat, rec_net)

    # Check the results
    assert isinstance(roc, np.ndarray)
    assert isinstance(auc, float)
    assert isinstance(rank_c, float)
    assert 0 <= auc <= 1
    assert 0 <= rank_c <= 1


def test_roc_auc_rank_c_not_numpy_array():
    """Test roc_auc_rank_c with inputs that are not numpy arrays."""
    n_nodes = 3
    valid_array = np.zeros((n_nodes, n_nodes))

    # Test with list instead of numpy array
    with pytest.raises(TypeError):
        roc_auc_rank_c([[0, 0, 0], [0, 0, 0], [0, 0, 0]], valid_array, valid_array)

    with pytest.raises(TypeError):
        roc_auc_rank_c(valid_array, [[0, 0, 0], [0, 0, 0], [0, 0, 0]], valid_array)

    with pytest.raises(TypeError):
        roc_auc_rank_c(valid_array, valid_array, [[0, 0, 0], [0, 0, 0], [0, 0, 0]])


def test_roc_auc_rank_c_not_square_matrix():
    """Test roc_auc_rank_c with inputs that are not square matrices."""
    # Create non-square matrices
    non_square = np.zeros((3, 4))
    square = np.zeros((3, 3))

    # Test with non-square orig_net
    with pytest.raises(ValueError):
        roc_auc_rank_c(non_square, square, square)

    # Test with non-square weight_mat
    with pytest.raises(ValueError):
        roc_auc_rank_c(square, non_square, square)

    # Test with non-square rec_net
    with pytest.raises(ValueError):
        roc_auc_rank_c(square, square, non_square)


def test_roc_auc_rank_c_different_shapes():
    """Test roc_auc_rank_c with inputs that have different shapes."""
    # Create matrices with different shapes
    small_square = np.zeros((3, 3))
    large_square = np.zeros((4, 4))

    # Test with different shapes
    with pytest.raises(ValueError):
        roc_auc_rank_c(small_square, large_square, large_square)

    with pytest.raises(ValueError):
        roc_auc_rank_c(small_square, small_square, large_square)

    with pytest.raises(ValueError):
        roc_auc_rank_c(large_square, small_square, large_square)
