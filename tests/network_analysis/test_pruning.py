"""Tests for network pruning functionality."""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from delaynet.network_analysis.pruning import statistical_pruning


class TestStatisticalPruning:
    """Test statistical pruning functionality."""

    def test_statistical_pruning_basic_functionality(self):
        """Test basic statistical pruning functionality."""
        # Create test matrix with known p-values
        p_values = np.array([[1.0, 0.01, 0.8], [0.05, 1.0, 0.02], [0.9, 0.03, 1.0]])

        # Test with alpha=0.05
        mask = statistical_pruning(p_values, alpha=0.05)

        # Expected: only connections with p < 0.05 should be True
        expected_mask = np.array(
            [[False, True, False], [False, False, True], [False, True, False]]
        )

        assert_array_equal(mask, expected_mask)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool

    def test_statistical_pruning_different_alpha_values(self):
        """Test statistical pruning with different alpha values."""
        p_values = np.array(
            [
                [1.0, 0.001, 0.01, 0.05, 0.1],
                [0.001, 1.0, 0.02, 0.08, 0.2],
                [0.01, 0.02, 1.0, 0.06, 0.3],
                [0.05, 0.08, 0.06, 1.0, 0.4],
                [0.1, 0.2, 0.3, 0.4, 1.0],
            ]
        )

        # Test with alpha=0.01
        mask_001 = statistical_pruning(p_values, alpha=0.01)
        # Only p-values < 0.01 should be True (0.001 appears twice)
        assert np.sum(mask_001) == 2

        # Test with alpha=0.05
        mask_005 = statistical_pruning(p_values, alpha=0.05)
        # p-values < 0.05 should be True (0.001×2, 0.01×2, 0.02×2)
        assert np.sum(mask_005) == 6

        # Test with alpha=0.1
        mask_01 = statistical_pruning(p_values, alpha=0.1)
        # p-values < 0.1 should be True (0.001×2, 0.01×2, 0.02×2, 0.05×2, 0.06×2, 0.08×2)
        assert np.sum(mask_01) == 12

    def test_statistical_pruning_edge_cases(self):
        """Test statistical pruning edge cases."""
        # Test with alpha=0 (nothing should pass)
        p_values = np.array([[1.0, 0.01], [0.05, 1.0]])

        mask = statistical_pruning(p_values, alpha=0.0)
        expected_mask = np.array([[False, False], [False, False]])

        assert_array_equal(mask, expected_mask)

        # Test with alpha=1.1 (everything should pass since all p-values < 1.1)
        mask = statistical_pruning(p_values, alpha=1.1)
        expected_mask = np.array([[True, True], [True, True]])
        assert_array_equal(mask, expected_mask)

    def test_statistical_pruning_copies_input(self):
        """Test that statistical pruning doesn't modify input matrix."""
        p_values = np.array([[1.0, 0.01, 0.8], [0.05, 1.0, 0.02], [0.9, 0.03, 1.0]])

        # Store original values
        original_p_values = p_values.copy()

        # Apply pruning
        statistical_pruning(p_values, alpha=0.05)

        # Check that original matrix is unchanged
        assert_array_equal(p_values, original_p_values)

    def test_statistical_pruning_bonferroni_correction(self):
        """Test statistical pruning with Bonferroni correction."""
        p_values = np.array([[1.0, 0.01, 0.02], [0.01, 1.0, 0.03], [0.02, 0.03, 1.0]])

        # Without correction
        mask_no_corr = statistical_pruning(p_values, alpha=0.05)

        # With Bonferroni correction (alpha/n_comparisons)
        mask_bonf = statistical_pruning(p_values, alpha=0.05, correction="bonferroni")

        # Bonferroni should be more conservative (fewer True values)
        assert np.sum(mask_bonf) <= np.sum(mask_no_corr)

    def test_statistical_pruning_holm_correction(self):
        """Test statistical pruning with Holm correction."""
        p_values = np.array(
            [
                [1.0, 0.001, 0.01, 0.02],
                [0.001, 1.0, 0.03, 0.04],
                [0.01, 0.03, 1.0, 0.05],
                [0.02, 0.04, 0.05, 1.0],
            ]
        )

        mask = statistical_pruning(p_values, alpha=0.05, correction="holm")

        # Should return boolean mask
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.shape == p_values.shape

    def test_statistical_pruning_fdr_bh_correction(self):
        """Test statistical pruning with FDR (Benjamini-Hochberg) correction."""
        p_values = np.array(
            [
                [1.0, 0.001, 0.01, 0.02],
                [0.001, 1.0, 0.03, 0.04],
                [0.01, 0.03, 1.0, 0.05],
                [0.02, 0.04, 0.05, 1.0],
            ]
        )

        mask = statistical_pruning(p_values, alpha=0.05, correction="fdr_bh")

        # Should return boolean mask
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.shape == p_values.shape

    def test_statistical_pruning_multipletests_kwargs(self):
        """Test statistical pruning with additional multipletests kwargs."""
        p_values = np.array([[1.0, 0.01, 0.02], [0.01, 1.0, 0.03], [0.02, 0.03, 1.0]])

        # Test with additional kwargs passed to multipletests
        mask = statistical_pruning(
            p_values,
            alpha=0.05,
            correction="fdr_bh",
            returnsorted=False,  # This is a valid kwarg for multipletests
        )

        # Should return boolean mask
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.shape == p_values.shape

    def test_statistical_pruning_invalid_correction(self):
        """Test statistical pruning with invalid correction method."""
        p_values = np.array([[1.0, 0.01], [0.05, 1.0]])

        with pytest.raises(ValueError, match="method not recognized"):
            statistical_pruning(p_values, correction="invalid_method")

    def test_statistical_pruning_rectangular_matrix(self):
        """Test statistical pruning works with rectangular matrices."""
        # The current implementation doesn't require square matrices
        p_values = np.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]])

        mask = statistical_pruning(p_values, alpha=0.05)

        # Should work and return mask of same shape
        assert mask.shape == p_values.shape
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool

        # All values < 0.05 should be True
        expected_mask = p_values < 0.05
        assert_array_equal(mask, expected_mask)

    def test_statistical_pruning_empty_matrix(self):
        """Test statistical pruning with empty matrix."""
        p_values = np.array([]).reshape(0, 0)

        mask = statistical_pruning(p_values, alpha=0.05)

        assert mask.shape == (0, 0)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool

    def test_statistical_pruning_single_value(self):
        """Test statistical pruning with single value matrix."""
        p_values = np.array([[0.03]])

        mask = statistical_pruning(p_values, alpha=0.05)

        assert mask.shape == (1, 1)
        assert mask[0, 0] == True  # 0.03 < 0.05

        mask = statistical_pruning(p_values, alpha=0.01)
        assert mask[0, 0] == False  # 0.03 > 0.01

    @pytest.mark.parametrize(
        "correction_method",
        [
            "bonferroni",
            "sidak",
            "holm",
            "holm-sidak",
            "simes-hochberg",
            "hommel",
            "fdr_bh",
            "fdr_by",
            "fdr_tsbh",
            "fdr_tsbky",
        ],
    )
    def test_statistical_pruning_correction_methods(self, correction_method):
        """Test statistical pruning with various correction methods using parametrize."""
        p_values = np.array(
            [
                [1.0, 0.001, 0.01, 0.02, 0.03],
                [0.001, 1.0, 0.02, 0.04, 0.05],
                [0.01, 0.02, 1.0, 0.06, 0.07],
                [0.02, 0.04, 0.06, 1.0, 0.08],
                [0.03, 0.05, 0.07, 0.08, 1.0],
            ]
        )

        # Test that the correction method works without errors
        mask = statistical_pruning(p_values, alpha=0.05, correction=correction_method)

        # Basic validation
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.shape == p_values.shape

        # Diagonal should always be False (p=1.0 for self-connections)
        assert not np.any(np.diag(mask))

        # Compare with no correction - corrected should be more conservative
        mask_no_corr = statistical_pruning(p_values, alpha=0.05)
        assert np.sum(mask) <= np.sum(mask_no_corr)

    @pytest.mark.parametrize(
        "multipletests_kwargs",
        [
            {"returnsorted": False},
            {"returnsorted": True},
            {"is_sorted": False},
            {"is_sorted": True},
            {"returnsorted": False, "is_sorted": False},
            {"returnsorted": True, "is_sorted": False},
        ],
    )
    def test_statistical_pruning_multipletests_kwargs_parametrized(
        self, multipletests_kwargs
    ):
        """Test statistical pruning with various multipletests kwargs using parametrize."""
        p_values = np.array(
            [
                [1.0, 0.001, 0.01, 0.02],
                [0.001, 1.0, 0.03, 0.04],
                [0.01, 0.03, 1.0, 0.05],
                [0.02, 0.04, 0.05, 1.0],
            ]
        )

        # Test that the kwargs work without errors
        mask = statistical_pruning(
            p_values, alpha=0.05, correction="fdr_bh", **multipletests_kwargs
        )

        # Basic validation
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.shape == p_values.shape

        # Should return a valid boolean mask (no NaN or invalid values)
        assert not np.any(np.isnan(mask.astype(float)))

        # Test passes if function executes without error and returns valid mask
        # (Different kwargs may produce different results, which is expected behavior)
