"""Tests specifically for the normalization in betweenness_centrality."""

import numpy as np
from unittest.mock import patch, MagicMock
from delaynet.network_analysis.metrics import betweenness_centrality


def test_betweenness_centrality_normalization_direct():
    """Test betweenness centrality normalization directly.

    This test specifically targets the normalization code at lines 90-93
    in metrics.py, ensuring that the centrality values are properly
    normalized when max_betweenness > 0.
    """
    # Create a mock for the Graph class
    mock_graph = MagicMock()

    # Set up a known betweenness result that will need normalization
    # For a 4-node directed graph, max_betweenness = (n-1)*(n-2) = 3*2 = 6
    mock_graph.betweenness.return_value = [6.0, 3.0, 0.0, 0.0]

    # Create a 4-node network
    weights = np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])

    # Mock the Graph.Weighted_Adjacency constructor to return our mock graph
    with patch("igraph.Graph.Weighted_Adjacency", return_value=mock_graph):
        # Test with normalize=True to trigger the normalization code
        centrality = betweenness_centrality(weights, directed=True, normalize=True)

        # The normalization should divide each value by max_betweenness (6)
        expected = np.array([1.0, 0.5, 0.0, 0.0])
        np.testing.assert_array_almost_equal(centrality, expected)

        # Verify that without normalization, we get the original values
        centrality_unnorm = betweenness_centrality(
            weights, directed=True, normalize=False
        )
        expected_unnorm = np.array([6.0, 3.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(centrality_unnorm, expected_unnorm)
