"""Tests for network metrics functionality."""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from delaynet.network_analysis.metrics import (
    betweenness_centrality,
    link_density,
    isolated_nodes_inbound,
    isolated_nodes_outbound,
    global_efficiency,
    transitivity,
    reciprocity,
    eigenvector_centrality,
)


class TestBetweennessCentrality:
    """Test betweenness centrality functionality."""

    @pytest.mark.parametrize(
        "weights, directed, normalize, expected_centrality, test_description",
        [
            (
                np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
                True,
                True,
                np.array([0.0, 1.0, 0.0]),
                "linear path - middle node has highest betweenness",
            ),
            (
                np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
                False,
                True,
                np.array([0.0, 0.0, 0.0]),
                "triangle network - no betweenness (all nodes connected)",
            ),
            (
                np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]),
                False,
                True,
                np.array([1.0, 0.0, 0.0, 0.0]),
                "star network - center node has all betweenness",
            ),
            (
                np.zeros((3, 3)),
                True,
                True,
                np.array([0.0, 0.0, 0.0]),
                "network with no connections",
            ),
            (np.array([[0]]), True, True, np.array([0.0]), "single node network"),
            (
                np.array([[0, 1], [1, 0]]),
                False,
                True,
                np.array([0.0, 0.0]),
                "two node network - no intermediate paths",
            ),
            (
                np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]),
                True,
                True,
                np.array([0.0, 2.0 / 3.0, 2.0 / 3.0, 0.0]),
                "linear path with 4 nodes - middle nodes have equal betweenness",
            ),
            (
                np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]]),
                False,
                True,
                np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]),
                "4-node cycle - all nodes have equal betweenness",
            ),
        ],
    )
    def test_betweenness_centrality_scenarios(
        self, weights, directed, normalize, expected_centrality, test_description
    ):
        """Test betweenness centrality calculation for various network scenarios."""
        centrality = betweenness_centrality(
            weights, directed=directed, normalize=normalize
        )
        (
            assert_array_almost_equal(centrality, expected_centrality, decimal=5),
            f"Failed for {test_description}",
        )

    def test_betweenness_centrality_normalization(self):
        """Test betweenness centrality with and without normalization."""
        # Linear path: 0-1-2
        weights = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

        # Normalized version
        centrality_norm = betweenness_centrality(weights, directed=True, normalize=True)

        # Unnormalized version
        centrality_unnorm = betweenness_centrality(
            weights, directed=True, normalize=False
        )

        # For a 3-node directed path, the middle node should have betweenness = 1 (normalized) or 2 (unnormalized)
        # The normalization factor for directed graphs with n=3 is (n-1)*(n-2) = 2*1 = 2
        assert centrality_norm[1] == pytest.approx(
            1.0
        ), "Normalized betweenness should be 1.0 for middle node"
        assert centrality_unnorm[1] == pytest.approx(
            2.0
        ), "Unnormalized betweenness should be 2.0 for middle node"

    def test_betweenness_centrality_normalization_max_betweenness(self):
        """Test betweenness centrality normalization when max_betweenness > 0."""
        # Create a network where normalization will occur (n_nodes > 2)
        # Use a star network where the center node has non-zero betweenness
        weights = np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])

        # Mock the igraph.Graph.betweenness method to return a known value
        from unittest.mock import patch

        # Return a value that will definitely be normalized
        mock_result = [3.0, 0.0, 0.0, 0.0]  # Center node has betweenness of 3.0

        with patch("igraph.Graph.betweenness", return_value=mock_result):
            # This should trigger the normalization code where max_betweenness > 0
            centrality = betweenness_centrality(weights, directed=True, normalize=True)

            # For a 4-node directed graph, max_betweenness = (n-1)*(n-2) = 3*2 = 6
            # So the normalized value should be 3.0/6.0 = 0.5
            assert centrality[0] == pytest.approx(0.5)

            # Verify the result is normalized (all values should be between 0 and 1)
            assert np.all(centrality >= 0)
            assert np.all(centrality <= 1)

    def test_betweenness_centrality_normalization_branch_coverage(self):
        """Test betweenness centrality normalization for branch coverage.

        This test specifically targets the branch condition at line 90:
        if max_betweenness > 0:
        """
        from unittest.mock import patch, MagicMock

        # Create a network
        weights = np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])

        # Create a mock for the Graph class
        mock_graph = MagicMock()
        mock_graph.betweenness.return_value = [3.0, 0.0, 0.0, 0.0]

        # Mock the Graph.Weighted_Adjacency constructor to return our mock graph
        with patch("igraph.Graph.Weighted_Adjacency", return_value=mock_graph):
            # Test with normalize=True to trigger the normalization code
            centrality = betweenness_centrality(weights, directed=True, normalize=True)

            # Verify the result is normalized
            assert centrality[0] == pytest.approx(0.5)

            # Test with normalize=False to skip the normalization code
            centrality_unnorm = betweenness_centrality(
                weights, directed=True, normalize=False
            )

            # Verify the result is not normalized
            assert centrality_unnorm[0] == pytest.approx(3.0)

    def test_betweenness_centrality_normalization_max_betweenness_zero(self):
        """Test betweenness centrality normalization when max_betweenness is exactly 0.

        This test specifically targets the branch condition at line 90:
        if max_betweenness > 0:

        We test the case where max_betweenness is exactly 0, which should skip
        the normalization step even if normalize=True.
        """
        from unittest.mock import patch, MagicMock

        # Create a network with 2 nodes
        # For undirected graphs with n=2, max_betweenness = (n-1)*(n-2)/2 = 0
        weights = np.array([[0, 1], [1, 0]])

        # Create a mock for the Graph class
        mock_graph = MagicMock()
        mock_graph.betweenness.return_value = [
            0.0,
            0.0,
        ]  # Both nodes have 0 betweenness

        # Mock the Graph.Weighted_Adjacency constructor to return our mock graph
        with patch("igraph.Graph.Weighted_Adjacency", return_value=mock_graph):
            # Test with normalize=True, but max_betweenness will be 0
            centrality = betweenness_centrality(weights, directed=False, normalize=True)

            # Verify the result is not changed (no normalization)
            assert_array_equal(centrality, np.array([0.0, 0.0]))

    def test_betweenness_centrality_normalization_max_betweenness_positive(self):
        """Test betweenness centrality normalization when max_betweenness is positive.

        This test specifically targets the branch condition at line 90:
        if max_betweenness > 0:

        We test the case where max_betweenness is positive, which should trigger
        the normalization step when normalize=True.
        """
        from unittest.mock import patch, MagicMock

        # Create a network with 4 nodes (star topology)
        weights = np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])

        # Create a mock for the Graph class
        mock_graph = MagicMock()
        # Center node has betweenness of 3.0
        mock_graph.betweenness.return_value = [3.0, 0.0, 0.0, 0.0]

        # Mock the Graph.Weighted_Adjacency constructor to return our mock graph
        with patch("igraph.Graph.Weighted_Adjacency", return_value=mock_graph):
            # Test with normalize=True to trigger the normalization code
            centrality = betweenness_centrality(weights, directed=True, normalize=True)

            # For a 4-node directed graph, max_betweenness = (n-1)*(n-2) = 3*2 = 6
            # So the normalized value should be 3.0/6.0 = 0.5
            assert centrality[0] == pytest.approx(0.5)

            # Test with normalize=False to skip the normalization code
            centrality_unnorm = betweenness_centrality(
                weights, directed=True, normalize=False
            )

            # Verify the result is not normalized
            assert centrality_unnorm[0] == pytest.approx(3.0)

    def test_betweenness_centrality_input_validation(self):
        """Test input validation for betweenness centrality."""
        non_square_weights = np.array([[1, 0, 1]])
        with pytest.raises(ValueError, match="must be square"):
            betweenness_centrality(non_square_weights)


class TestLinkDensity:
    """Test link density functionality."""

    @pytest.mark.parametrize(
        "weights, directed, expected_density, test_description",
        [
            (
                np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]),
                True,
                2.0 / 6.0,
                "basic directed network",
            ),
            (
                np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
                False,
                2.0 / 3.0,
                "undirected network",
            ),
            (
                np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
                True,
                1.0,
                "fully connected directed network",
            ),
            (
                np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
                False,
                1.0,
                "fully connected undirected network",
            ),
            (np.zeros((3, 3)), True, 0.0, "network with no connections"),
            (np.array([[0]]), True, 0.0, "single node network"),
        ],
    )
    def test_link_density_scenarios(
        self, weights, directed, expected_density, test_description
    ):
        """Test link density calculation for various network scenarios."""
        density = link_density(weights, directed=directed)
        assert density == pytest.approx(
            expected_density
        ), f"Failed for {test_description}"

    def test_link_density_input_validation(self):
        """Test input validation for link density."""
        non_square_weights = np.array([[1, 0, 1]])
        with pytest.raises(ValueError, match="must be square"):
            link_density(non_square_weights)


class TestIsolatedNodes:
    """Test isolated nodes functionality."""

    @pytest.mark.parametrize(
        "weights, expected_inbound, expected_outbound, test_description",
        [
            (
                np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]),
                1,  # Node 0 has no inbound connections
                1,  # Node 2 has no outbound connections
                "basic network with isolated nodes",
            ),
            (
                np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
                0,  # All nodes have inbound connections
                0,  # All nodes have outbound connections
                "network with no isolated nodes",
            ),
            (
                np.zeros((3, 3)),
                3,  # All nodes isolated inbound
                3,  # All nodes isolated outbound
                "network with all nodes isolated",
            ),
        ],
    )
    def test_isolated_nodes_scenarios(
        self, weights, expected_inbound, expected_outbound, test_description
    ):
        """Test isolated nodes counting for various network scenarios."""
        inbound_count = isolated_nodes_inbound(weights)
        outbound_count = isolated_nodes_outbound(weights)

        assert (
            inbound_count == expected_inbound
        ), f"Inbound count failed for {test_description}"
        assert (
            outbound_count == expected_outbound
        ), f"Outbound count failed for {test_description}"

    def test_isolated_nodes_input_validation(self):
        """Test input validation for isolated nodes functions."""
        non_square_weights = np.array([[1, 0, 1]])

        with pytest.raises(ValueError, match="must be square"):
            isolated_nodes_inbound(non_square_weights)

        with pytest.raises(ValueError, match="must be square"):
            isolated_nodes_outbound(non_square_weights)


class TestGlobalEfficiency:
    """Test global efficiency functionality."""

    @pytest.mark.parametrize(
        "weights, directed, expected_efficiency, test_description",
        [
            (
                np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
                False,
                1.0,
                "fully connected undirected network",
            ),
            (np.zeros((3, 3)), True, 0.0, "network with no connections"),
            (np.array([[0]]), True, 0.0, "single node network"),
            (
                np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]),
                True,
                (1.0 + 0.5 + 1.0) / 6.0,
                "directed path network",
            ),
        ],
    )
    def test_global_efficiency_scenarios(
        self, weights, directed, expected_efficiency, test_description
    ):
        """Test global efficiency calculation for various network scenarios."""
        efficiency = global_efficiency(weights, directed=directed)
        assert efficiency == pytest.approx(
            expected_efficiency
        ), f"Failed for {test_description}"

    def test_global_efficiency_pair_count_zero(self):
        """Test global efficiency when pair_count is zero.

        This is a special case that's hard to trigger in practice but is handled
        in the code as a defensive check.
        """
        from unittest.mock import patch, MagicMock

        # Create a simple network
        weights = np.array([[0, 1], [1, 0]])

        # We need to mock the entire calculation to ensure pair_count is exactly 0
        # This requires a more complex approach to bypass the loop that increments pair_count

        # Create a mock for the Graph class
        mock_graph = MagicMock()
        mock_graph.distances.return_value = [
            [float("inf"), float("inf")],
            [float("inf"), float("inf")],
        ]

        # Mock the Graph.Weighted_Adjacency constructor to return our mock graph
        with patch("igraph.Graph.Weighted_Adjacency", return_value=mock_graph):
            # Also patch the loop that increments pair_count to ensure it's 0
            original_range = range

            def mock_range(*args, **kwargs):
                # Return an empty range for the specific loop that increments pair_count
                if len(args) > 0 and args[0] == 2:  # n_nodes = 2
                    return []
                return original_range(*args, **kwargs)

            with patch("builtins.range", side_effect=mock_range):
                efficiency = global_efficiency(weights)
                assert efficiency == 0.0

    def test_global_efficiency_input_validation(self):
        """Test input validation for global efficiency."""
        non_square_weights = np.array([[1, 0, 1]])
        with pytest.raises(ValueError, match="must be square"):
            global_efficiency(non_square_weights)


class TestTransitivity:
    """Test transitivity functionality."""

    @pytest.mark.parametrize(
        "weights, expected_transitivity, test_description",
        [
            (
                np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
                1.0,
                "triangle network with perfect transitivity",
            ),
            (
                np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]),
                0.0,
                "star network with no triangles",
            ),
            (np.zeros((3, 3)), 0.0, "network with no connections"),
            (np.array([[0]]), 0.0, "single node network"),
            (np.array([[0, 1], [1, 0]]), 0.0, "two node network"),
            (
                np.array([[0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]]),
                0.6,
                "4-node network with one complete triangle (transitivity = 0.6)",
            ),
            (
                np.array(
                    [
                        [0, 1, 1, 0, 0],
                        [1, 0, 1, 1, 0],
                        [1, 1, 0, 0, 0],
                        [0, 1, 0, 0, 1],
                        [0, 0, 0, 1, 0],
                    ]
                ),
                0.5,
                "5-node network with partial connectivity (transitivity = 0.5)",
            ),
            (
                np.array([[0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 1], [1, 0, 1, 0]]),
                0.75,
                "4-node network with multiple triangles (transitivity = 0.75)",
            ),
            (
                np.array(
                    [
                        [0, 1, 1, 1, 0, 0],
                        [1, 0, 1, 0, 1, 0],
                        [1, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 1],
                        [0, 1, 0, 1, 0, 1],
                        [0, 0, 1, 1, 1, 0],
                    ]
                ),
                1.0 / 3.0,
                "6-node network with mixed connectivity (transitivity ≈ 0.33)",
            ),
            (
                # Directed graph that should be treated as undirected
                np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]]),
                1.0,
                "directed graph treated as undirected (forms a triangle when undirected)",
            ),
            (
                # Directed graph with a triangle when treated as undirected
                np.array([[0, 1, 1], [0, 0, 1], [1, 0, 0]]),
                1.0,
                "directed graph with a triangle when treated as undirected",
            ),
        ],
    )
    def test_transitivity_scenarios(
        self, weights, expected_transitivity, test_description
    ):
        """Test transitivity calculation for various network scenarios.

        Note: The directed parameter is ignored as transitivity is always
        calculated on the undirected version of the graph.
        """
        trans = transitivity(weights)
        assert trans == pytest.approx(
            expected_transitivity
        ), f"Failed for {test_description}"

    def test_transitivity_nan_result(self):
        """Test transitivity when igraph returns NaN.

        This tests the case where the transitivity calculation returns NaN,
        which should be handled by returning 0.0.
        """
        from unittest.mock import patch
        import numpy as np

        # Create a network with some connections
        weights = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

        # Mock the igraph.Graph.transitivity_undirected method to return NaN
        with patch("igraph.Graph.transitivity_undirected", return_value=float("nan")):
            trans = transitivity(weights)
            assert trans == 0.0

    def test_transitivity_input_validation(self):
        """Test input validation for transitivity."""
        non_square_weights = np.array([[1, 0, 1]])
        with pytest.raises(ValueError, match="must be square"):
            transitivity(non_square_weights)


class TestReciprocity:
    """Test reciprocity functionality."""

    @pytest.mark.parametrize(
        "weights, expected_reciprocity, test_description",
        [
            (
                np.array([[0, 1, 0], [1, 0, 1], [0, 0, 0]]),
                2.0 / 3.0,
                "3-node network with one reciprocated edge out of two (reciprocity = 2/3)",
            ),
            (
                np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
                0.0,
                "3-node directed cycle with no reciprocated edges (reciprocity = 0.0)",
            ),
            (
                np.array([[0, 1, 1], [1, 0, 0], [0, 1, 0]]),
                0.5,
                "3-node network with 1 of 2 edges reciprocated (reciprocity = 0.5)",
            ),
            (
                np.array([[0, 1, 1, 0], [1, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]]),
                0.4,
                "4-node network with 2 of 5 edges reciprocated (reciprocity = 0.4)",
            ),
            (
                # Use a matrix with all zeros except for a very small value that's treated as zero
                np.array([[0, 1e-10, 0], [0, 0, 0], [0, 0, 0]]),
                0.0,
                "network with no connections (reciprocity = 0.0)",
            ),
            (
                np.array([[0]]),
                0.0,
                "single node network (reciprocity = 0.0)",
            ),
        ],
    )
    def test_reciprocity_scenarios(
        self, weights, expected_reciprocity, test_description
    ):
        """Test reciprocity calculation for various network scenarios."""
        recip = reciprocity(weights)
        assert recip == pytest.approx(
            expected_reciprocity
        ), f"Failed for {test_description}"

    def test_reciprocity_no_connections(self):
        """Test reciprocity with a directed network that has no connections."""
        # Create a directed network with no connections
        # The matrix must be non-symmetric to pass the directed check
        # We'll use a matrix with a very small value to make it non-symmetric
        # but still effectively have no connections for the reciprocity calculation
        weights = np.zeros((3, 3))
        weights[2, 0] = 1e-10  # Very small value to make it non-symmetric

        # Mock the np_all function to return False for the symmetry check
        # but True for the no connections check
        from unittest.mock import patch

        def mock_np_all(condition):
            # Return False for the symmetry check (weights == weights.T)
            # but True for the no connections check (weights == 0)
            if np.array_equal(condition, weights == weights.T):
                return False
            return True

        with patch("delaynet.network_analysis.metrics.np_all", side_effect=mock_np_all):
            recip = reciprocity(weights)
            assert recip == 0.0

    def test_reciprocity_nan_result(self):
        """Test reciprocity when igraph returns NaN.

        This tests the case where the reciprocity calculation returns NaN,
        which should be handled by returning 0.0.
        """
        from unittest.mock import patch

        # Create a directed network
        weights = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

        # Mock the igraph.Graph.reciprocity method to return NaN
        with patch("igraph.Graph.reciprocity", return_value=float("nan")):
            recip = reciprocity(weights)
            assert recip == 0.0

    def test_reciprocity_input_validation(self):
        """Test input validation for reciprocity."""
        # Test non-square matrix
        non_square_weights = np.array([[1, 0, 1]])
        with pytest.raises(ValueError, match="must be square"):
            reciprocity(non_square_weights)

        # Test undirected (symmetric) matrix
        symmetric_weights = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        with pytest.raises(ValueError, match="only defined for directed networks"):
            reciprocity(symmetric_weights)


class TestEigenvectorCentrality:
    """Test eigenvector centrality functionality."""

    def test_eigenvector_centrality_symmetric(self):
        """Test eigenvector centrality for symmetric network."""
        weights = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

        centrality = eigenvector_centrality(weights, directed=False)

        # For symmetric fully connected network, all nodes should have equal centrality
        assert len(centrality) == 3
        assert_array_almost_equal(centrality, centrality[0] * np.ones(3), decimal=5)

    def test_eigenvector_centrality_star_network(self):
        """Test eigenvector centrality for star network."""
        # Node 0 is the center of the star
        weights = np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])

        centrality = eigenvector_centrality(weights, directed=False)

        # Center node should have highest centrality
        assert centrality[0] > centrality[1]
        assert centrality[0] > centrality[2]
        assert centrality[0] > centrality[3]

        # Peripheral nodes should have equal centrality
        assert_array_almost_equal(centrality[1:], centrality[1] * np.ones(3), decimal=5)

    @pytest.mark.parametrize(
        "weights, directed, expected_centrality, test_description",
        [
            (np.zeros((3, 3)), True, np.zeros(3), "network with no connections"),
            (np.array([[0]]), True, np.array([1.0]), "single node network"),
            (np.array([]).reshape(0, 0), True, np.array([]), "empty network"),
        ],
    )
    def test_eigenvector_centrality_simple_scenarios(
        self, weights, directed, expected_centrality, test_description
    ):
        """Test eigenvector centrality for simple network scenarios."""
        centrality = eigenvector_centrality(weights, directed=directed)
        (
            assert_array_equal(centrality, expected_centrality),
            f"Failed for {test_description}",
        )

    def test_eigenvector_centrality_normalization(self):
        """Test that eigenvector centrality is properly normalized."""
        weights = np.array([[0, 2, 3], [2, 0, 1], [3, 1, 0]])

        centrality = eigenvector_centrality(weights, directed=False)

        # Should be normalized to unit length
        norm = np.linalg.norm(centrality)
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_eigenvector_centrality_normalization_positive_norm(self):
        """Test eigenvector centrality normalization when norm > 0.

        This test explicitly verifies that the normalization step works
        when the norm of the centrality vector is positive.
        """
        from unittest.mock import patch, MagicMock
        import numpy as np

        # Create a network
        weights = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

        # Mock igraph's eigenvector_centrality to return a known vector
        # with a positive norm
        mock_result = [0.5, 0.5, 0.5]  # This has a norm of sqrt(0.75)
        mock_result_array = np.array(mock_result)

        # Create a mock for the Graph class
        mock_graph = MagicMock()
        mock_graph.eigenvector_centrality.return_value = mock_result

        # Store the original norm function to avoid recursion
        original_norm = np.linalg.norm

        # Create a custom norm function that returns a fixed value for our mock result
        def custom_norm(x, *args, **kwargs):
            # Check if this is our mock result array
            if (
                isinstance(x, np.ndarray)
                and x.shape == mock_result_array.shape
                and np.all(x == mock_result_array)
            ):
                return np.sqrt(0.75)
            # For any other input, use the real norm function
            return original_norm(x, *args, **kwargs)

        # Mock the Graph.Weighted_Adjacency constructor to return our mock graph
        with patch("igraph.Graph.Weighted_Adjacency", return_value=mock_graph):
            # Use our custom norm function
            with patch("numpy.linalg.norm", side_effect=custom_norm):
                centrality = eigenvector_centrality(weights)

                # The result should be normalized to unit length
                # Use the original norm function to check
                assert original_norm(centrality) == pytest.approx(1.0)

                # Each element should be the original value divided by the norm
                expected = mock_result_array / np.sqrt(0.75)
                assert_array_almost_equal(centrality, expected)

    def test_eigenvector_centrality_normalization_branch_coverage(self):
        """Test eigenvector centrality normalization for branch coverage.

        This test specifically targets the branch condition at line 540:
        if norm > 0:
        """
        from unittest.mock import patch, MagicMock
        import numpy as np

        # Create a network
        weights = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

        # Create a mock for the Graph class
        mock_graph = MagicMock()

        # Test both branches of the normalization condition

        # Case 1: norm > 0 (should normalize)
        mock_result_positive = [0.5, 0.5, 0.5]  # Non-zero vector
        mock_graph.eigenvector_centrality.return_value = mock_result_positive

        # Mock the Graph.Weighted_Adjacency constructor to return our mock graph
        with patch("igraph.Graph.Weighted_Adjacency", return_value=mock_graph):
            # Test the positive norm case
            centrality_positive = eigenvector_centrality(weights)

            # The result should be normalized to unit length
            assert np.linalg.norm(centrality_positive) == pytest.approx(1.0)

        # Case 2: norm == 0 (should not normalize)
        mock_result_zero = [0.0, 0.0, 0.0]  # Zero vector
        mock_graph.eigenvector_centrality.return_value = mock_result_zero

        # Mock the Graph.Weighted_Adjacency constructor to return our mock graph
        with patch("igraph.Graph.Weighted_Adjacency", return_value=mock_graph):
            # Test the zero norm case
            centrality_zero = eigenvector_centrality(weights)

            # The result should be the same as the input (zero vector)
            assert_array_equal(centrality_zero, np.array(mock_result_zero))

    def test_eigenvector_centrality_input_validation(self):
        """Test input validation for eigenvector centrality."""
        non_square_weights = np.array([[1, 0, 1]])
        with pytest.raises(ValueError, match="must be square"):
            eigenvector_centrality(non_square_weights)
