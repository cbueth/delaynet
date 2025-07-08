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
        assert centrality_norm[1] == pytest.approx(1.0), (
            "Normalized betweenness should be 1.0 for middle node"
        )
        assert centrality_unnorm[1] == pytest.approx(2.0), (
            "Unnormalized betweenness should be 2.0 for middle node"
        )

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
        assert density == pytest.approx(expected_density), (
            f"Failed for {test_description}"
        )

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

        assert inbound_count == expected_inbound, (
            f"Inbound count failed for {test_description}"
        )
        assert outbound_count == expected_outbound, (
            f"Outbound count failed for {test_description}"
        )

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
        assert efficiency == pytest.approx(expected_efficiency), (
            f"Failed for {test_description}"
        )

    def test_global_efficiency_input_validation(self):
        """Test input validation for global efficiency."""
        non_square_weights = np.array([[1, 0, 1]])
        with pytest.raises(ValueError, match="must be square"):
            global_efficiency(non_square_weights)


class TestTransitivity:
    """Test transitivity functionality."""

    @pytest.mark.parametrize(
        "weights, directed, expected_transitivity, test_description",
        [
            (
                np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
                False,
                1.0,
                "triangle network with perfect transitivity",
            ),
            (
                np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]),
                False,
                0.0,
                "star network with no triangles",
            ),
            (np.zeros((3, 3)), True, 0.0, "network with no connections"),
            (np.array([[0]]), True, 0.0, "single node network"),
            (np.array([[0, 1], [1, 0]]), False, 0.0, "two node network"),
            (
                np.array([[0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]]),
                False,
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
                False,
                0.5,
                "5-node network with partial connectivity (transitivity = 0.5)",
            ),
            (
                np.array([[0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 1], [1, 0, 1, 0]]),
                False,
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
                False,
                1.0 / 3.0,
                "6-node network with mixed connectivity (transitivity ≈ 0.33)",
            ),
        ],
    )
    def test_transitivity_scenarios(
        self, weights, directed, expected_transitivity, test_description
    ):
        """Test transitivity calculation for various network scenarios."""
        trans = transitivity(weights, directed=directed)
        assert trans == pytest.approx(expected_transitivity), (
            f"Failed for {test_description}"
        )

    def test_transitivity_input_validation(self):
        """Test input validation for transitivity."""
        non_square_weights = np.array([[1, 0, 1]])
        with pytest.raises(ValueError, match="must be square"):
            transitivity(non_square_weights)


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

    def test_eigenvector_centrality_input_validation(self):
        """Test input validation for eigenvector centrality."""
        non_square_weights = np.array([[1, 0, 1]])
        with pytest.raises(ValueError, match="must be square"):
            eigenvector_centrality(non_square_weights)
