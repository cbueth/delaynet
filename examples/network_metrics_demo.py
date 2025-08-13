"""Demonstration of network analysis metrics in delaynet.

This script showcases all the network analysis metrics implemented in delaynet,
including link density, isolated nodes counting, global efficiency, transitivity,
and eigenvector centrality. Different network topologies are used to demonstrate
how these metrics behave in various scenarios.
"""

import numpy as np
from delaynet.network_analysis.metrics import (
    link_density,
    isolated_nodes_inbound,
    isolated_nodes_outbound,
    global_efficiency,
    transitivity,
    eigenvector_centrality,
)
from delaynet.preparation import gen_delayed_causal_network


def create_example_networks():
    """Create various example networks for demonstration.

    :return: Dictionary of network examples with descriptions.
    :rtype: dict
    """
    networks = {}

    # 1. Fully connected network
    networks["fully_connected"] = {
        "matrix": np.array(
            [
                [0.0, 0.8, 0.6, 0.7],
                [0.5, 0.0, 0.9, 0.4],
                [0.3, 0.7, 0.0, 0.8],
                [0.6, 0.2, 0.5, 0.0],
            ]
        ),
        "description": "Fully connected network (all nodes connected)",
    }

    # 2. Star network (hub and spokes)
    networks["star"] = {
        "matrix": np.array(
            [
                [0.0, 0.8, 0.6, 0.7, 0.5],
                [0.3, 0.0, 0.0, 0.0, 0.0],
                [0.4, 0.0, 0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0, 0.0, 0.0],
                [0.6, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        "description": "Star network (central hub connected to all others)",
    }

    # 3. Linear chain
    networks["chain"] = {
        "matrix": np.array(
            [
                [0.0, 0.8, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.7, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.6, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.9],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        "description": "Linear chain (nodes connected in sequence)",
    }

    # 4. Disconnected network
    networks["disconnected"] = {
        "matrix": np.array(
            [
                [0.0, 0.8, 0.0, 0.0],
                [0.6, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.7],
                [0.0, 0.0, 0.5, 0.0],
            ]
        ),
        "description": "Disconnected network (two separate components)",
    }

    # 5. Triangle network
    networks["triangle"] = {
        "matrix": np.array([[0.0, 0.8, 0.6], [0.7, 0.0, 0.9], [0.5, 0.4, 0.0]]),
        "description": "Triangle network (perfect clustering)",
    }

    return networks


def analyze_network(name, network_data, directed=True):
    """Analyze a single network with all metrics.

    :param name: Name of the network.
    :type name: str
    :param network_data: Dictionary containing matrix and description.
    :type network_data: dict
    :param directed: Whether to treat network as directed.
    :type directed: bool
    :return: Dictionary of computed metrics.
    :rtype: dict
    """
    matrix = network_data["matrix"]
    description = network_data["description"]

    print(f"\n{'=' * 60}")
    print(f"Network: {name.upper()}")
    print(f"Description: {description}")
    print(f"Directed: {directed}")
    print(f"{'=' * 60}")

    print("\nWeight Matrix:")
    print(matrix)

    # Compute all metrics
    results = {}

    # Link Density
    results["link_density"] = link_density(matrix, directed=directed)
    print(f"\nLink Density: {results['link_density']:.4f}")
    print("  → Ratio of existing connections to maximum possible connections")

    # Isolated Nodes
    results["isolated_inbound"] = isolated_nodes_inbound(matrix)
    results["isolated_outbound"] = isolated_nodes_outbound(matrix)
    print(f"\nIsolated Nodes:")
    print(f"  Inbound (no incoming connections): {results['isolated_inbound']}")
    print(f"  Outbound (no outgoing connections): {results['isolated_outbound']}")

    # Global Efficiency
    results["global_efficiency"] = global_efficiency(matrix, directed=directed)
    print(f"\nGlobal Efficiency: {results['global_efficiency']:.4f}")
    print("  → Average inverse shortest path length (information flow efficiency)")

    # Transitivity
    results["transitivity"] = transitivity(matrix, directed=directed)
    print(f"\nTransitivity: {results['transitivity']:.4f}")
    print("  → Global clustering coefficient (triangle density)")

    # Eigenvector Centrality
    results["eigenvector_centrality"] = eigenvector_centrality(
        matrix, directed=directed
    )
    print(f"\nEigenvector Centrality:")
    for i, centrality in enumerate(results["eigenvector_centrality"]):
        print(f"  Node {i}: {centrality:.4f}")
    print("  → Influence based on connections to high-scoring nodes")

    return results


def compare_directed_vs_undirected():
    """Compare metrics for directed vs undirected interpretation of the same network."""
    print(f"\n{'=' * 80}")
    print("DIRECTED vs UNDIRECTED COMPARISON")
    print(f"{'=' * 80}")

    # Use a simple asymmetric network
    matrix = np.array([[0.0, 0.8, 0.0], [0.3, 0.0, 0.9], [0.0, 0.6, 0.0]])

    print("\nNetwork Matrix:")
    print(matrix)

    print(f"\n{'Metric':<25} {'Directed':<15} {'Undirected':<15}")
    print("-" * 55)

    # Compare each metric
    metrics_to_compare = [
        ("Link Density", lambda m, d: link_density(m, directed=d)),
        ("Global Efficiency", lambda m, d: global_efficiency(m, directed=d)),
        ("Transitivity", lambda m, d: transitivity(m, directed=d)),
    ]

    for metric_name, metric_func in metrics_to_compare:
        directed_val = metric_func(matrix, True)
        undirected_val = metric_func(matrix, False)
        print(f"{metric_name:<25} {directed_val:<15.4f} {undirected_val:<15.4f}")


def demonstrate_with_random_network():
    """Demonstrate metrics on a randomly generated network."""
    print(f"\n{'=' * 80}")
    print("RANDOM NETWORK DEMONSTRATION")
    print(f"{'=' * 80}")

    # Generate a random network using delaynet's built-in function
    print("Generating random delayed causal network...")
    n_nodes = 6
    ts_len = 1000
    link_density_param = 0.3
    weight_range = (0.5, 1.5)

    adjacency_matrix, weight_matrix, time_series = gen_delayed_causal_network(
        ts_len=ts_len,
        n_nodes=n_nodes,
        l_dens=link_density_param,
        wm_min_max=weight_range,
        rng=42,  # For reproducibility
    )

    print(f"\nGenerated network parameters:")
    print(f"  Nodes: {n_nodes}")
    print(f"  Target link density: {link_density_param}")
    print(f"  Time series length: {ts_len}")

    # Analyze the generated network
    network_data = {
        "matrix": weight_matrix,
        "description": f"Randomly generated network with {n_nodes} nodes",
    }

    results = analyze_network("random_generated", network_data, directed=True)

    # Show the actual adjacency structure
    print(f"\nTrue Adjacency Matrix (ground truth):")
    print(adjacency_matrix.astype(int))

    return results


def main():
    """Main demonstration function."""
    print("DELAYNET NETWORK METRICS DEMONSTRATION")
    print("=" * 80)
    print("This demo showcases all network analysis metrics available in delaynet:")
    print("• Link Density: Ratio of existing to possible connections")
    print("• Isolated Nodes: Nodes with no inbound/outbound connections")
    print("• Global Efficiency: Average inverse shortest path length")
    print("• Transitivity: Global clustering coefficient")
    print("• Eigenvector Centrality: Node influence based on network structure")

    # Create and analyze example networks
    networks = create_example_networks()
    all_results = {}

    for network_name, network_data in networks.items():
        all_results[network_name] = analyze_network(
            network_name, network_data, directed=True
        )

    # Compare directed vs undirected
    compare_directed_vs_undirected()

    # Demonstrate with random network
    random_results = demonstrate_with_random_network()

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print("All network metrics have been successfully demonstrated!")
    print("Key observations:")
    print("• Fully connected networks have maximum link density (1.0)")
    print("• Star networks have high centralization but low transitivity")
    print("• Chain networks have low efficiency due to long paths")
    print("• Triangle networks have perfect transitivity (1.0)")
    print("• Disconnected networks have reduced global efficiency")

    print(f"\nFor more details on each metric, see the delaynet documentation.")
    print(
        "All metrics accept weight matrices and can handle both directed and undirected networks."
    )


if __name__ == "__main__":
    main()
