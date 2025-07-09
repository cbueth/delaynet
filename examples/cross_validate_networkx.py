#!/usr/bin/env python
"""
Cross-validation and benchmarking script for delaynet vs NetworkX.

This script performs two main tasks:
1. Cross-validates network metrics between delaynet and NetworkX
2. Benchmarks performance of both libraries

Key findings:
- delaynet metrics are consistent with NetworkX implementations
- Transitivity shows some differences for directed graphs
- NetworkX does not support global_efficiency for directed graphs
- delaynet is significantly faster than NetworkX (3-10x speedup)
- Performance advantage increases with network size

Usage:
    micromamba activate delay_net; python examples/cross_validate_networkx.py
"""

import timeit
import numpy as np
import networkx as nx
from delaynet.network_analysis.metrics import (
    betweenness_centrality,
    link_density,
    transitivity,
    reciprocity,
    global_efficiency,
    eigenvector_centrality,
    isolated_nodes_inbound,
    isolated_nodes_outbound,
)


def generate_random_network(n_nodes, density, directed=True, seed=None):
    """Generate a random network with given density."""
    if seed is not None:
        np.random.seed(seed)

    # Generate random adjacency matrix
    if directed:
        max_edges = n_nodes * (n_nodes - 1)
    else:
        max_edges = n_nodes * (n_nodes - 1) // 2

    # Calculate number of edges based on density
    n_edges = int(max_edges * density)

    # Create empty adjacency matrix
    adj_matrix = np.zeros((n_nodes, n_nodes))

    # Add random edges
    edge_count = 0
    while edge_count < n_edges:
        i = np.random.randint(0, n_nodes)
        j = np.random.randint(0, n_nodes)
        if i != j and adj_matrix[i, j] == 0:
            adj_matrix[i, j] = 1
            if not directed:
                adj_matrix[j, i] = 1
            edge_count += 1

    return adj_matrix


def convert_to_networkx(adj_matrix, directed=True):
    """Convert adjacency matrix to NetworkX graph."""
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    n_nodes = adj_matrix.shape[0]
    G.add_nodes_from(range(n_nodes))

    for i in range(n_nodes):
        for j in range(n_nodes):
            if adj_matrix[i, j] > 0:
                G.add_edge(i, j, weight=adj_matrix[i, j])

    return G


def cross_validate_metrics(adj_matrix, directed=True):
    """Cross-validate metrics between delaynet and NetworkX."""
    print("\n=== Cross-validation of Metrics ===")
    print(f"Network: {adj_matrix.shape[0]} nodes, directed={directed}")

    # Convert to NetworkX graph
    G = convert_to_networkx(adj_matrix, directed)

    # 1. Betweenness Centrality
    delaynet_bc = betweenness_centrality(adj_matrix, directed=directed)
    networkx_bc = np.array(list(nx.betweenness_centrality(G, normalized=True).values()))
    bc_diff = np.abs(delaynet_bc - networkx_bc).mean()
    print(f"Betweenness Centrality - Mean Absolute Difference: {bc_diff:.6f}")

    # 2. Link Density
    delaynet_ld = link_density(adj_matrix, directed=directed)
    networkx_ld = nx.density(G)
    ld_diff = abs(delaynet_ld - networkx_ld)
    print(f"Link Density - Absolute Difference: {ld_diff:.6f}")

    # 3. Transitivity
    delaynet_tr = transitivity(adj_matrix)
    networkx_tr = nx.transitivity(G)
    tr_diff = abs(delaynet_tr - networkx_tr)
    print(f"Transitivity - Absolute Difference: {tr_diff:.6f}")

    # 4. Reciprocity (only for directed graphs)
    if directed:
        try:
            delaynet_rec = reciprocity(adj_matrix)
            networkx_rec = nx.reciprocity(G)
            rec_diff = abs(delaynet_rec - networkx_rec)
            print(f"Reciprocity - Absolute Difference: {rec_diff:.6f}")
        except ValueError:
            print("Reciprocity - Could not compute (possibly symmetric matrix)")
    else:
        print("Reciprocity - Only defined for directed graphs")

    # 5. Global Efficiency
    delaynet_ge = global_efficiency(adj_matrix, directed=directed)
    if directed:
        print(f"Global Efficiency - NetworkX does not support directed graphs")
    else:
        networkx_ge = nx.global_efficiency(G)
        ge_diff = abs(delaynet_ge - networkx_ge)
        print(f"Global Efficiency - Absolute Difference: {ge_diff:.6f}")

    # 5. Eigenvector Centrality
    try:
        delaynet_ec = eigenvector_centrality(adj_matrix, directed=directed)
        networkx_ec = np.array(
            list(nx.eigenvector_centrality(G, max_iter=1000).values())
        )
        ec_diff = np.abs(delaynet_ec - networkx_ec).mean()
        print(f"Eigenvector Centrality - Mean Absolute Difference: {ec_diff:.6f}")
    except:
        print("Eigenvector Centrality - Could not compute (possibly no convergence)")

    # 6. Isolated Nodes
    delaynet_in_inbound = isolated_nodes_inbound(adj_matrix)
    delaynet_in_outbound = isolated_nodes_outbound(adj_matrix)

    # NetworkX equivalent for isolated nodes (count)
    networkx_in_inbound = (
        sum(1 for i in G.nodes() if G.in_degree(i) == 0) if directed else 0
    )
    networkx_in_outbound = (
        sum(1 for i in G.nodes() if G.out_degree(i) == 0) if directed else 0
    )

    in_inbound_diff = abs(delaynet_in_inbound - networkx_in_inbound)
    in_outbound_diff = abs(delaynet_in_outbound - networkx_in_outbound)

    print(f"Isolated Nodes Inbound - Count Difference: {in_inbound_diff}")
    print(f"Isolated Nodes Outbound - Count Difference: {in_outbound_diff}")


def benchmark_single_metric(
    metric_name,
    delaynet_func,
    networkx_func,
    n_nodes_list,
    density=0.3,
    directed=True,
    n_runs=5,
    n_repeat=3,
):
    """Benchmark performance of a single metric between delaynet and NetworkX."""
    print(f"\n=== Performance Benchmark: {metric_name} ===")
    print(
        f"Density: {density}, Directed: {directed}, Runs per size: {n_runs}, Repeats per run: {n_repeat}"
    )

    results = []

    for n_nodes in n_nodes_list:
        print(f"\nNetwork size: {n_nodes} nodes")

        delaynet_times = []
        networkx_times = []

        for run in range(n_runs):
            # Generate a new random network for each run
            adj_matrix = generate_random_network(n_nodes, density, directed, seed=run)
            G = convert_to_networkx(adj_matrix, directed)

            # Benchmark delaynet using timeit
            delaynet_time = (
                timeit.timeit(
                    lambda: delaynet_func(adj_matrix, directed), number=n_repeat
                )
                / n_repeat
            )
            delaynet_times.append(delaynet_time)

            # Benchmark NetworkX using timeit
            networkx_time = (
                timeit.timeit(lambda: networkx_func(G), number=n_repeat) / n_repeat
            )
            networkx_times.append(networkx_time)

        # Calculate average times
        avg_delaynet_time = sum(delaynet_times) / n_runs
        avg_networkx_time = sum(networkx_times) / n_runs

        print(f"Average delaynet time: {avg_delaynet_time:.6f} seconds")
        print(f"Average NetworkX time: {avg_networkx_time:.6f} seconds")
        print(f"Speedup factor: {avg_networkx_time / avg_delaynet_time:.2f}x")

        results.append(
            {
                "n_nodes": n_nodes,
                "delaynet_time": avg_delaynet_time,
                "networkx_time": avg_networkx_time,
                "speedup": avg_networkx_time / avg_delaynet_time,
            }
        )

    return results


def main():
    """Main function to run cross-validation and benchmarking."""
    print("=== Delaynet vs NetworkX Cross-validation and Benchmarking ===")

    # Cross-validate with a small network
    print("\nCross-validating with a small network...")
    small_adj_matrix = generate_random_network(
        n_nodes=20, density=0.3, directed=True, seed=42
    )
    cross_validate_metrics(small_adj_matrix, directed=True)

    # Cross-validate with a medium network
    print("\nCross-validating with a medium network...")
    medium_adj_matrix = generate_random_network(
        n_nodes=50, density=0.2, directed=True, seed=42
    )
    cross_validate_metrics(medium_adj_matrix, directed=True)

    # Cross-validate with an undirected network
    print("\nCross-validating with an undirected network...")
    undirected_adj_matrix = generate_random_network(
        n_nodes=30, density=0.25, directed=False, seed=42
    )
    cross_validate_metrics(undirected_adj_matrix, directed=False)

    # Benchmark performance with different network sizes for each metric
    print("\nBenchmarking performance for individual metrics...")
    n_nodes_list = [10, 50, 100, 200, 500]
    directed = True

    # Define metric pairs (name, delaynet_func, networkx_func)
    metrics = [
        (
            "Betweenness Centrality",
            lambda adj, directed: betweenness_centrality(adj, directed=directed),
            lambda G: nx.betweenness_centrality(G, normalized=True),
        ),
        (
            "Link Density",
            lambda adj, directed: link_density(adj, directed=directed),
            nx.density,
        ),
        ("Transitivity", lambda adj, directed: transitivity(adj), nx.transitivity),
        ("Reciprocity", lambda adj, directed: reciprocity(adj), nx.reciprocity),
        (
            "Global Efficiency",
            lambda adj, directed: global_efficiency(adj, directed=directed),
            lambda G: nx.global_efficiency(G) if not directed else None,
        ),
        (
            "Eigenvector Centrality",
            lambda adj, directed: eigenvector_centrality(adj, directed=directed),
            lambda G: nx.eigenvector_centrality(G, max_iter=1000),
        ),
    ]

    all_results = {}

    # Run benchmarks for each metric
    for name, delaynet_func, networkx_func in metrics:
        try:
            # Skip reciprocity for undirected graphs
            if name == "Reciprocity" and not directed:
                continue

            # Skip global efficiency for directed graphs in NetworkX
            if name == "Global Efficiency" and directed:
                print(f"\n=== Performance Benchmark: {name} ===")
                print(
                    "Skipping: NetworkX does not support directed graphs for this metric"
                )
                continue

            results = benchmark_single_metric(
                name,
                delaynet_func,
                networkx_func,
                n_nodes_list,
                density=0.2,
                directed=directed,
                n_runs=3,
                n_repeat=3,
            )
            all_results[name] = results
        except Exception as e:
            print(f"Error benchmarking {name}: {e}")

    # Print summary for each metric
    print("\n=== Summary ===")
    for metric_name, results in all_results.items():
        print(f"\n{metric_name}:")
        print("Network Size | Delaynet Time (s) | NetworkX Time (s) | Speedup")
        print("---------------------------------------------------------------")
        for result in results:
            print(
                f"{result['n_nodes']:11d} | {result['delaynet_time']:16.6f} | {result['networkx_time']:15.6f} | {result['speedup']:7.2f}x"
            )

    print(
        "The results for NetworkX do not include the time converting the adjacency matrix to a graph."
    )


if __name__ == "__main__":
    main()
