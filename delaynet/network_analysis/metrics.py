"""Basic network metrics for delaynet.

This module provides functions to compute fundamental network metrics
from reconstructed and potentially pruned networks.
"""

from igraph import Graph
from numpy import (
    ndarray,
    array,
    sum as np_sum,
    all as np_all,
    fill_diagonal,
    triu,
    isnan,
    zeros,
    linalg,
)


def betweenness_centrality(
    weight_matrix: ndarray,
    directed: bool = True,
    normalize: bool = True,
) -> ndarray:
    """
    Compute betweenness centrality for each node in the network.

    Betweenness centrality measures how often a node lies on the shortest
    paths between other nodes in the network.

    :param weight_matrix: Matrix of connection weights. Non-zero values
                          indicate connections. For weighted networks,
                          weights are interpreted as connection strengths.
    :type weight_matrix: numpy.ndarray, shape (n_nodes, n_nodes)
    :param directed: If True, treat the network as directed.
    :type directed: bool
    :param normalize: If True, normalize by the maximum possible betweenness.
    :type normalize: bool
    :return: Array of betweenness centrality values for each node.
    :rtype: numpy.ndarray, shape (n_nodes,)
    :raises ValueError: If weight_matrix is not square.

    Example:
    --------
    >>> import numpy as np
    >>> from delaynet.network_analysis.metrics import betweenness_centrality
    >>> # Example binary adjacency matrix
    >>> weights = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    >>> centrality = betweenness_centrality(weights)
    >>> centrality.shape
    (3,)
    """
    # Validate input
    if weight_matrix.shape[0] != weight_matrix.shape[1]:
        raise ValueError(
            f"weight_matrix must be square, got shape {weight_matrix.shape}"
        )
    n_nodes = weight_matrix.shape[0]

    # Handle edge cases
    if n_nodes <= 1:
        return zeros(n_nodes)

    # Check if there are any connections
    if np_all(weight_matrix == 0):
        return zeros(n_nodes)

    # Create igraph graph from weight matrix
    g = Graph.Weighted_Adjacency(
        weight_matrix.tolist(), mode="directed" if directed else "undirected"
    )

    # Calculate betweenness centrality using igraph
    result = g.betweenness(directed=directed)

    # Convert to numpy array and handle any NaN values
    centrality = array(result)
    centrality[isnan(centrality)] = 0.0

    # Apply normalization if requested
    if normalize and n_nodes > 2:
        if directed:
            # For directed graphs: max betweenness = (n-1)*(n-2)
            max_betweenness = (n_nodes - 1) * (n_nodes - 2)
        else:
            # For undirected graphs: max betweenness = (n-1)*(n-2)/2
            max_betweenness = (n_nodes - 1) * (n_nodes - 2) / 2.0

        if max_betweenness > 0:
            centrality = centrality / max_betweenness

    return centrality


def link_density(
    weight_matrix: ndarray,
    directed: bool = True,
) -> float:
    """
    Compute the link density of the network.

    Link density is the ratio of existing connections to the maximum
    possible number of connections in the network. This is equivalent
    to network density.

    :param weight_matrix: Matrix of connection weights. Non-zero values
                          indicate connections.
    :type weight_matrix: numpy.ndarray, shape (n_nodes, n_nodes)
    :param directed: If True, treat the network as directed.
    :type directed: bool
    :return: Link density value between 0 and 1.
    :rtype: float
    :raises ValueError: If weight_matrix is not square.

    Example:
    --------
    >>> import numpy as np
    >>> from delaynet.network_analysis.metrics import link_density
    >>> # Example binary adjacency matrix
    >>> weights = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    >>> density = link_density(weights)
    >>> isinstance(density, float)
    True
    """
    # Validate input
    if weight_matrix.shape[0] != weight_matrix.shape[1]:
        raise ValueError(
            f"weight_matrix must be square, got shape {weight_matrix.shape}"
        )

    n_nodes = weight_matrix.shape[0]

    # Count existing connections (non-zero entries, excluding diagonal)
    adjacency_matrix = (weight_matrix != 0).astype(int)
    fill_diagonal(adjacency_matrix, 0)  # Exclude self-loops

    existing_connections = np_sum(adjacency_matrix)

    # Calculate maximum possible connections
    if directed:
        max_connections = n_nodes * (n_nodes - 1)
    else:
        max_connections = n_nodes * (n_nodes - 1) // 2
        # For undirected networks, count only upper triangle
        existing_connections = np_sum(triu(adjacency_matrix, k=1))

    if max_connections == 0:
        return 0.0

    return existing_connections / max_connections


def isolated_nodes_inbound(weight_matrix: ndarray) -> int:
    """
    Count the number of nodes with no inbound links.

    These are nodes that do not receive delays from other nodes.

    :param weight_matrix: Matrix of connection weights. Non-zero values
                          indicate connections. Rows represent sources,
                          columns represent targets.
    :type weight_matrix: numpy.ndarray, shape (n_nodes, n_nodes)
    :return: Number of nodes with no inbound connections.
    :rtype: int
    :raises ValueError: If weight_matrix is not square.

    Example:
    --------
    >>> import numpy as np
    >>> from delaynet.network_analysis.metrics import isolated_nodes_inbound
    >>> # Example adjacency matrix
    >>> weights = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    >>> count = isolated_nodes_inbound(weights)
    >>> isinstance(count, int)
    True
    """
    # Validate input
    if weight_matrix.shape[0] != weight_matrix.shape[1]:
        raise ValueError(
            f"weight_matrix must be square, got shape {weight_matrix.shape}"
        )

    # Create adjacency matrix (non-zero entries indicate connections)
    adjacency_matrix = (weight_matrix != 0).astype(int)

    # For inbound connections, sum over rows (sources) for each column (target)
    # Exclude diagonal (self-loops)
    fill_diagonal(adjacency_matrix, 0)
    inbound_degrees = np_sum(adjacency_matrix, axis=0)

    # Count nodes with zero inbound degree
    isolated_inbound = np_sum(inbound_degrees == 0)

    return int(isolated_inbound)


def isolated_nodes_outbound(weight_matrix: ndarray) -> int:
    """
    Count the number of nodes with no outbound links.

    These are nodes that do not propagate delays to other nodes,
    they just receive them.

    :param weight_matrix: Matrix of connection weights. Non-zero values
                          indicate connections. Rows represent sources,
                          columns represent targets.
    :type weight_matrix: numpy.ndarray, shape (n_nodes, n_nodes)
    :return: Number of nodes with no outbound connections.
    :rtype: int
    :raises ValueError: If weight_matrix is not square.

    Example:
    --------
    >>> import numpy as np
    >>> from delaynet.network_analysis.metrics import isolated_nodes_outbound
    >>> # Example adjacency matrix
    >>> weights = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    >>> count = isolated_nodes_outbound(weights)
    >>> isinstance(count, int)
    True
    """
    # Validate input
    if weight_matrix.shape[0] != weight_matrix.shape[1]:
        raise ValueError(
            f"weight_matrix must be square, got shape {weight_matrix.shape}"
        )

    # Create adjacency matrix (non-zero entries indicate connections)
    adjacency_matrix = (weight_matrix != 0).astype(int)

    # For outbound connections, sum over columns (targets) for each row (source)
    # Exclude diagonal (self-loops)
    fill_diagonal(adjacency_matrix, 0)
    outbound_degrees = np_sum(adjacency_matrix, axis=1)

    # Count nodes with zero outbound degree
    isolated_outbound = np_sum(outbound_degrees == 0)

    return int(isolated_outbound)


def global_efficiency(weight_matrix: ndarray, directed: bool = True) -> float:
    """
    Compute the global efficiency of the network.

    Global efficiency is the average of the inverse shortest path lengths
    between all pairs of nodes. It measures how efficiently information
    can be exchanged over the network.

    :param weight_matrix: Matrix of connection weights. Non-zero values
                          indicate connections. For weighted networks,
                          weights are interpreted as connection strengths.
    :type weight_matrix: numpy.ndarray, shape (n_nodes, n_nodes)
    :param directed: If True, treat the network as directed.
    :type directed: bool
    :return: Global efficiency value between 0 and 1.
    :rtype: float
    :raises ValueError: If weight_matrix is not square.

    Example:
    --------
    >>> import numpy as np
    >>> from delaynet.network_analysis.metrics import global_efficiency
    >>> # Example binary adjacency matrix
    >>> weights = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    >>> efficiency = global_efficiency(weights)
    >>> isinstance(efficiency, float)
    True
    """
    # Validate input
    if weight_matrix.shape[0] != weight_matrix.shape[1]:
        raise ValueError(
            f"weight_matrix must be square, got shape {weight_matrix.shape}"
        )

    n_nodes = weight_matrix.shape[0]

    if n_nodes <= 1:
        return 0.0

    # Remove self-loops for efficiency calculation
    weight_matrix_copy = weight_matrix.copy()
    fill_diagonal(weight_matrix_copy, 0)

    # Check if there are any connections
    if np_all(weight_matrix_copy == 0):
        return 0.0

    # For efficiency calculation, we need distances = 1/weight
    # Create distance matrix where distance = 1/abs(weight) for non-zero weights
    distance_matrix = zeros(weight_matrix_copy.shape)
    nonzero_mask = weight_matrix_copy != 0
    distance_matrix[nonzero_mask] = 1.0 / abs(weight_matrix_copy[nonzero_mask])

    # Create igraph graph using Weighted_Adjacency with distance matrix
    g = Graph.Weighted_Adjacency(
        distance_matrix.tolist(), mode="directed" if directed else "undirected"
    )

    # Calculate shortest path distances for all pairs
    dist_matrix = g.distances(weights="weight")

    # Calculate global efficiency
    total_efficiency = 0.0
    pair_count = 0

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                distance = dist_matrix[i][j]
                if distance < float("inf") and distance > 0:
                    total_efficiency += 1.0 / distance
                pair_count += 1

    if pair_count == 0:
        return 0.0

    return total_efficiency / pair_count


def transitivity(weight_matrix: ndarray) -> float:
    r"""
    Compute the transitivity (global clustering coefficient) of the network.

    Transitivity measures the fraction of all possible triangles present in the graph.
    Following the NetworkX definition, transitivity is calculated as:

    .. math::

        T = 3 \frac{\text{number of triangles}}{\text{number of triads}}


    where triads are sets of 3 nodes with at least 2 edges between them.

    This implementation uses igraph's
    :doc:`Graph.transitivity_undirected() <igraph:analysis>` method,
    which correctly implements the above definition. For directed graphs,
    the network is first converted to undirected before calculation.

    Note that for directed networks, the direction of edges is ignored when
    calculating transitivity, as the concept of triangles is defined for
    undirected graphs. For directed networks, consider using :func:`reciprocity`
    to measure the tendency of vertex pairs to form mutual connections.

    Due to the way NetworkX handles directed networks when calculating transitivity,
    transitivity calculated with this method differs on undirected graphs.
    This implementation collapses the directed network into an undirected network
    and calculates transitivity using igraph's method.

    :param weight_matrix: Matrix of connection weights. Non-zero values
                          indicate connections.
    :type weight_matrix: numpy.ndarray, shape (n_nodes, n_nodes)
    :return: Transitivity value between 0 and 1.
    :rtype: float
    :raises ValueError: If weight_matrix is not square.

    Example:
    --------
    >>> import numpy as np
    >>> from delaynet.network_analysis.metrics import transitivity
    >>> # Example binary adjacency matrix
    >>> weights = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    >>> trans = transitivity(weights)
    >>> isinstance(trans, float)
    True
    """
    # Validate input
    if weight_matrix.shape[0] != weight_matrix.shape[1]:
        raise ValueError(
            f"weight_matrix must be square, got shape {weight_matrix.shape}"
        )
    n_nodes = weight_matrix.shape[0]

    if n_nodes <= 2:
        return 0.0

    # Check if there are any connections
    if np_all(weight_matrix == 0):
        return 0.0

    # Create igraph graph from weight matrix
    # Always create as directed first to handle non-symmetric matrices
    g = Graph.Weighted_Adjacency(weight_matrix.tolist(), mode="directed")

    # Always convert to undirected for transitivity calculation
    g_undirected = g.as_undirected(mode="collapse")
    result = g_undirected.transitivity_undirected()

    # Handle case where igraph returns nan (no edges)
    if isnan(result):
        return 0.0

    return float(result)


def reciprocity(weight_matrix: ndarray) -> float:
    r"""
    Compute the reciprocity of a directed network.

    Reciprocity measures the tendency of vertex pairs to form mutual connections.
    It is defined as the fraction of edges that are reciprocated in a directed
    network. Formally, the reciprocity is calculated as:

    .. math::

        R = \frac{1}{m} \sum_{i,j} A_{i,j} A_{j,i} = \frac{1}{m} \mathrm{Tr} \, A^2

    where :math:`m` is the number of edges and :math:`A` is the adjacency matrix
    of the graph. Note that :math:`A_{i,j} A_{j,i} = 1` if and only if :math:`i`
    links to :math:`j` and vice versa.

    This implementation uses igraph's :doc:`Graph.reciprocity() <igraph:analysis>`
    method.

    For undirected networks, reciprocity is not defined and this function will
    raise a ValueError.

    :param weight_matrix: Matrix of connection weights. Non-zero values
                          indicate connections.
    :type weight_matrix: numpy.ndarray, shape (n_nodes, n_nodes)
    :return: Reciprocity value between 0 and 1.
    :rtype: float
    :raises ValueError: If weight_matrix is not square or if the network is undirected.

    Example:
    --------
    >>> import numpy as np
    >>> from delaynet.network_analysis.metrics import reciprocity
    >>> # Example directed adjacency matrix
    >>> weights = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    >>> recip = reciprocity(weights)
    >>> isinstance(recip, float)
    True

    References:
    -----------
    .. [1] https://www.sci.unich.it/~francesc/teaching/network/transitivity.html
    """
    # Validate input
    if weight_matrix.shape[0] != weight_matrix.shape[1]:
        raise ValueError(
            f"weight_matrix must be square, got shape {weight_matrix.shape}"
        )

    n_nodes = weight_matrix.shape[0]

    # Handle special cases
    if n_nodes <= 1:
        return 0.0

    # Check if the network is undirected (symmetric matrix)
    if np_all(weight_matrix == weight_matrix.T):
        raise ValueError(
            "Reciprocity is only defined for directed networks. "
            "For undirected networks, all connections are reciprocal by definition."
        )

    # Check if there are any connections
    if np_all(weight_matrix == 0):
        return 0.0

    # Create igraph graph from weight matrix
    g = Graph.Weighted_Adjacency(weight_matrix.tolist(), mode="directed")

    # Calculate reciprocity using igraph
    result = g.reciprocity()

    # Handle case where igraph returns nan (no edges)
    if isnan(result):
        return 0.0

    return float(result)


def eigenvector_centrality(
    weight_matrix: ndarray,
    directed: bool = True,
) -> ndarray:
    """
    Compute eigenvector centrality for each node in the network.

    Eigenvector centrality measures the influence of a node in a network.
    It assigns relative scores to all nodes based on the concept that
    connections to high-scoring nodes contribute more to the score of
    the node in question than equal connections to low-scoring nodes.

    :param weight_matrix: Matrix of connection weights. Non-zero values
                          indicate connections.
    :type weight_matrix: numpy.ndarray, shape (n_nodes, n_nodes)
    :param directed: If True, treat the network as directed.
    :type directed: bool
    :return: Array of eigenvector centrality values for each node.
    :rtype: numpy.ndarray, shape (n_nodes,)
    :raises ValueError: If weight_matrix is not square.

    Example:
    --------
    >>> import numpy as np
    >>> from delaynet.network_analysis.metrics import eigenvector_centrality
    >>> # Example binary adjacency matrix
    >>> weights = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    >>> centrality = eigenvector_centrality(weights)
    >>> centrality.shape
    (3,)
    """
    # Validate input
    if weight_matrix.shape[0] != weight_matrix.shape[1]:
        raise ValueError(
            f"weight_matrix must be square, got shape {weight_matrix.shape}"
        )

    n_nodes = weight_matrix.shape[0]

    # Handle edge cases
    if n_nodes == 0:
        return array([])

    if n_nodes == 1:
        return array([1.0])

    # Check if the matrix has any connections
    adjacency_matrix = (weight_matrix != 0).astype(int)
    fill_diagonal(adjacency_matrix, 0)  # Remove self-loops

    if np_all(adjacency_matrix == 0):
        return zeros(n_nodes)

    # Create igraph graph from weight matrix
    g = Graph.Weighted_Adjacency(
        weight_matrix.tolist(), mode="directed" if directed else "undirected"
    )

    # Use igraph's eigenvector centrality
    centrality = g.eigenvector_centrality(weights="weight", directed=directed)

    # Convert to numpy array and normalize to unit length
    centrality = array(centrality)
    norm = linalg.norm(centrality)
    if norm > 0:
        centrality = centrality / norm

    return centrality
