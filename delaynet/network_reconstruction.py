"""Network reconstruction module for delaynet.

This module provides functionality to reconstruct networks from time series data
by applying connectivity measures to pairs of time series.
"""

import numpy as np
from numpy import ndarray
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory

from .connectivity import connectivity, Metric


def _compute_pair_connectivity_shared(args):
    """Compute connectivity for a single (i,j) pair using shared memory.

    :param args: Tuple containing (i, j, shm_name, shape, dtype, connectivity_measure, lag_steps, kwargs)
    :type args: tuple
    :return: Tuple containing (i, j, p_value, optimal_lag)
    :rtype: tuple[int, int, float, int]
    """
    i, j, shm_name, shape, dtype, connectivity_measure, lag_steps, kwargs = args

    # Attach to shared memory
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    time_series = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

    # Extract the required time series (these are views of shared memory)
    ts_i = time_series[:, i]
    ts_j = time_series[:, j]

    # Compute connectivity
    result = connectivity(
        ts_i, ts_j, connectivity_measure, lag_steps=lag_steps, **kwargs
    )

    existing_shm.close()  # Don't unlink, main process will do that
    return i, j, result[0], result[1]


def reconstruct_network(
    time_series: ndarray,
    connectivity_measure: Metric,
    lag_steps: int | list[int] | None = None,
    workers: int = None,
    **kwargs,
) -> tuple[ndarray, ndarray]:
    """
    Reconstruct a network from time series data.

    This function applies a connectivity measure to all pairs of time series
    to construct a network represented by weight and lag matrices.

    :param time_series: Array of time series data with shape (n_time, n_nodes).
                       Each column represents a time series for one node.
    :type time_series: numpy.ndarray
    :param connectivity_measure: Connectivity measure to use. Can be either a string
                                name of a built-in measure or a callable function.
                                Available string measures can be found using
                                :func:`delaynet.connectivity.show_connectivity_metrics`.
                                If a callable is provided, it should take two
                                time series as input and return a tuple of (float, int).
    :type connectivity_measure: str or Callable
    :param lag_steps: The number of lag steps to consider. Required.
                      Can be integer for [1, ..., num], or a list of integers.
    :type lag_steps: int | list[int] | None
    :param workers: Number of workers to use for parallel computation.
    :type workers: int | None
    :param kwargs: Additional keyword arguments passed to the connectivity measure.
    :type kwargs: dict
    :return: Tuple containing:

             - weight_matrix: Matrix of p-values with shape (n_nodes, n_nodes).
               Lower p-values indicate stronger connections.
             - lag_matrix: Matrix of optimal time lags with shape (n_nodes, n_nodes).
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    :raises ValueError: If time_series has incorrect dimensions.
    :raises ValueError: If ``connectivity_measure`` is unknown (when given as string).
    :raises ValueError: If ``connectivity_measure`` returns invalid value (when given as callable).
    :raises ValueError: If ``connectivity_measure`` is neither string nor callable.

    Example:
    --------
    >>> import numpy as np
    >>> from delaynet.network_reconstruction import reconstruct_network
    >>> # Generate sample data: 100 time points, 5 nodes
    >>> data = np.random.randn(100, 5)
    >>>
    >>> # Using string metric
    >>> weights, lags = reconstruct_network(data, "linear_correlation", lag_steps=5)
    >>> weights.shape
    (5, 5)
    >>> lags.shape
    (5, 5)
    >>>
    >>> # Using callable metric
    >>> def custom_metric(ts1, ts2, lag_steps=None):
    ...     # Using numpy cov function
    ...     all_values = [np.cov(ts1[: -lag or None], ts2[lag:])[0,1] for lag in lag_steps]
    ...     idx_optimal = min(range(len(all_values)), key=all_values.__getitem__)
    ...     return all_values[idx_optimal], lag_steps[idx_optimal]
    >>> weights, lags = reconstruct_network(data, custom_metric, lag_steps=5)
    >>> weights.shape
    (5, 5)

    Note:
    -----
    The diagonal elements of the weight matrix are set to 1.0 by default,
    indicating no significant self-connection.
    """
    # Validate input
    if time_series.ndim != 2:
        raise ValueError(
            f"time_series must be 2-dimensional, got {time_series.ndim} dimensions"
        )

    n_time, n_nodes = time_series.shape

    if n_time < 2:
        raise ValueError(f"time_series must have at least 2 time points, got {n_time}")

    if n_nodes < 2:
        raise ValueError(f"time_series must have at least 2 nodes, got {n_nodes}")

    # Initialize output matrices
    weight_matrix = np.zeros((n_nodes, n_nodes), dtype=float)
    lag_matrix = np.zeros((n_nodes, n_nodes), dtype=int)

    # Set diagonal elements to p=1.0 (no significant self-connection)
    np.fill_diagonal(weight_matrix, 1.0)

    # Compute connectivity for all pairs
    if workers is None or workers == 1:
        # Sequential execution (current implementation)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:  # Skip self-connections - perfect correlation (p=0) at lag 0
                    # Compute connectivity
                    result = connectivity(
                        time_series[:, i],
                        time_series[:, j],
                        connectivity_measure,
                        lag_steps=lag_steps,
                        **kwargs,
                    )
                    # Connectivity measure returns (p_value, lag)
                    weight_matrix[i, j] = result[0]
                    lag_matrix[i, j] = result[1]
    else:
        # Parallel execution using shared memory
        # Create shared memory once
        shm = shared_memory.SharedMemory(create=True, size=time_series.nbytes)
        shared_array = np.ndarray(
            time_series.shape, dtype=time_series.dtype, buffer=shm.buf
        )
        shared_array[:] = time_series[:]  # Copy data to shared memory once

        try:
            # Prepare jobs: only pass indices and shared memory info
            jobs = []
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j:
                        jobs.append(
                            (
                                i,
                                j,
                                shm.name,
                                time_series.shape,
                                time_series.dtype,
                                connectivity_measure,
                                lag_steps,
                                kwargs,
                            )
                        )

            # Execute in parallel
            with ProcessPoolExecutor(max_workers=workers) as executor:
                results = executor.map(_compute_pair_connectivity_shared, jobs)

            # Populate matrices from results
            for i, j, weight, lag in results:
                weight_matrix[i, j] = weight
                lag_matrix[i, j] = lag

        finally:
            shm.close()
            shm.unlink()  # Clean up shared memory

    return weight_matrix, lag_matrix
