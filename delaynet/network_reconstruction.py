"""Network reconstruction module for delaynet.

This module provides functionality to reconstruct networks from time series data
by applying connectivity measures to pairs of time series.
"""

from os import environ
from time import time
from sys import stdout

import numpy as np
from numpy import ndarray
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory, Manager, get_context

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
    # Check if we're in a Sphinx build environment once
    is_sphinx = is_sphinx_build()
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

    total_pairs = n_nodes * (n_nodes - 1)  # Total number of pairs to process
    start_time = time()

    # Compute connectivity for all pairs
    if workers is None or workers == 1:
        # Sequential execution
        pairs_processed = 0
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
                    pairs_processed += 1
                    print_progress(
                        pairs_processed,
                        total_pairs,
                        start_time,
                        prefix="Sequential: ",
                        sphinx_mode=is_sphinx,
                    )
    else:
        # Parallel execution using shared memory
        # Create shared memory once
        shm = shared_memory.SharedMemory(create=True, size=time_series.nbytes)
        shared_array = np.ndarray(
            time_series.shape, dtype=time_series.dtype, buffer=shm.buf
        )
        shared_array[:] = time_series[:]  # Copy data to shared memory once

        try:
            # Create a shared counter and lock for progress tracking
            with Manager() as manager:
                counter = manager.Value("i", 0)
                lock = manager.Lock()

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

                # Execute in parallel with progress tracking
                results_list = []
                # Use 'spawn' start method to avoid fork() warnings in multi-threaded processes
                mp_context = get_context("spawn")
                with ProcessPoolExecutor(
                    max_workers=workers, mp_context=mp_context
                ) as executor:
                    futures = [
                        executor.submit(
                            _compute_with_progress,
                            job,
                            counter,
                            lock,
                            total_pairs,
                            start_time,
                            sphinx_mode=is_sphinx,
                        )
                        for job in jobs
                    ]

                    for future in futures:
                        result = future.result()
                        results_list.append(result)

                # Populate matrices from results
                for i, j, weight, lag in results_list:
                    weight_matrix[i, j] = weight
                    lag_matrix[i, j] = lag

        finally:
            shm.close()
            shm.unlink()  # Clean up shared memory

    # Check if we're in Sphinx mode before printing a newline
    if not is_sphinx:
        print()  # New line after completion only if not in Sphinx mode

    return weight_matrix, lag_matrix


def format_time(seconds):
    """Format time in appropriate units (seconds, minutes, hours)."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def is_sphinx_build():
    """Check if the code is running in a Sphinx build environment."""
    # Check for SPHINX_MYST_NB_BUILD and read the docs variables
    return environ.get("SPHINX_MYST_NB_BUILD") == "1" or "READTHEDOCS" in environ


def print_progress(current, total, start_time, prefix="", sphinx_mode=None):
    """Print progress bar with percentage, counts, and estimated time remaining.

    :param current: Current progress value
    :param total: Total value to reach
    :param start_time: Start time for ETA calculation
    :param prefix: Prefix string for the progress bar
    :param sphinx_mode: How to handle progress in Sphinx documentation.
                       If True, only print progress when current == total.
                       If None, automatically detect Sphinx environment.
    """
    # Skip intermediate updates in Sphinx documentation
    if sphinx_mode is None:
        sphinx_mode = is_sphinx_build()

    if sphinx_mode and current < total:
        return

    progress = current / total
    bar_length = 30
    filled_length = int(bar_length * progress)
    bar = "#" * filled_length + "-" * (bar_length - filled_length)

    # Calculate elapsed time and estimate remaining time
    elapsed = time() - start_time

    # Show elapsed time when at 100%, otherwise show ETA
    if current >= total:
        time_str = f"Time: {format_time(elapsed)}"
    elif progress > 0:
        eta = elapsed * (total / current - 1)
        time_str = f"ETA: {format_time(eta)}"
    else:
        time_str = "ETA: ..."

    # Create the progress line with both percentage and counts
    percent = progress * 100

    # Always use carriage return for terminal-friendly output
    line = f"\r{prefix}[{bar}] {current}/{total} ({percent:.1f}%) {time_str}"

    stdout.write(line)
    stdout.flush()


def update_progress(counter, total, start_time, prefix, sphinx_mode=None):
    """Update progress from worker processes

    :param counter: Shared counter value
    :param total: Total number of items to process
    :param start_time: Start time for ETA calculation
    :param prefix: Prefix string for the progress bar
    :param sphinx_mode: How to handle progress in Sphinx documentation
    """
    print_progress(counter.value, total, start_time, prefix, sphinx_mode)
    stdout.flush()


def _compute_with_progress(
    job, counter, lock, total_pairs, start_time, sphinx_mode=None
):
    """Wrapper function to compute connectivity and update progress.

    :param job: Job parameters for connectivity computation
    :param counter: Shared counter for progress tracking
    :param lock: Lock for thread-safe counter updates
    :param total_pairs: Total number of pairs to process
    :param start_time: Start time for ETA calculation
    :param sphinx_mode: How to handle progress in Sphinx documentation
    :return: Computation result
    """
    result = _compute_pair_connectivity_shared(job)
    with lock:
        counter.value += 1
        # Only update progress inside the lock to prevent race conditions
        print_progress(
            counter.value, total_pairs, start_time, "Parallel:   ", sphinx_mode
        )
    return result
