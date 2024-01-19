"""Module to provide a unified interface for all causality metrics."""
from collections.abc import Callable

from numpy import ndarray


from .causalities import __all_causality_metrics_names__


def causality(
    ts1: ndarray,
    ts2: ndarray,
    metric: str | Callable[[ndarray, ndarray, ...], float | tuple[float, int]],
    **kwargs,
) -> float | tuple[float, int]:
    """
    Calculate causality between two time series using a given metric.

    Keyword arguments are forwarded to the metric function.

    The metrics can be either a string or a function, implementing a causality metric.
    The following metrics are available (case-insensitive):
        - COP: Continuous Ordinal Patterns
        - GC: Granger Causality
        - GC_Bi: Bi-directional Granger Causality
        - TE: Transfer Entropy
        - MI_KA: Mutual Information
        - RC: Rank Correlation
        - OS: Ordinal Synchronisation
        - Naive: Sum

    (Find all in submodule `delaynet.causalities`, names are stored in
    `delaynet.causalities.__all_causality_metrics__`)

    If a `callable` is given, it should take two time series as input and return a
    `float`, or a `tuple` of `float` and `int`.

    :param ts1: First time series.
    :type ts1: ndarray
    :param ts2: Second time series.
    :type ts2: ndarray
    :param metric: Metric to use.
    :type metric: str or Callable
    :param kwargs: Keyword arguments forwarded to the metric function, see documentation
                   of the metrics.
    :return: Causality value and lag (if applicable).
    :rtype: float or tuple of float and int
    :raises ValueError: If the metric is unknown. Given as string.
    :raises ValueError: If the metric returns an invalid value. Given a Callable.
    :raises ValueError: If the metric is neither a string nor a Callable.
    """

    if isinstance(metric, str):
        metric = metric.lower()

        if metric not in __all_causality_metrics_names__:
            raise ValueError(f"Unknown metric: {metric}")

        return __all_causality_metrics_names__[metric](ts1, ts2, **kwargs)

    if not callable(metric):
        raise ValueError("Invalid metric. Must be string or callable.")

    result = metric(ts1, ts2, **kwargs)
    if isinstance(result, float):
        return result
    if isinstance(result, tuple) and len(result) == 2:
        if isinstance(result[0], float) and isinstance(result[1], int):
            return result[0], result[1]
        raise ValueError(
            "Invalid return value of metric function. "
            "First value of tuple must be float, second must be int."
        )
    raise ValueError("Metric function must return float or tuple of float and int.")
