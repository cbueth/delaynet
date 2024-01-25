"""Module to provide a unified interface for all connectivity metrics."""
from collections.abc import Callable

from numpy import ndarray


from .connectivities import __all_connectivity_metrics_names__
from .connectivities.connectivity import connectivity as connectivity_decorator

Metric = str | Callable[[ndarray, ndarray, ...], float | tuple[float, int]]


def connectivity(
    ts1: ndarray,
    ts2: ndarray,
    metric: Metric,
    *args,
    **kwargs,
) -> float | tuple[float, int]:
    """
    Calculate connectivity between two time series using a given metric.

    Keyword arguments are forwarded to the metric function.

    The metrics can be either a string or a function, implementing a connectivity
    metric.
    The following metrics are available (case-insensitive):
        - COP: Continuous Ordinal Patterns
        - GC: Granger Causality
        - GC_Bi: Bi-directional Granger Causality
        - TE: Transfer Entropy
        - MI_KA: Mutual Information
        - RC: Rank Correlation
        - OS: Ordinal Synchronisation
        - Naive: Sum

    (Find all in submodule :mod:`delaynet.connectivities`, names are stored in
    :attr:`delaynet.connectivities.__all_connectivity_metrics__`)

    If a `callable` is given, it should take two time series as input and return a
    `float`, or a `tuple` of `float` and `int`.

    :param ts1: First time series.
    :type ts1: ndarray
    :param ts2: Second time series.
    :type ts2: ndarray
    :param metric: Metric to use.
    :type metric: str or Callable
    :param args: Positional arguments forwarded to the connectivity function, see
                 documentation.
    :type args: list
    :param kwargs: Keyword arguments forwarded to the connectivity function, see
                   documentation.
    :return: Connectivity value and lag (if applicable).
    :rtype: float or tuple of float and int
    :raises ValueError: If the metric is unknown. Given as string.
    :raises ValueError: If the metric returns an invalid value. Given a Callable.
    :raises ValueError: If the metric is neither a string nor a Callable.
    """

    if isinstance(metric, str):
        metric = metric.lower()

        if metric not in __all_connectivity_metrics_names__:
            raise ValueError(f"Unknown metric: {metric}")

        return __all_connectivity_metrics_names__[metric](ts1, ts2, **kwargs)

    if not callable(metric):
        raise ValueError(
            f"Invalid connectivity metric: {metric}. Must be string or callable."
        )

    # connectivity metric is a callable,
    # add decorator to assure correct kwargs, type and shape
    mcb_kwargs = kwargs.pop("mcb_kwargs", None)
    return connectivity_decorator(mcb_kwargs=mcb_kwargs)(metric)(
        ts1, ts2, *args, **kwargs
    )
