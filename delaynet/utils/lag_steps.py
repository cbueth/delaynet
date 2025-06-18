"""Utility functions to handle lag steps."""

from typing import ParamSpec, Protocol

import numpy as np

from .logging import logging


def assure_lag_list(lag_steps: int | list[int]) -> list[int]:
    """Ensure that ``lag_steps`` is a list of lags.

    If ``lag_steps`` is an integer, it will be
    converted to a list containing integers from 1 to ``lag_steps``.
    If ``lag_steps`` is already a list, it will be checked to ensure that
    all elements are integers.

    :param lag_steps: An integer >= 1 or a list of integers.
    :type lag_steps: int | list[int]
    :return: A list of integers
    :type: list[int]
    :raises ValueError: If ``lag_steps`` is not an integer >= 1 or a list of integers.
    """
    if isinstance(lag_steps, int):
        if lag_steps <= 0:
            raise ValueError("`lag_steps` must be a positive integer or list of such.")
        lag_steps = list(range(1, lag_steps + 1))
    elif not isinstance(lag_steps, list) or (
        isinstance(lag_steps, list) and not all(isinstance(x, int) for x in lag_steps)
    ):
        raise ValueError("`lag_steps` must be an integer or a list of integers.")
    if any(x <= 0 for x in lag_steps):
        logging.warning(
            "Some elements in `lag_steps` are non-positive. "
            "This might produce unscientific results."
        )
    return lag_steps


P = ParamSpec("P")


class Connectivity(Protocol[P]):
    """A protocol for connectivity metrics.

    Connectivity metrics are rigidly typed in their first three parameters:
    the time series and one lag step. The rest are optional keyword arguments.
    """

    def __call__(
        self,
        ts1: np.ndarray,
        ts2: np.ndarray,
        lag: int,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> float: ...


def find_optimal_lag(
    metric_func: Connectivity,
    ts1,
    ts2,
    lag_steps: list,
    op=min,
    **kwargs,
):
    """Find the optimal value and lag for a given metric function.

    The optimal value and lag are determined by applying a given operation
    to a list of values obtained by applying ``metric_func``
    for each lag in ``lag_steps``.
    The operation can be `min`, `max`, or any other operation that takes a list of
    values and returns a single value.
    If ``metric_func`` returns a *p*-value, the operator should be the minimum
    (default optional parameter).

    :param metric_func: Function to compute the metric for a given lag step.
                        Accepts time series ``ts1`` and ``ts2``, a lag step ``lag``,
                        and any additional keyword arguments.
    :type metric_func: Connectivity
    :param ts1: First time series.
    :type ts1: numpy.ndarray
    :param ts2: Second time series.
    :type ts2: numpy.ndarray
    :param lag_steps: Time lags to consider.
                      Needs to be a list of integers.
    :type lag_steps: list
    :param op: Operator to find the optimal lag step
               (e.g., default :py:func:`min` or :py:func:`max`).
    :type op: Callable
    :param kwargs: Additional keyword arguments to pass to the metric function.
    :return: Optimal metric value and corresponding lag step.
    :rtype: tuple[float, int]
    """
    all_values = [metric_func(ts1, ts2, lag_step, **kwargs) for lag_step in lag_steps]
    idx_optimal = op(range(len(all_values)), key=all_values.__getitem__)
    return all_values[idx_optimal], lag_steps[idx_optimal]
