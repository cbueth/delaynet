"""Linear correlation (LC) connectivity metric."""

from numpy import argmax, abs as np_abs
from scipy.stats import pearsonr

from .connectivity import connectivity


@connectivity
def linear_correlation(
    ts1, ts2, max_lag_steps: int = 0, return_abs: bool = True, **pr_kwargs
):
    """
    Linear correlation (LC) connectivity metric.

    :param ts1: First time series.
    :type ts1: ndarray
    :param ts2: Second time series.
    :type ts2: ndarray
    :param max_lag_steps: Maximum time lag to consider.
    :type max_lag_steps: int
    :param return_abs: Return absolute value of correlation.
    :type return_abs: bool
    :param pr_kwargs: Keyword arguments forwarded to :func:`scipy.stats.pearsonr`.
    :return: Mutual information value and time lag.
    :rtype: tuple[float, int]
    """
    all_c = [
        pearsonr(ts1[: -k or None], ts2[k:], **pr_kwargs)[0]
        for k in range(max_lag_steps + 1)
    ]
    idx_max = argmax(np_abs(all_c))
    return np_abs(all_c[idx_max]) if return_abs else all_c[idx_max], idx_max
