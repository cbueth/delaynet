"""Rank correlation (RC) connectivity metric."""

from scipy.stats import spearmanr
from numpy import argmin

from .connectivity import connectivity


@connectivity
def rank_correlation(ts1, ts2, max_lag_steps: int = 0):
    """
    Rank correlation (RC) connectivity metric.

    :param ts1: First time series.
    :type ts1: ndarray
    :param ts2: Second time series.
    :type ts2: ndarray
    :param max_lag_steps: Maximum time lag to consider.
    :type max_lag_steps: int
    :return: Mutual information value and time lag.
    :rtype: tuple[float, int]
    """
    all_c = [spearmanr(ts1[: -k or None], ts2[k:])[1] for k in range(max_lag_steps + 1)]
    idx_min = argmin(all_c)
    return all_c[idx_min], idx_min
