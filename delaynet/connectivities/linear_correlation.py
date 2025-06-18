"""Linear correlation (LC) connectivity metric."""

from numpy import argmax, abs as np_abs
from scipy.stats import pearsonr

from ..decorators import connectivity
from ..utils.lag_steps import find_optimal_lag


@connectivity
def linear_correlation(ts1, ts2, lag_steps: int | list = None, **pr_kwargs):
    """
    Linear correlation (LC) connectivity metric.

    LC measures the linear correlation between two time series over
    a specified number of time lags.
    The relevant value is the *p*-value of the Pearson correlation coefficient.
    This is returned by :py:func:`scipy.stats.pearsonr` as the second element of the
    returned tuple.

    :param ts1: First time series.
    :type ts1: numpy.ndarray
    :param ts2: Second time series.
    :type ts2: numpy.ndarray
    :param lag_steps: Time lags to consider.
                      Can be a single integer or a list of integers.
                      An integer will consider lags [1, ..., lag_steps].
                      A list will consider the specified values as lags.
    :type lag_steps: int | list
    :param pr_kwargs: Keyword arguments forwarded to :func:`scipy.stats.pearsonr`.
    :return: Best *p*-value and corresponding lag.
    :rtype: tuple[float, int]
    """
    return find_optimal_lag(_pearson_pval, ts1, ts2, lag_steps, **pr_kwargs)


def _pearson_pval(ts1, ts2, lag, **pr_kwargs):
    return pearsonr(ts1[: -lag or None], ts2[lag:], **pr_kwargs)[1]
