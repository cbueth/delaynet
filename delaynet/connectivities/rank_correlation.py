"""Rank correlation (RC) connectivity metric."""

from scipy.stats import spearmanr

from ..decorators import connectivity
from ..utils.lag_steps import find_optimal_lag


@connectivity
def rank_correlation(ts1, ts2, lag_steps: int | list = None, **sr_kwargs):
    """
    Rank correlation (RC) connectivity metric.

    RC measures the spearman rank correlation coefficient between two time series
    over specified time lags.
    The interesting value is the *p*-value of the statistic, which is returned as the
    second element of the returned tuple in :func:`scipy.stats.spearmanr`.

    :param ts1: First time series.
    :type ts1: numpy.ndarray
    :param ts2: Second time series.
    :type ts2: numpy.ndarray
    :param lag_steps: Time lags to consider.
                      Can be a single integer or a list of integers.
                      An integer will consider lags [1, ..., lag_steps].
                      A list will consider the specified values as lags.
    :type lag_steps: int | list
    :param sr_kwargs: Keyword arguments forwarded to :func:`scipy.stats.spearmanr`.
    :return: Best *p*-value and corresponding lag.
    :rtype: tuple[float, int]
    """
    return find_optimal_lag(_rc_pval, ts1, ts2, lag_steps, **sr_kwargs)


def _rc_pval(ts1, ts2, lag, **sr_kwargs):
    return spearmanr(ts1[: -lag or None], ts2[lag:], **sr_kwargs)[1]
