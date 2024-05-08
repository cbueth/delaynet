"""Gravity connectivity metric."""

from numpy import sum as npsum, exp, argmin, array

from ..decorators import connectivity


@connectivity
def gravity(ts1, ts2, max_lag_steps: int = 5):
    """Gravity connectivity (GC) metric.

    :param ts1: First time series.
    :type ts1: numpy.ndarray
    :param ts2: Second time series.
    :type ts2: numpy.ndarray
    :param max_lag_steps: Maximum time lag to consider.
    :type max_lag_steps: int
    :return: Mutual information value and time lag.
    :rtype: tuple[float, int]
    """
    # q1 = npsum(exp(ts1))
    # q2 = npsum(exp(ts2))
    #
    # return q1 * q2

    all_g = array([gravity_single(ts1, ts2, k) for k in range(max_lag_steps + 1)])
    idx_min = argmin(all_g)
    return all_g[idx_min], idx_min


def gravity_single(ts1, ts2, lag_step):
    """Helper function for gravity connectivity metric.

    :param ts1: First time series.
    :type ts1: numpy.ndarray
    :param ts2: Second time series.
    :type ts2: numpy.ndarray
    :param lag_step: Time lag to consider.
    :type lag_step: int
    :return: Mutual information value and time lag.
    :rtype: tuple[float, int]
    """
    q1 = npsum(exp(ts1[: -lag_step or None]))
    q2 = npsum(exp(ts2[lag_step:]))

    return q1 * q2
