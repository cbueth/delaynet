"""Second difference (2diff) detrending."""

from numpy import copy

from ..decorators import detrending_method


@detrending_method(check_shape=False)
def second_difference(ts):
    """Second difference (2diff) detrending.

    :param ts: Time series to detrend.
    :type ts: numpy.ndarray
    :return: Detrended time series (length is reduced by 2).
    :rtype: numpy.ndarray
    """
    t_ts = copy(ts)
    t_ts = t_ts[1:] - t_ts[:-1]
    t_ts = t_ts[1:] - t_ts[:-1]
    return t_ts
