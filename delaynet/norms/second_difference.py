"""Second difference (2diff) norm."""
from numpy import ravel

from .norm import norm


@norm(check_shape=False)
def second_difference(ts):
    """Second difference (2diff) norm.

    :param ts: Time series to normalize.
    :type ts: ndarray
    :return: Normalized time series (length is reduced by 2).
    :rtype: ndarray
    """
    t_ts = ravel(ts)
    t_ts = t_ts[1:] - t_ts[:-1]
    t_ts = t_ts[1:] - t_ts[:-1]
    return t_ts
