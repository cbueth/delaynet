"""Delta norm."""

from numpy import copy, ravel, size, mean

from ..decorators import norm


@norm
def delta(ts, window_size: int = 10):
    """Delta norm.

    :param ts: Time series to normalize.
    :type ts: ndarray
    :param window_size: Window size to use for calculating the mean.
    :type window_size: int
    :return: Normalized time series.
    :rtype: ndarray
    """
    t_ts = ravel(ts)
    t_ts2 = copy(t_ts)
    for k in range(size(t_ts)):
        off1 = k - window_size
        off1 = max(off1, 0)
        sub_ts = t_ts[off1 : (k + window_size)]

        t_ts2[k] = t_ts[k] - mean(sub_ts)
    norm_ts1 = t_ts2

    return norm_ts1
