"""Delta norm."""

from numpy import copy, ravel, size, mean

from ..decorators import norm


@norm
def delta(ts, window_size: int = 10):
    r"""Delta norm.

    Local mean subtraction.
    Subtract the local mean, mean([x_{t - w}, ..., x_{t + w}]), from each value x_t.

    .. math::
        x_t' = x_t - \left(2w + 1\right)^{-1} \sum_{k = t - w}^{t + w} x_k

    :param ts: Time series to normalise.
    :type ts: numpy.ndarray
    :param window_size: Window size to use for calculating the mean.
    :type window_size: int
    :return: Normalised time series.
    :rtype: numpy.ndarray
    """
    ts2 = copy(ts)
    for k in range(size(ts)):
        off1 = k - window_size
        off1 = max(off1, 0)
        sub_ts = ts[off1 : (k + window_size)]

        ts2[k] = ts[k] - mean(sub_ts)

    return ts2
