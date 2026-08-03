"""Delta detrending."""

from numpy import empty, empty_like, integer, cumsum

from ..decorators import detrending_method


@detrending_method
def delta(ts, window_size: int = 10):
    r"""Delta detrending.

    Local mean subtraction.
    Subtract the local mean, mean([x_{t - w}, ..., x_{t + w}]), from each value x_t.

    .. math::
        x_t' = x_t - \left(2w + 1\right)^{-1} \sum_{k = t - w}^{t + w} x_k

    :param ts: Time series to detrend.
    :type ts: numpy.ndarray
    :param window_size: Window size to use for calculating the mean. Must be a positive integer.
    :type window_size: int
    :return: Detrended time series.
    :rtype: numpy.ndarray
    :raises ValueError: If the window_size is not a positive integer.
    """
    # Validate window_size
    if not isinstance(window_size, (int, integer)) or window_size <= 0:
        raise ValueError(f"window_size must be a positive integer, not {window_size}.")

    n = ts.shape[0]
    c = empty(n + 1, dtype=ts.dtype)
    c[0] = 0
    c[1:] = cumsum(ts)

    ts2 = empty_like(ts)
    for k in range(n):
        left = max(0, k - window_size)
        right = min(n, k + window_size)
        ts2[k] = ts[k] - (c[right] - c[left]) / (right - left)

    return ts2
