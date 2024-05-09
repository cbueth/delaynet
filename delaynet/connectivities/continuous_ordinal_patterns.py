"""Continuous Ordinal Patterns (COP) connectivity metric."""

import numpy as np
from numba import njit, prange

from .granger import gt_multi_lag

from ..decorators import connectivity


@connectivity
def random_patterns(
    ts1, ts2, p_size=5, num_rnd_patterns=50, linear=True, max_lag_steps=5
):
    """
    Continuous Ordinal Patterns (COP) connectivity metric.

    :param ts1: First time series.
    :type ts1: numpy.ndarray
    :param ts2: Second time series.
    :type ts2: numpy.ndarray
    :param p_size: Size of the ordinal pattern.
    :type p_size: int
    :param num_rnd_patterns: Number of random patterns to consider.
    :type num_rnd_patterns: int
    :param linear: Start with the identity pattern.
    :type linear: bool
    :param max_lag_steps: Maximum time lag to consider.
    :type max_lag_steps: int
    :return: Mutual information value and time lag.
    :rtype: tuple[float, int]
    """
    best_pv = 1.0
    best_lag = 0

    if linear:
        best_pv, all_pv = gt_multi_lag(ts1, ts2, max_lag_steps=max_lag_steps)
        best_lag = np.argmin(all_pv)

    for _ in range(num_rnd_patterns):
        best_ret = np.random.uniform(0.0, 1.0, p_size)
        best_ret = norm_window(best_ret)

        t_ts1 = tranf_ts(np.copy(ts1), best_ret)
        t_ts2 = tranf_ts(np.copy(ts2), best_ret)

        # p_v = GC.GT_SingleLag( t_ts1, t_ts2, max_lag_steps = max_lag_steps )
        # if best_pv > p_v:
        #     best_pv = p_v

        p_v, all_pv = gt_multi_lag(t_ts1, t_ts2, max_lag_steps=max_lag_steps)
        if best_pv > p_v:
            best_pv = p_v
            best_lag = np.argmin(all_pv)

    return best_pv, best_lag


@njit(cache=True, nogil=True)
def norm_window(ts: np.ndarray) -> np.ndarray:  # pragma: no cover
    """Normalize a window to values between -1 and 1."""
    new_ts = np.copy(ts)
    new_ts -= np.min(new_ts)
    new_ts /= np.max(new_ts)
    new_ts = (new_ts - 0.5) * 2.0

    new_ts[np.isnan(new_ts)] = 0.0

    return new_ts


@njit(cache=True, nogil=True)
def tranf_ts(ts, patt):  # pragma: no cover
    """Transform a time series using a pattern."""
    ts_l = ts.shape[0]
    w = np.zeros((ts_l - patt.shape[0] + 1))

    for t in range(ts_l - patt.shape[0] + 1):
        n_w = norm_window(ts[t : (t + patt.shape[0])])
        for l in range(patt.shape[0]):
            w[t] += np.abs(n_w[l] - patt[l])

        w[t] = w[t] / patt.shape[0] / 2.0

    return w


# Second implementation


@njit(nogil=True, parallel=True)
def pattern_transform(
    ts: np.ndarray, patterns: np.ndarray
) -> np.ndarray:  # pragma: no cover
    """Transform time series using patterns.

    Multiple time series can be transformed with multiple patterns at once.
    Patterns need to have the same length.

    :param ts: Time series.
    :type ts: numpy.ndarray, shape=(n_ts, ts_len)
    :param patterns: Patterns.
    :type patterns: numpy.ndarray, shape=(n_patterns, pattern_len)
    :return: Transformed time series.
    :rtype: numpy.ndarray, shape=(n_ts, n_patterns, ts_len - pattern_len + 1)
    """
    transformed = np.zeros(
        (ts.shape[0], patterns.shape[0], ts.shape[1] - patterns.shape[1] + 1)
    )
    for i in range(ts.shape[0]):
        normed_windows = norm_windows(ts[i], patterns.shape[1])
        for j in range(patterns.shape[0]):
            transformed[i, j] = pattern_distance(normed_windows, patterns[j])
    return transformed


@njit(nogil=True, parallel=True)
def norm_windows(ts: np.ndarray, window_size: int) -> np.ndarray:  # pragma: no cover
    """Normalize sliding windows of a time series to values between -1 and 1.

    :param ts: Time series.
    :type ts: numpy.ndarray
    :param window_size: Size of the window.
    :type window_size: int
    :return: Normalized windows.
    :rtype: numpy.ndarray
    """
    # Create a sliding window view of the input array
    # windows = np.lib.stride_tricks.sliding_window_view(
    #     x=ts, window_shape=window_size, writeable=False
    # )
    windows = np.lib.stride_tricks.as_strided(
        x=ts,
        strides=(ts.strides[0], ts.strides[0]),
        shape=(ts.shape[0] - window_size + 1, window_size),
    )
    normed_windows = np.zeros_like(windows)
    # Normalize each window to [-1, 1]
    for i in prange(windows.shape[0]):  # pylint: disable=not-an-iterable
        normed_windows[i] = norm_window(windows[i])
    return normed_windows


@njit(nogil=True, parallel=True)
def pattern_distance(
    windows: np.ndarray, pattern: np.ndarray
) -> np.ndarray:  # pragma: no cover
    """Compute the distance between the windows and a pattern.

    :param windows: Normalized windows.
    :type windows: numpy.ndarray
    :param pattern: Pattern.
    :type pattern: numpy.ndarray
    :return: Distance between the windows and the pattern.
    :rtype: numpy.ndarray
    """
    distances = np.zeros(windows.shape[0])
    for i in prange(windows.shape[0]):  # pylint: disable=not-an-iterable
        for j in prange(pattern.shape[0]):  # pylint: disable=not-an-iterable
            distances[i] += np.abs(windows[i, j] - pattern[j])
    return distances / pattern.shape[0] / 2.0
    # equiv. to np.sum(np.abs(windows - pattern), axis=1) / pattern.shape[0] / 2.0
