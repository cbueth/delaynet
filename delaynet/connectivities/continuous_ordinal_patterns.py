"""Continuous Ordinal Patterns (COP) connectivity metric."""

import numpy as np
from numba import njit, prange

from ..decorators import connectivity
from .granger import gt_multi_lag


@connectivity
def random_patterns(
    ts1, ts2, p_size=5, num_rnd_patterns=50, linear=True, lag_steps: int | list = None
):
    """
    Continuous Ordinal Patterns (COP) connectivity metric
    :cite:p:`zaninContinuousOrdinalPatterns2023,olivaresEvaluatingMethodsDetrending2025`.

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
    :param lag_steps: Time lags to consider.
                      Can be a single integer or a list of integers.
                      An integer will consider lags [1, ..., lag_steps].
                      A list will consider the specified values as lags.
    :type lag_steps: int | list
    :return: Best *p*-value and corresponding lag.
    :rtype: tuple[float, int]
    """
    if p_size + max(lag_steps) - 1 > ts1.shape[0]:
        raise ValueError(
            "Pattern size + lag-steps cannot be larger than the time series length."
        )

    if linear:
        best_pv, best_lag = gt_multi_lag(ts1, ts2, lag_steps=lag_steps)
    else:
        best_pv, best_lag = np.inf, 0

    rnd_patterns = np.random.uniform(0.0, 1.0, (num_rnd_patterns, p_size))
    # rnd_patterns = np.linspace(0, 1, p_size)
    rnd_patterns = np.tile(rnd_patterns, (num_rnd_patterns, 1))
    for i in range(num_rnd_patterns):
        rnd_patterns[i] = norm_window(rnd_patterns[i])

    t_ts1 = pattern_transform(np.copy(ts1), rnd_patterns)
    t_ts2 = pattern_transform(np.copy(ts2), rnd_patterns)

    for i in range(num_rnd_patterns):
        p_v, pv_idx = gt_multi_lag(t_ts1[i, :], t_ts2[i, :], lag_steps=lag_steps)
        if best_pv > p_v:
            best_pv = p_v
            best_lag = pv_idx
    return best_pv, best_lag


@njit(cache=True, nogil=True)
def norm_window(ts: np.ndarray) -> np.ndarray:  # pragma: no cover
    """Normalise a window to values between -1 and 1."""
    new_ts = np.copy(ts)
    new_ts -= np.min(new_ts)
    new_ts /= np.max(new_ts)
    new_ts = (new_ts - 0.5) * 2.0

    new_ts[np.isnan(new_ts)] = 0.0

    return new_ts


def pattern_transform(ts: np.ndarray, patterns: np.ndarray) -> np.ndarray:
    """Transform time series using patterns.

    Multiple time series can be transformed with multiple patterns at once.
    Patterns need to have the same length.

    This function also accepts 1D time series and patterns.
    Wrapper for :func:`pattern_transform_2d`.

    :param ts: Time series.
    :type ts: numpy.ndarray, shape=(n_ts, ts_len) or shape=(ts_len,)
    :param patterns: Patterns.
    :type patterns: numpy.ndarray, shape=(n_patterns, pattern_len)
                    or shape=(pattern_len,)
    :return: Transformed time series.
    :rtype: numpy.ndarray, shape=(n_ts, n_patterns, ts_len - pattern_len + 1)
            or if ``ts`` and/or ``patterns`` are 1D, the squeesed shape.
            For both 1D, the shape is (ts_len - pattern_len + 1).
    """
    if ts.ndim == 1:
        ts = ts.reshape(1, -1)
    if patterns.ndim == 1:
        patterns = patterns.reshape(1, -1)
    return pattern_transform_2d(ts, patterns).squeeze()


@njit(nogil=True, parallel=True)
def pattern_transform_2d(
    ts: np.ndarray, patterns: np.ndarray
) -> np.ndarray:  # pragma: no cover
    """Transform time series using patterns.

    Multiple time series can be transformed with multiple patterns at once.
    Patterns need to have the same length.

    Use :func:`pattern_transform` for convenience if you only have one time series or
    one pattern to transform.

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
    """Normalise sliding windows of a time series to values between -1 and 1.

    :param ts: Time series.
    :type ts: numpy.ndarray
    :param window_size: Size of the window.
    :type window_size: int
    :return: Normalised windows.
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
    # Normalise each window to [-1, 1]
    for i in prange(windows.shape[0]):
        normed_windows[i] = norm_window(windows[i])
    return normed_windows


@njit(nogil=True, parallel=True)
def pattern_distance(
    windows: np.ndarray, pattern: np.ndarray
) -> np.ndarray:  # pragma: no cover
    """Compute the distance between the windows and a pattern.

    :param windows: Normalised windows.
    :type windows: numpy.ndarray
    :param pattern: Pattern.
    :type pattern: numpy.ndarray
    :return: Distance between the windows and the pattern.
    :rtype: numpy.ndarray
    """
    distances = np.zeros(windows.shape[0])
    for i in prange(windows.shape[0]):
        for j in prange(pattern.shape[0]):
            distances[i] += np.abs(windows[i, j] - pattern[j])
    return distances / pattern.shape[0] / 2.0
    # equiv. to np.sum(np.abs(windows - pattern), axis=1) / pattern.shape[0] / 2.0
