"""Continuous Ordinal Patterns (COP) connectivity metric."""

import numpy as np
from numba import jit

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


@jit(cache=True, nopython=True, nogil=True)
def norm_window(ts):
    """Normalize a window to values between -1 and 1."""
    new_ts = np.copy(ts)
    new_ts -= np.min(new_ts)
    new_ts /= np.max(new_ts)
    new_ts = (new_ts - 0.5) * 2.0

    new_ts[np.isnan(new_ts)] = 0.0

    return new_ts


@jit(cache=True, nopython=True, nogil=True)
def tranf_ts(ts, patt):
    """Transform a time series using a pattern."""
    ts_l = ts.shape[0]
    w = np.zeros((ts_l - patt.shape[0] + 1))

    for t in range(ts_l - patt.shape[0] + 1):
        n_w = norm_window(ts[t : (t + patt.shape[0])])
        for l in range(patt.shape[0]):
            w[t] += np.abs(n_w[l] - patt[l])

        w[t] = w[t] / patt.shape[0] / 2.0

    return w
