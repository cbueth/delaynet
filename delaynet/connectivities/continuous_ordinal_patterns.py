"""Continuous Ordinal Patterns (COP) connectivity metric."""

import numpy as np
from numba import jit

from .granger import gt_multi_lag

from .connectivity import connectivity


@connectivity
def random_patterns(ts1, ts2, max_lag=5, p_size=5, num_rnd_patterns=50, linear=True):
    best_pv = 1.0
    best_lag = 0

    if linear:
        best_pv, all_pv = gt_multi_lag(ts1, ts2, max_lag=max_lag)
        best_lag = np.argmin(all_pv)

    for k in range(num_rnd_patterns):
        best_ret = np.random.uniform(0.0, 1.0, (p_size))
        best_ret = norm_window(best_ret)

        t_ts1 = tranf_ts(np.copy(ts1), best_ret)
        t_ts2 = tranf_ts(np.copy(ts2), best_ret)

        # pV = GC.GT_SingleLag( t_ts1, t_ts2, maxlag = maxlag )
        # if best_pv > pV:
        #     best_pv = pV

        pV, all_pv = gt_multi_lag(t_ts1, t_ts2, max_lag)
        if best_pv > pV:
            best_pv = pV
            best_lag = np.argmin(all_pv)

    return best_pv, best_lag


@jit(cache=True, nopython=True, nogil=True)
def norm_window(ts):
    new_ts = np.copy(ts)
    new_ts -= np.min(new_ts)
    new_ts /= np.max(new_ts)
    new_ts = (new_ts - 0.5) * 2.0

    new_ts[np.isnan(new_ts)] = 0.0

    return new_ts


@jit(cache=True, nopython=True, nogil=True)
def tranf_ts(ts, patt):
    tsL = ts.shape[0]
    W = np.zeros((tsL - patt.shape[0] + 1))

    for t in range(tsL - patt.shape[0] + 1):
        nW = norm_window(ts[t : (t + patt.shape[0])])
        for l in range(patt.shape[0]):
            W[t] += np.abs(nW[l] - patt[l])

        W[t] = W[t] / patt.shape[0] / 2.0

    return W
