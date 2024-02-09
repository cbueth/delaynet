"""Granger Causality (GC) connectivity metric."""

from contextlib import redirect_stdout

from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.tsatools import lagmat2ds
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import numpy as np

from .connectivity import connectivity


@connectivity
def gt_single_lag(ts1, ts2, lag_step: int = 5):
    """Granger Causality (GC) connectivity metric with fixed time lag.

    :param ts1: First time series.
    :type ts1: ndarray
    :param ts2: Second time series.
    :type ts2: ndarray
    :param lag_step: Time lag to consider.
    :type lag_step: int
    :return: Mutual information value and time lag.
    :rtype: tuple[float, int]
    """
    full_ts = np.array([ts2, ts1]).T

    dta = lagmat2ds(full_ts, lag_step, trim="both", dropex=1)
    dtajoint = add_constant(dta[:, 1:], prepend=False)

    res2djoint = OLS(dta[:, 0], dtajoint).fit()

    rconstr = np.column_stack(
        (
            np.zeros((lag_step, lag_step)),
            np.eye(lag_step, lag_step),
            np.zeros((lag_step, 1)),
        )
    )
    ftres = res2djoint.f_test(rconstr)

    return np.squeeze(ftres.pvalue)[()]


def gt_multi_lag(ts1, ts2, max_lag_steps: int = 5):
    """Granger Causality (GC) connectivity metric with variable time lag.

    :param ts1: First time series.
    :type ts1: ndarray
    :param ts2: Second time series.
    :type ts2: ndarray
    :param max_lag_steps: Maximum time lag to consider.
    :type max_lag_steps: int
    :return: Mutual information value and time lag.
    :rtype: tuple[float, int]
    """
    all_p_values = [
        gt_single_lag(ts1, ts2, lag_step) for lag_step in range(1, max_lag_steps + 1)
    ]
    idx_min = min(range(len(all_p_values)), key=all_p_values.__getitem__)
    return np.min(all_p_values), idx_min
    # TODO: or should statsmodels.tsa.stattools.grangercausalitytests be used?


def gt_bi_multi_lag(ts1, ts2, max_lag_steps: int = 5):
    """Bidirectional Granger Causality (GC) connectivity metric with variable time lag.

    Uses :func:`grangercausalitytests` from statsmodels.

    :param ts1: First time series.
    :type ts1: ndarray
    :param ts2: Second time series.
    :type ts2: ndarray
    :param max_lag_steps: Maximum time lag to consider.
    :type max_lag_steps: int
    :return: Mutual information value and time lag.
    :rtype: tuple[float, int]
    """
    full_ts = np.array([ts2, ts1]).T
    full_ts_r = np.array([ts2[::-1], ts1[::-1]]).T

    all_p_values = np.zeros((max_lag_steps, 1))

    for lag_step in range(1, max_lag_steps + 1):
        with redirect_stdout(None):
            gc_res = grangercausalitytests(full_ts, [lag_step])
        v1 = np.log(gc_res[lag_step][1][0].ssr / gc_res[lag_step][1][1].ssr)

        with redirect_stdout(None):
            gc_res = grangercausalitytests(full_ts_r, [lag_step])
        v2 = np.log(gc_res[lag_step][1][0].ssr / gc_res[lag_step][1][1].ssr)

        all_p_values[lag_step - 1] = v1 - v2

    return np.max(all_p_values), all_p_values
