"""
Granger Causality (GC) connectivity metric.

This module implements the Granger causality test,
a renowned metric for assessing predictive causality between elements of a system.
The Granger causality test operates on a simple and intuitive assumption:
if element B is causing element A,
then the past of B should contain information that aids in predicting the future of A.
In other words, B has a role in shaping the future of A.
:cite:p:`grangerInvestigatingCausalRelations1969,dieboldElementsForecasting1997`

Since its inception, the Granger causality test has found wide-ranging applications
in various fields such as economics, engineering, sociology, biology, and neuroscience.
It has also been adapted to cater to different situations and types of data.
:cite:p:`zaninAssessingGrangerCausality2021`

Before using the Granger causality test, be sure to detrend the time series data.
:cite:p:`Bessler01061984`

This module provides three implementations of the Granger causality test:
a single lag version, a multi-lag version, and a bidirectional multi-lag version.

.. footbibliography::

"""

from contextlib import redirect_stdout

from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.tsatools import lagmat2ds
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import numpy as np

from ..decorators import connectivity


@connectivity
def gt_single_lag(ts1, ts2, lag_step: int = 5):
    """Granger Causality (GC) connectivity metric with fixed time lag.

    Testing causality of ts1 -> ts2 with a fixed time lag.

    :param ts1: First time series.
    :type ts1: numpy.ndarray
    :param ts2: Second time series.
    :type ts2: numpy.ndarray
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


@connectivity
def gt_multi_lag(ts1, ts2, max_lag_steps: int = 5):
    """Granger Causality connectivity metric with variable time lag.

    Testing for various time lags and selecting the one with the lowest p-value.

    :param ts1: First time series.
    :type ts1: numpy.ndarray
    :param ts2: Second time series.
    :type ts2: numpy.ndarray
    :param max_lag_steps: Maximum time lag to consider.
    :type max_lag_steps: int
    :return: Mutual information value and time lag.
    :rtype: tuple[float, int]
    """
    all_p_values = [
        gt_single_lag(ts1, ts2, lag_step) for lag_step in range(1, max_lag_steps + 1)
    ]
    idx_min = min(range(len(all_p_values)), key=all_p_values.__getitem__)
    return all_p_values[idx_min], idx_min


@connectivity
def gt_multi_lag_statsmodels(ts1, ts2, max_lag_steps: int = 5):
    """Granger Causality (GC) connectivity metric with variable time lag.

    Uses :func:`statsmodels.tsa.stattools.grangercausalitytests` from statsmodels.

    :param ts1: First time series.
    :type ts1: numpy.ndarray
    :param ts2: Second time series.
    :type ts2: numpy.ndarray
    :param max_lag_steps: Maximum time lag to consider.
    :type max_lag_steps: int
    :return: Mutual information value and time lag.
    :rtype: tuple[float, int]
    """
    all_p_values = [
        grangercausalitytests(np.array([ts2, ts1]).T, [lag_step], verbose=False)[
            lag_step
        ][0]["ssr_ftest"][1]
        for lag_step in range(1, max_lag_steps + 1)
    ]
    idx_min = min(range(len(all_p_values)), key=all_p_values.__getitem__)
    return all_p_values[idx_min], idx_min


@connectivity
def gt_bi_multi_lag(ts1, ts2, max_lag_steps: int = 5):
    """Bidirectional Granger Causality (GC) connectivity metric with variable time lag.

    Testing for various time lags and selecting the one with the lowest p-value.
    Both ts1 -> ts2 and ts2 -> ts1 are tested, and the causality is accepted only if the
    former is stronger than the latter.
    Contrary to the name, this metric is exactly excluding bi-directional causalities.

    Uses :func:`statsmodels.tsa.stattools.grangercausalitytests` from statsmodels.

    :param ts1: First time series.
    :type ts1: numpy.ndarray
    :param ts2: Second time series.
    :type ts2: numpy.ndarray
    :param max_lag_steps: Maximum time lag to consider.
    :type max_lag_steps: int
    :return: Mutual information value and time lag.
    :rtype: tuple[float, int]
    """
    full_ts = np.array([ts2, ts1]).T
    full_ts_r = np.array([ts2[::-1], ts1[::-1]]).T
    full_ts_xy = np.array([ts1, ts2]).T
    full_ts_xy_r = np.array([ts1[::-1], ts2[::-1]]).T

    all_p_values = np.zeros((max_lag_steps, 1))

    for lag_step in range(1, max_lag_steps + 1):
        with redirect_stdout(None):
            gc_res = grangercausalitytests(full_ts, [lag_step])
        f_yx = np.log(gc_res[lag_step][1][0].ssr / gc_res[lag_step][1][1].ssr)

        with redirect_stdout(None):
            gc_res = grangercausalitytests(full_ts_r, [lag_step])
        ft_yx = np.log(gc_res[lag_step][1][0].ssr / gc_res[lag_step][1][1].ssr)

        with redirect_stdout(None):
            gc_res = grangercausalitytests(full_ts_xy, [lag_step])
        f_xy = np.log(gc_res[lag_step][1][0].ssr / gc_res[lag_step][1][1].ssr)

        with redirect_stdout(None):
            gc_res = grangercausalitytests(full_ts_xy_r, [lag_step])
        ft_xy = np.log(gc_res[lag_step][1][0].ssr / gc_res[lag_step][1][1].ssr)

        all_p_values[lag_step - 1] = (ft_xy - ft_yx) - (f_xy - f_yx)

    # Determine the maximal difference between the two directions, i.e. a->b and b->a
    idx_max = max(range(len(all_p_values)), key=all_p_values.__getitem__)
    return all_p_values[idx_max][0], idx_max
