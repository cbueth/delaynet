"""
Granger Causality (GC) connectivity metric.

This module implements the Granger causality test,
a renowned metric for assessing predictive causality between elements of a system.
The Granger causality test operates on a simple and intuitive assumption:
if element B is causing element A,
then the past of B should contain information that aids in predicting the future of A.
In other words, B has a role in shaping the future of A.
:cite:p:`grangerInvestigatingCausalRelations1969,dieboldElementsForecasting1997,kirchgassnerIntroductionModernTime2013`

Since its inception, the Granger causality test has found wide-ranging applications
in various fields such as economics, engineering, sociology, biology, and neuroscience.
It has also been adapted to cater to different situations and types of data.
:cite:p:`zaninAssessingGrangerCausality2021`

Before using the Granger causality test, be sure to detrend the time series data.
:cite:p:`Bessler01061984`

This module provides three implementations of the Granger causality test:
a single-lag version, a multi-lag version, and a bidirectional multi-lag version.
"""

import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import f as f_dist
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.tsatools import lagmat2ds

from ..decorators import connectivity
from ..utils.lag_steps import find_optimal_lag


def gt_single_lag(ts1, ts2, lag_step):
    """Granger Causality (GC) connectivity metric with fixed time lag.

    Testing causality of ts1 -> ts2 with a fixed time lag.

    :param ts1: First time series.
    :type ts1: numpy.ndarray
    :param ts2: Second time series.
    :type ts2: numpy.ndarray
    :param lag_step: Time lag to consider.
    :type lag_step: int
    :return: Granger causality test *p*-value.
    :rtype: float
    """
    from statsmodels.tools.tools import add_constant
    from statsmodels.tsa.tsatools import lagmat2ds

    full_ts = np.array([ts2, ts1]).T
    dta = lagmat2ds(full_ts, lag_step, trim="both", dropex=1)
    dtajoint = add_constant(dta[:, 1:], prepend=False)

    y = dta[:, 0]
    X = dtajoint

    n, k = X.shape
    L = lag_step

    XTX = X.T @ X
    L = lag_step

    try:
        beta = np.linalg.solve(XTX, X.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]

    resid = y - X @ beta
    rss = float(resid @ resid)
    dof = n - k
    sigma2 = rss / dof

    try:
        XTX_inv = np.linalg.inv(XTX)
    except np.linalg.LinAlgError:
        XTX_inv = np.linalg.pinv(XTX)

    R = np.column_stack(
        (
            np.zeros((L, L)),
            np.eye(L, L),
            np.zeros((L, 1)),
        )
    )

    Rb = R @ beta
    R_vcov_R = R @ XTX_inv @ R.T
    R_vcov_R *= sigma2

    try:
        inner = np.linalg.solve(R_vcov_R, Rb)
    except np.linalg.LinAlgError:
        inner = np.linalg.lstsq(R_vcov_R, Rb, rcond=None)[0]

    f_stat = float(Rb @ inner) / L
    p_value = float(f_dist.sf(f_stat, L, dof))

    return p_value


@connectivity
def gt_multi_lag(ts1, ts2, lag_steps: int | list = None):
    """Granger Causality connectivity metric with variable time lag.

    Testing for various time lags and selecting the one with the lowest p-value.

    :param ts1: First time series.
    :type ts1: numpy.ndarray
    :param ts2: Second time series.
    :type ts2: numpy.ndarray
    :param lag_steps: Time lags to consider.
                      Can be a single integer or a list of integers.
                      An integer will consider lags [1, ..., lag_steps].
                      A list will consider the specified values as lags.
    :type lag_steps: int | list
    :return: Best *p*-value and corresponding lag.
    :rtype: tuple[float, int]
    """
    return find_optimal_lag(gt_single_lag, ts1, ts2, lag_steps)
