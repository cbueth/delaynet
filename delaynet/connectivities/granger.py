"""Granger Causality (GC) connectivity metric."""

from statsmodels.tsa.tsatools import lagmat2ds
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import numpy as np


def gt_single_lag(ts1, ts2, max_lag):  # TODO: unused function
    full_ts = np.array([ts2, ts1]).T

    # TODO: Repeated code block
    dta = lagmat2ds(full_ts, max_lag, trim="both", dropex=1)
    dtajoint = add_constant(dta[:, 1:], prepend=False)

    res2djoint = OLS(dta[:, 0], dtajoint).fit()

    rconstr = np.column_stack(
        (
            np.zeros((max_lag - 1, max_lag - 1)),
            np.eye(max_lag - 1, max_lag - 1),
            np.zeros((max_lag - 1, 1)),
        )
    )  # TODO: unused statement
    rconstr = np.column_stack(
        (np.zeros((max_lag, max_lag)), np.eye(max_lag, max_lag), np.zeros((max_lag, 1)))
    )
    ftres = res2djoint.f_test(rconstr)
    pValue = np.squeeze(ftres.pvalue)[()]

    return pValue


def gt_multi_lag(ts1, ts2, max_lag=5):
    full_ts = np.array([ts2, ts1]).T

    allPValues = np.zeros((max_lag, 1))

    topLag = max_lag
    for max_lag in range(1, topLag + 1):
        # TODO: Use gt_single_lag here
        dta = lagmat2ds(full_ts, max_lag, trim="both", dropex=1)
        dtajoint = add_constant(dta[:, 1:], prepend=False)

        res2djoint = OLS(dta[:, 0], dtajoint).fit()

        rconstr = np.column_stack(
            (
                np.zeros((max_lag - 1, max_lag - 1)),
                np.eye(max_lag - 1, max_lag - 1),
                np.zeros((max_lag - 1, 1)),
            )
        )  # TODO: unused statement
        rconstr = np.column_stack(
            (
                np.zeros((max_lag, max_lag)),
                np.eye(max_lag, max_lag),
                np.zeros((max_lag, 1)),
            )
        )
        ftres = res2djoint.f_test(rconstr)
        allPValues[max_lag - 1] = np.squeeze(ftres.pvalue)[()]

    return np.min(allPValues), allPValues


def gt_bi_multi_lag(ts1, ts2, max_lag):
    full_ts = np.array([ts2, ts1]).T
    full_ts_r = np.array([ts2[::-1], ts1[::-1]]).T

    all_p_values = np.zeros((max_lag, 1))

    top_lag = max_lag
    for max_lag in range(1, top_lag + 1):
        from statsmodels.tsa.stattools import grangercausalitytests
        import contextlib

        with contextlib.redirect_stdout(None):
            gc_res = grangercausalitytests(full_ts, [max_lag])
        v1 = np.log(gc_res[max_lag][1][0].ssr / gc_res[max_lag][1][1].ssr)

        with contextlib.redirect_stdout(None):
            gc_res = grangercausalitytests(full_ts_r, [max_lag])
        v2 = np.log(gc_res[max_lag][1][0].ssr / gc_res[max_lag][1][1].ssr)

        all_p_values[max_lag - 1] = v1 - v2

    return np.max(all_p_values), all_p_values
