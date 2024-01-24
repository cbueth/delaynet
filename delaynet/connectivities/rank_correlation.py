"""Rank correlation (RC) connectivity metric."""

from scipy.stats import spearmanr
from numpy import zeros, argmin, min as npmin

from .connectivity import connectivity


@connectivity
def rank_correlation(ts1, ts2):
    all_c = zeros(6)

    all_c[0] = spearmanr(ts1, ts2)[1]
    for k in range(1, 6):
        all_c[k] = spearmanr(ts1[:-k], ts2[k:])[1]

    return npmin(all_c), argmin(all_c)
