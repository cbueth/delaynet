"""Naive sum causality metric."""

from numpy import sum as npsum, exp


def naive_sum(ts1, ts2):
    q1 = npsum(exp(ts1))
    q2 = npsum(exp(ts2))

    return q1 * q2
