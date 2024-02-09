"""Ordinal Synchronisation (OS) connectivity metric."""

import numpy as np

from .connectivity import connectivity


@connectivity
def ordinal_synchronisation(ts1, ts2, d: int = 3, tau: int = 1, max_lag_steps: int = 0):
    """
    Ordinal synchronisation (OS) connectivity metric.

    :param ts1: First time series.
    :type ts1: ndarray
    :param ts2: Second time series.
    :type ts2: ndarray
    :param d: Embedding dimension / delay dimension.
    :type d: int
    :param tau: OS time delay.
    :type tau: int
    :param max_lag_steps: Maximum time series lag to consider, in steps of ``tau``.
    :type max_lag_steps: int
    :return: Mutual information value and time lag.
    :rtype: tuple[float, int]
    """
    os = [
        os_metric(ts1[: -k or None], ts2[k:], n=np.size(ts1) - k, d=d, tau=tau)
        for k in range(max_lag_steps + 1)
    ]
    idx_max = np.argmax(np.abs(os))
    return 1.0 / np.max(np.abs(os)), idx_max


# pylint: disable=too-many-locals
def os_metric(x1, x2, n, d, tau):
    """
    Ordinal synchronisation metric, helper function.

    :param x1: First time series.
    :type x1: ndarray
    :param x2: Second time series.
    :type x2: ndarray
    :param n: Sample size / number of observations.
    :type n: int
    :param d: Embedding dimension / delay dimension.
    :type d: int
    :param tau: Time delay.
    :type tau: int
    :return: Ordinal synchronisation metric.
    :rtype: float
    """
    v0 = np.arange(0, d)
    norm = np.sqrt(np.dot(v0, v0))
    min_val = np.dot(np.arange(0, d), np.flip(np.arange(0, d))) / np.dot(
        np.arange(0, d), np.arange(0, d)
    )

    os_aux = np.zeros(((d - 1) * tau + 1, 1))
    ios = np.zeros((n // d, 1))

    for i in range((d - 1) * tau + 1):
        x11 = x1[i:i:tau]
        x22 = x2[i:i:tau]
        x11 = x11[: len(x11) // d * d]
        v0 = np.vstack(np.hsplit(x11, len(x11) // d))
        x22 = x22[: len(x22) // d * d]
        w0 = np.vstack(np.hsplit(x22, len(x22) // d))
        n2 = len(x11)
        del x11
        del x22

        iv = np.argsort(v0) / norm
        iw = np.argsort(w0) / norm

        for t in range(n2 // d):
            ios_raw = np.dot(iv[t, :], iw[t, :])
            ios[t] = 2 * ((ios_raw - min_val) / (1 - min_val) - 0.5)

        os_aux[i] = np.mean(ios)

    # TODO: Optimise this
    # - Precompute x11, x22
    # - Replace hsplit and vstack with reshape
    #   - To split x11 and x22 into chunks of size d and stack them vertically
    # - Calculate ios mean with np.mean(ios, axis=1) - outside the loop
    # - Use list comprehension instead of for loop
    # - Use numba(?)
    # - Write tests to check that the optimised version gives the same results

    return np.mean(os_aux)


# def os_metric(x1, x2, d, tau):
#     """
#     Ordinal synchronisation metric, helper function.
#
#     :param x1: First time series.
#     :type x1: ndarray
#     :param x2: Second time series.
#     :type x2: ndarray
#     :param d: Embedding dimension / delay dimension.
#     :type d: int
#     :param tau: Time delay.
#     :type tau: int
#     :return: Ordinal synchronisation metric.
#     :rtype: float
#     """
#     v0 = np.arange(0, d)
#     norm = np.sqrt(np.dot(v0, v0))
#     min_val = np.dot(np.arange(0, d), np.flip(np.arange(0, d))) / np.dot(
#         np.arange(0, d), np.arange(0, d)
#     )
#
#     x11 = x1[:(d - 1) * tau + 1:tau]
#     x22 = x2[:(d - 1) * tau + 1:tau]
#     x11 = x11[: len(x11) // d * d].reshape(-1, d)
#     x22 = x22[: len(x22) // d * d].reshape(-1, d)
#
#     os_aux = [compute_os_aux(i, x11, x22, norm, min_val) for i in range((d - 1) * tau + 1)]
#     return np.mean(os_aux)
#
#
# def compute_os_aux(i, x11, x22, norm, min_val):
#     v0 = x11[i]
#     w0 = x22[i]
#     iv = np.argsort(v0) / norm
#     iw = np.argsort(w0) / norm
#
#     ios = 2 * ((np.dot(iv, iw) - min_val) / (1 - min_val) - 0.5)
#     return np.mean(ios)
