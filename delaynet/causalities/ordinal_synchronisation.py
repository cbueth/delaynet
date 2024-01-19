"""Ordinal Synchronisation (OS) causality metric."""

import numpy as np


def ordinal_synchronisation(ts1, ts2):
    os = np.zeros(6)

    for k in range(1, 6):
        os[k] = os_metric(ts1[:(-k)], ts2[k:], N=np.size(ts1) - k, D=3, tau=1)

    return 1.0 / np.max(np.abs(os)), np.argmax(np.abs(os))


def os_metric(x1, x2, N, D, tau):
    v0 = np.arange(0, D)
    norm = np.sqrt(np.dot(v0, v0))
    min_val = np.dot(np.arange(0, D), np.flip(np.arange(0, D))) / np.dot(
        np.arange(0, D), np.arange(0, D)
    )

    os_aux = np.zeros(((D - 1) * tau + 1, 1))
    ios = np.zeros((N // D, 1))

    for n in range((D - 1) * tau + 1):
        x11 = x1[n:N:tau]
        x22 = x2[n:N:tau]
        x11 = x11[: len(x11) // D * D]
        v0 = np.vstack(np.hsplit(x11, len(x11) // D))
        x22 = x22[: len(x22) // D * D]
        w0 = np.vstack(np.hsplit(x22, len(x22) // D))
        n2 = len(x11)
        del x11
        del x22

        iv = np.argsort(v0) / norm
        iw = np.argsort(w0) / norm

        for t in range(n2 // D):
            ios_raw = np.dot(iv[t, :], iw[t, :])
            ios[t] = 2 * ((ios_raw - min_val) / (1 - min_val) - 0.5)

        os_aux[n] = np.mean(ios)
    return np.mean(os_aux)
