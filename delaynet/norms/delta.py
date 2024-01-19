"""Delta norm."""

from numpy import copy, ravel, size, mean


def delta(vol_data):
    t_ts = ravel(vol_data)
    t_ts2 = copy(t_ts)
    for k in range(size(t_ts)):
        off1 = k - 10
        if off1 < 0:
            off1 = 0
        sub_ts = t_ts[off1 : (k + 10)]

        t_ts2[k] = t_ts[k] - mean(sub_ts)
    norm_ts1 = t_ts2

    return norm_ts1
