"""Second difference (2diff) norm."""
from numpy import hstack, empty, copy, ravel

from .norm import norm


@norm
def second_difference(vol_data):
    norm_ts1 = empty(0)

    t_ts = ravel(vol_data)
    t_ts = t_ts[1:] - t_ts[:-1]
    t_ts = t_ts[1:] - t_ts[:-1]
    norm_ts1 = hstack((norm_ts1, copy(t_ts)))

    return norm_ts1
