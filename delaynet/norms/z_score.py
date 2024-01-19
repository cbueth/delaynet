"""Z-Score (ZS) normalization."""

from numpy import copy, empty, hstack, mean, mod, ravel, size, std
from .norm import norm


@norm
def z_score(vol_data, airport, phase=0):
    norm_ts1 = empty((0))

    for year in range(5):
        for month in range(12):
            num_days = 31
            if (month + 1) in [4, 6, 9, 11]:
                num_days = 30
            if (month + 1) == 2:
                num_days = 28

            t_ts = ravel(vol_data[year, month, :num_days, airport, :, phase])
            t_ts2 = copy(t_ts)
            for k in range(size(t_ts)):
                in_offset = mod(k, 7 * 24)
                sub_ts = t_ts[in_offset :: 7 * 24]
                st_dev = std(sub_ts)

                if st_dev > 0:
                    t_ts2[k] = (t_ts[k] - mean(sub_ts)) / st_dev
            norm_ts1 = hstack((norm_ts1, copy(t_ts2)))

    return norm_ts1
