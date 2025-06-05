"""Second difference (2diff) norm."""

from numpy import ravel

from ..decorators import norm


@norm(check_shape=False)
def second_difference(ts):
    """Second difference (2diff) norm.

    :param ts: Time series to normalise.
    :type ts: numpy.ndarray
    :return: Normalised time series (length is reduced by 2).
    :rtype: numpy.ndarray
    """
    t_ts = ravel(
        ts
    )  # TODO: norms allow 1D and 2D arrays. This only works for 1D, add 2D compatability
    t_ts = t_ts[1:] - t_ts[:-1]
    t_ts = t_ts[1:] - t_ts[:-1]
    return t_ts
