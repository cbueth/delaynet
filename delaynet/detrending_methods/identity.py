"""Identity."""

import logging

from ..decorators import detrending_method


@detrending_method
def identity(ts):
    """Identity 'detrending' - no detrending.

    :param ts: Time series to detrend.
    :type ts: numpy.ndarray
    :return: 'detrended' time series.
    :rtype: numpy.ndarray
    """
    logging.warning(
        "Identity function is not detrending. "
        "Only use for testing or if data is already detrended."
    )
    return ts
