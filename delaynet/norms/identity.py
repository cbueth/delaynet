"""Identity / Normalization."""

import logging

from ..decorators import norm


@norm
def identity(ts):
    """Identity 'normalization'.

    :param ts: Time series to normalize.
    :type ts: numpy.ndarray
    :return: 'Normalized' time series.
    :rtype: numpy.ndarray
    """
    logging.warning(
        "Identity norm is not a normalization. "
        "Only use for testing or if data is already normalized."
    )
    return ts
