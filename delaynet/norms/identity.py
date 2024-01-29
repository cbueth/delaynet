"""Identity / Normalization."""

from .norm import norm
from ..utils.logging import logger


@norm
def identity(ts):
    """Identity 'normalization'.

    :param ts: Time series to normalize.
    :type ts: ndarray
    :return: 'Normalized' time series.
    :rtype: ndarray
    """
    logger.warning("Identity norm is not a normalization. Only use for testing.")
    return ts
