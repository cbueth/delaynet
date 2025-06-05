"""Identity / Normalisation."""

import logging

from ..decorators import norm


@norm
def identity(ts):
    """Identity 'normalisation'.

    :param ts: Time series to normalise.
    :type ts: numpy.ndarray
    :return: 'Normalised' time series.
    :rtype: numpy.ndarray
    """
    logging.warning(
        "Identity norm is not a normalisation. "
        "Only use for testing or if data is already normalised."
    )
    return ts
