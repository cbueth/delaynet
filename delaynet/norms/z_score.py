"""Z-Score (ZS) normalization."""

from numpy import copy, mean as npmean, mod, size, std, ndarray, arange, integer
from .norm import norm
from ..utils.logging import logger


@norm
def z_score(ts: ndarray, periodicity: int) -> ndarray:
    """Z-Score (ZS) normalization.

    :param ts: Time series to normalize.
    :type ts: ndarray
    :param periodicity: Periodicity of the time series - reoccurrence of the same
                        pattern.
    :type periodicity: int > 0
    :return: Normalized time series.
    :rtype: ndarray
    :raises ValueError: If the periodicity is not a positive integer.
    """
    if not isinstance(periodicity, (int, integer)) or periodicity <= 0:
        raise ValueError(f"periodicity must be a positive integer, not {periodicity}.")
    # Warn if periodicity >= len(ts)
    if periodicity >= ts.size:
        logger.warning(
            f"For periodicity ({periodicity}) >= len(ts) ({ts.size}), "
            f"Z-Score normalization is equivalent to Identity function."
        )

    ts2 = copy(ts)
    for k in range(size(ts)):
        in_offset = mod(k, periodicity)
        sub_ts = ts[in_offset::periodicity]
        st_dev = std(sub_ts)
        if st_dev > 0:
            ts2[k] = (ts[k] - npmean(sub_ts)) / st_dev
    return ts2


@norm
def z_score_vectorized(ts: ndarray, periodicity: int) -> ndarray:
    """Z-Score (ZS) normalization.

    This version is optimized to avoid the for loop in the original version.
    Using reshape and mean/std on the reshaped array, we can compute the mean and
    standard deviation for each periodicity at once.

    :param ts: Time series to normalize.
    :type ts: ndarray
    :param periodicity: Periodicity of the time series - reoccurrence of the same
                        pattern.
    :type periodicity: int
    :return: Normalized time series.
    :rtype: ndarray
    """
    # Create an array of indices for each periodicity
    indices = arange(len(ts)).reshape(-1, periodicity)

    # Use these indices to create a 2D array
    ts2 = ts[indices]

    # Compute the mean and standard deviation for each periodicity
    mean = npmean(ts2, axis=0)
    std_dev = std(ts2, axis=0)

    # Normalize
    ts2 = (ts2 - mean) / std_dev

    # Flatten the 2D array back into a 1D array
    return ts2.ravel()
