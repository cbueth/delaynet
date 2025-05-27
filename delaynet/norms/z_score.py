"""Z-Score (ZS) normalization."""

from numba import prange, njit, int64
from numpy import mean as npmean, mod, std, ndarray, arange, integer, copy, zeros

from ..decorators import norm

from ..utils.logging import logging


@norm
def z_score(ts: ndarray, periodicity: int = 1, max_periods: int = -1) -> ndarray:
    r"""Z-Score (ZS) normalization.

    The Z-Score (ZS) normalization is a standard score that measures the number of
    standard deviations a data point is from the mean. It is calculated as:

    .. math::
        Z = \frac{X - \mu}{\sigma}

    where :math:`X` is the data point, :math:`\mu` is the mean, and :math:`\sigma` is
    the standard deviation.

    The Z-Score normalization is applied to a time series by computing the mean and
    standard deviation of a sub time series around each data point.
    The sub time series is defined by the periodicity and the maximum number of periods
    to consider before and after the current value.
    Mean and standard deviation are computed for each sub time series and used to
    normalize the data point.
    The current value is removed from the sub time series to avoid bias.

    For a valid Z-Score, :math:`2\times\texttt{periodicity}+1 \leq \texttt{len(ts)}`
    needs to be satisfied.
    Also, :math:`\texttt{max\_periods} \times \texttt{periodicity} \geq\texttt{len(ts)}`
    results in including all periods.

    :param ts: Time series to normalize.
    :type ts: numpy.ndarray
    :param periodicity: Periodicity of the time series - reoccurrence of the same
                        pattern.
    :type periodicity: int > 0
    :param max_periods: Maximum number of periods to consider before and after the
                        current value. If -1, all periods are considered.
    :type max_periods: int >= -1
    :return: Normalized time series.
    :rtype: numpy.ndarray
    :raises ValueError: If the ``periodicity`` is not a positive integer.
    :raises ValueError: If the ``max_periods`` is not a positive integer or -1.
    :raises ValueError: If :math:`2\times\texttt{periodicity}+1 \leq \texttt{len(ts)}`
                        is not satisfied.
    """
    if not isinstance(periodicity, (int, integer)) or periodicity <= 0:
        raise ValueError(f"periodicity must be a positive integer, not {periodicity}.")
    if not isinstance(max_periods, (int, integer)) or max_periods < -1:
        raise ValueError(
            f"`max_periods` must be a positive integer or -1, not {max_periods}."
        )
    if 2 * periodicity + 1 > ts.size:
        raise ValueError(
            f"Periodicity ({periodicity}) is too large for the time series of length "
            f"{ts.size}. Need 2*periodicity+1 <= len(ts)."
        )

    # If max_periods * periodicity + 1 >= len(ts), then max_periods = -1
    if max_periods * periodicity + 1 >= ts.size:
        logging.debug(
            "max_periods (%s) is larger than or equal to the "
            "available periods (%s). Setting max_periods to -1 (all periods).",
            max_periods,
            ts.size / periodicity,
        )
        max_periods = -1

    if max_periods == -1 and periodicity == 1:
        # Simple case, no slicing needed
        ts_std = std(ts)
        if ts_std == 0:
            logging.warning(
                "Standard deviation of the whole time series is 0, "
                "returning a zero array of the same size."
            )
            return zeros(ts.size)
        return (ts - npmean(ts)) / ts_std

    ts2 = (
        _z_score_loop_all(periodicity, ts)
        if max_periods == -1
        else _z_score_loop_partial(periodicity, max_periods, ts)
    )

    return ts2


@njit(parallel=True)
def _z_score_loop_all(periodicity, ts):  # pragma: no cover
    ts2 = copy(ts)
    for k in prange(ts.size):
        sub_ts = _get_sub_ts_all_periods(ts, k, periodicity)
        st_dev = std(sub_ts)
        if st_dev > 0:
            ts2[k] = (ts[k] - npmean(sub_ts)) / st_dev
    return ts2


@njit(parallel=True)
def _z_score_loop_partial(periodicity, max_periods, ts):  # pragma: no cover
    ts2 = copy(ts)
    for k in prange(ts.size):
        sub_ts = _get_sub_ts_partial(ts, k, periodicity, max_periods)
        st_dev = std(sub_ts)
        if st_dev > 0:
            ts2[k] = (ts[k] - npmean(sub_ts)) / st_dev
    return ts2


@njit
def _get_sub_ts_all_periods(ts, k, periodicity):  # pragma: no cover
    """Get the sub time series for all periods."""
    in_offset = mod(k, periodicity)
    sub_slice = slice(in_offset, None, periodicity)
    sub_ts = ts[sub_slice]
    # Remove the current index
    mask = arange(len(ts)) != k
    return sub_ts[mask[sub_slice]]


@njit
def _get_sub_ts_partial(ts, k, periodicity, max_periods):  # pragma: no cover
    """Get the sub time series for a limited number of periods."""
    remainder = mod(k, periodicity)
    start_index = max(remainder, k - (max_periods * periodicity))
    end_index = min(len(ts) - remainder, k + ((max_periods + 1) * periodicity))
    indices = arange(start_index, end_index, periodicity, dtype=int64)
    # Remove the current index
    indices = indices[indices != k]
    return ts[indices]
