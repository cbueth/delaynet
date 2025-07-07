"""Gravity connectivity metric."""

from numpy import exp, array
from numpy import sum as npsum
from numpy.random import default_rng, Generator

from ..decorators import connectivity
from ..utils.lag_steps import find_optimal_lag


@connectivity
def gravity(
    ts1,
    ts2,
    lag_steps: int | list = None,
    n_tests: int = 20,
    rng: Generator | None = None,
):
    """Gravity connectivity (GC) metric.

    :param ts1: First time series.
    :type ts1: numpy.ndarray
    :param ts2: Second time series.
    :type ts2: numpy.ndarray
    :param lag_steps: Time lags to consider.
                      Can be a single integer or a list of integers.
                      An integer will consider lags [1, ..., lag_steps].
                      A list will consider the specified values as lags.
    :type lag_steps: int | list
    :param n_tests: Number of iterations or resamples to perform within the hypothesis
                    test.
    :type n_tests: int
    :param rng: Random number generator to resample from.
                If None, a default generator will be used.
    :type rng: Generator | None
    :return: Best *p*-value and corresponding lag.
    :rtype: tuple[float, int]
    """
    if not isinstance(rng, Generator):
        rng = default_rng(rng)
    return find_optimal_lag(
        gravity_single, ts1, ts2, lag_steps, n_tests=n_tests, rng=rng
    )


def gravity_single(
    ts1, ts2, lag_step, n_tests: int = None, rng: Generator = None
) -> float:
    """Helper function for gravity connectivity metric.

    This uses a permutation test to determine the *p*-value.

    :param ts1: First time series.
    :type ts1: numpy.ndarray
    :param ts2: Second time series.
    :type ts2: numpy.ndarray
    :param lag_step: Time lag to consider.
    :type lag_step: int
    :param n_tests: Number of iterations or resamples to perform within the hypothesis
                    test.
    :type n_tests: int
    :param rng: Random number generator to resample from.
    :type rng: Generator
    :return: *p*-value of gravity connectivity.
    :rtype: float
    """
    gravity = grav(ts1[: -lag_step or None], ts2[lag_step:])
    gravity_permuted = [
        grav(ts1[: -lag_step or None], rng.permutation(ts2[lag_step:]))
        for _ in range(n_tests)
    ]

    # p_val = num(perm_val > gravity) / n_tests
    return npsum(array(gravity_permuted) > gravity) / n_tests


def grav(a, b):
    q1 = npsum(exp(a))
    q2 = npsum(exp(b))
    return q1 * q2
