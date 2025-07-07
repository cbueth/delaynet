"""Transfer Entropy (TE) connectivity metric."""

from infomeasure import estimator

from ..decorators import connectivity
from ..utils.lag_steps import find_optimal_lag


@connectivity(
    # mcb_kwargs={"n_bins": 2, "alphabet": "ordinal", "strategy": "quantile"},
)
def transfer_entropy(
    ts1,
    ts2,
    approach: str = "",
    lag_steps: int | list = None,
    hypothesis_type: str = "permutation_test",
    n_tests: int = 20,
    te_kwargs=None,
):
    r"""
    Transfer Entropy (TE) connectivity metric.

    :param ts1: First time series.
    :type ts1: numpy.ndarray
    :param ts2: Second time series.
    :type ts2: numpy.ndarray
    :param approach: Approach to use. See :func:`infomeasure.transfer_entropy` for
                     available approaches.
    :type approach: str
    :param lag_steps: Time lags to consider.
                      Can be a single integer or a list of integers.
                      An integer will consider lags [1, ..., lag_steps].
                      A list will consider the specified values as lags.
    :type lag_steps: int | list
    :param hypothesis_type: Type of hypothesis test to use.
                            Either 'permutation_test' or 'bootstrap'.
                            Default is 'permutation_test'.
    :type hypothesis_type: str
    :param n_tests: Number of iterations or resamples to perform within the hypothesis
                    test.
    :type n_tests: int
    :param te_kwargs: Additional keyword arguments for the transfer entropy estimator.
    :type te_kwargs: dict
    :return: Best *p*-value and corresponding lag.
    :rtype: tuple[float, int]
    """

    if approach == "":
        from infomeasure import (
            transfer_entropy as im_te,
        )

        raise ValueError(
            "The approach parameter must be given. "
            "See `infomeasure.transfer_entropy` for available approaches. \n"
            f"help(infomeasure.transfer_entropy):\n{im_te.__doc__}"
        )

    if te_kwargs is None:
        te_kwargs = {}

    def te_p_value(x, y, lag, **kwargs):
        est = estimator(
            x,
            y,
            measure="transfer_entropy",
            approach=approach,
            prop_time=lag,
            **kwargs,
        )
        test = est.statistical_test(method=hypothesis_type, n_tests=n_tests)
        return test.p_value

    return find_optimal_lag(te_p_value, ts1, ts2, lag_steps, **te_kwargs)
