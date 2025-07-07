"""Mutual information (MI) connectivity metric."""

from PIL.ImageChops import offset
from infomeasure import estimator

from ..decorators import connectivity
from ..utils.lag_steps import find_optimal_lag


@connectivity(
    # mcb_kwargs={"n_bins": 3, "alphabet": "ordinal", "strategy": "quantile"},
)
def mutual_information(
    ts1,
    ts2,
    approach: str = "",
    lag_steps: int | list = None,
    hypothesis_type: str = "permutation_test",
    n_tests: int = 20,
    mi_kwargs=None,
):
    r"""Mutual Information (MI) connectivity metric.


    :param ts1: First time series.
    :type ts1: numpy.ndarray
    :param ts2: Second time series.
    :type ts2: numpy.ndarray
    :param approach: Approach to use. See :func:`infomeasure.mutual_information` for
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
    :param mi_kwargs: Additional keyword arguments for the mutual information estimator.
    :type mi_kwargs: dict
    :return: Best *p*-value and corresponding lag.
    :rtype: tuple[float, int]
    :raises ValueError: If ``approach`` is not given.
    """

    if approach == "":
        from infomeasure import mutual_information as im_mi

        raise ValueError(
            "The approach parameter must be given. "
            "See `infomeasure.mutual_information` for available approaches. \n"
            f"help(infomeasure.mutual_information):\n{im_mi.__doc__}"
        )

    if mi_kwargs is None:
        mi_kwargs = {}

    def mi_p_value(x, y, lag, **kwargs):
        est = estimator(
            x,
            y,
            measure="mutual_information",
            approach=approach,
            prop_time=lag,
            **kwargs,
        )
        test = est.statistical_test(method=hypothesis_type, n_tests=n_tests)
        return test.p_value

    return find_optimal_lag(mi_p_value, ts1, ts2, lag_steps, **mi_kwargs)
