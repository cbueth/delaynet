"""Mutual information (MI) connectivity metric."""

from infomeasure import mutual_information as im_mi

from ..decorators import connectivity


@connectivity(
    # check_symbolic=True,
    entropy_like=True,
    # mcb_kwargs={"n_bins": 3, "alphabet": "ordinal", "strategy": "quantile"},
)
def mutual_information(
    ts1, ts2, approach: str = "", max_lag_steps: int = 5, mi_kwargs=None
):
    r"""Mutual Information (MI) connectivity metric.


    :param ts1: First time series.
    :type ts1: numpy.ndarray
    :param ts2: Second time series.
    :type ts2: numpy.ndarray
    :param approach: Approach to use. See :func:`infomeasure.mutual_information` for
                     available approaches.
    :type approach: str
    :param max_lag_steps: Maximum time lag to consider.
    :type max_lag_steps: int
    :param mi_kwargs: Additional keyword arguments for the mutual information estimator.
    :type mi_kwargs: dict
    :return: Mutual information value and time lag.
    :rtype: tuple[float, int]

    :raises ValueError: If `approach` is not given.
    """

    if approach == "":
        raise ValueError(
            "The approach parameter must be given. "
            "See `infomeasure.mutual_information` for available approaches. \n"
            f"help(infomeasure.mutual_information):\n{im_mi.__doc__}"
        )

    if mi_kwargs is None:
        mi_kwargs = {}
    mi_values = []
    for t in range(0, max_lag_steps + 1):
        mi = im_mi(ts1, ts2, approach=approach, offset=t, **mi_kwargs)
        if isinstance(mi, tuple):
            mi_values.append(mi[0])
        else:
            mi_values.append(mi)

    # TODO: or do we need the p-Value?
    # change to min p-value
    idx_max = max(range(len(mi_values)), key=mi_values.__getitem__)
    return -mi_values[idx_max], idx_max


# TODO: transfer tests
# Test the function
# var1 = array([0, 1, 0, 1, 1])
# var2 = array([1, 0, 1, 1, 0])
# print("Average MI:", compute_average_mi(var1, var2, 2, 2))
