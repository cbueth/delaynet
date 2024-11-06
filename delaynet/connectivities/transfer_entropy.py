"""Transfer Entropy (TE) connectivity metric."""

from infomeasure import estimator

from ..decorators import connectivity


@connectivity(
    # check_symbolic=True,
    entropy_like=True,
    # mcb_kwargs={"n_bins": 2, "alphabet": "ordinal", "strategy": "quantile"},
)
def transfer_entropy(ts1, ts2, approach: str = "", te_kwargs=None):
    r"""
    Transfer Entropy (TE) connectivity metric.
    :param ts1: First time series.
    :type ts1: numpy.ndarray
    :param ts2: Second time series.
    :type ts2: numpy.ndarray
    :param approach: Approach to use. See :func:`infomeasure.transfer_entropy` for
                     available approaches.
    :type approach: str
    :param te_kwargs: Additional keyword arguments for the transfer entropy estimator.
    :type te_kwargs: dict
    :return: Transfer Entropy value.
    :rtype: float
    """

    if approach == "":
        from infomeasure import (  # pylint: disable=import-outside-toplevel
            transfer_entropy as im_te,
        )

        raise ValueError(
            "The approach parameter must be given. "
            "See `infomeasure.transfer_entropy` for available approaches. \n"
            f"help(infomeasure.transfer_entropy):\n{im_te.__doc__}"
        )
    # TODO: implement max_lag_steps=5 ?

    if te_kwargs is None:
        te_kwargs = {}

    return -(
        estimator(
            source=ts1,
            dest=ts2,
            measure="transfer_entropy",
            approach=approach,
            **te_kwargs,
        ).effective_val()
        - estimator(
            source=ts2,
            dest=ts1,
            measure="transfer_entropy",
            approach=approach,
            **te_kwargs,
        ).effective_val()
    )


# TODO: Transfer tests
# Test the function
# source = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
# dest = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])

# k = 2  # Embedding length for destination
# l = 2  # Embedding length for source
# delay = 1  # Time delay between source and destination

# te_value = compute_transfer_entropy(source, dest, k, l, delay)
# te_value
