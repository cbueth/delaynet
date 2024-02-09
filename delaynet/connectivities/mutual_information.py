"""Mutual information (MI) connectivity metric."""

from numpy import zeros, log2, ndarray

from .connectivity import connectivity


@connectivity(mcb_kwargs={"n_bins": 3, "alphabet": "ordinal", "strategy": "quantile"})
def mutual_information(
    ts1, ts2, base1: int = 3, base2: int = 3, max_lag_steps: int = 5
):
    r"""Mutual Information (MI) connectivity metric.

    1. This function incrementally counts the occurrences of each pair of states
       :math:`(i)` and :math:`(j)` in the two variables, as well as the
       occurrences of individual states in each variable.

    2. Then it computes the average mutual information (MI) using the formula:

    .. math::

       MI = \sum_{i,j} p(i, j) \cdot \log_2 \left( \frac{p(i, j)}{p(i) \cdot p(j)}
       \right)

    Where :math:`(p(i, j))` is the joint probability of :math:`(i)` and :math:`(j)`,
    :math:`(p(i))` and :math:`(p(j))` are the marginal probabilities of :math:`(i)`
    and :math:`(j)`, respectively.

    3. The time difference :math:`(\text{timeDiff})` specifies how far apart in time
       the two variables are when calculating their mutual information.

    Specifically, for each time step :math:`(t)`, the value of the first variable
    at :math:`(t - \text{timeDiff})` is paired with the value of the second variable
    at :math:`(t)`. This is useful in capturing temporal dependencies and understanding
    how past states of one variable may influence the current state of another.

    The MI equation remains the same, but with the joint and marginal probabilities
    calculated using the time-lagged pairs:

    .. math::

       MI = \sum_{i,j} p(i[t-\text{timeDiff}], j[t])
       \cdot \log_2 \left( \frac{p(i[t-\text{timeDiff}], j[t])}{p(i[t-\text{timeDiff}])
                           \cdot p(j[t])} \right)

    Here :math:`(p(i[t-\text{timeDiff}], j[t]))` is the joint probability
    of observing :math:`(i)` at time :math:`(t - \text{timeDiff})`
             and :math:`(j)` at time :math:`(t)`.


    :param ts1: First time series.
    :type ts1: ndarray
    :param ts2: Second time series.
    :type ts2: ndarray
    :param base1: Number of states for first time series.
    :type base1: int
    :param base2: Number of states for second time series.
    :type base2: int
    :param max_lag_steps: Maximum time lag to consider.
    :type max_lag_steps: int
    :return: Mutual information value and time lag.
    :rtype: tuple[float, int]
    """
    p_values = [
        compute_average_mi(ts1[:(-t)], ts2[t:], base1, base2)
        for t in range(0, max_lag_steps + 1)
    ]
    idx_max = max(range(len(p_values)), key=p_values.__getitem__)
    return -p_values[idx_max], idx_max


def compute_average_mi(  # pylint: disable=too-many-locals
    var1: ndarray, var2: ndarray, base1: int, base2: int, time_diff: int = 0
) -> float:
    """
    Compute the average mutual information between two variables.

    :param var1: Array containing the states of the first variable.
    :type var1: ndarray
    :param var2: Array containing the states of the second variable.
    :type var2: ndarray
    :param base1: Number of states for the first variable.
    :type base1: int
    :param base2: Number of states for the second variable.
    :type base2: int
    :param time_diff: Time difference between the variables.
    :type time_diff: int
    :return: Average mutual information.
    :rtype: float
    """
    i_count, j_count, joint_count, observations = count_occurrences(
        var1, var2, base1, base2, time_diff
    )

    # Compute MI
    mi = 0.0
    for i in range(base1):
        prob_i = i_count[i] / observations
        for j in range(base2):
            prob_j = j_count[j] / observations
            joint_prob = joint_count[i][j] / observations

            if joint_prob * prob_i * prob_j > 0:
                local_value = log2(joint_prob / (prob_i * prob_j))
                mi += joint_prob * local_value

    return mi


def count_occurrences(
    var1: ndarray, var2: ndarray, base1: int, base2: int, time_diff: int
) -> tuple[ndarray, ndarray, ndarray, int]:
    """
    Count occurrences of states and state pairs in two variables.

    :param var1: Array containing the states of the first variable.
    :type var1: ndarray
    :param var2: Array containing the states of the second variable.
    :type var2: ndarray
    :param base1: Number of states for the first variable.
    :type base1: int
    :param base2: Number of states for the second variable.
    :type base2: int
    :param time_diff: Time difference between the variables.
    :type time_diff: int
    :return: Counts of individual states and state pairs, and number of observations.
    :rtype: tuple[ndarray, ndarray, ndarray, int]
    """
    observations = len(var1) - time_diff
    joint_count = zeros((base1, base2), dtype=int)
    i_count = zeros(base1, dtype=int)
    j_count = zeros(base2, dtype=int)
    # Count occurrences with time difference
    for t in range(time_diff, len(var1)):
        i = var1[t - time_diff]
        j = var2[t]
        joint_count[i][j] += 1
        i_count[i] += 1
        j_count[j] += 1
    return i_count, j_count, joint_count, observations


# TODO: transfer tests
# Test the function
# var1 = array([0, 1, 0, 1, 1])
# var2 = array([1, 0, 1, 1, 0])
# print("Average MI:", compute_average_mi(var1, var2, 2, 2))
