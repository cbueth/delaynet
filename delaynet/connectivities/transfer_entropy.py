"""Transfer Entropy (TE) connectivity metric."""

import numpy as np

from .connectivity import connectivity


@connectivity(mcb_kwargs={"n_bins": 2, "alphabet": "ordinal", "strategy": "quantile"})
def transfer_entropy(ts1, ts2):
    r"""
    Transfer Entropy (TE) connectivity metric.

    Transfer Entropy for Discrete variables,
    from the source to the destination time-series.

    This function computes Transfer Entropy based on the counts obtained from
    :func:`count_tuples` function. The formula for Transfer Entropy used is:

    .. math::

       TE = \sum p(s_{t-l}, d_{t}, d_{t-k}) \log_2 \left(
       \frac{p(d_{t} | s_{t-l}, d_{t-k})}{p(d_{t} | d_{t-k})}
       \right)

    Where:

    - :math:`TE`: Transfer Entropy
    - :math:`s_{t-l}`: Source state at time :math:`t-l`
    - :math:`d_{t}`: Destination state at time :math:`t`
    - :math:`d_{t-k}`: Past state of the destination at time :math:`t-k`
    - :math:`p(s_{t-l}, d_{t}, d_{t-k})`: Joint probability of
                                          :math:`s_{t-l`, :math:`d_{t}`, :math:`d_{t-k}`
    - :math:`p(d_{t} | s_{t-l}, d_{t-k})`: Conditional probability of :math:`d_{t}`
                                           given :math:`s_{t-l`, :math:`d_{t-k}`
    - :math:`p(d_{t} | d_{t-k})`: Conditional probability of :math:`d_{t}`
                                  given :math:`d_{t-k}`

    :param ts1: First time series.
    :type ts1: ndarray
    :param ts2: Second time series.
    :type ts2: ndarray
    :return: Transfer Entropy value.
    :rtype: float
    """
    # ts1 = np.array( ts1 > np.median( ts1 ), dtype = int )
    # ts2 = np.array( ts2 > np.median( ts2 ), dtype = int )

    # te_effective_1_2 = np.zeros((6))  # TODO: implement max_lag_steps=5 ?

    te_o_1_2 = compute_transfer_entropy(ts1, ts2, 2, 1, 0)
    te_p_1_2 = compute_transfer_entropy(np.random.permutation(ts1), ts2, 2, 1, 0)

    te_o_2_1 = compute_transfer_entropy(ts2, ts1, 2, 1, 0)
    te_p_2_1 = compute_transfer_entropy(np.random.permutation(ts2), ts1, 2, 1, 0)
    return -np.max(te_o_1_2 - te_p_1_2 - te_o_2_1 + te_p_2_1)


def compute_transfer_entropy(  # pylint: disable=too-many-locals
    source, dest, k, l, delay
):
    r"""
    Compute Transfer Entropy from source to destination.

    Transfer Entropy formula used:
    .. math::

        TE = \sum p(s_{t-l}, d_{t}, d_{t-k}) \log \left(
        \frac{p(d_{t} | s_{t-l}, d_{t-k})}{p(d_{t} | d_{t-k})}
        \right)

    :param source: Source time-series data.
    :type source: ndarray
    :param dest: Destination time-series data.
    :type dest: ndarray
    :param k: Embedding length for the destination variable.
    :type k: int
    :param l: Embedding length for the source variable.
    :type l: int
    :param delay: Time delay between source and destination.
    :type delay: int
    :return: Transfer Entropy from source to destination.
    :rtype: float
    """
    (
        source_next_past_count,
        source_past_count,
        next_past_count,
        past_count,
        observations,
    ) = count_tuples(source, dest, k, l, delay)

    te = 0
    for (s_t, d_t, d_t_k), p_s_t_d_t_d_t_k in source_next_past_count.items():
        p_s_t_d_t_d_t_k /= observations

        p_s_t_d_t_k = source_past_count[s_t, d_t_k] / observations
        p_d_t_d_t_k = next_past_count[d_t, d_t_k] / past_count[d_t_k]
        p_d_t_k = past_count[d_t_k] / observations

        log_term = (p_d_t_d_t_k / p_d_t_k) / (p_s_t_d_t_k / p_d_t_k)
        local_value = np.log(log_term)

        te += p_s_t_d_t_d_t_k * local_value

    # Convert to base 2 logarithm
    te /= np.log(2)

    return te


def count_tuples(source, dest, k, l, delay):  # pylint: disable=too-many-locals
    """
    Count tuples for Transfer Entropy computation.

    :param source: Source time-series data.
    :type source: ndarray
    :param dest: Destination time-series data.
    :type dest: ndarray
    :param k: Embedding length for the destination variable.
    :type k: int
    :param l: Embedding length for the source variable.
    :type l: int
    :param delay: Time delay between source and destination.
    :type delay: int
    :return: Several counts for Transfer Entropy computation.
             - source_next_past_count: Count for source, next state of destination,
                                       and past state of destination.
             - source_past_count: Count for source and past state of destination.
             - next_past_count: Count for next state and past state of destination.
             - past_count: Count for past state of destination.
    :rtype: tuple[dict, dict, dict, dict, int]
    """
    source_next_past_count = {}
    source_past_count = {}
    next_past_count = {}
    past_count = {}
    observations = 0

    # Initialize past states

    for t in range(max(k, l + delay), len(dest)):
        # Next state for the destination variable
        next_state_dest = dest[t]

        # Update past states
        past_state_dest = dest[t - k : t]
        past_state_source = source[t - delay - l + 1 : t - delay + 1]

        # Convert arrays to tuple to use as dictionary keys
        past_state_dest_t = tuple(past_state_dest)
        past_state_source_t = tuple(past_state_source)

        # Update counts
        if (
            past_state_source_t,
            next_state_dest,
            past_state_dest_t,
        ) in source_next_past_count:
            source_next_past_count[
                past_state_source_t, next_state_dest, past_state_dest_t
            ] += 1
        else:
            source_next_past_count[
                past_state_source_t, next_state_dest, past_state_dest_t
            ] = 1

        if (past_state_source_t, past_state_dest_t) in source_past_count:
            source_past_count[past_state_source_t, past_state_dest_t] += 1
        else:
            source_past_count[past_state_source_t, past_state_dest_t] = 1

        if (next_state_dest, past_state_dest_t) in next_past_count:
            next_past_count[next_state_dest, past_state_dest_t] += 1
        else:
            next_past_count[next_state_dest, past_state_dest_t] = 1

        if past_state_dest_t in past_count:
            past_count[past_state_dest_t] += 1
        else:
            past_count[past_state_dest_t] = 1

        observations += 1

    return (
        source_next_past_count,
        source_past_count,
        next_past_count,
        past_count,
        observations,
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
