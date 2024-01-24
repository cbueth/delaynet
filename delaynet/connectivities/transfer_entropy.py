"""Transfer Entropy (TE) connectivity metric."""
from itertools import product

# # Transfer Entropy for Discrete variables

# The Transfer Entropy from the source to the destination time-series
#
# The two Python functions, `count_tuples` and `compute_transfer_entropy`, to perform the calculations:
#
# 1. **`count_tuples`**: This function counts the occurrences of different states and transitions for the source and destination time-series. The counts are stored in dictionaries.
#
# 2. **`compute_transfer_entropy`**: This function computes Transfer Entropy based on the counts obtained from `count_tuples`. The formula for Transfer Entropy used is:
#
# $$
# TE = \sum p(s_{t-l}, d_{t}, d_{t-k}) \log_2 \left( \frac{p(d_{t} | s_{t-l}, d_{t-k})}{p(d_{t} | d_{t-k})} \right)
# $$
#
# - $TE$: Transfer Entropy
# - $\left(s_{t-l}\right)$: Source state at time $t-l$
# - $d_{t}$: Destination state at time $t$
# - $d_{t-k}$: Past state of the destination at time $t-k$
# - $p(s_{t-l}, d_{t}, d_{t-k})$: Joint probability of $s_{t-l}$, $d_{t}$, $d_{t-k}$
# - $p(d_{t} | s_{t-l}, d_{t-k})$: Conditional probability of $d_{t}$ given $s_{t-l}$, $d_{t-k}$
# - $p(d_{t} | d_{t-k})$: Conditional probability of $d_{t}$ given $d_{t-k}$
#

import numpy as np

from ..utils.multiple_coeff_binning import MultipleCoefficientBinning


def transfer_entropy(ts1, ts2):
    transformer = MultipleCoefficientBinning(
        n_bins=2, alphabet="ordinal", strategy="quantile"
    )
    transformer.fit(ts1.reshape(-1, 1))
    ts1 = transformer.transform(ts1.reshape(-1, 1))[:, 0]

    transformer = MultipleCoefficientBinning(
        n_bins=2, alphabet="ordinal", strategy="quantile"
    )
    transformer.fit(ts2.reshape(-1, 1))
    ts2 = transformer.transform(ts2.reshape(-1, 1))[:, 0]

    # ts1 = np.array( ts1 > np.median( ts1 ), dtype = int )
    # ts2 = np.array( ts2 > np.median( ts2 ), dtype = int )

    te_effective_1_2 = np.zeros((6))

    te_o_1_2 = compute_transfer_entropy(ts1, ts2, 2, 1, 0)
    te_p_1_2 = compute_transfer_entropy(np.random.permutation(ts1), ts2, 2, 1, 0)
    te_effective_1 = te_o_1_2 - te_p_1_2

    te_o_2_1 = compute_transfer_entropy(ts2, ts1, 2, 1, 0)
    te_p_2_1 = compute_transfer_entropy(np.random.permutation(ts2), ts1, 2, 1, 0)
    te_effective_2 = te_o_2_1 - te_p_2_1

    te_effective_1_2[0] = te_effective_1 - te_effective_2

    return -np.max(te_effective_1_2)


def compute_transfer_entropy(source, dest, k, l, delay):
    """
    Compute Transfer Entropy from source to destination.

    Transfer Entropy formula used:
    \[ TE = \sum p(s_{t-l}, d_{t}, d_{t-k}) \log \left( \frac{p(d_{t} | s_{t-l}, d_{t-k})}{p(d_{t} | d_{t-k})} \right) \]

    Parameters:
    - source (np.array): Source time-series data.
    - dest (np.array): Destination time-series data.
    - k (int): Embedding length for the destination variable.
    - l (int): Embedding length for the source variable.
    - delay (int): Time delay between source and destination.

    Returns:
    - te (float): Transfer Entropy from source to destination.
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


def count_tuples(source, dest, k, l, delay):
    """
    Count tuples for Transfer Entropy computation.

    Parameters:
    - source (np.array): Source time-series data.
    - dest (np.array): Destination time-series data.
    - k (int): Embedding length for the destination variable.
    - l (int): Embedding length for the source variable.
    - delay (int): Time delay between source and destination.

    Returns:
    - source_next_past_count (dict): Count for source, next state of destination, and past state of destination.
    - source_past_count (dict): Count for source and past state of destination.
    - next_past_count (dict): Count for next state and past state of destination.
    - past_count (dict): Count for past state of destination.
    """
    source_next_past_count = {}
    source_past_count = {}
    next_past_count = {}
    past_count = {}
    observations = 0

    # Initialize past states
    past_state_dest = dest[:k]
    past_state_source = source[:l]

    for t in range(max(k, l + delay), len(dest)):
        # Next state for the destination variable
        next_state_dest = dest[t]

        # State for the source variable
        state_source = source[t - delay - l + 1 : t - delay + 1]

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
