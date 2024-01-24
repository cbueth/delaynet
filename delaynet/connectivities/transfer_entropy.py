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


def transfer_entropy_pyif(ts1, ts2):
    from PyIF import te_compute as te

    p_value = te.te_compute(ts1, ts2, k=20, embedding=2, safetyCheck=False, GPU=False)

    if p_value == 0:
        p_value = 10**-5

    return 1.0 / p_value


def transfer_entropy_2(ts1, ts2):
    # from MCB import MultipleCoefficientBinning

    # transformer = MultipleCoefficientBinning(n_bins=3, alphabet = 'ordinal', strategy = 'quantile' )
    # transformer.fit(ts1.reshape(-1, 1))
    # ts1 = transformer.transform(ts1.reshape(-1, 1))[:, 0]

    # transformer = MultipleCoefficientBinning(n_bins=3, alphabet = 'ordinal', strategy = 'quantile' )
    # transformer.fit(ts2.reshape(-1, 1))
    # ts2 = transformer.transform(ts2.reshape(-1, 1))[:, 0]

    ts1 = np.array(ts1 > np.median(ts1), dtype=int)
    ts2 = np.array(ts2 > np.median(ts2), dtype=int)

    pValue = np.zeros((6))
    for t in [1]:
        pValue[t] = te(4, 4, t, ts1, ts2)
        if pValue[t] <= 10**-5:
            pValue[t] = 10**-5

    return 1.0 / np.max(pValue)


def joint_probability(k, l, h, a, b):
    """
    k B time horizon
    l A time horizon
    h instant in the future of serie B

    a, b array type"""

    # Alarm Series A (cause), B (effect), same len
    # tested
    sizeSeries = a.size
    transEntropy = 0
    numStates = 2 ** (k + l + 1)
    combinations = list(map(list, product([0, 1], repeat=k + l + 1)))
    counting = np.zeros(numStates)
    prob_cnjt = np.zeros(numStates)
    a_prob_ind = []
    b_prob_ind = []
    # joitn probability p(i_sub_t+1), i_sub_t**k, j_sub_t**l)
    inicio = np.max([k, l]) - 1
    for i in np.arange(inicio, sizeSeries - h):
        for hk in np.arange(0, k):
            b_prob_ind.append(b[i - hk])
        for hl in np.arange(0, l):
            a_prob_ind.append(a[i - hl])

        # print(a.size, b.size, a.size -1)
        ab = [b[i + h]] + b_prob_ind + a_prob_ind
        index_comb = combinations.index(ab)
        counting[index_comb] = counting[index_comb] + 1

        a_prob_ind = []
        b_prob_ind = []

    total = sum(counting)

    prob_cnjt = counting / total

    return prob_cnjt


def joint_prob_ih_ik(k, l, joint_prob_ih_ik_jl):
    states_ith_ik = list(map(list, product([0, 1], repeat=k + 1)))
    combinations = list(map(list, product([0, 1], repeat=k + l + 1)))
    p_jnt_ith_ik = np.zeros(2 ** (k + 1))

    for i, state in enumerate(states_ith_ik):
        for j, comb in enumerate(combinations):
            if comb[0 : k + 1] == state:
                p_jnt_ith_ik[i] = p_jnt_ith_ik[i] + joint_prob_ih_ik_jl[j]
    return p_jnt_ith_ik


def conditional_prob(k, l, joint_prob):
    states = list(map(list, product([0, 1], repeat=k + l)))
    combinations = list(map(list, product([0, 1], repeat=k + l + 1)))

    size = int(joint_prob.size / 2)
    conditional = np.zeros(2 ** (k + l + 1))

    for i, state in enumerate(states):
        index_zero = combinations.index([0] + state)
        prob_zero = joint_prob[index_zero]

        index_one = combinations.index([1] + state)
        prob_one = joint_prob[index_one]

        if prob_zero + prob_one != 0:
            conditional[i] = prob_zero / (prob_zero + prob_one)
            conditional[i + 2 ** (k + l)] = prob_one / (prob_zero + prob_one)
    return conditional


def conditional_div(k, l, conditional_num, conditional_den):
    combinations = list(map(list, product([0, 1], repeat=k + l + 1)))
    conditional_division = np.zeros(conditional_num.size)
    states_den = list(map(list, product([0, 1], repeat=1 + k)))
    for j, comb in enumerate(combinations):
        if conditional_den[states_den.index(comb[0 : k + 1])] != 0:
            conditional_division[j] = (
                conditional_num[j] / conditional_den[states_den.index(comb[0 : k + 1])]
            )
    return conditional_division


def te(k, l, h, ts1, ts2):
    """
    transentropy a->b
    te(k,l,h,a,b)
    k - dimension of ts2, number of samples of the past of ts2
    l - dimension of ts1, number of samples of the passt of ts1
    h -> instant in the future of b
    """
    joint_p_ih_ik_jl = joint_probability(k, l, h, ts1, ts2)

    joint_p_ih_ik = joint_prob_ih_ik(k, l, joint_p_ih_ik_jl)
    conditional_num = conditional_prob(k, l, joint_p_ih_ik_jl)
    conditional_den = conditional_prob(k, 0, joint_p_ih_ik)
    div = conditional_div(k, l, conditional_num, conditional_den)

    # log2 from the division of the conditionals -> #p(i_sub_t+h|i_sub_t**k, j_sub_t**l) /p(i_sub_t+h|i_t**k)

    log2_div_cond = np.log2(div[div != 0])
    return np.sum(joint_p_ih_ik_jl[div != 0] * log2_div_cond)
