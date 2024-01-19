"""Mutual information (MI) causality metric."""


# # Mutual Information calculator for discrete data

# 1. Lets incrementally count the occurrences of each pair of states \(i\) and \(j\) in the two variables, as well as the occurrences of individual states in each variable.
#
# 2. Then computes the average mutual information (MI) using the formula:
#
# $$
# \text{MI} = \sum_{i,j} p(i, j) \cdot \log_2 \left( \frac{p(i, j)}{p(i) \cdot p(j)} \right)
# $$
#
# Where \( p(i, j) \) is the joint probability of \(i\) and \(j\), \( p(i) \) and \( p(j) \) are the marginal probabilities of \(i\) and \(j\), respectively. The function also computes the standard deviation of the local MI values.
#

# -  3. The time difference \( \text{timeDiff} \) specifies how far apart in time the two variables are when calculating their mutual information.
#
# Specifically, for each time step \( t \), the value of the first variable at \( t - \text{timeDiff} \) is paired with the value of the second variable at \( t \). This is useful in capturing temporal dependencies and understanding how past states of one variable may influence the current state of another.
#
# The MI equation remains the same, but with the joint and marginal probabilities calculated using the time-lagged pairs:
#
# $$
# \text{MI} = \sum_{i,j} p(i[t-\text{timeDiff}], j[t]) \cdot \log_2 \left( \frac{p(i[t-\text{timeDiff}], j[t])}{p(i[t-\text{timeDiff}]) \cdot p(j[t])} \right)
# $$
#
# Here \( p(i[t-\text{timeDiff}], j[t]) \) is the joint probability of observing \( i \) at time \( t - \text{timeDiff} \) and \( j \) at time \( t \).
#

from numpy import zeros, log2

from ..utils.multiple_coeff_binning import MultipleCoefficientBinning


def mutual_information(ts1, ts2):
    transformer = MultipleCoefficientBinning(
        n_bins=3, alphabet="ordinal", strategy="quantile"
    )
    transformer.fit(ts1.reshape(-1, 1))
    ts1 = transformer.transform(ts1.reshape(-1, 1))[:, 0]

    transformer = MultipleCoefficientBinning(
        n_bins=3, alphabet="ordinal", strategy="quantile"
    )
    transformer.fit(ts2.reshape(-1, 1))
    ts2 = transformer.transform(ts2.reshape(-1, 1))[:, 0]

    # ts1 = array( ts1 > median( ts1 ), dtype = int )
    # ts2 = array( ts2 > median( ts2 ), dtype = int )

    pValue = zeros((6))
    for t in range(0, 6):
        if t == 0:
            pValue[t] = compute_average_mi(ts1, ts2, 3, 3)
        else:
            pValue[t] = compute_average_mi(ts1[:(-t)], ts2[t:], 3, 3)

    return -max(pValue)


def compute_average_mi(var1, var2, base1, base2, timeDiff=0):
    """
    Compute the average mutual information between two variables.

    Parameters:
    - var1, var2: Arrays containing the states of the two variables.
    - base1, base2: Number of states for each variable.
    - timeDiff: Time difference between the variables.

    Returns:
    - mi: Average mutual information.
    """
    observations = len(var1) - timeDiff  # Adjust for time difference
    joint_count = zeros((base1, base2), dtype=int)
    i_count = zeros(base1, dtype=int)
    j_count = zeros(base2, dtype=int)

    # Count occurrences with time difference
    for t in range(timeDiff, len(var1)):
        i = var1[t - timeDiff]
        j = var2[t]
        joint_count[i][j] += 1
        i_count[i] += 1
        j_count[j] += 1

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


# TODO: transfer tests
# Test the function
# var1 = array([0, 1, 0, 1, 1])
# var2 = array([1, 0, 1, 1, 0])
# print("Average MI:", compute_average_mi(var1, var2, 2, 2))
