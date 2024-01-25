"""Generate example data for DelayNet."""

from numpy import random, zeros, ndarray, integer, floating


def gen_rand_data(
    n_nodes: int,
    ts_len: int,
    l_dens: float,
    wm_min_max: tuple[float, float] = (0.5, 1.5),
    rng: random.Generator | int | None = None,
) -> tuple[ndarray[bool], ndarray[float], ndarray[float]]:
    """
    Generate random data for DelayNet.


    :param n_nodes: Number of nodes.
    :type n_nodes: int
    :param ts_len: Length of time series.
    :type ts_len: int
    :param l_dens: Density of the adjacency matrix.
    :type l_dens: float
    :param wm_min_max: Minimum and maximum of the weight matrix.
    :type wm_min_max: tuple[float, float]
    :param rng: Random number generator, seed or None. When `None`, the default
                numpy random number generator is used.
    :return: Adjacency matrix, weight matrix, time series.
    :rtype: tuple[ndarray[bool], ndarray[float], ndarray[float]]

    :raises ValueError: When `n_nodes` is not a positive integer.
    :raises ValueError: When `ts_len` is not a positive integer.
    :raises ValueError: When `l_dens` is not in [0, 1].
    :raises ValueError: When `wm_min_max` is not a tuple of floats with length 2 and
                        `wm_min_max[0] <= wm_min_max[1]`.
    :raises TypeError: When `rng` is not `None`, an `int` or a `numpy.random.Generator`.
    """
    # Set random number generator
    if rng is None:
        rng = random.default_rng()
    elif isinstance(rng, (int, integer)):
        rng = random.default_rng(rng)
    elif not isinstance(rng, random.Generator):
        raise TypeError(
            "rng must be None, an int or a numpy.random.Generator, "
            f"but is {type(rng)}."
        )

    # Check input
    if not isinstance(n_nodes, (int, integer)) or n_nodes < 1:
        raise ValueError(f"n_nodes must be a positive integer, but is {n_nodes}.")
    if not isinstance(ts_len, (int, integer)) or ts_len < 1:
        raise ValueError(f"ts_len must be a positive integer, but is {ts_len}.")
    if not isinstance(l_dens, (float, floating)) or not 0.0 <= l_dens <= 1.0:
        raise ValueError(f"l_dens must be a float in [0, 1], but is {l_dens}.")
    if not (
        isinstance(wm_min_max, tuple)
        and len(wm_min_max) == 2
        and isinstance(wm_min_max[0], (float, floating))
        and isinstance(wm_min_max[1], (float, floating))
        and wm_min_max[0] <= wm_min_max[1]
    ):
        raise ValueError(
            f"wm_min_max must be a tuple of floats with length 2 and "
            f"wm_min_max[0] <= wm_min_max[1], but is {wm_min_max}."
        )

    # Generate adjacency matrix
    am = rng.uniform(0.0, 1.0, (n_nodes, n_nodes)) < l_dens
    am[range(n_nodes), range(n_nodes)] = False  # no self-loops

    # Generate weight matrix
    wm = rng.uniform(*wm_min_max, (n_nodes, n_nodes))
    wm *= am  # set weights of non-edges to 0

    # Generate lag matrix
    lag = random.randint(1, 5, (n_nodes, n_nodes))

    # Generate time series
    all_ts = zeros((n_nodes, ts_len))
    for a1 in range(n_nodes):
        for a2 in range(n_nodes):
            if am[a1, a2]:
                for t in range(ts_len):
                    if rng.uniform(0.0, 1.0) > 0.2:
                        continue

                    v = rng.exponential(1.0) * wm[a1, a2] + 1
                    all_ts[a1, t] += v

                    if t + lag[a1, a2] < ts_len:
                        all_ts[a2, t + lag[a1, a2]] += v

    return am, wm, all_ts
