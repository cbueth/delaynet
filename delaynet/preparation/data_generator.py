"""Generate example data for delaynet."""

from numpy import (
    zeros,
    ndarray,
    integer,
    floating,
    max as np_max,
    array as np_array,
    arange,
    convolve,
    size,
    fill_diagonal,
)
from numpy.random import default_rng, Generator
from scipy.stats import gamma
from numba import jit


def gen_data(
    generation_method: str,
    ts_len: int,
    n_nodes: int = 1,
    **kwargs,
) -> ndarray | tuple[ndarray, ...]:
    """Wrapper for the different data generation approaches.

    Supported generation methods:

    - 'dcn': Delayed causal network time series. :func:`gen_delayed_causal_network`
    - 'fmri': fMRI time series. :func:`gen_fmri`

    :param generation_method: Type of time series to generate.
    :type generation_method: str
    :param ts_len: Length of time series.
    :type ts_len: int
    :param n_nodes: Number of nodes (i.e., time series). If not specified,
                    defaults to 1.
    :type n_nodes: int
    :param kwargs: Additional arguments for the data generation functions.
    :return: Time series, possibly with additional data.
    :rtype: numpy.ndarray or tuple[ndarray, ...]
    :raises ValueError: If the generation method is unknown.
    """
    if generation_method.lower() == "dcn":
        return gen_delayed_causal_network(ts_len, n_nodes, **kwargs)
    if generation_method.lower() == "fmri":
        if n_nodes == 1:
            return gen_fmri(ts_len, **kwargs)
        return gen_fmri_multiple(ts_len, n_nodes, **kwargs)
    raise ValueError(f"Unknown generation method: {generation_method}.")


def gen_delayed_causal_network(
    ts_len: int,
    n_nodes: int,
    l_dens: float,
    wm_min_max: tuple[float, float] = (0.5, 1.5),
    rng=None,
) -> tuple[ndarray[bool], ndarray[float], ndarray[float]]:
    """
    Generate delayed causal network data for delaynet.


    :param ts_len: Length of time series.
    :type ts_len: int
    :param n_nodes: Number of nodes (i.e., time series).
    :type n_nodes: int
    :param l_dens: Density of the adjacency matrix.
    :type l_dens: float
    :param wm_min_max: Minimum and maximum of the weight matrix.
    :type wm_min_max: tuple[float, float]
    :param rng: Random number generator or seed,
                passed to :func:`numpy.random.default_rng`.
    :return: Adjacency matrix, weight matrix, time series.
    :rtype: tuple[ndarray[bool], ndarray[float], ndarray[float]]

    :raises ValueError: When ``n_nodes`` is not a positive integer.
    :raises ValueError: When ``ts_len`` is not a positive integer.
    :raises ValueError: When ``l_dens`` is not in [0, 1].
    :raises ValueError: When ``wm_min_max`` is not a tuple of floats with length 2 and
                        ``wm_min_max[0] <= wm_min_max[1]``.
    """
    rng = default_rng(rng)

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
    lag = rng.integers(1, 5, (n_nodes, n_nodes))

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


def gen_fmri(
    ts_len: int = 1000,
    downsampling_factor: int = 2,
    time_resolution: float = 0.2,
    coupling_strength: float = 2.0,
    noise_initial_sd: float = 1.0,
    noise_final_sd: float = 0.1,
    rng=None,
):
    """
    Generate fMRI time series.

    This function generates random fMRI time series.
    It is based on the studies by :cite:t:`roebroeckMappingDirectedInfluence2005`
    and :cite:t:`rajapakseLearningEffectiveBrain2007`.

    :param ts_len: Length of the time series.
    :type ts_len: int
    :param downsampling_factor: Downsampling factor.
    :type downsampling_factor: int
    :param time_resolution: Time resolution.
    :type time_resolution: float
    :param coupling_strength: Coupling strength.
    :type coupling_strength: float
    :param noise_initial_sd: Standard deviation of the noise for
                             the initial time series.
    :type noise_initial_sd: float
    :param noise_final_sd: Standard deviation of the noise for the final time series.
    :type noise_final_sd: float
    :param rng: Random number generator or seed,
                passed to :func:`numpy.random.default_rng`.
    :return: fMRI time series.
    :rtype: numpy.ndarray[float]
    :author: Massimiliano Zanin and Carson Büth
    """
    rng = default_rng(rng)

    # Generate initial time series
    coupling_matrix = np_array([[-0.9, 0], [coupling_strength, -0.9]], dtype=float)
    ts_init = __initial_ts(ts_len, noise_initial_sd, coupling_matrix, rng)

    # Generate fMRI time series
    hrf_vals = __hrf(arange(0, 30, time_resolution))
    # Convolve the initial time series with the Hemodynamic Response Function (HRF)
    ts_convolve = zeros((ts_len + size(hrf_vals, 0) - 1, 2))
    ts_convolve[:, 0] = convolve(ts_init[:, 0], hrf_vals)
    ts_convolve[:, 1] = convolve(ts_init[:, 1], hrf_vals)
    # Downsample the time series
    ts_convolve = ts_convolve[::downsampling_factor]
    # Add noise
    ts_convolve += rng.normal(0.0, noise_final_sd, (size(ts_convolve, 0), 2))

    return ts_convolve


def gen_fmri_multiple(
    ts_len: int = 1000,
    n_nodes: int = 2,
    downsampling_factor: int = 2,
    time_resolution: float = 0.2,
    coupling_strength: float = 2.0,
    noise_initial_sd: float = 1.0,
    noise_final_sd: float = 0.1,
    rng=None,
):
    """
    Generate fMRI time series for multiple nodes.

    This function works similarly to :func:`gen_fmri`,
    but generates multiple time series at once.

    :param ts_len: Length of the time series.
    :type ts_len: int
    :param n_nodes: Number of nodes (i.e., time series).
    :type n_nodes: int
    :param downsampling_factor: Downsampling factor.
    :type downsampling_factor: int
    :param time_resolution: Time resolution.
    :type time_resolution: float
    :param coupling_strength: Coupling strength.
    :type coupling_strength: float
    :param noise_initial_sd: Standard deviation of the noise for the initial time
                             series.
    :type noise_initial_sd: float
    :param noise_final_sd: Standard deviation of the noise for the final time series.
    :type noise_final_sd: float
    :param rng: Random number generator or seed,
                passed to :func:`numpy.random.default_rng`.
    :return: fMRI time series. First dimension is time, second dimension is nodes.
    :rtype: numpy.ndarray[float], shape = (num_nodes, ts_len)
    """
    rng = default_rng(rng)

    # Generate initial time series
    coupling_matrix = zeros((n_nodes, n_nodes))
    fill_diagonal(coupling_matrix, -0.9)
    coupling_matrix[0, 1:] = coupling_strength
    ts_init = __initial_ts_var_num_nodes(
        ts_len, n_nodes, noise_initial_sd, coupling_matrix, rng
    )

    # Generate fMRI time series
    hrf_at_trs = __hrf(arange(0, 30, time_resolution))
    # Convolve the initial time series with the Hemodynamic Response Function (HRF)
    ts_convolve = zeros((n_nodes, ts_len + size(hrf_at_trs, 0) - 1))
    for k in range(n_nodes):
        ts_convolve[k, :] = convolve(ts_init[:, k], hrf_at_trs)
    # ts_convolve[:, :] = convolve(ts_init, hrf_at_trs) # TODO: check if equivalent
    # Downsample the time series
    ts_convolve = ts_convolve[:, ::downsampling_factor]
    # Add noise
    ts_convolve += rng.normal(0.0, noise_final_sd, (n_nodes, size(ts_convolve, 1)))

    return ts_convolve


@jit(cache=True, nopython=True, nogil=True)
def __initial_ts(
    ts_len: int,
    noise: float,
    coupling_matrix: ndarray[float],
    rng: Generator,
):
    """
    Generate initial time series.

    Returns two normal distributed, coupled time series.

    :param ts_len: Length of time series.
    :type ts_len: int
    :param noise: Standard deviation of the noise.
    :type noise: float
    :param coupling_matrix: Coupling matrix.
    :type coupling_matrix: numpy.ndarray[float], shape = (2, 2)
    :param rng: Random number generator.
    :return: Time series.
    :rtype: numpy.ndarray[float], shape = (ts_len, 2)
    """
    ts = zeros((ts_len, 2))

    for k in range(ts_len):
        if k == 0:
            ts[k, 0] = rng.normal(0.0, noise)
            ts[k, 1] = rng.normal(0.0, noise)
            continue

        ts[k, 0] = (
            ts[k - 1, 0] * coupling_matrix[0, 0]
            + ts[k - 1, 1] * coupling_matrix[0, 1]
            + rng.normal(0.0, 1.0) * noise
        )
        ts[k, 1] = (
            ts[k - 1, 0] * coupling_matrix[1, 0]
            + ts[k - 1, 1] * coupling_matrix[1, 1]
            + rng.normal(0.0, 1.0) * noise
        )

    return ts


def __hrf(times: ndarray[float], rng: Generator = None) -> ndarray[float]:
    """
    Hemodynamic Response Function (HRF).

    :param times: Time points.
    :type times: numpy.ndarray[float]
    :return: HRF values.
    :rtype: numpy.ndarray[float]
    """
    # gamma = gamma_gen(seed=rng)
    gamma.random_state = rng
    # Generate Peak and Undershoot
    peak_values = gamma.pdf(times, 6)
    undershoot_values = gamma.pdf(times, 12)
    # Normalise
    values = peak_values - 0.35 * undershoot_values
    return values / np_max(values) * 0.6


@jit(cache=True, nopython=True, nogil=True)
def __initial_ts_var_num_nodes(
    ts_len: int,
    num_ts: int,
    noise: float,
    coupling_matrix: ndarray[float],
    rng: Generator,
):
    """
    Generate initial time series for multiple nodes.
    :param ts_len: Length of time series.
    :type ts_len: int
    :param num_ts: Number of time series.
    :type num_ts: int
    :param noise: Standard deviation of the noise.
    :type noise: float
    :param coupling_matrix: Coupling matrix.
    :type coupling_matrix: numpy.ndarray[float], shape = (num_ts, num_ts)
    :param rng: Random number generator.
    :type rng: Generator
    :return: Time series.
    :rtype: numpy.ndarray[float], shape = (ts_len, num_ts)
    """
    ts = zeros((ts_len, num_ts))

    for k in range(ts_len):
        if k == 0:
            for l in range(num_ts):
                ts[k, l] = rng.normal(0.0, noise)
            continue

        for l in range(num_ts):
            ts[k, l] = ts[k - 1, l] * coupling_matrix[l, l] + rng.normal(0.0, noise)

            for l2 in range(num_ts):
                if l == l2:
                    continue

                ts[k, l] += ts[k - 1, l2] * coupling_matrix[l2, l]

    return ts
