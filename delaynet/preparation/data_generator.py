"""Generate example data for delaynet.

This module provides functions to generate synthetic data for testing and
demonstrating the delaynet package. It includes:

1. Delayed causal network generation (gen_delayed_causal_network)
2. fMRI-like time series generation (gen_fmri, gen_fmri_multiple)
3. SynthATDelays-based transportation delay generation:
   - Random connectivity scenario (gen_synthatdelays_random_connectivity)
   - Independent operations with trends scenario (gen_synthatdelays_independent_operations_with_trends)

The SynthATDelays generators create realistic transportation delay data based on
simulated airport and flight operations, allowing for testing of delay propagation
detection methods in controlled scenarios. These functions return a Results_Class
object containing delay data matrices and statistics.
"""

import numpy as np
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

from synthatdelays import (
    AnalyseResults,
    ExecSimulation,
    Results_Class,
    Scenario_RandomConnectivity,
    Scenario_IndependentOperationsWithTrends,
)


def gen_synthatdelays_random_connectivity(
    sim_time: int,
    num_airports: int,
    num_aircraft: int,
    buffer_time: float,
    seed: int = 0,
) -> Results_Class:
    """Generate delay data using SynthATDelays Random Connectivity scenario.

    This scenario is composed of a set of airports, randomly connected by a set of
    independent flights, and with random and homogeneous enroute delays.

    :param sim_time: Simulation time in days.
    :type sim_time: int
    :param num_airports: Number of simulated airports.
    :type num_airports: int
    :param num_aircraft: Number of simulated aircraft.
    :type num_aircraft: int
    :param buffer_time: Buffer time between subsequent operations, in hours.
    :type buffer_time: float
    :param seed: Seed for random number generation.
    :type seed: int
    :return: Results object containing delay data and statistics.
    :rtype: synthatdelays.Classes.Results_Class

    The returned Results_Class object contains several attributes:

    - avgArrivalDelay: Average arrival delay per airport and time window.
      Shape: (num_windows, num_airports)
    - avgDepartureDelay: Average departure delay per airport and time window.
      Shape: (num_windows, num_airports)
    - numArrivalFlights: Number of arrival flights per airport and time window.
    - numDepartureFlights: Number of departure flights per airport and time window.
    - totalArrivalDelay: Total arrival delay across all flights.
    - totalDepartureDelay: Total departure delay across all flights.

    Example usage:

    >>> import numpy as np
    >>> from delaynet.preparation.data_generator import gen_synthatdelays_random_connectivity
    >>>
    >>> # Generate delay data for 5 airports over 10 days
    >>> results = gen_synthatdelays_random_connectivity(
    ...     sim_time=10,
    ...     num_airports=5,
    ...     num_aircraft=10,
    ...     buffer_time=0.8
    ... )
    >>>
    >>> # Access the average arrival delay matrix
    >>> arrival_delays = results.avgArrivalDelay
    >>> arrival_delays.shape
    (240, 5)  # 240 time windows (24 hours * 10 days), 5 airports
    """
    # Validate inputs
    if not isinstance(sim_time, (int, np.integer)) or sim_time < 1:
        raise ValueError(f"sim_time must be a positive integer, but is {sim_time}.")

    try:
        # Create the scenario with the specified parameters
        options = Scenario_RandomConnectivity(
            numAirports=num_airports,
            numAircraft=num_aircraft,
            bufferTime=buffer_time,
            seed=seed,
        )

        # Override simulation time with the requested value
        options.simTime = sim_time

        # Run the simulation
        executed_flights = ExecSimulation(options)
        all_results = AnalyseResults(executed_flights, options)

        return all_results
    except Exception as e:
        raise RuntimeError(f"Error in SynthATDelays simulation: {str(e)}")


def gen_synthatdelays_independent_operations_with_trends(
    sim_time: int,
    activate_trend: bool,
    seed: int = 0,
) -> Results_Class:
    """Generate delay data using SynthATDelays Independent Operations with Trends scenario.

    This scenario is composed of two groups of two airports. Flights connect airports within
    the same group, but not across groups; hence, no propagations can happen between groups.
    When trends are activated, delays are added at specific hours, generating spurious
    causality relations.

    :param sim_time: Simulation time in days.
    :type sim_time: int
    :param activate_trend: If true, delays are added at 12:00 and 14:00, generating spurious causalities.
    :type activate_trend: bool
    :param seed: Seed for random number generation.
    :type seed: int
    :param _testing: Internal parameter for testing, do not use.
    :type _testing: bool
    :return: Results object containing delay data and statistics.
    :rtype: synthatdelays.Classes.Results_Class

    The returned Results_Class object contains several attributes:

    - avgArrivalDelay: Average arrival delay per airport and time window.
      Shape: (num_windows, num_airports)
    - avgDepartureDelay: Average departure delay per airport and time window.
      Shape: (num_windows, num_airports)
    - numArrivalFlights: Number of arrival flights per airport and time window.
    - numDepartureFlights: Number of departure flights per airport and time window.
    - totalArrivalDelay: Total arrival delay across all flights.
    - totalDepartureDelay: Total departure delay across all flights.

    Example usage:

    >>> from delaynet.preparation.data_generator import gen_synthatdelays_independent_operations_with_trends
    >>>
    >>> # Generate delay data with trends activated
    >>> results = gen_synthatdelays_independent_operations_with_trends(
    ...     sim_time=10,
    ...     activate_trend=True
    ... )
    >>>
    >>> # Access the average departure delay matrix
    >>> departure_delays = results.avgDepartureDelay
    >>> departure_delays.shape
    (240, 4)  # 240 time windows (24 hours * 10 days), 4 airports
    """
    # Validate inputs
    if not isinstance(sim_time, (int, np.integer)) or sim_time < 1:
        raise ValueError(f"sim_time must be a positive integer, but is {sim_time}.")
    if not isinstance(activate_trend, bool):
        raise ValueError(f"activate_trend must be a boolean, but is {activate_trend}.")

    try:
        # Create the scenario with the specified parameters
        options = Scenario_IndependentOperationsWithTrends(
            activateTrend=activate_trend, seed=seed
        )

        # Override simulation time with the requested value
        options.simTime = sim_time

        # Run the simulation
        executed_flights = ExecSimulation(options)
        all_results = AnalyseResults(executed_flights, options)

        return all_results
    except Exception as e:
        raise RuntimeError(f"Error in SynthATDelays simulation: {str(e)}")


def extract_airport_delay_time_series(
    results: Results_Class,
    delay_type: str = "arrival",
) -> np.ndarray:
    """Extract airport delay time series from SynthATDelays results.

    :param results: Results object from SynthATDelays simulation.
    :type results: synthatdelays.Classes.Results_Class
    :param delay_type: Type of delay to extract ('arrival' or 'departure').
    :type delay_type: str
    :return: Delay time series for each airport.
    :rtype: numpy.ndarray, shape = (num_time_windows, num_airports)
    :raises ValueError: If delay_type is not 'arrival' or 'departure'.
    :raises ImportError: If synthatdelays is not installed.

    Example usage:

    >>> results = gen_synthatdelays_random_connectivity(10, 5, 10, 0.8)
    >>> arrival_delays = extract_airport_delay_time_series(results, "arrival")
    >>> arrival_delays.shape
    (240, 5)  # 240 time windows (24 hours * 10 days), 5 airports
    """

    if delay_type.lower() == "arrival":
        return results.avgArrivalDelay
    elif delay_type.lower() == "departure":
        return results.avgDepartureDelay
    else:
        raise ValueError(
            f"Unknown delay type: {delay_type}. Use 'arrival' or 'departure'."
        )


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
):  # pragma: no cover
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
):  # pragma: no cover
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
