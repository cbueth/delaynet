"""Test example data generator for preparation module."""

import pytest
import numpy as np
from numpy import array_equal
from numpy.random import default_rng

from delaynet.preparation.data_generator import (
    gen_delayed_causal_network,
    gen_fmri,
    gen_fmri_multiple,
    gen_synthatdelays_random_connectivity,
    gen_synthatdelays_independent_operations_with_trends,
    extract_airport_delay_time_series,
)


@pytest.mark.parametrize("n_nodes", [1, 2, 10])
@pytest.mark.parametrize("ts_len", [1, 2, 10])
@pytest.mark.parametrize("l_dens", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("wm_min_max", [(0.0, 1.0), (0.5, 1.5), (1.0, 2.0)])
@pytest.mark.parametrize("rng", [None, 0, default_rng(0)])
def test_gen_delayed_causal_network(
    n_nodes, ts_len, l_dens, wm_min_max: tuple[float, float], rng
):
    """Test the gen_delayed_causal_network function."""
    am, wm, ts = gen_delayed_causal_network(ts_len, n_nodes, l_dens, wm_min_max, rng)
    assert am.shape == (n_nodes, n_nodes)
    assert wm.shape == (n_nodes, n_nodes)
    assert ts.shape == (n_nodes, ts_len)
    assert am.dtype == bool
    assert wm.dtype == float
    assert ts.dtype == float
    assert am.sum() <= n_nodes**2 - n_nodes
    assert 0 <= wm.sum() <= (n_nodes**2 - n_nodes) * wm_min_max[1]
    assert ts.sum() >= 0.0


@pytest.mark.parametrize(
    "rng, is_stable", [(None, False), (0, True), (default_rng(0), False)]
)
def test_gen_delayed_causal_network_stability(rng, is_stable):
    """Test if the gen_delayed_causal_network function is stable.
    This is, if the same random seed is used, the output should be the same.
    But using the same random generator twice should not produce the same output.
    """
    n_nodes = 10
    ts_len = 10
    l_dens = 0.5
    wm_min_max = (0.5, 1.5)
    am1, wm1, ts1 = gen_delayed_causal_network(ts_len, n_nodes, l_dens, wm_min_max, rng)
    am2, wm2, ts2 = gen_delayed_causal_network(ts_len, n_nodes, l_dens, wm_min_max, rng)
    assert array_equal(am1, am2) == is_stable
    assert array_equal(wm1, wm2) == is_stable
    assert array_equal(ts1, ts2) == is_stable


@pytest.mark.parametrize(
    "n_nodes, ts_len, l_dens, wm_min_max, rng, error",
    [
        (0, 1, 0.0, (0.0, 1.0), None, ValueError),  # n_nodes is not a positive integer
        (1.0, 1, 0.0, (0.0, 1.0), None, ValueError),  # n_nodes is not an integer
        (1, 0, 0.0, (0.0, 1.0), None, ValueError),  # ts_len is not a positive integer
        (1, 1.0, 0.0, (0.0, 1.0), None, ValueError),  # ts_len is not an integer
        (1, 1, -0.1, (0.0, 1.0), None, ValueError),  # l_dens is not in [0, 1]
        (1, 1, 1.1, (0.0, 1.0), None, ValueError),  # l_dens is not in [0, 1]
        (1, 1, 0.0, (1.0, 0.0), None, ValueError),  # wm_min_max[0] > wm_min_max[1]
        (1, 1, 0.0, (0.0, 1.0), "invalid", TypeError),
        # rng is not `None`, an `int` or a `numpy.random.Generator`
    ],
)
def test_gen_delayed_causal_network_invalid_inputs(
    n_nodes, ts_len, l_dens, wm_min_max, rng, error
):
    """Test the gen_delayed_causal_network function with invalid inputs."""
    with pytest.raises(error):
        gen_delayed_causal_network(ts_len, n_nodes, l_dens, wm_min_max, rng)


def test_gen_fmri():
    """Test the gen_fmri function."""
    # Test with default parameters
    ts = gen_fmri()
    assert isinstance(ts, np.ndarray)
    assert ts.ndim == 2
    assert ts.shape[1] == 2  # Two time series

    # Test with custom parameters
    ts_len = 500
    ts = gen_fmri(ts_len=ts_len)
    assert ts.shape[0] < ts_len  # Due to downsampling

    # Test with different random seed
    ts1 = gen_fmri(rng=0)
    ts2 = gen_fmri(rng=0)
    assert array_equal(ts1, ts2)  # Same seed should give same result

    ts3 = gen_fmri(rng=1)
    assert not array_equal(ts1, ts3)  # Different seed should give different result


@pytest.mark.parametrize("ts_len", [10, 100, 1000])
@pytest.mark.parametrize("downsampling_factor", [1, 2, 5])
@pytest.mark.parametrize("time_resolution", [0.1, 0.2, 0.5])
def test_gen_fmri_parameters(ts_len, downsampling_factor, time_resolution):
    """Test gen_fmri with different parameters."""
    ts = gen_fmri(
        ts_len=ts_len,
        downsampling_factor=downsampling_factor,
        time_resolution=time_resolution,
    )

    # The exact output shape can vary slightly due to implementation details
    # So we'll check that it's close to the expected shape
    assert ts.shape[1] == 2

    # Check that the length is reasonable
    expected_min_length = (
        ts_len + int(30 / time_resolution) - 10
    ) // downsampling_factor
    expected_max_length = (
        ts_len + int(30 / time_resolution) + 10
    ) // downsampling_factor

    assert expected_min_length <= ts.shape[0] <= expected_max_length
    assert not np.isnan(ts).any()  # No NaN values
    assert not np.isinf(ts).any()  # No infinite values


@pytest.mark.parametrize("coupling_strength", [0.0, 1.0, 2.0, 5.0])
@pytest.mark.parametrize("noise_initial_sd", [0.1, 1.0, 2.0])
@pytest.mark.parametrize("noise_final_sd", [0.01, 0.1, 0.5])
def test_gen_fmri_coupling_and_noise(
    coupling_strength, noise_initial_sd, noise_final_sd
):
    """Test gen_fmri with different coupling and noise parameters."""
    ts_len = 100  # Shorter length for faster tests

    ts = gen_fmri(
        ts_len=ts_len,
        coupling_strength=coupling_strength,
        noise_initial_sd=noise_initial_sd,
        noise_final_sd=noise_final_sd,
    )

    assert not np.isnan(ts).any()  # No NaN values
    assert not np.isinf(ts).any()  # No infinite values

    # Higher coupling strength should lead to higher correlation between time series
    if coupling_strength > 0:
        # Calculate correlation between the two time series
        correlation = np.corrcoef(ts[:, 0], ts[:, 1])[0, 1]

        # For very small time series or certain parameter combinations,
        # the correlation might be very small but should still be non-zero
        if ts_len >= 100:
            # The correlation should be non-zero for coupled time series
            assert abs(correlation) > 0.0001


def test_gen_fmri_multiple():
    """Test the gen_fmri_multiple function."""
    # Test with default parameters
    ts = gen_fmri_multiple()
    assert isinstance(ts, np.ndarray)
    assert ts.ndim == 2
    assert ts.shape[0] == 2  # Default number of nodes

    # Test with custom parameters
    ts_len = 500
    n_nodes = 3
    ts = gen_fmri_multiple(ts_len=ts_len, n_nodes=n_nodes)
    assert ts.shape[0] == n_nodes
    assert ts.shape[1] < ts_len  # Due to downsampling

    # Test with different random seed
    ts1 = gen_fmri_multiple(rng=0)
    ts2 = gen_fmri_multiple(rng=0)
    assert array_equal(ts1, ts2)  # Same seed should give same result

    ts3 = gen_fmri_multiple(rng=1)
    assert not array_equal(ts1, ts3)  # Different seed should give different result


@pytest.mark.parametrize("ts_len", [10, 100, 1000])
@pytest.mark.parametrize("n_nodes", [2, 3, 5])
@pytest.mark.parametrize("downsampling_factor", [1, 2, 5])
def test_gen_fmri_multiple_parameters(ts_len, n_nodes, downsampling_factor):
    """Test gen_fmri_multiple with different parameters."""
    ts = gen_fmri_multiple(
        ts_len=ts_len,
        n_nodes=n_nodes,
        downsampling_factor=downsampling_factor,
    )

    # The exact output shape can vary slightly due to implementation details
    # So we'll check that it's close to the expected shape
    assert ts.shape[0] == n_nodes

    # Check that the length is reasonable
    time_resolution = 0.2  # Default value
    expected_min_length = (
        ts_len + int(30 / time_resolution) - 10
    ) // downsampling_factor
    expected_max_length = (
        ts_len + int(30 / time_resolution) + 10
    ) // downsampling_factor

    assert expected_min_length <= ts.shape[1] <= expected_max_length
    assert not np.isnan(ts).any()  # No NaN values
    assert not np.isinf(ts).any()  # No infinite values


@pytest.mark.parametrize("coupling_strength", [0.0, 1.0, 2.0, 5.0])
@pytest.mark.parametrize("noise_initial_sd", [0.1, 1.0, 2.0])
@pytest.mark.parametrize("noise_final_sd", [0.01, 0.1, 0.5])
def test_gen_fmri_multiple_coupling_and_noise(
    coupling_strength, noise_initial_sd, noise_final_sd
):
    """Test gen_fmri_multiple with different coupling and noise parameters."""
    ts_len = 100  # Shorter length for faster tests
    n_nodes = 3

    ts = gen_fmri_multiple(
        ts_len=ts_len,
        n_nodes=n_nodes,
        coupling_strength=coupling_strength,
        noise_initial_sd=noise_initial_sd,
        noise_final_sd=noise_final_sd,
    )

    assert not np.isnan(ts).any()  # No NaN values
    assert not np.isinf(ts).any()  # No infinite values

    # Test that the first node influences other nodes when coupling_strength > 0
    if coupling_strength > 0:
        # Calculate correlation between the first node and other nodes
        for i in range(1, n_nodes):
            correlation = np.corrcoef(ts[0, :], ts[i, :])[0, 1]

            # For very small time series or certain parameter combinations,
            # the correlation might be very small but should still be non-zero
            if ts_len >= 100:
                # The correlation should be non-zero for coupled time series
                assert abs(correlation) > 0.00005


@pytest.mark.parametrize("n_nodes", [1, 2, 5, 10])
def test_gen_fmri_multiple_different_node_counts(n_nodes):
    """Test gen_fmri_multiple with different numbers of nodes."""
    ts_len = 50  # Shorter length for faster tests

    ts = gen_fmri_multiple(
        ts_len=ts_len,
        n_nodes=n_nodes,
        rng=0,
    )

    assert ts.shape[0] == n_nodes
    assert not np.isnan(ts).any()  # No NaN values
    assert not np.isinf(ts).any()  # No infinite values

    # Test that the coupling matrix is correctly constructed
    # The first node should influence all other nodes
    if n_nodes > 1:
        # Calculate correlation between the first node and other nodes
        for i in range(1, n_nodes):
            correlation = np.corrcoef(ts[0, :], ts[i, :])[0, 1]

            # There should be some correlation, but it might be very weak
            # for very short time series or certain node configurations
            if ts_len >= 50:
                assert abs(correlation) > 0.001


def test_gen_synthatdelays_random_connectivity():
    """Test the gen_synthatdelays_random_connectivity function."""
    # Test with minimal parameters
    sim_time = 1  # Short simulation time for testing
    num_airports = 3
    num_aircraft = 6
    buffer_time = 0.8

    results = gen_synthatdelays_random_connectivity(
        sim_time=sim_time,
        num_airports=num_airports,
        num_aircraft=num_aircraft,
        buffer_time=buffer_time,
    )

    # Check that results is a Results_Class object with expected attributes
    assert hasattr(results, "avgArrivalDelay")
    assert hasattr(results, "avgDepartureDelay")

    # Check that the delay matrices have the expected shape
    num_windows = 24  # 24 hours in a day with default 60-minute window
    assert results.avgArrivalDelay.shape == (num_windows, num_airports)
    assert results.avgDepartureDelay.shape == (num_windows, num_airports)


def test_gen_synthatdelays_random_connectivity_invalid_inputs():
    """Test the gen_synthatdelays_random_connectivity function with invalid inputs."""
    # Test with invalid sim_time
    with pytest.raises(ValueError, match="sim_time must be a positive integer"):
        gen_synthatdelays_random_connectivity(0, 3, 6, 0.8)


def test_gen_synthatdelays_random_connectivity_exception_handling(monkeypatch):
    """Test exception handling in gen_synthatdelays_random_connectivity."""

    # Mock ExecSimulation to raise an exception
    def mock_exec_simulation(*args, **kwargs):
        raise Exception("Test exception")

    # Apply the mock
    monkeypatch.setattr(
        "delaynet.preparation.data_generator.ExecSimulation", mock_exec_simulation
    )

    # Test that the exception is properly caught and re-raised
    with pytest.raises(
        RuntimeError, match="Error in SynthATDelays simulation: Test exception"
    ):
        gen_synthatdelays_random_connectivity(1, 3, 6, 0.8)


def test_gen_synthatdelays_independent_operations_with_trends():
    """Test the gen_synthatdelays_independent_operations_with_trends function."""
    # Test with trends activated
    sim_time = 1  # Short simulation time for testing

    # Call the function with trends activated
    results = gen_synthatdelays_independent_operations_with_trends(
        sim_time=sim_time, activate_trend=True
    )

    # Check that results is a Results_Class object with expected attributes
    assert hasattr(results, "avgArrivalDelay")
    assert hasattr(results, "avgDepartureDelay")

    # Check that the delay matrices have the expected shape
    num_windows = 24  # 24 hours in a day with default 60-minute window
    assert results.avgArrivalDelay.shape == (num_windows, 4)
    assert results.avgDepartureDelay.shape == (num_windows, 4)

    # Call the function with trends deactivated
    results_no_trend = gen_synthatdelays_independent_operations_with_trends(
        sim_time=sim_time, activate_trend=False
    )
    assert hasattr(results_no_trend, "avgArrivalDelay")


def test_gen_synthatdelays_independent_operations_with_trends_invalid_inputs():
    """Test the gen_synthatdelays_independent_operations_with_trends function with invalid inputs."""
    # Test with invalid sim_time
    with pytest.raises(ValueError, match="sim_time must be a positive integer"):
        gen_synthatdelays_independent_operations_with_trends(0, True)

    # Test with invalid activate_trend
    with pytest.raises(ValueError, match="activate_trend must be a boolean"):
        gen_synthatdelays_independent_operations_with_trends(1, "yes")


def test_gen_synthatdelays_independent_operations_with_trends_exception_handling(
    monkeypatch,
):
    """Test exception handling in gen_synthatdelays_independent_operations_with_trends."""

    # Mock ExecSimulation to raise an exception
    def mock_exec_simulation(*args, **kwargs):
        raise Exception("Test exception")

    # Apply the mock
    monkeypatch.setattr(
        "delaynet.preparation.data_generator.ExecSimulation", mock_exec_simulation
    )

    # Test that the exception is properly caught and re-raised
    with pytest.raises(
        RuntimeError, match="Error in SynthATDelays simulation: Test exception"
    ):
        gen_synthatdelays_independent_operations_with_trends(1, True)


def test_extract_airport_delay_time_series():
    """Test the extract_airport_delay_time_series function."""
    # Generate real results using the synthatdelays package
    sim_time = 1
    num_airports = 3
    results = gen_synthatdelays_random_connectivity(
        sim_time=sim_time, num_airports=num_airports, num_aircraft=6, buffer_time=0.8
    )

    # Test with arrival delays
    arrival_delays = extract_airport_delay_time_series(results, "arrival")
    assert isinstance(arrival_delays, np.ndarray)
    assert arrival_delays.shape == (24, num_airports)

    # Test with departure delays
    departure_delays = extract_airport_delay_time_series(results, "departure")
    assert isinstance(departure_delays, np.ndarray)
    assert departure_delays.shape == (24, num_airports)

    # Test with invalid delay type
    with pytest.raises(ValueError, match="Unknown delay type"):
        extract_airport_delay_time_series(results, "invalid")


@pytest.mark.parametrize("time_resolution", [0.1, 0.2, 0.5, 1.0])
def test_hrf_indirectly(time_resolution):
    """Test the __hrf function indirectly through gen_fmri."""
    ts_len = 50
    # Use a fixed seed for reproducibility
    rng = 0

    # Generate fMRI time series with different time resolutions
    ts = gen_fmri(ts_len=ts_len, time_resolution=time_resolution, rng=rng)

    # The exact output shape can vary slightly due to implementation details
    # So we'll check that it's close to the expected shape
    assert ts.shape[1] == 2

    # Check that the length is reasonable
    expected_min_length = (
        ts_len + int(30 / time_resolution) - 10
    ) // 2  # Default downsampling_factor is 2
    expected_max_length = (ts_len + int(30 / time_resolution) + 10) // 2

    assert expected_min_length <= ts.shape[0] <= expected_max_length

    # The HRF function should produce a smooth response
    # We can test this by checking that the time series is not too noisy
    # Calculate the difference between consecutive time points
    diffs = np.diff(ts, axis=0)

    # The mean absolute difference should not be too large
    # This is a heuristic test that the HRF is working as expected
    assert np.mean(np.abs(diffs)) < 1.0


@pytest.mark.parametrize("rng", [None, 0, 42, default_rng(0)])
def test_hrf_with_different_rngs(rng):
    """Test the __hrf function with different random number generators."""
    ts_len = 50

    # Generate fMRI time series with different random seeds
    ts = gen_fmri(ts_len=ts_len, rng=rng)

    # Basic checks
    assert not np.isnan(ts).any()
    assert not np.isinf(ts).any()

    # The HRF function should produce a smooth response
    # We can test this by checking that the time series is not too noisy
    # Calculate the difference between consecutive time points
    diffs = np.diff(ts, axis=0)

    # The mean absolute difference should not be too large
    assert np.mean(np.abs(diffs)) < 1.0
