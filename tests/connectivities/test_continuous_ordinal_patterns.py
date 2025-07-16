"""Tests for the continuous ordinal patterns connectivity measure."""

import pytest
from numpy import array, allclose, linspace, random, roll

from delaynet import connectivity
from delaynet.connectivities.continuous_ordinal_patterns import (
    norm_window,
    norm_windows,
    pattern_distance,
    pattern_transform,
    random_patterns,
)


@pytest.mark.parametrize(
    "ts, expected",
    [
        ([0.0], [0.0]),  # Single value
        ([-20.0], [0.0]),  # Single value
        ([1.0, 1.0], [0.0, 0.0]),  # Constant values
        ([0.0, 1.0], [-1.0, 1.0]),
        ([1.0, 0.0], [1.0, -1.0]),
        ([10.0, 11.0, 9.0], [0.0, 1.0, -1.0]),
        (linspace(0, 1, 50), linspace(-1, 1, 50)),
    ],
)
def test_norm_window(ts, expected):
    """Test the normalisation of a window by design."""
    assert allclose(norm_window(array(ts)), array(expected))


@pytest.mark.parametrize(
    "ts",
    [None, [1, 2, 3], True, "string", 1, 1.0, [1, 2, "string"], [1, 2, None]],
)
def test_norm_window_typing_error(ts):
    """Test the normalisation of a window with invalid input."""
    with pytest.raises(Exception):
        ts = array(ts)
        norm_window(ts)


@pytest.mark.parametrize(
    "ts, window_size, expected",
    [
        ([0.0], 1, [[0.0]]),
        ([1.0, 1.0], 2, [[0.0, 0.0]]),
        ([0.0, 1.0], 1, [[0.0], [0.0]]),
        ([0.0, 1.0, 2.0], 2, [[-1.0, 1.0], [-1.0, 1.0]]),
        ([0.0, 1.0, 2.0, 3.0], 2, [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]),
        ([0.0, 1.0, 2.0, 3.0], 3, [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]),
        (linspace(0, 1, 50), 10, [linspace(-1, 1, 10)] * 41),
    ],
)
def test_norm_windows(ts, window_size, expected):
    """Test the normalisation of windows."""
    assert allclose(norm_windows(array(ts), window_size), array(expected))


@pytest.mark.parametrize(
    "windows, pattern, expected",
    [
        ([[0.0]], [0.0], [0.0]),
        ([[-1.0, 1.0]], [-1.0, 1.0], [0.0]),
        ([[0.0, 0.5, 1.0]], [1.0, 0.5, 0.0], [2 / 6]),
        ([[0.0, 0.4, 1.0]], [1.0, 0.5, 0.0], [2.1 / 6]),
        ([[0.0, 1.0], [1.0, 0.0]], [0.0, 1.0], [0.0, 2 / 4]),
    ],
)
def test_pattern_distance(windows, pattern, expected):
    """Test the computation of the distance between windows and a pattern."""
    assert allclose(pattern_distance(array(windows), array(pattern)), array(expected))


@pytest.mark.parametrize(
    "ts, patterns, expected",
    [
        (
            [[0.0, 1.0, 2.0]],
            [[-1.0, 1.0]],
            [[0.0, 0.0]],
        ),
        (
            [linspace(0, 1, 50)],
            [[-1.0, 1.0]],
            [[0.0] * (50 - 2 + 1)],
        ),
        (
            [[0.0, 1.0, 2.0, 3.0]],
            [[-1.0, 1.0], [1.0, -1.0]],
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        ),
        (
            [[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]],
            [[-1.0, 1.0], [1.0, -1.0]],
            [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]],
        ),
        (
            [linspace(0, 1, 50) for _ in range(20)],
            [linspace(-1, 1, 8) for _ in range(3)],
            [[[0.0] * (50 - 8 + 1)] * 3] * 20,
        ),
        (
            [[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]],
            [-1.0, 1.0],  # Single pattern, 1D
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ),
        (
            [0.0, 1.0, 2.0, 3.0],  # Single time series, 1D
            [[-1.0, 1.0], [1.0, -1.0]],
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        ),
        (
            [0.0, 1.0, 2.0, 3.0],  # Single time series, 1D
            [1.0, -1.0],  # Single pattern, 1D
            [1.0, 1.0, 1.0],
        ),
    ],
)
def test_pattern_transform(ts, patterns, expected):
    """Test the transformation of time series using patterns."""
    assert allclose(pattern_transform(array(ts), array(patterns)), array(expected))

def test_random_patterns_too_large():
    """Test Pattern size + lag-steps larger than the time series length"""
    with pytest.raises(ValueError, match="Pattern size \+ lag-steps"):
        random_patterns(array([1, 2, 3]), array([1, 2, 3]), lag_steps=10)


@pytest.mark.parametrize("p_size,num_rnd_patterns", [(5, 3), (2, 10)])
@pytest.mark.parametrize("linear", [True, False])
def test_random_patterns(p_size, num_rnd_patterns, linear):
    # Generate some test time series data
    random.seed(24567)
    ts1 = random.normal(0, 1, size=100)
    ts2 = roll(ts1, 2) + random.normal(0, 0.1, size=100)  # Create causally related series

    # Test the connectivity with default parameters
    result = connectivity(
        ts1,
        ts2,
        metric="random_patterns",
        lag_steps=5,
        p_size=p_size,
        num_rnd_patterns=num_rnd_patterns,
        linear=linear,
    )

    # Assert that the function returns expected format
    assert isinstance(result, tuple), "Result should be a tuple"
    assert len(result) == 2, "Result should contain two elements"

    # Test with expected value
    # The lag should be 2 since we created ts2 by rolling ts1 by 2
    p_value, lag = result
    assert lag == 2, f"Expected lag to be 2, got {lag}"
