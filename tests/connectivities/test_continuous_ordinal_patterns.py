"""Tests for the continuous ordinal patterns connectivity measure."""

import numba
import pytest
from numpy import array, allclose, linspace

from delaynet.connectivities.continuous_ordinal_patterns import (
    norm_window,
    norm_windows,
    pattern_distance,
    pattern_transform,
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
    """Test the normalization of a window by design."""
    assert allclose(norm_window(array(ts)), array(expected))


@pytest.mark.parametrize(
    "ts",
    [None, [1, 2, 3], True, "string", 1, 1.0, [1, 2, "string"], [1, 2, None]],
)
def test_norm_window_typing_error(ts):
    """Test the normalization of a window with invalid input."""
    with pytest.raises(numba.TypingError):
        norm_window(array(ts))


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
    """Test the normalization of windows."""
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
    ],
)
def test_pattern_transform(ts, patterns, expected):
    """Test the transformation of time series using patterns."""
    assert allclose(pattern_transform(array(ts), array(patterns)), array(expected))
