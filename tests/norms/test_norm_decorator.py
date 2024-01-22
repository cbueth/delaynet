"""Test the norm decorator."""

import pytest
from numpy import ndarray, array, array_equal, hstack

from delaynet.norms.norm import norm


def test_norm_decorator_simple():
    """Test the norm decorator by designing a simple norm."""

    @norm
    def simple_norm(ts: ndarray) -> ndarray:
        """Increment all values by one."""
        return ts + 1

    assert array_equal(simple_norm(array([1, 2, 3])), array([2, 3, 4]))


@pytest.mark.parametrize(
    "a, expected",
    [
        (-10, [-9, -8, -7]),
        (0, [1, 2, 3]),
        (1, [2, 3, 4]),
        (2, [3, 4, 5]),
        (3, [4, 5, 6]),
    ],
)
def test_norm_decorator_kwargs(a, expected):
    """Test the norm decorator by designing a simple norm with kwargs."""

    @norm
    def simple_norm(ts: ndarray, a: int = 1) -> ndarray:
        """Increment all values by one."""
        return ts + a

    assert array_equal(simple_norm(array([1, 2, 3]), a=a), array(expected))


def test_norm_decorator_kwargs_unknown():
    """Test the norm decorator by designing a simple norm with unknown kwargs."""

    @norm
    def simple_norm(ts: ndarray, a: int = 1) -> ndarray:
        """Increment all values by one."""
        return ts + a

    with pytest.raises(TypeError, match="got an unexpected keyword argument 'b'"):
        simple_norm(array([1, 2, 3]), b=2)


def test_norm_decorator_kwargs_unknown_ignored():
    """Test the norm decorator by designing a simple norm with unknown kwargs
    and kwarg checker off."""

    @norm
    def simple_norm(ts: ndarray, a: int = 1) -> ndarray:
        """Increment all values by one."""
        return ts + a

    assert array_equal(
        simple_norm(array([1, 2, 3]), check_kwargs=False, b=2), array([2, 3, 4])
    )


def test_norm_decorator_required_kwonly():
    """Test the norm decorator by designing a
    norm with required keyword-only arguments."""

    @norm
    def simple_norm(ts: ndarray, *, a: int) -> ndarray:
        """Increment all values by one."""
        return ts + a

    assert array_equal(simple_norm(array([1, 2, 3]), a=2), array([3, 4, 5]))


@pytest.mark.parametrize("check_kwargs", [True, False])
def test_norm_decorator_mixed_args(check_kwargs):
    """Test the norm decorator with a function that has mixed arguments."""

    @norm
    def mixed_args_norm(ts: ndarray, a=1, *, b: int) -> ndarray:
        """Increment all values by a and b."""
        return ts + a + b

    # Test with positional and keyword arguments
    assert array_equal(mixed_args_norm(array([1, 2, 3]), 2, b=3), array([6, 7, 8]))

    # Test with missing required keyword argument
    with pytest.raises(TypeError, match="missing a required argument: 'b'"):
        mixed_args_norm(array([1, 2, 3]), 2)

    # Test with unknown keyword argument
    if check_kwargs:
        with pytest.raises(TypeError, match="got an unexpected keyword argument 'c'"):
            mixed_args_norm(array([1, 2, 3]), 2, b=3, c=4)
    else:
        assert array_equal(
            mixed_args_norm(array([1, 2, 3]), 2, check_kwargs=check_kwargs, b=3, c=4),
            array([6, 7, 8]),
        )
        assert array_equal(
            mixed_args_norm(array([1, 2, 3]), 2, b=3, c=4, check_kwargs=check_kwargs),
            array([6, 7, 8]),
        )


def test_norm_decorator_faulty_input_type():
    """Test the norm decorator by designing a norm with a non-ndarray input."""

    @norm
    def non_ndarray_norm(ts: list) -> ndarray:
        """Increment all values by one."""
        return array(ts) + 1

    with pytest.raises(
        TypeError, match="ts must be of type ndarray, not <class 'list'>"
    ):
        non_ndarray_norm([1, 2, 3])


# when input is ndarray, but output is not ndarray
def test_norm_decorator_faulty_output_type():
    """Test the norm decorator by designing a norm with a non-ndarray output."""

    @norm
    def non_ndarray_norm(ts: ndarray) -> list:
        """Increment all values by one."""
        return list(ts + 1)

    with pytest.raises(
        ValueError,
        match="Norm function non_ndarray_norm must return an ndarray, "
        "not <class 'list'>.",
    ):
        non_ndarray_norm(array([1, 2, 3]))


@pytest.mark.parametrize(
    "faulty_norm",
    [
        lambda ts: array([]),
        lambda ts: hstack((ts, ts)),
    ],
)
def test_norm_decorator_shape_mismatch(faulty_norm):
    """Test the norm decorator by designing a norm with a shape mismatch."""
    with pytest.raises(ValueError, match="Shape of normalised time series"):
        norm(faulty_norm)(array([1, 2, 3]))
