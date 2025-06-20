"""Test the norm decorator."""

from sys import version_info

import pytest
from numpy import (
    arange,
    array,
    array_equal,
    hstack,
    inf,
    isinf,
    isnan,
    nan,
    ndarray,
    ones,
    random,
)

from delaynet.decorators import norm


# Shared norm functions used across multiple tests
@norm
def axis_dependent_norm(ts: ndarray, axis: int = -1) -> ndarray:
    """Multiply values by their position index along the specified axis."""
    # Create arange based on the shape along the specified axis
    axis_length = ts.shape[axis]
    multiplier = arange(axis_length)

    # Create the proper shape for broadcasting
    shape = [1] * ts.ndim
    shape[axis] = axis_length
    multiplier = multiplier.reshape(shape)

    return ts * multiplier


@norm
def axis_independent_norm(ts: ndarray) -> ndarray:
    """Same as axis_dependent_norm, but without the axis parameter."""
    return ts * arange(ts.shape[0])


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

    error_msg = (
        "missing a required argument: 'b'"
        if version_info[:2] < (3, 12)
        else "missing a required keyword-only argument: 'b'"
    )

    # Test with missing required keyword argument
    with pytest.raises(TypeError, match=error_msg):
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


@pytest.mark.parametrize("check_nan", [True, False])
@pytest.mark.parametrize("check_inf", [True, False])
@pytest.mark.parametrize("replace", [nan, inf, -inf])
@pytest.mark.parametrize(
    "test_ts",
    [
        array([1, 2, 3]),
        array([nan, 2, 3]),
        array([-inf, 2, 3]),
        array([inf, 2, nan]),
    ],
)
def test_norm_decorator_check_nans(test_ts, check_nan, check_inf, replace):
    """Test the norm decorator by designing a norm introducing a NaN."""

    @norm(check_nan=check_nan, check_inf=check_inf)
    def norm_with_nans(ts: ndarray) -> ndarray:
        """Assign the second value NaN."""
        # make array allow NaNs
        ts = ts.astype(float)
        ts[1] = replace
        return ts

    nan_condition = check_nan and (isnan(replace) or isnan(test_ts).any())
    inf_condition = check_inf and (isinf(replace) or isinf(test_ts).any())
    if nan_condition or inf_condition:
        with pytest.raises(
            ValueError,
            match="Normalised time series contains "
            + (
                ", ".join(
                    msg
                    for msg, check in zip(
                        ["NaNs", "Infs"],
                        [nan_condition, inf_condition],
                    )
                    if check
                )
            )
            # match any normed_ts
            + ": .*"
            + (
                "Input time series contained "
                + (
                    ", ".join(
                        msg
                        for msg, check in zip(
                            ["NaNs", "Infs"],
                            [
                                check_nan and isnan(test_ts).any(),
                                check_inf and isinf(test_ts).any(),
                            ],
                        )
                        if check
                    )
                )
            ),
        ):
            norm_with_nans(test_ts)
    else:
        assert norm_with_nans(test_ts) is not None


@pytest.mark.parametrize("check_shape", [True, False])
@pytest.mark.parametrize(
    "test_norm, shortening",
    [
        (lambda ts: ts, False),  # identity
        (lambda ts: ts[1:], True),  # remove first value
        (lambda ts: ts[:-1], True),  # remove last value
        (lambda ts: ts[1:-1], True),  # remove first and last value
    ],
)
def test_norm_decorator_check_shape(time_series, test_norm, shortening, check_shape):
    """Test the norm decorator by designing a norm that shortens the time series."""

    @norm(check_shape=check_shape)
    def norm_shortening(ts: ndarray) -> ndarray:
        """Shorten the time series."""
        return test_norm(ts)

    # For multidimensional arrays, we need to provide an axis parameter
    kwargs = {}
    if time_series.ndim > 1:
        kwargs["axis"] = 1  # Use axis=1 for backward compatibility

    if shortening and check_shape:
        with pytest.raises(
            ValueError,
            match="Shape of normalised time series",
        ):
            norm_shortening(time_series, **kwargs)
    else:
        assert norm_shortening(time_series, **kwargs) is not None


@pytest.mark.parametrize("dim_diff", [1, 2])
def test_norm_decorator_check_shape_dimensionality(time_series, dim_diff):
    """Test the norm decorator by designing a norm that changes the dimensionality."""

    @norm(check_shape=False)
    def add_dimensions(ts: ndarray) -> ndarray:
        """Add dimensions to the time series."""
        return ts.reshape(ts.shape + (1,) * dim_diff)

    # For multidimensional arrays, we need to provide an axis parameter
    kwargs = {}
    if time_series.ndim > 1:
        kwargs["axis"] = 1  # Use axis=1 for backward compatibility

    with pytest.raises(
        ValueError,
        match="Dimensionality of normalised time series",
    ):
        add_dimensions(time_series, **kwargs)


@pytest.mark.parametrize(
    "input_array, axis, expected",
    [
        (
            array([1, 2, 3]),
            0,
            array([0, 2, 6]),
        ),
        (
            array([[1, 2, 3], [4, 5, 6]]),
            1,
            array([[0, 2, 6], [0, 5, 12]]),
        ),
        (
            array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
            2,
            array([[[0, 2], [0, 4]], [[0, 6], [0, 8]]]),
        ),
        (
            array([[1, 2, 3], [4, 5, 6]]),
            0,
            array([[0, 0, 0], [4, 5, 6]]),
        ),
        (
            array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
            0,
            array([[[0, 0], [0, 0]], [[5, 6], [7, 8]]]),
        ),
        (
            array([[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]]]),
            0,
            array([[[0, 0], [0, 0]], [[1, 1], [1, 1]], [[2, 2], [2, 2]]]),
        ),
        (
            array([[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]]]),
            1,
            array([[[0, 0], [1, 1]], [[0, 0], [1, 1]], [[0, 0], [1, 1]]]),
        ),
        (
            array([[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]]]),
            2,
            array([[[0, 1], [0, 1]], [[0, 1], [0, 1]], [[0, 1], [0, 1]]]),
        ),
        (
            array([[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]]]),
            -1,  # equivalent to axis 2
            array([[[0, 1], [0, 1]], [[0, 1], [0, 1]], [[0, 1], [0, 1]]]),
        ),
        (
            array([[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]]]),
            -2,  # equivalent to axis 1
            array([[[0, 0], [1, 1]], [[0, 0], [1, 1]], [[0, 0], [1, 1]]]),
        ),
        (
            array([[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]]]),
            -3,  # equivalent to axis 0
            array([[[0, 0], [0, 0]], [[1, 1], [1, 1]], [[2, 2], [2, 2]]]),
        ),
    ],
)
def test_norm_decorator_multidimensional_arrays(input_array, axis, expected):
    """Test the norm decorator with arrays of various dimensions."""
    result = axis_dependent_norm(input_array, axis=axis)
    assert array_equal(result, expected)


@pytest.mark.parametrize(
    "input_array, description",
    [
        (
            array([[1, 2, 3], [4, 5, 6]]),
            "2D_array_without_axis_parameter_should_raise_error",
        ),
        (
            array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
            "3D_array_without_axis_parameter_should_raise_error",
        ),
        (
            array(
                [
                    [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                    [[[9, 10], [11, 12]], [[13, 14], [15, 16]]],
                ]
            ),
            "4D_array_without_axis_parameter_should_raise_error",
        ),
    ],
    ids=[
        "2D_array_without_axis_parameter_should_raise_error",
        "3D_array_without_axis_parameter_should_raise_error",
        "4D_array_without_axis_parameter_should_raise_error",
    ],
)
def test_norm_decorator_multidimensional_missing_axis(input_array, description):
    """Test that multidimensional arrays require an axis parameter."""
    with pytest.raises(ValueError, match="axis.*kwarg must be specified"):
        axis_dependent_norm(input_array)


@pytest.mark.parametrize(
    "input_array, axis, expected, description",
    [
        (
            array([1, 2, 3]),
            None,
            array([2, 3, 4]),
            "1D_array_no_axis_parameter_add_1_to_all_elements",
        ),
        (
            array([[1, 2, 3], [4, 5, 6]]),
            1,
            array([[2, 3, 4], [5, 6, 7]]),
            "2D_array_axis_1_function_handles_axis_itself",
        ),
        (
            array([[1, 2, 3], [4, 5, 6]]),
            0,
            array([[2, 3, 4], [5, 6, 7]]),
            "2D_array_axis_0_function_handles_axis_itself",
        ),
        (
            array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
            2,
            array([[[2, 3], [4, 5]], [[6, 7], [8, 9]]]),
            "3D_array_axis_2_function_handles_axis_itself",
        ),
    ],
    ids=[
        "1D_array_no_axis_parameter_add_1_to_all_elements",
        "2D_array_axis_1_function_handles_axis_itself",
        "2D_array_axis_0_function_handles_axis_itself",
        "3D_array_axis_2_function_handles_axis_itself",
    ],
)
def test_norm_decorator_with_axis_parameter(input_array, axis, expected, description):
    """Test norm function that has an axis parameter in its signature."""

    @norm
    def norm_with_axis(ts: ndarray, axis: int = None) -> ndarray:
        """Norm function that handles an axis parameter itself."""
        if axis is None:
            return ts + 1
        else:
            # Simple example: add 1 along the specified axis
            return ts + 1

    if axis is None:
        result = norm_with_axis(input_array)
    else:
        result = norm_with_axis(input_array, axis=axis)
    assert array_equal(result, expected)


@pytest.mark.parametrize(
    "input_array, axis, expected, description",
    [
        (
            array([1, 2, 3]),
            None,
            array([2, 3, 4]),
            "1D_array_no_axis_needed_add_1_to_all_elements",
        ),
        (
            array([[1, 2, 3], [4, 5, 6]]),
            1,
            array([[2, 3, 4], [5, 6, 7]]),
            "2D_array_axis_1_decorator_uses_apply_along_axis",
        ),
        (
            array([[1, 2, 3], [4, 5, 6]]),
            0,
            array([[2, 3, 4], [5, 6, 7]]),
            "2D_array_axis_0_decorator_uses_apply_along_axis",
        ),
        (
            array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
            1,
            array([[[2, 3], [4, 5]], [[6, 7], [8, 9]]]),
            "3D_array_axis_1_decorator_uses_apply_along_axis",
        ),
    ],
    ids=[
        "1D_array_no_axis_needed_add_1_to_all_elements",
        "2D_array_axis_1_decorator_uses_apply_along_axis",
        "2D_array_axis_0_decorator_uses_apply_along_axis",
        "3D_array_axis_1_decorator_uses_apply_along_axis",
    ],
)
def test_norm_decorator_without_axis_parameter(
    input_array, axis, expected, description
):
    """Test norm function that doesn't have an axis parameter
    - uses apply_along_axis.
    """

    @norm
    def norm_without_axis(ts: ndarray) -> ndarray:
        """Norm function that works on 1D arrays only."""
        return ts + 1

    if axis is None:
        result = norm_without_axis(input_array)
    else:
        result = norm_without_axis(input_array, axis=axis)
    assert array_equal(result, expected)


@pytest.mark.parametrize(
    "input_shape, axis, expected_multiplier, description",
    [
        (
            (2, 3, 4),
            0,
            array([0, 1]).reshape(2, 1, 1),
            "3D_array_axis_0_multiply_by_0_1_along_first_dimension",
        ),
        (
            (2, 3, 4),
            1,
            array([0, 1, 2]).reshape(1, 3, 1),
            "3D_array_axis_1_multiply_by_0_1_2_along_second_dimension",
        ),
        (
            (2, 3, 4),
            2,
            array([0, 1, 2, 3]).reshape(1, 1, 4),
            "3D_array_axis_2_multiply_by_0_1_2_3_along_third_dimension",
        ),
        (
            (3, 5),
            0,
            array([0, 1, 2]).reshape(3, 1),
            "2D_array_axis_0_multiply_by_0_1_2_along_rows",
        ),
        (
            (3, 5),
            1,
            array([0, 1, 2, 3, 4]).reshape(1, 5),
            "2D_array_axis_1_multiply_by_0_1_2_3_4_along_columns",
        ),
        (
            (2, 2, 3, 4),
            3,
            array([0, 1, 2, 3]).reshape(1, 1, 1, 4),
            "4D_array_axis_3_multiply_by_0_1_2_3_along_last_dimension",
        ),
    ],
    ids=[
        "3D_array_axis_0_multiply_by_0_1_along_first_dimension",
        "3D_array_axis_1_multiply_by_0_1_2_along_second_dimension",
        "3D_array_axis_2_multiply_by_0_1_2_3_along_third_dimension",
        "2D_array_axis_0_multiply_by_0_1_2_along_rows",
        "2D_array_axis_1_multiply_by_0_1_2_3_4_along_columns",
        "4D_array_axis_3_multiply_by_0_1_2_3_along_last_dimension",
    ],
)
def test_norm_decorator_different_axes(
    input_shape, axis, expected_multiplier, description
):
    """Test norm decorator with different axis values."""
    input_array = ones(input_shape)
    result = axis_dependent_norm(input_array, axis=axis)
    expected = ones(input_shape) * expected_multiplier

    assert array_equal(result, expected)
    assert result.shape == input_array.shape


@pytest.mark.parametrize(
    "input_array, axis, multiplier, offset, expected, description",
    [
        (
            array([1, 2, 3]),
            None,
            3.0,
            2,
            array([5, 8, 11]),
            "1D_array_multiply_by_3_add_2",
        ),
        (
            array([[1, 2], [3, 4]]),
            1,
            2.0,
            1,
            array([[3, 5], [7, 9]]),
            "2D_array_axis_1_multiply_by_2_add_1",
        ),
        (
            array([[1, 2], [3, 4]]),
            0,
            1.5,
            3,
            array([[4.5, 6.0], [7.5, 9.0]]),
            "2D_array_axis_0_multiply_by_1_5_add_3",
        ),
        (
            array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
            2,
            0.5,
            10,
            array([[[10.5, 11.0], [11.5, 12.0]], [[12.5, 13.0], [13.5, 14.0]]]),
            "3D_array_axis_2_multiply_by_0_5_add_10",
        ),
        (
            array([1, 2, 3, 4, 5]),
            None,
            1.0,
            0,
            array([1, 2, 3, 4, 5]),
            "1D_array_identity_multiply_by_1_add_0",
        ),
    ],
    ids=[
        "1D_array_multiply_by_3_add_2",
        "2D_array_axis_1_multiply_by_2_add_1",
        "2D_array_axis_0_multiply_by_1_5_add_3",
        "3D_array_axis_2_multiply_by_0_5_add_10",
        "1D_array_identity_multiply_by_1_add_0",
    ],
)
def test_norm_decorator_complex_norm_with_parameters(
    input_array, axis, multiplier, offset, expected, description
):
    """Test norm decorator with a complex norm function having multiple parameters."""

    @norm
    def complex_norm(ts: ndarray, multiplier: float = 2.0, offset: int = 1) -> ndarray:
        return ts * multiplier + offset

    if axis is None:
        result = complex_norm(input_array, multiplier=multiplier, offset=offset)
    else:
        result = complex_norm(
            input_array, axis=axis, multiplier=multiplier, offset=offset
        )
    assert array_equal(result, expected)


@pytest.mark.parametrize(
    "shape, axis, description",
    [
        ((5,), None, "1D_array_shape_5_identity_norm_preserves_shape"),
        ((3, 4), 0, "2D_array_shape_3x4_axis_0_identity_norm_preserves_shape"),
        ((3, 4), 1, "2D_array_shape_3x4_axis_1_identity_norm_preserves_shape"),
        ((2, 3, 4), 0, "3D_array_shape_2x3x4_axis_0_identity_norm_preserves_shape"),
        ((2, 3, 4), 1, "3D_array_shape_2x3x4_axis_1_identity_norm_preserves_shape"),
        ((2, 3, 4), 2, "3D_array_shape_2x3x4_axis_2_identity_norm_preserves_shape"),
        (
            (2, 2, 2, 2),
            0,
            "4D_array_shape_2x2x2x2_axis_0_identity_norm_preserves_shape",
        ),
        (
            (2, 2, 2, 2),
            3,
            "4D_array_shape_2x2x2x2_axis_3_identity_norm_preserves_shape",
        ),
        ((6, 7, 8), 1, "3D_array_shape_6x7x8_axis_1_identity_norm_preserves_shape"),
    ],
    ids=[
        "1D_array_shape_5_identity_norm_preserves_shape",
        "2D_array_shape_3x4_axis_0_identity_norm_preserves_shape",
        "2D_array_shape_3x4_axis_1_identity_norm_preserves_shape",
        "3D_array_shape_2x3x4_axis_0_identity_norm_preserves_shape",
        "3D_array_shape_2x3x4_axis_1_identity_norm_preserves_shape",
        "3D_array_shape_2x3x4_axis_2_identity_norm_preserves_shape",
        "4D_array_shape_2x2x2x2_axis_0_identity_norm_preserves_shape",
        "4D_array_shape_2x2x2x2_axis_3_identity_norm_preserves_shape",
        "3D_array_shape_6x7x8_axis_1_identity_norm_preserves_shape",
    ],
)
def test_norm_decorator_preserves_shape(shape, axis, description):
    """Test that the decorator preserves array shapes correctly."""

    @norm
    def identity_norm(ts: ndarray) -> ndarray:
        return ts

    input_array = random.randn(*shape)

    if axis is None:
        result = identity_norm(input_array)
    else:
        result = identity_norm(input_array, axis=axis)

    assert result.shape == input_array.shape
    assert array_equal(result, input_array)


def test_norm_decorator_backward_compatibility():
    """Test that existing 2D behaviour is preserved when axis=1."""

    # Test that 2D array with axis=1 gives an expected axis-dependent result
    input_2d = array([[1, 2, 3], [4, 5, 6]])
    result = axis_dependent_norm(input_2d, axis=1)
    expected = array([[0, 2, 6], [0, 5, 12]])  # [[1*0, 2*1, 3*2], [4*0, 5*1, 6*2]]
    assert array_equal(result, expected)


@pytest.mark.parametrize("n_dim", list(range(1, 6)))
@pytest.mark.parametrize("n_axis", list(range(-6, 6)))
def test_axis_independent_and_dependent_norm_same_output(n_dim, n_axis):
    """Test that axis_independent_norm and axis_dependent_norm have the same output."""
    data = ones((5,) * n_dim)

    # Skip invalid axis values for the given dimensionality
    if n_axis < -n_dim or n_axis >= n_dim:
        return

    assert array_equal(
        axis_independent_norm(data, axis=n_axis),
        axis_dependent_norm(data, axis=n_axis),
    )
