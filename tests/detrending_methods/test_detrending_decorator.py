"""Test the detrend decorator."""

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

from delaynet.decorators import detrending_method


# Shared detrend functions used across multiple tests
@detrending_method
def axis_dependent_detrend(ts: ndarray, axis: int = -1) -> ndarray:
    """Multiply values by their position index along the specified axis."""
    # Create arange based on the shape along the specified axis
    axis_length = ts.shape[axis]
    multiplier = arange(axis_length)

    # Create the proper shape for broadcasting
    shape = [1] * ts.ndim
    shape[axis] = axis_length
    multiplier = multiplier.reshape(shape)

    return ts * multiplier


@detrending_method
def axis_independent_detrend(ts: ndarray) -> ndarray:
    """Same as axis_dependent_detrend, but without the axis parameter."""
    return ts * arange(ts.shape[0])


@detrending_method
def simple_detrend(ts: ndarray) -> ndarray:
    """Increment all values by one."""
    return ts + 1


def test_detrend_decorator_simple():
    """Test the detrend decorator by designing a simple detrend."""

    @detrending_method
    def simple_detrend(ts: ndarray) -> ndarray:
        """Increment all values by one."""
        return ts + 1

    assert array_equal(simple_detrend(array([1, 2, 3])), array([2, 3, 4]))


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
def test_detrend_decorator_kwargs(a, expected):
    """Test the detrend decorator by designing a simple detrend with kwargs."""

    @detrending_method
    def simple_detrend(ts: ndarray, a: int = 1) -> ndarray:
        """Increment all values by one."""
        return ts + a

    assert array_equal(simple_detrend(array([1, 2, 3]), a=a), array(expected))


def test_detrend_decorator_kwargs_unknown():
    """Test the detrend decorator by designing a simple detrend with unknown kwargs."""

    @detrending_method
    def simple_detrend(ts: ndarray, a: int = 1) -> ndarray:
        """Increment all values by one."""
        return ts + a

    with pytest.raises(TypeError, match="got an unexpected keyword argument 'b'"):
        simple_detrend(array([1, 2, 3]), b=2)


def test_detrend_decorator_kwargs_unknown_ignored():
    """Test the detrend decorator by designing a simple detrend with unknown kwargs
    and kwarg checker off."""

    @detrending_method
    def simple_detrend(ts: ndarray, a: int = 1) -> ndarray:
        """Increment all values by one."""
        return ts + a

    assert array_equal(
        simple_detrend(array([1, 2, 3]), check_kwargs=False, b=2), array([2, 3, 4])
    )


def test_detrend_decorator_required_kwonly():
    """Test the detrend decorator by designing a
    detrend with required keyword-only arguments."""

    @detrending_method
    def simple_detrend(ts: ndarray, *, a: int) -> ndarray:
        """Increment all values by one."""
        return ts + a

    assert array_equal(simple_detrend(array([1, 2, 3]), a=2), array([3, 4, 5]))


@pytest.mark.parametrize("check_kwargs", [True, False])
def test_detrend_decorator_mixed_args(check_kwargs):
    """Test the detrend decorator with a function that has mixed arguments."""

    @detrending_method
    def mixed_args_detrend(ts: ndarray, a=1, *, b: int) -> ndarray:
        """Increment all values by a and b."""
        return ts + a + b

    # Test with positional and keyword arguments
    assert array_equal(mixed_args_detrend(array([1, 2, 3]), 2, b=3), array([6, 7, 8]))

    error_msg = (
        "missing a required argument: 'b'"
        if version_info[:2] < (3, 12)
        else "missing a required keyword-only argument: 'b'"
    )

    # Test with missing required keyword argument
    with pytest.raises(TypeError, match=error_msg):
        mixed_args_detrend(array([1, 2, 3]), 2)

    # Test with unknown keyword argument
    if check_kwargs:
        with pytest.raises(TypeError, match="got an unexpected keyword argument 'c'"):
            mixed_args_detrend(array([1, 2, 3]), 2, b=3, c=4)
    else:
        assert array_equal(
            mixed_args_detrend(array([1, 2, 3]), 2, check_kwargs=check_kwargs, b=3, c=4),
            array([6, 7, 8]),
        )
        assert array_equal(
            mixed_args_detrend(array([1, 2, 3]), 2, b=3, c=4, check_kwargs=check_kwargs),
            array([6, 7, 8]),
        )


def test_detrend_decorator_faulty_input_type():
    """Test the detrend decorator by designing a detrend with a non-ndarray input."""

    @detrending_method
    def non_ndarray_detrend(ts: list) -> ndarray:
        """Increment all values by one."""
        return array(ts) + 1

    with pytest.raises(
        TypeError, match="ts must be of type ndarray, not <class 'list'>"
    ):
        non_ndarray_detrend([1, 2, 3])


# when input is ndarray, but output is not ndarray
def test_detrend_decorator_faulty_output_type():
    """Test the detrend decorator by designing a detrend with a non-ndarray output."""

    @detrending_method
    def non_ndarray_detrend(ts: ndarray) -> list:
        """Increment all values by one."""
        return list(ts + 1)

    with pytest.raises(
        ValueError,
        match="Detrending function non_ndarray_detrend must return an ndarray, "
        "not <class 'list'>.",
    ):
        non_ndarray_detrend(array([1, 2, 3]))


@pytest.mark.parametrize(
    "faulty_detrend",
    [
        lambda ts: array([]),
        lambda ts: hstack((ts, ts)),
    ],
)
def test_detrend_decorator_shape_mismatch(faulty_detrend):
    """Test the detrend decorator by designing a detrend with a shape mismatch."""
    with pytest.raises(ValueError, match="Shape of detrended time series"):
        detrending_method(faulty_detrend)(array([1, 2, 3]))


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
def test_detrend_decorator_check_nans(test_ts, check_nan, check_inf, replace):
    """Test the detrend decorator by designing a detrend introducing a NaN."""

    @detrending_method(check_nan=check_nan, check_inf=check_inf)
    def detrend_with_nans(ts: ndarray) -> ndarray:
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
            match="Detrended time series contains "
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
            # match any detrended_ts
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
            detrend_with_nans(test_ts)
    else:
        assert detrend_with_nans(test_ts) is not None


@pytest.mark.parametrize("check_shape", [True, False])
@pytest.mark.parametrize(
    "test_detrend, shortening",
    [
        (lambda ts: ts, False),  # identity
        (lambda ts: ts[1:], True),  # remove first value
        (lambda ts: ts[:-1], True),  # remove last value
        (lambda ts: ts[1:-1], True),  # remove first and last value
    ],
)
def test_detrend_decorator_check_shape(time_series, test_detrend, shortening, check_shape):
    """Test the detrend decorator by designing a detrend that shortens the time series."""

    @detrending_method(check_shape=check_shape)
    def detrend_shortening(ts: ndarray) -> ndarray:
        """Shorten the time series."""
        return test_detrend(ts)

    # For multidimensional arrays, we need to provide an axis parameter
    kwargs = {}
    if time_series.ndim > 1:
        kwargs["axis"] = 1  # Use axis=1 for backward compatibility

    if shortening and check_shape:
        with pytest.raises(
            ValueError,
            match="Shape of detrended time series",
        ):
            detrend_shortening(time_series, **kwargs)
    else:
        assert detrend_shortening(time_series, **kwargs) is not None


@pytest.mark.parametrize("dim_diff", [1, 2])
def test_detrend_decorator_check_shape_dimensionality(time_series, dim_diff):
    """Test the detrend decorator by designing a detrend that changes the dimensionality."""

    @detrending_method(check_shape=False)
    def add_dimensions(ts: ndarray) -> ndarray:
        """Add dimensions to the time series."""
        return ts.reshape(ts.shape + (1,) * dim_diff)

    # For multidimensional arrays, we need to provide an axis parameter
    kwargs = {}
    if time_series.ndim > 1:
        kwargs["axis"] = 1  # Use axis=1 for backward compatibility

    with pytest.raises(
        ValueError,
        match="Dimensionality of detrended time series",
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
def test_detrend_decorator_multidimensional_arrays(input_array, axis, expected):
    """Test the detrend decorator with arrays of various dimensions."""
    result = axis_dependent_detrend(input_array, axis=axis)
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
def test_detrend_decorator_multidimensional_missing_axis(input_array, description):
    """Test that multidimensional arrays require an axis parameter."""
    with pytest.raises(ValueError, match="axis.*kwarg must be specified"):
        axis_dependent_detrend(input_array)


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
def test_detrend_decorator_with_axis_parameter(input_array, axis, expected, description):
    """Test detrend function that has an axis parameter in its signature."""

    @detrending_method
    def detrend_with_axis(ts: ndarray, axis: int = None) -> ndarray:
        """Detrend function that handles an axis parameter itself."""
        if axis is None:
            return ts + 1
        else:
            # Simple example: add 1 along the specified axis
            return ts + 1

    if axis is None:
        result = detrend_with_axis(input_array)
    else:
        result = detrend_with_axis(input_array, axis=axis)
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
def test_detrend_decorator_without_axis_parameter(
    input_array, axis, expected, description
):
    """Test detrend function that doesn't have an axis parameter
    - uses apply_along_axis.
    """

    @detrending_method
    def detrend_without_axis(ts: ndarray) -> ndarray:
        """Detrend function that works on 1D arrays only."""
        return ts + 1

    if axis is None:
        result = detrend_without_axis(input_array)
    else:
        result = detrend_without_axis(input_array, axis=axis)
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
def test_detrend_decorator_different_axes(
    input_shape, axis, expected_multiplier, description
):
    """Test detrend decorator with different axis values."""
    input_array = ones(input_shape)
    result = axis_dependent_detrend(input_array, axis=axis)
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
def test_detrend_decorator_complex_detrend_with_parameters(
    input_array, axis, multiplier, offset, expected, description
):
    """Test detrend decorator with a complex detrend function having multiple parameters."""

    @detrending_method
    def complex_detrend(ts: ndarray, multiplier: float = 2.0, offset: int = 1) -> ndarray:
        return ts * multiplier + offset

    if axis is None:
        result = complex_detrend(input_array, multiplier=multiplier, offset=offset)
    else:
        result = complex_detrend(
            input_array, axis=axis, multiplier=multiplier, offset=offset
        )
    assert array_equal(result, expected)


@pytest.mark.parametrize(
    "shape, axis, description",
    [
        ((5,), None, "1D_array_shape_5_identity_detrend_preserves_shape"),
        ((3, 4), 0, "2D_array_shape_3x4_axis_0_identity_detrend_preserves_shape"),
        ((3, 4), 1, "2D_array_shape_3x4_axis_1_identity_detrend_preserves_shape"),
        ((2, 3, 4), 0, "3D_array_shape_2x3x4_axis_0_identity_detrend_preserves_shape"),
        ((2, 3, 4), 1, "3D_array_shape_2x3x4_axis_1_identity_detrend_preserves_shape"),
        ((2, 3, 4), 2, "3D_array_shape_2x3x4_axis_2_identity_detrend_preserves_shape"),
        (
            (2, 2, 2, 2),
            0,
            "4D_array_shape_2x2x2x2_axis_0_identity_detrend_preserves_shape",
        ),
        (
            (2, 2, 2, 2),
            3,
            "4D_array_shape_2x2x2x2_axis_3_identity_detrend_preserves_shape",
        ),
        ((6, 7, 8), 1, "3D_array_shape_6x7x8_axis_1_identity_detrend_preserves_shape"),
    ],
    ids=[
        "1D_array_shape_5_identity_detrend_preserves_shape",
        "2D_array_shape_3x4_axis_0_identity_detrend_preserves_shape",
        "2D_array_shape_3x4_axis_1_identity_detrend_preserves_shape",
        "3D_array_shape_2x3x4_axis_0_identity_detrend_preserves_shape",
        "3D_array_shape_2x3x4_axis_1_identity_detrend_preserves_shape",
        "3D_array_shape_2x3x4_axis_2_identity_detrend_preserves_shape",
        "4D_array_shape_2x2x2x2_axis_0_identity_detrend_preserves_shape",
        "4D_array_shape_2x2x2x2_axis_3_identity_detrend_preserves_shape",
        "3D_array_shape_6x7x8_axis_1_identity_detrend_preserves_shape",
    ],
)
def test_detrend_decorator_preserves_shape(shape, axis, description):
    """Test that the decorator preserves array shapes correctly."""

    @detrending_method
    def identity_detrend(ts: ndarray) -> ndarray:
        return ts

    input_array = random.randn(*shape)

    if axis is None:
        result = identity_detrend(input_array)
    else:
        result = identity_detrend(input_array, axis=axis)

    assert result.shape == input_array.shape
    assert array_equal(result, input_array)


def test_detrend_decorator_backward_compatibility():
    """Test that existing 2D behaviour is preserved when axis=1."""

    # Test that 2D array with axis=1 gives an expected axis-dependent result
    input_2d = array([[1, 2, 3], [4, 5, 6]])
    result = axis_dependent_detrend(input_2d, axis=1)
    expected = array([[0, 2, 6], [0, 5, 12]])  # [[1*0, 2*1, 3*2], [4*0, 5*1, 6*2]]
    assert array_equal(result, expected)


@pytest.mark.parametrize("n_dim", list(range(1, 6)))
@pytest.mark.parametrize("n_axis", list(range(-6, 6)))
def test_axis_independent_and_dependent_detrend_same_output(n_dim, n_axis):
    """Test that axis_independent_detrend and axis_dependent_detrend have the same output."""
    data = ones((5,) * n_dim)

    # Skip invalid axis values for the given dimensionality
    if n_axis < -n_dim or n_axis >= n_dim:
        return

    assert array_equal(
        axis_independent_detrend(data, axis=n_axis),
        axis_dependent_detrend(data, axis=n_axis),
    )


def test_detrend_decorator_empty_array():
    """Test that the detrend decorator raises ValueError for empty arrays."""

    @detrending_method
    def simple_detrend(ts: ndarray) -> ndarray:
        return ts + 1

    # Test with empty 1D array
    empty_array = array([])
    with pytest.raises(ValueError, match="ts must not be empty"):
        simple_detrend(empty_array)

    # Test with empty 2D array
    empty_2d_array = array([]).reshape(0, 5)
    with pytest.raises(ValueError, match="ts must not be empty"):
        simple_detrend(empty_2d_array, axis=0)


@pytest.mark.parametrize(
    "axis, array_dim",
    [
        (1, 1),    # axis=1 for 1D array (valid range: -1 to 0)
        (-2, 1),   # axis=-2 for 1D array (valid range: -1 to 0)
        (2, 1),    # axis=2 for 1D array (valid range: -1 to 0)
    ],
)
def test_detrend_decorator_axis_out_of_bounds_1d(axis, array_dim):
    """Test that the detrend decorator raises ValueError for out of bounds axis on 1D arrays."""

    @detrending_method
    def simple_detrend(ts: ndarray) -> ndarray:
        return ts + 1

    test_array = array([1, 2, 3])  # 1D array

    with pytest.raises(
        ValueError, 
        match=f"axis {axis} is out of bounds for array of dimension {array_dim}"
    ):
        simple_detrend(test_array, axis=axis)


@pytest.mark.parametrize(
    "axis, array_shape, array_dim",
    [
        (2, (3, 4), 2),      # axis=2 for 2D array (valid range: -2 to 1)
        (-3, (3, 4), 2),     # axis=-3 for 2D array (valid range: -2 to 1)
        (3, (2, 3, 4), 3),   # axis=3 for 3D array (valid range: -3 to 2)
        (-4, (2, 3, 4), 3),  # axis=-4 for 3D array (valid range: -3 to 2)
    ],
)
def test_detrend_decorator_axis_out_of_bounds_multidimensional(axis, array_shape, array_dim):
    """Test that the detrend decorator raises ValueError for out of bounds axis on multidimensional arrays."""

    @detrending_method
    def simple_detrend(ts: ndarray) -> ndarray:
        return ts + 1

    test_array = ones(array_shape)

    with pytest.raises(
        ValueError, 
        match=f"axis {axis} is out of bounds for array of dimension {array_dim}"
    ):
        simple_detrend(test_array, axis=axis)
