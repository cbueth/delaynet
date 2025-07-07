"""Decorators for the delaynet package."""

from functools import wraps
from collections.abc import Callable
from inspect import signature

from numpy import ndarray, isnan, isinf, apply_along_axis, floating, integer

from .utils.bind_args import bind_args
from .utils.lag_steps import assure_lag_list

Connectivity = Callable[[ndarray, ndarray, ...], tuple[float, int]]
Detrend = Callable[[ndarray, ...], ndarray]


def connectivity(
    *args,
):
    """Decorator for the connectivity functions.

    As the connectivity functions take different arguments, this decorator is used to
    make sure that the connectivity functions are called with the correct positional
    and keyword arguments.

    The decorated function only needs to take the two ndarrays as the first positional
    arguments and can take any number of further arguments.

    The decorator checks that all passed keyword arguments exist and that all positional
    arguments are passed.

    Shape of the input time series must be equal.

    :return: The decorated function.
    :rtype: Callable
    :raises TypeError: If ``mcb_kwargs`` is not ``None`` or a ``dict``.
    """

    def connectivity_outer(connectivity_func: Connectivity) -> Connectivity:
        """Outer function of the decorator.

        :param connectivity_func: The connectivity function to decorate.
        :type connectivity_func: Callable
        :return: The decorated connectivity function.
        :rtype: Callable
        """
        # Check if the decorated function accepts 'lag_steps' as a keyword argument
        sig = signature(connectivity_func)
        if "lag_steps" not in sig.parameters:
            raise TypeError(
                f"The decorated function `{connectivity_func.__name__}` does not accept"
                " 'lag_steps' as a keyword argument."
            )

        @wraps(connectivity_func)
        def wrapper(
            ts1: ndarray,
            ts2: ndarray,
            *args,
            **kwargs,
        ) -> tuple[float, int]:
            """Wrapper for the connectivity functions.

            If kwargs have a key ``check_kwargs`` with value ``False``, the kwargs are
            not checked for availability.
            This is useful if you want to pass unused keyword.

            ``lag_steps`` will explicitly be checked, if not ``None``.

            :param ts1: The first time series.
            :type ts1: numpy.ndarray
            :param ts2: The second time series.
            :type ts2: numpy.ndarray
            :param args: The args to pass to the connectivity function.
            :type args: list
            :param lag_steps: The number of lag steps to consider. Required.
                              Can be integer for [1, ..., num], or a list of integers.
            :type lag_steps: int | list[int] | None
            :param kwargs: The kwargs to pass to the connectivity function.
            :type kwargs: dict
            :return: Connectivity value and lag (if applicable).
            :rtype: tuple[float, int]
            :raises TypeError: type of ``ts1`` or ``ts2`` is not ndarray.
            :raises ValueError: If ``ts1`` and ``ts2`` do not have the same shape.
            :raises ValueError: If an argument is missing.
            :raises TypeError: If an unknown kwarg is passed.
            :raises TypeError: If ``lag_steps`` is not ``None`` and
                               ``connectivity_func`` does not provide it.
            """
            # Check if ts1 and ts2 are ndarrays
            if not isinstance(ts1, ndarray) and not isinstance(ts2, ndarray):
                raise TypeError(
                    f"`ts1` and `ts2` must be of type ndarray, not {type(ts1)} and "
                    f"{type(ts2)}."
                )
            # Check if ts1 and ts2 have the same shape
            if ts1.shape != ts2.shape:
                raise ValueError(
                    "`ts1` and `ts2` must have the same shape, "
                    f"but have shapes {ts1.shape} and {ts2.shape}."
                )

            # Prepare lag steps for the connectivity function
            # num -> [1, ..., num], or keeping list if already a list
            # lag_steps = assure_lag_list(lag_steps)
            if kwargs.get("lag_steps") is None:
                raise ValueError(
                    "`lag_steps` must be passed to the connectivity function "
                    "as keyword argument."
                )
            kwargs["lag_steps"] = assure_lag_list(kwargs.get("lag_steps"))

            # Bind args for the connectivity function
            # - makes sure only existing args are passed to it
            bound_args = bind_args(connectivity_func, [ts1, ts2, *args], kwargs)
            # Call the connectivity function with the bound arguments
            conn_value = connectivity_func(*bound_args.args, **bound_args.kwargs)

            if not (isinstance(conn_value, tuple) and len(conn_value) == 2):
                raise ValueError("Metric function must return tuple of float and int.")
            if not (
                isinstance(conn_value[0], (float, floating))
                and isinstance(conn_value[1], (int, integer))
            ):
                raise ValueError(
                    "Invalid return value of connectivity function. "
                    "First value of tuple must be float, second must be int. "
                    f"Got {type(conn_value[0])} and {type(conn_value[1])}."
                )
            return conn_value

        return wrapper

    # Usage without parentheses (@connectivity)
    if args and callable(args[0]):
        return connectivity_outer(args[0])
    # Usage with parentheses (@connectivity(), @connectivity(...))
    return connectivity_outer


def detrending_method(
    *args,
    check_nan: bool = True,
    check_inf: bool = True,
    check_shape: bool = True,
):
    """Decorator for the detrending functions.

    Input time series can be of any shape. For 1D arrays,
    the method is applied directly.
    For higher dimensional arrays, an 'axis' kwarg must be provided to specify along
    which axis to apply the detrending.

    The decorator automatically detects if the detrending function has an 'axis'
    parameter in its signature. If it does, the axis is passed directly to the function.
    If not, apply_along_axis is used to apply the function along the specified axis.

    Input must be non-empty.
    If ``check_shape`` is ``True``, the shape of the output time series is checked.
    Otherwise, only the dimensionality must remain the same, so the length of the time
    series can change.

    As the detrending methods take different arguments, this decorator is used to make
    sure that the detrending methods are called with the correct positional and keyword
    arguments.

    The decorated function only needs to take the ndarray as the first positional
    argument and can take any number of further arguments.

    The decorator checks that all passed keyword arguments exist and that all positional
    arguments are passed.

    :param check_nan: If ``True``, check if the detrended time series contains NaNs.
    :type check_nan: bool
    :param check_inf: If ``True``, check if the detrended time series contains Infs.
    :type check_inf: bool
    :param check_shape: If ``True``, check if the shape of the detrended time series is
                        equal to the shape of the input time series.
    :type check_shape: bool
    :param detrend_func: The detrending function to decorate.
    :type detrend_func: Callable
    :return: The decorated function.
    :rtype: Callable
    """

    def detrend_outer(detrend_func: Detrend) -> Detrend:
        """Outer function of the decorator.

        :param detrend_func: The detrend function to decorate.
        :type detrend_func: Callable
        :return: The decorated detrend function.
        :rtype: Callable
        """

        @wraps(detrend_func)
        def wrapper(ts: ndarray, *args, **kwargs) -> ndarray:
            """Wrapper for the detrending functions.

            If kwargs have a key ``check_kwargs`` with value ``False``,
            the kwargs are not checked for availability.
            This is useful if you want to pass unused keyword.

            :param ts: The Time series to detrend.
            :type ts: numpy.ndarray of any shape
            :param args: The args to pass to the detrending function.
            :type args: list
            :param kwargs: The kwargs to pass to the detrending function.
            :type kwargs: dict
            :return: The Detrended time series.
            :rtype: numpy.ndarray
            :raises TypeError: Type of ts is not ndarray.
            :raises ValueError: If the input time series is empty.
            :raises ValueError: If axis kwarg is missing for multidimensional arrays.
            :raises ValueError: If an argument is missing.
            :raises TypeError: If an unknown kwarg is passed.
            :raises ValueError: If the shape of the detrended output is not equal to the
                                shape of the input time series. Only if ``check_shape``
                                is ``True``.
            :raises ValueError: If the dimensionality of the detrended output is not
                                equal to the dimensionality of the input time series.
            :raises ValueError: If the detrended output contains NaNs.
                                Only if ``check_nan`` is ``True``.
            :raises ValueError: If the detrended output contains Infs.
                                Only if ``check_inf`` is ``True``.
            """
            # Check if ts is an ndarray
            if not isinstance(ts, ndarray):
                raise TypeError(f"ts must be of type ndarray, not {type(ts)}.")

            # Check if ts is empty
            if ts.size == 0:
                raise ValueError("ts must not be empty.")

            # Check if detrend_func has an 'axis' parameter in its signature
            sig = signature(detrend_func)
            has_axis_param = "axis" in sig.parameters

            # For 1D arrays, apply detrending directly
            if ts.ndim == 1:
                # Validate axis if provided
                if "axis" in kwargs:
                    axis = kwargs["axis"]
                    # Validate axis bounds for 1D array
                    if axis < -ts.ndim or axis >= ts.ndim:
                        raise ValueError(
                            f"axis {axis} is out of bounds for array of dimension {ts.ndim}"
                        )

                if has_axis_param:
                    bound_args = bind_args(detrend_func, [ts, *args], kwargs)
                    detrended_ts = detrend_func(*bound_args.args, **bound_args.kwargs)
                else:
                    # Remove 'axis' from kwargs if function doesn't accept it
                    kwargs_without_axis = {
                        k: v for k, v in kwargs.items() if k != "axis"
                    }
                    bound_args = bind_args(
                        detrend_func, [ts, *args], kwargs_without_axis
                    )
                    detrended_ts = detrend_func(*bound_args.args, **bound_args.kwargs)
            else:
                # For higher dimensional arrays, require axis kwarg
                if "axis" not in kwargs:
                    raise ValueError(
                        f"For {ts.ndim}D arrays, 'axis' kwarg must be specified to "
                        f"indicate along which axis to apply the detrending function."
                    )

                axis = kwargs["axis"]

                # Validate axis bounds
                if axis < -ts.ndim or axis >= ts.ndim:
                    raise ValueError(
                        f"axis {axis} is out of bounds for array of dimension {ts.ndim}"
                    )

                if has_axis_param:
                    # Pass axis directly to the detrending function
                    bound_args = bind_args(detrend_func, [ts, *args], kwargs)
                    detrended_ts = detrend_func(*bound_args.args, **bound_args.kwargs)
                else:
                    # Use apply_along_axis, removing 'axis' from kwargs for the detrending function
                    kwargs_without_axis = {
                        k: v for k, v in kwargs.items() if k != "axis"
                    }
                    bound_args = bind_args(
                        detrend_func, [ts, *args], kwargs_without_axis
                    )
                    detrended_ts = apply_along_axis(
                        detrend_func,  # func1d
                        axis,  # axis
                        ts,  # arr
                        *bound_args.args[1:],  # args for func1d
                        **bound_args.kwargs,  # kwargs for func1d
                    )

            # Get the shape/dimensionality of the input time series
            shape_dim = ts.shape if check_shape else ts.ndim
            # Check output type
            if not isinstance(detrended_ts, ndarray):
                raise ValueError(
                    f"Detrending function {detrend_func.__name__} must return an "
                    f"ndarray, not {type(detrended_ts)}."
                )
            # Check if the shape of the output time series is equal to the
            # shape of the input time series
            if shape_dim != (detrended_ts.shape if check_shape else detrended_ts.ndim):
                raise ValueError(
                    (
                        f"Shape of detrended time series ({detrended_ts.shape}) "
                        f"does not match shape of input time series ({shape_dim})."
                    )
                    if check_shape
                    else (
                        f"Dimensionality of detrended time series ({detrended_ts.ndim})"
                        " does not match dimensionality of input time series "
                        f"({shape_dim})."
                    )
                )
            all_checks = {
                "check_nan": {"func": isnan, "msg": "NaNs", "check": check_nan},
                "check_inf": {"func": isinf, "msg": "Infs", "check": check_inf},
            }
            # keep if key == check_nan and check_nan is True
            checks = {
                # extend val by result of check
                key: {**val, "result": val["func"](detrended_ts).any()}
                for key, val in all_checks.items()
                if val["check"]
            }
            if any(val["result"] for val in checks.values()):
                # lazy calculate for input ts
                all_checks = {
                    key: {**val, "result": val["func"](ts).any()}
                    for key, val in all_checks.items()
                    if val["check"]
                }
                detrend_val = (val["msg"] for val in checks.values() if val["result"])
                input_val = (val["msg"] for val in all_checks.values() if val["result"])
                raise ValueError(
                    f"Detrended time series contains "
                    f"{', '.join(detrend_val)}: "
                    f"{detrended_ts}. " + f"Input time series contained "
                    f"{', '.join(input_val)}."
                )
            return detrended_ts

        return wrapper

    # Usage without parentheses (@detrending_method)
    if args and callable(args[0]):
        return detrend_outer(args[0])
    # Usage with parentheses (@detrending_method(), @detrending_method(...))
    return detrend_outer
