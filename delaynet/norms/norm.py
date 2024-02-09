"""Decorator for the norm functions."""

from functools import wraps
from collections.abc import Callable

from numpy import ndarray, isnan, isinf, apply_along_axis

from ..utils.bind_args import bind_args

Norm = Callable[[ndarray, ...], ndarray]


def norm(
    *args,
    check_nan: bool = True,
    check_inf: bool = True,
    check_shape: bool = True,
):
    """Decorator for the norm functions.

    Input time series can be 1D or 2D. If 2D, the norm is applied to each row.
    Input must be non-empty.
    If ``check_shape`` is ``True``, the shape of the output time series is checked.
    Otherwise, only the dimensionality must remain the same, so the length of the time
    series can change.

    As the norms take different arguments, this decorator is used to make sure that
    the norms are called with the correct positional and keyword arguments.

    The decorated function only needs to take the ndarray as the first positional
    argument and can take any number of further arguments.

    The decorator checks that all passed keyword arguments exist and that all positional
    arguments are passed.

    :param check_nan: If ``True``, check if the normed time series contains NaNs.
    :type check_nan: bool
    :param check_inf: If ``True``, check if the normed time series contains Infs.
    :type check_inf: bool
    :param check_shape: If ``True``, check if the shape of the normed time series is
                        equal to the shape of the input time series.
    :type check_shape: bool
    :param norm_func: The norm function to decorate.
    :type norm_func: Callable
    :return: The decorated function.
    :rtype: Callable
    """

    def norm_outer(norm_func: Norm) -> Norm:
        """Outer function of the decorator.

        :param norm_func: The norm function to decorate.
        :type norm_func: Callable
        :return: The decorated norm function.
        :rtype: Callable
        """

        @wraps(norm_func)
        def wrapper(ts: ndarray, *args, **kwargs) -> ndarray:
            """Wrapper for the norm functions.

            If kwargs have a key ``check_kwargs`` with value ``False``,
            the kwargs are not checked for availability.
            This is useful if you want to pass unused keyword.

            :param ts: The time series to normalise.
            :type ts: ndarray, shape (n,) or (m, n)
            :param args: The args to pass to the norm function.
            :type args: list
            :param kwargs: The kwargs to pass to the norm function.
            :type kwargs: dict
            :return: The normalised time series.
            :rtype: ndarray
            :raises TypeError: Type of ts is not ndarray of dimension 1 or 2.
            :raises ValueError: If the input time series is empty.
            :raises ValueError: If an argument is missing.
            :raises TypeError: If an unknown kwarg is passed.
            :raises ValueError: If the shape of the norm output is not equal to the
                                shape of the input time series. Only if ``check_shape``
                                is ``True``.
            :raises ValueError: If the dimensionality of the norm output is not equal
                                to the dimensionality of the input time series.
            :raises ValueError: If the norm output contains NaNs. Only if ``check_nan``
                                is ``True``.
            :raises ValueError: If the norm output contains Infs. Only if ``check_inf``
                                is ``True``.
            """
            # Check if ts is an ndarray
            if not isinstance(ts, ndarray):
                raise TypeError(f"ts must be of type ndarray, not {type(ts)}.")
            # Check if ts is 1D or 2D
            if ts.ndim not in [1, 2]:
                raise TypeError(f"ts must be of dimension 1 or 2, not {ts.ndim}.")
            # Check if ts is empty
            if ts.size == 0:
                raise ValueError("ts must not be empty.")

            bound_args = bind_args(norm_func, [ts, *args], kwargs)
            # Call the norm function with the bound arguments
            if ts.ndim == 1:
                normed_ts = norm_func(*bound_args.args, **bound_args.kwargs)
            else:
                normed_ts = apply_along_axis(
                    norm_func,  # func1d
                    1,  # axis
                    ts,  # arr
                    *bound_args.args[1:],  # args for func1d
                    **bound_args.kwargs,  # kwargs for func1d
                )

            # Get the shape/dimensionality of the input time series
            shape_dim = ts.shape if check_shape else ts.ndim
            # Check output type
            if not isinstance(normed_ts, ndarray):
                raise ValueError(
                    f"Norm function {norm_func.__name__} must return an ndarray, "
                    f"not {type(normed_ts)}."
                )
            # Check if the shape of the output time series is equal to the
            # shape of the input time series
            if shape_dim != (normed_ts.shape if check_shape else normed_ts.ndim):
                raise ValueError(
                    (
                        f"Shape of normalised time series ({normed_ts.shape}) "
                        f"does not match shape of input time series ({shape_dim})."
                    )
                    if check_shape
                    else (
                        f"Dimensionality of normalised time series ({normed_ts.ndim}) "
                        "does not match dimensionality of input time series "
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
                key: {**val, "result": val["func"](normed_ts).any()}
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
                norm_val = (val["msg"] for val in checks.values() if val["result"])
                input_val = (val["msg"] for val in all_checks.values() if val["result"])
                raise ValueError(
                    f"Normalised time series contains "
                    f"{', '.join(norm_val)}: "
                    f"{normed_ts}. " + f"Input time series contained "
                    f"{', '.join(input_val)}."
                )
            return normed_ts

        return wrapper

    # Usage without parentheses
    if args and callable(args[0]):
        return norm_outer(args[0])
    # Usage with parentheses
    return norm_outer
