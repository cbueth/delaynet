"""Decorator for the norm functions."""

from functools import wraps
from collections.abc import Callable

from numpy import ndarray, isnan, isinf

from ..utils.bind_args import bind_args

Norm = Callable[[ndarray, ...], ndarray]


def norm(
    *args,
    check_nan: bool = True,
    check_inf: bool = True,
):
    """Decorator for the norm functions.

    As the norms take different arguments, this decorator is used to make sure that
    the norms are called with the correct positional and keyword arguments.

    The decorated function only needs to take the ndarray as the first positional
    argument and can take any number of further arguments.

    The decorator checks that all passed keyword arguments exist and that all positional
    arguments are passed.

    Shape of the input and output time series must be equal.

    :param check_nan: If ``True``, check if the normed time series contains NaNs.
    :type check_nan: bool
    :param check_inf: If ``True``, check if the normed time series contains Infs.
    :type check_inf: bool
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

            If kwargs have a key ``check_kwargs`` with value ``False``, the kwargs are not
            checked for availability. This is useful if you want to pass unused keyword.

            :param ts: The time series to normalise.
            :type ts: ndarray
            :param args: The args to pass to the norm function.
            :type args: list
            :param kwargs: The kwargs to pass to the norm function.
            :type kwargs: dict
            :return: The normalised time series.
            :rtype: ndarray
            :raises TypeError: type of ts is not ndarray.
            :raises ValueError: If an argument is missing.
            :raises TypeError: If an unknown kwarg is passed.
            :raises ValueError: If the shape of the norm output is not equal to the shape of
                                the input time series.
            :raises ValueError: If the norm output contains NaNs. Only if ``check_nan``
                                is ``True``.
            :raises ValueError: If the norm output contains Infs. Only if ``check_inf``
                                is ``True``.
            """
            # Check if ts is an ndarray
            if not isinstance(ts, ndarray):
                raise TypeError(f"ts must be of type ndarray, not {type(ts)}.")

            bound_args = bind_args(norm_func, [ts, *args], kwargs)
            # Call the norm function with the bound arguments
            normed_ts = norm_func(*bound_args.args, **bound_args.kwargs)

            # Get the shape of the input time series
            shape = ts.shape
            # Check output type
            if not isinstance(normed_ts, ndarray):
                raise ValueError(
                    f"Norm function {norm_func.__name__} must return an ndarray, "
                    f"not {type(normed_ts)}."
                )
            # Check if the shape of the output time series is equal to the
            # shape of the input time series
            if normed_ts.shape != shape:
                raise ValueError(
                    f"Shape of normalised time series ({normed_ts.shape}) "
                    f"does not match shape of input time series ({shape})."
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
                raise ValueError(
                    f"Normalised time series contains "
                    f"{', '.join(val['msg'] for val in checks.values() if val['result'])}: "
                    f"{normed_ts}. "
                    + (
                        f"Input time series contained "
                        f"{', '.join(val['msg'] for val in all_checks.values() if val['result'])}."
                    )
                )
            return normed_ts

        return wrapper

    # Usage without parentheses
    if args and callable(args[0]):
        return norm_outer(args[0])
    # Usage with parentheses
    return norm_outer
