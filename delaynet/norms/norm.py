"""Decorator for the norm functions."""

from functools import wraps
from inspect import signature
from collections.abc import Callable

from numpy import ndarray


def norm(
    norm_func: Callable[[ndarray, ...], ndarray]
) -> Callable[[ndarray, ...], ndarray]:
    """Decorator for the norm functions.

    As the norms take different arguments, this decorator is used to make sure that
    the norms are called with the correct positional and keyword arguments.

    The decorated function only needs to take the ndarray as the first positional
    argument and can take any number of further arguments.

    The decorator checks that all passed keyword arguments exist and that all positional
    arguments are passed.

    Shape of the input and output time series must be equal.

    :param norm_func: The norm function to decorate.
    :type norm_func: Callable
    :return: The decorated function.
    :rtype: Callable
    :raises ValueError: If an argument is missing.
    :raises TypeError: If an unknown kwarg is passed.
    :raises ValueError: If the shape of the norm output is not equal to the shape of
                        the input time series.
    """

    @wraps(norm_func)
    def wrapper(ts: ndarray, *args, **kwargs) -> ndarray:
        """Wrapper for the norm functions.

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
        """
        # Check if ts is an ndarray
        if not isinstance(ts, ndarray):
            raise TypeError(f"ts must be of type ndarray, not {type(ts)}.")
        # Get the signature of the norm function
        sig = signature(norm_func)
        # Bind the arguments to the parameters
        # This will automatically raise a TypeError if a required argument is missing
        # or an unknown argument is passed
        bound_args = sig.bind(ts, *args, **kwargs)
        bound_args.apply_defaults()
        # Get the shape of the input time series
        shape = ts.shape
        # Call the norm function with the bound arguments
        normed_ts = norm_func(*bound_args.args, **bound_args.kwargs)
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
        return normed_ts

    return wrapper
