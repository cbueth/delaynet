"""Module to provide unified interface for all detrending methods."""

from collections.abc import Callable

from numpy import ndarray

from .detrending_methods import __all_detrending_names__, __all_detrending_names_simple__
from .decorators import detrending_method


def detrend(
    ts: ndarray,
    /,
    method: str | Callable[[ndarray, ...], ndarray],
    *args,
    axis: int = None,
    **kwargs,
) -> ndarray:
    """
    Detrend time series using a given detrending method.

    Keyword arguments are forwarded to the detrend function.

    If ``check_kwargs`` is passed in kwargs with value ``False``, the kwargs are not
    checked for availability. This is useful if you want to pass unused values in
    generic functions.

    The detrending methods can be either a string or a function,
    implementing a detrending method.
    Find the method string specifier using :func:`show_detrending_methods`.

    (Find all in submodule :mod:`delaynet.detrending_methods`, names are stored in
    :attr:`delaynet.detrending_methods.__all_detrending__`)

    If a ``callable`` is given, it should take a time series as input and return
    the Detrended time series.

    :param ts: Time series to detrend. 1D or 2D. Positional only.
    :type ts: numpy.ndarray
    :param method: Method to use for detrending.
    :type method: str or Callable
    :param args: Positional arguments forwarded to the detrending function,
                 see documentation of the detrending methods.
    :type args: list
    :param axis: Axis to detrend along. Needs to be passed for ts of higher
                 dimensions. If 1D ts, ``axis`` is ignored.
    :param axis: int | None, optional
    :param kwargs: Keyword arguments forwarded to the detrending function,
                   see documentation of the detrending methods.
    :type kwargs: dict
    :return: Detrended time series.
    :rtype: numpy.ndarray
    :raises ValueError: If the method is unknown. Given as string.
    :raises ValueError: If the method returns an invalid value. Given a Callable.
    :raises ValueError: If the method is neither a string nor a Callable.
    :raises ValueError: If the shape of the method output is not equal to the shape of
                        the input time series.
    """
    if axis is not None:
        kwargs["axis"] = axis

    if isinstance(method, str):
        method = method.lower()

        if method not in __all_detrending_names__:
            raise ValueError(f"Unknown detrending method: {method}")

        return __all_detrending_names__[method](ts, *args, **kwargs)

    if not callable(method):
        raise ValueError(
            f"Unknown detrending method: {method}."
            "Must be either a string or a callable."
        )

    # method is callable, add decorator to assure correct kwargs, type and shape
    return detrending_method(method)(ts, *args, **kwargs)


def show_detrending_methods():
    """Pretty print all available detrending methods."""
    print("Available detrending methods:")
    for method, aliases in __all_detrending_names_simple__.items():
        print(f"\nDetrending method: {method}")
        print("Aliases:")
        for alias in aliases:
            print(f" - {alias}")
    print("\n")
