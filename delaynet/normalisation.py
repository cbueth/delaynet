"""Module to provide unified interface for all norms."""
from collections.abc import Callable

from numpy import ndarray

from .norms import __all_norms_names__
from .norms.norm import norm as norm_decorator


def normalise(
    ts: ndarray,
    norm: str | Callable[[ndarray, ...], ndarray],
    *args,
    **kwargs,
) -> ndarray:
    """
    Normalise a time series using a given norm.

    Keyword arguments are forwarded to the norm function.

    If `check_kwargs` is passed in kwargs with value `False`, the kwargs are not
    checked for availability. This is useful if you want to pass unused values in
    generic functions.

    The norms can be either a string or a function, implementing a norm.
    The following norms are available (case-insensitive):
        - ZS: Z-Score
        - DT: Delta
        - 2DT: Second Difference
        - ID: Identity

    (Find all in submodule `delaynet.norms`, names are stored in
    `delaynet.norms.__all_norms__`)

    If a `callable` is given, it should take a time series as input and return
    the normalised time series.

    :param ts: Time series to normalise.
    :type ts: ndarray
    :param norm: Norm to use.
    :type norm: str or Callable
    :param args: Positional arguments forwarded to the norm function, see documentation
                 of the norms.
    :type args: tuple
    :param kwargs: Keyword arguments forwarded to the norm function, see documentation
                   of the norms.
    :type kwargs: dict
    :return: Normalised time series.
    :rtype: ndarray
    :raises ValueError: If the norm is unknown. Given as string.
    :raises ValueError: If the norm returns an invalid value. Given a Callable.
    :raises ValueError: If the norm is neither a string nor a Callable.
    :raises ValueError: If the shape of the norm output is not equal to the shape of
                        the input time series.
    """

    if isinstance(norm, str):
        norm = norm.lower()

        if norm not in __all_norms_names__:
            raise ValueError(f"Unknown norm: {norm}")

        return __all_norms_names__[norm](ts, *args, **kwargs)

    if not callable(norm):
        raise ValueError(
            f"Unknown norm: {norm}. Must be either a string or a callable."
        )

    # norm is callable, add decorator to assure correct kwargs, type and shape
    return norm_decorator(norm)(ts, *args, **kwargs)
