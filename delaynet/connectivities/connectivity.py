"""Decorator for the connectivity functions."""

from functools import wraps
from collections.abc import Callable
from inspect import signature

from numpy import ndarray

from ..utils.bind_args import bind_args
from ..utils.multiple_coeff_binning import MultipleCoefficientBinning

Connectivity = Callable[[ndarray, ndarray, ...], float | tuple[float, int]]


def connectivity(
    mcb_kwargs: dict | None = None,
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
    :param mcb_kwargs: Keyword arguments for the :class:`MultipleCoefficientBinning`
                       transformer. If ``None``, no binning is applied.
    :type mcb_kwargs: dict | None
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

        @wraps(connectivity_func)
        def wrapper(
            ts1: ndarray, ts2: ndarray, *args, **kwargs
        ) -> float | tuple[float, int]:
            """Wrapper for the connectivity functions.

            If kwargs have a key ``check_kwargs`` with value ``False``, the kwargs are not
            checked for availability. This is useful if you want to pass unused keyword.

            ``max_lag_steps`` will explicitly be checked, if not ``None``.

            :param ts1: The first time series.
            :type ts1: ndarray
            :param ts2: The second time series.
            :type ts2: ndarray
            :param args: The args to pass to the connectivity function.
            :type args: list
            :param kwargs: The kwargs to pass to the connectivity function.
            :type kwargs: dict
            :return: Connectivity value and lag (if applicable).
            :rtype: float | tuple[float, int]
            :raises TypeError: type of ``ts1`` or ``ts2`` is not ndarray.
            :raises ValueError: If ``ts1`` and ``ts2`` do not have the same shape.
            :raises ValueError: If an argument is missing.
            :raises TypeError: If an unknown kwarg is passed.
            :raises TypeError: If ``max_lag_steps`` is not ``None`` and
                               ``connectivity_func`` does not provide it.
            """
            # Check if ts1 and ts2 are ndarrays
            if not isinstance(ts1, ndarray):
                raise TypeError(f"`ts1` must be of type ndarray, not {type(ts1)}.")
            if not isinstance(ts2, ndarray):
                raise TypeError(f"`ts2` must be of type ndarray, not {type(ts2)}.")
            # Check if ts1 and ts2 have the same shape
            if ts1.shape != ts2.shape:
                raise ValueError(
                    "`ts1` and `ts2` must have the same shape, "
                    f"but have shapes {ts1.shape} and {ts2.shape}."
                )

            # Multiple Coefficient Binning (MCB)
            if mcb_kwargs is not None:
                ts1 = bin_timeseries(ts1, mcb_kwargs)
                ts2 = bin_timeseries(ts2, mcb_kwargs)

            # Check if max_lag_steps in function signature
            if "max_lag_steps" in kwargs and kwargs["max_lag_steps"] is not None:
                if "max_lag_steps" not in signature(connectivity_func).parameters:
                    raise TypeError(
                        f"{connectivity_func.__name__} does not provide finding the "
                        "optimal time lag."
                    )

            bound_args = bind_args(connectivity_func, [ts1, ts2, *args], kwargs)
            # Call the norm function with the bound arguments
            conn_value = connectivity_func(*bound_args.args, **bound_args.kwargs)

            if isinstance(conn_value, float):
                return conn_value
            if isinstance(conn_value, tuple) and len(conn_value) == 2:
                if isinstance(conn_value[0], float) and isinstance(conn_value[1], int):
                    return conn_value[0], conn_value[1]
                raise ValueError(
                    "Invalid return value of connectivity function. "
                    "First value of tuple must be float, second must be int."
                )
            raise ValueError(
                "Metric function must return float or tuple of float and int."
            )

        def bin_timeseries(ts: ndarray, binning_kwargs: dict) -> ndarray:
            """Bin time series using the MultipleCoefficientBinning transformer.

            :param ts: Time series to bin.
            :type ts: ndarray
            :param binning_kwargs: Keyword arguments for
                                   the :class:`MultipleCoefficientBinning` transformer.
            :return: Binned time series.
            :rtype: ndarray
            :raises ValueError: If the binning_kwargs are invalid.
            """
            if not isinstance(binning_kwargs, dict):
                raise ValueError(
                    f"binning_kwargs must be a dict, not {type(binning_kwargs)}."
                )
            transformer = MultipleCoefficientBinning(**binning_kwargs)
            transformer.fit(ts.reshape(-1, 1))
            ts = transformer.transform(ts.reshape(-1, 1))[:, 0]
            return ts

        return wrapper

    return connectivity_outer
