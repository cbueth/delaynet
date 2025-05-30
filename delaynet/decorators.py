"""Decorators for the delaynet package."""

from functools import wraps
from collections.abc import Callable
from inspect import signature

from numpy import ndarray, isnan, isinf, apply_along_axis, floating, integer

from .utils.bind_args import bind_args
from .utils.multi_coeff_binning import MultipleCoefficientBinning
from .preprocess.symbolic import check_symbolic_pairwise, to_symbolic

Connectivity = Callable[[ndarray, ndarray, ...], float | tuple[float, int]]
Norm = Callable[[ndarray, ...], ndarray]


def connectivity(
    *args,
    entropy_like: bool = False,  # TODO: remove this functionality to make package usage more explicit
    check_symbolic: bool | int | None = False,
    default_to_symbolic: dict
    | None = None,  # TODO: remove this functionality to make package usage more explicit
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

    :param entropy_like: If ``True``, mark the connectivity as entropy-like.
    :type entropy_like: bool
    :param check_symbolic: If ``True``, check if the connectivity values are symbolic.
                           A specific number of unique symbols can be set (``None`` for
                           no limit).
                           Necessary for entropy-based connectivities.
    :type check_symbolic: bool | int
    :param default_to_symbolic: Default configuration for converting to symbolic.
                                Can be overridden by the ``symbolic_conversion``
                                argument, when calling the decorated function.
    :type default_to_symbolic: dict
    :param mcb_kwargs: Keyword arguments for the :py:class:`MultipleCoefficientBinning`
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
            ts1: ndarray,
            ts2: ndarray,
            *args,
            symbolic_conversion: dict | None = None,
            **kwargs,
        ) -> float | tuple[float, int]:
            """Wrapper for the connectivity functions.

            If kwargs have a key ``check_kwargs`` with value ``False``, the kwargs are
            not checked for availability.
            This is useful if you want to pass unused keyword.

            ``max_lag_steps`` will explicitly be checked, if not ``None``.

            :param ts1: The first time series.
            :type ts1: numpy.ndarray
            :param ts2: The second time series.
            :type ts2: numpy.ndarray
            :param args: The args to pass to the connectivity function.
            :type args: list
            :param symbolic_conversion: Keyword arguments for converting to symbolic.
                                        Overrides the default configuration.
            :type symbolic_conversion: dict | None
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

            if symbolic_conversion not in [None, {}]:
                if not isinstance(symbolic_conversion, dict):
                    raise TypeError(
                        f"`symbolic_conversion` must be a dict, not "
                        f"{type(symbolic_conversion)}."
                    )
                ts1 = to_symbolic(ts1, **symbolic_conversion)
                ts2 = to_symbolic(ts2, **symbolic_conversion)
            elif default_to_symbolic not in [None, {}]:
                ts1 = to_symbolic(ts1, **default_to_symbolic)
                ts2 = to_symbolic(ts2, **default_to_symbolic)

            # Check if the time series are symbolic
            if check_symbolic or check_symbolic is None:
                check_symbolic_pairwise(
                    ts1,
                    ts2,
                    max_symbols=None if check_symbolic is True else check_symbolic,
                    # if check_symbolic is True, no limit is set
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

            if isinstance(conn_value, (float, floating)):
                return conn_value
            if isinstance(conn_value, tuple) and len(conn_value) == 2:
                if isinstance(conn_value[0], (float, floating)) and isinstance(
                    conn_value[1], (int, integer)
                ):
                    return conn_value[0], conn_value[1]
                raise ValueError(
                    "Invalid return value of connectivity function. "
                    "First value of tuple must be float, second must be int. "
                    f"Got {type(conn_value[0])} and {type(conn_value[1])}."
                )
            raise ValueError(
                "Metric function must return float or tuple of float and int."
            )

        def bin_timeseries(ts: ndarray, binning_kwargs: dict) -> ndarray:
            """Bin time series using the MultipleCoefficientBinning transformer.

            :param ts: Time series to bin.
            :type ts: numpy.ndarray
            :param binning_kwargs: Keyword arguments for
                                   the :class:`MultipleCoefficientBinning` transformer.
            :return: Binned time series.
            :rtype: numpy.ndarray
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

        # Mark connectivity as entropy-likeness
        wrapper.is_entropy_like = entropy_like

        return wrapper

    # Usage without parentheses (@connectivity)
    if args and callable(args[0]):
        return connectivity_outer(args[0])
    # Usage with parentheses (@connectivity(), @connectivity(...))
    return connectivity_outer


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
            :type ts: numpy.ndarray, shape (n,) or (m, n)
            :param args: The args to pass to the norm function.
            :type args: list
            :param kwargs: The kwargs to pass to the norm function.
            :type kwargs: dict
            :return: The normalised time series.
            :rtype: numpy.ndarray
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
            else:  # must be 2D, due to previous check
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

    # Usage without parentheses (@norm)
    if args and callable(args[0]):
        return norm_outer(args[0])
    # Usage with parentheses (@norm(), @norm(...))
    return norm_outer
