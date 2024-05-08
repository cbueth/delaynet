"""Function to check if numpy array(s) are symbolic.
Used for entropy-based connectivity metrics.
"""

from operator import gt
from collections.abc import Callable

from numpy import unique, concatenate, ndarray, quantile, linspace, digitize

from delaynet.utils.logging import logging


def check_symbolic_pairwise(
    array1: ndarray, array2: ndarray, max_symbols: int | None = None
) -> None:
    """
    Check if two numpy arrays are symbolic.

    Accepted are integer arrays, signed or unsigned, but not float arrays.

    Together, both arrays should not have more than ``max_symbols`` unique symbols.

    :param array1: The first numpy array to check.
    :type array1: numpy.ndarray
    :param array2: The second numpy array to check.
    :type array2: numpy.ndarray
    :param max_symbols: The maximum number of unique symbols allowed.
    :type max_symbols: int
    :raises ValueError: If the arrays are not symbolic.
    :raises ValueError: If ``max_symbols`` is <= 0.
    :raises ValueError: If the arrays have more than ``max_symbols`` unique symbols.

    """
    if array1.dtype.kind == "f" or array2.dtype.kind == "f":
        logging.error(
            "Set `symbolic_bins` in connectivity() to explicitly convert to symbolic."
        )
        raise ValueError("Input arrays cannot be of float type.")
    if array1.dtype.kind in "iu" and array2.dtype.kind in "iu":
        if max_symbols is None:
            return  # If no limit is set, return
        if max_symbols <= 0:
            raise ValueError("max_symbols must be greater than 0.")
        unique_symbols = unique(concatenate((unique(array1), unique(array2))))
        if len(unique_symbols) > max_symbols:
            raise ValueError(
                f"Input arrays have more than {max_symbols} unique symbols."
            )
    else:
        raise ValueError("Input arrays must be of integer type.")


def to_symbolic(
    array: ndarray,
    method: str = "quantilize",
    **kwargs,
) -> ndarray[int]:
    """
    Convert a numpy array to symbolic based on the specified method.

    Method can be one of:

    - :py:func:`quantize`: Convert to symbolic by digitizing into ``max_symbols`` bins.
    - :py:func:`quantilize`: Convert to symbolic by quantilizing into ``num_quantiles``
      bins.
    - :py:func:`binarize`: Convert to binary based on a threshold.
    - :py:func:`round_to_int`: Round to the nearest integer.

    :param array: The numpy array to convert.
    :type array: numpy.ndarray
    :param method: The method to use for conversion.
    :type method: str
    :param kwargs: Additional keyword arguments for the method.
    :return: The symbolic numpy array.
    """
    if method == "quantize":
        return quantize(array, **kwargs)
    if method == "quantilize":
        return quantilize(array, **kwargs)
    if method == "binarize":
        return binarize(array, **kwargs)
    if method == "round_to_int":
        return round_to_int(array)
    raise ValueError("Invalid method for symbolic conversion.")


def round_to_int(array: ndarray) -> ndarray[int]:
    """
    Round a numpy array to the nearest integer.

    :param array: The numpy array to round.
    :type array: numpy.ndarray
    :return: The rounded numpy array.
    :rtype: numpy.ndarray

    :example:
    >>> round_to_int(array([1.1, 2.2, 3.3]))
    array([1, 2, 3])
    """
    logging.debug("Rounding to nearest integer.")
    return array.round().astype(int)


def quantize(array: ndarray, max_symbols: int) -> ndarray[int]:
    """
    Convert a numpy array to symbolic by digitizing it into ``max_symbols`` bins.
    Digitizes the array into ``max_symbols`` bins on [0, max_symbols-1].

    :param array: The numpy array to convert. Must be numeric.
    :type array: numpy.ndarray
    :param max_symbols: The maximum number of unique symbols allowed.
    :type max_symbols: int
    :return: The symbolic numpy array.
    :rtype: numpy.ndarray
    :raises ValueError: If ``max_symbols`` is not a positive integer.
    :raises ValueError: If the array is not numeric.

    :example:
    >>> quantize(array([1.0, 2.0, 3.0, 4.0]), 2)
    array([0, 0, 1, 1])
    """
    if not (isinstance(max_symbols, int) and max_symbols > 0):
        raise ValueError("max_symbols must be a positive integer.")
    # Array needs to be numeric, (uint, int, float)
    if array.dtype.kind not in "iuf":
        raise ValueError("Input array must be of numeric type to convert.")

    logging.debug(f"Converting to symbolic with max_symbols={max_symbols}.")
    return (
        (  # Stretch the array linearly to [0, max_symbols-1]
            (array - array.min()) / (array.max() - array.min()) * (max_symbols - 1)
        )
        .round()
        .astype(int)
    )


def quantilize(array: ndarray, num_quantiles: int = 4) -> ndarray[int]:
    """
    Convert a numpy array to symbolic by quantilizing it into ``num_quantiles`` bins.

    This function works similar to quantize, but instead of linearly scaling and
    categorizing the array into equal bins, it uses quantiles to categorize the array.

    A rank-based method to convert the array into symbolic form.

    :param array: The numpy array to convert. Must be numeric.
    :type array: numpy.ndarray
    :param num_quantiles: The number of quantiles to use for categorization.
                          By default, the four quartiles are used.
    :type num_quantiles: int
    :return: The symbolic numpy array.
    :rtype: numpy.ndarray
    :raises ValueError: If ``num_quantiles`` is not a positive integer.
    :raises ValueError: If the array is not numeric.

    :example:
    >>> quantilize(array([1, 2, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7]), 3)
    array([0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2])
    """
    if not (isinstance(num_quantiles, int) and num_quantiles > 0):
        raise ValueError("num_quantiles must be a positive integer.")
    if array.dtype.kind not in "iuf":
        raise ValueError("Input array must be of numeric type to convert.")

    logging.debug(f"Converting to symbolic with num_quantiles={num_quantiles}.")
    bins = quantile(array, linspace(0, 1, num_quantiles + 1))[:-1]
    return digitize(array, bins, right=False)


def binarize(
    array: ndarray,
    threshold: float | Callable[[ndarray], float],
    operator: Callable[[ndarray, float], ndarray] = gt,
) -> ndarray[int]:
    """
    Convert a numpy array to binary based on a threshold.

    The threshold can be a float or a callable that takes the array as input,
    e.g., ``lambda x: median(x)`` or ``lambda x: x.mean()``.
    By default, the operator is greater than (gt).

    :param array: The numpy array to convert.
    :type array: numpy.ndarray
    :param threshold: The threshold to binarize the array.
    :type threshold: float or collections.abc.Callable[[numpy.ndarray], float]
    :param operator: The operator to use for binarization.
    :type operator: collections.abc.Callable[[numpy.ndarray, float], numpy.ndarray]
    :return: The binary numpy array.
    :rtype: numpy.ndarray

    :example:
    >>> binarize(array([1.0, 2.0, 3.0, 4.0]), 1.5)
    array([0, 1, 1, 1])
    >>> binarize(array([1.0, 2.0, 3.0, 4.0]), lambda x: x.mean())
    array([0, 0, 1, 1])
    >>> binarize(array([1.0, 2.0, 3.0, 4.0]), 2.5, operator=lt)
    array([1, 1, 0, 0])
    """
    if callable(threshold):
        threshold = threshold(array)
    logging.debug(f"Binarizing with threshold={threshold}.")
    return operator(array, threshold).astype(int)
