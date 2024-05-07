"""Function to check if numpy array(s) are symbolic.
Used for entropy-based connectivity metrics.
"""

from numpy import unique, concatenate, ndarray

from ..utils.logging import logging


def check_symbolic_pairwise(
    array1: ndarray, array2: ndarray, max_symbols: int | None = None
) -> None:
    """
    Check if two numpy arrays are symbolic.

    Accepted are integer arrays, signed or unsigned, but not float arrays.

    Together, both arrays should not have more than ``max_symbols`` unique symbols.

    :param array1: The first numpy array to check.
    :type array1: ndarray
    :param array2: The second numpy array to check.
    :type array2: ndarray
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


def to_symbolic(array: ndarray[float], max_symbols: int | None = None) -> ndarray[int]:
    """
    Convert a numpy array to symbolic. (float to int)

    Converts float arrays to integer arrays.
    Either rounding to the next integer, or when ``max_symbols`` is set,
    digitizing the array into ``max_symbols`` bins on [0, max_symbols-1].

    :param array: The numpy array to convert.
    :type array: ndarray
    :param max_symbols: The maximum number of unique symbols allowed.
                        Default is None, which means no limit.
    :type max_symbols: int | None
    :return: The symbolic numpy array.
    :rtype: ndarray
    :raises ValueError: If ``max_symbols`` is <= 0.
    :raises ValueError: If the array is not of float type.
    """
    if max_symbols is not None and max_symbols <= 0:
        raise ValueError("max_symbols must be greater than 0.")
    if array.dtype.kind == "f":
        if max_symbols is None:
            logging.warning("Converting to symbolic, only rounding to integer.")
            return array.round().astype(int)
        logging.warning(f"Converting to symbolic with max_symbols={max_symbols}.")
        return (
            (  # Stretch the array linearly to [0, max_symbols-1]
                (array - array.min()) / (array.max() - array.min()) * (max_symbols - 1)
            )
            .round()
            .astype(int)
        )
    raise ValueError("Input array must be of float type to convert.")
