"""Function to check if numpy array(s) are symbolic.
Used for entropy-based connectivity metrics.
"""

from numpy import unique, concatenate, ndarray


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
