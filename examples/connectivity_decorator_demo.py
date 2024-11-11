"""Technical demonstration of how the connectivity decorator works.

Re-builds the connectivity decorator from scratch, with some pre-processing
options.
"""

from collections.abc import Callable
from functools import wraps

from numpy import ndarray, array
from scipy.stats import pearsonr

Connectivity = Callable[[ndarray, ndarray, ...], float | tuple[float, int]]


def connectivity(*args, pre_processing_info: int | None = None) -> Callable:
    """Connectivity decorator."""

    def decorate(connectivity_func: Connectivity) -> Connectivity:
        @wraps(connectivity_func)
        def wrapper(
            ts1: ndarray, ts2: ndarray, *args, **kwargs
        ) -> float | tuple[float, int]:
            if pre_processing_info is not None:
                print(f"Pre-processing info: {pre_processing_info}")
            return connectivity_func(ts1, ts2, *args, **kwargs)

        return wrapper

    # Usage without parentheses
    if args and callable(args[0]):
        return decorate(args[0])

    # Usage with parentheses
    return decorate


@connectivity(pre_processing_info=5)
def linear_correlation(x: ndarray, y: ndarray):
    """Simplified linear correlation metric."""
    print("Calculating linear correlation...")
    stat = pearsonr(x, y)
    print(f"Linear correlation: {stat[0]}" f" (p-value: {stat[1]})")
    return stat[0]


@connectivity
def dot_product(x: ndarray, y: ndarray):
    """Simplified dot product metric with no pre-processing info."""
    print("Calculating dot product...")
    stat = x @ y
    print(f"Dot product: {stat}")
    return stat


__method_dict__ = {"lc": linear_correlation, "dp": dot_product}


if __name__ == "__main__":
    series1 = array([1, 2, 3, 4, 5])
    series2 = array([1, 2, 3, 4, 5])

    # Call wrapped connectivity function
    print(linear_correlation(series1, series2))
    # Call connectivity function from dict
    print(__method_dict__["lc"](series1, series2))

    # Call wrapped connectivity function
    print(dot_product(series1, series2))
    # Call connectivity function from dict
    print(__method_dict__["dp"](series1, series2))
