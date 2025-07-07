"""Detrending methods init, subpackage of delaynet."""

# Import all detrending methods
from .delta import delta
from .identity import identity
from .second_difference import second_difference
from .z_score import z_score

from ..utils.dict_lookup import dict_lookup

# Named detrending methods
__all_detrending_names__ = {
    "delta": delta,
    "dt": delta,
    "identity": identity,
    "id": identity,
    "second difference": second_difference,
    "2dt": second_difference,
    "z-score": z_score,
    "zs": z_score,
    "zscore": z_score,
}

# List of all available detrending methods
__all_detrending__ = set(__all_detrending_names__.values())

# Extend named detrending methods with the function name
# e.g. adds "second_difference": second_difference
for method in __all_detrending__:
    __all_detrending_names__[method.__name__] = method

# Convenient name dict: "method.__name__": ["method", "method short", ...]
# shows all names that point to the same nor,
__all_detrending_names_simple__ = dict_lookup(__all_detrending_names__)
__all_detrending_names_simple__ = {
    metric.__name__: names for metric, names in __all_detrending_names_simple__.items()
}
