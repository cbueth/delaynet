"""Norms init, subpackage of delaynet."""

# Import all norms
from .delta import delta
from .identity import identity
from .second_difference import second_difference
from .z_score import z_score

from ..utils.dict_lookup import dict_lookup

# Named norms
__all_norms_names__ = {
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

# List of all available norms
__all_norms__ = set(__all_norms_names__.values())

# Extend named norms with the function name
# e.g. adds "second_difference": second_difference
for norm in __all_norms__:
    __all_norms_names__[norm.__name__] = norm

# Convenient name dict: "norm.__name__": ["norm", "norm short", ...]
# shows all names that point to the same nor,
__all_norms_names_simple__ = dict_lookup(__all_norms_names__)
__all_norms_names_simple__ = {
    metric.__name__: names for metric, names in __all_norms_names_simple__.items()
}
