"""Identity / Normalization."""

from .norm import norm


@norm
def identity(vol_data):
    return vol_data
