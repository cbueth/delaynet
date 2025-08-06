"""Data preparation subpackage for delaynet."""

from .data_generator import (
    gen_delayed_causal_network,
    gen_fmri,
    gen_fmri_multiple,
    gen_synthatdelays_random_connectivity,
    gen_synthatdelays_independent_operations_with_trends,
    extract_airport_delay_time_series,
)
