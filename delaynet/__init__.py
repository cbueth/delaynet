"""DelayNet init."""
from ._version import __version__
from .utils.logging import logging

# Expose most common functions
from .connectivity import connectivity
from .connectivities import __all_connectivity_metrics_names_simple__
from .normalisation import normalise
from .norms import __all_norms_names_simple__

# Set package attributes
__author__ = "Carlson Büth"

logging.info("DelayNet version %s", __version__)
