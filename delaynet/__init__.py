"""DelayNet init."""
import logging
from ._version import __version__

# Expose most common functions
from .connectivity import connectivity
from .connectivities import __all_connectivity_metrics_names_simple__
from .normalisation import normalise
from .norms import __all_norms_names_simple__

# Set package attributes
__author__ = "Carlson Büth"

logging.info("DelayNet version %s", __version__)
