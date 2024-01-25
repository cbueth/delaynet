"""DelayNet init."""
from os.path import dirname, join
import logging.config
from ._version import __version__

# Expose most common functions
from .connectivity import connectivity
from .connectivities import __all_connectivity_metrics_names_simple__
from .normalisation import normalise
from .norms import __all_norms_names_simple__
from .utils.logging import logger

# Set package attributes
__author__ = "Carlson Büth"

logger.info("DelayNet version %s", __version__)
