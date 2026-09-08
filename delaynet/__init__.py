"""delaynet init."""

__version__ = "0.3.4"
from .utils.logging import logging

# Expose most common functions
from .connectivity import connectivity, show_connectivity_metrics
from .detrending import detrend, show_detrending_methods
from .network_reconstruction import reconstruct_network
from . import preparation
from . import network_analysis

# Set package attributes
__author__ = "Carlson Büth"

logging.debug("delaynet version %s", __version__)
