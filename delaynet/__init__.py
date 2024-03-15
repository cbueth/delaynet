"""DelayNet init."""

from ._version import __version__
from .utils.logging import logging

# Expose most common functions
from .connectivity import connectivity, show_connectivity_metrics
from .normalisation import normalise, show_norms
from . import preparation

# Set package attributes
__author__ = "Carlson Büth"

logging.info("DelayNet version %s", __version__)
