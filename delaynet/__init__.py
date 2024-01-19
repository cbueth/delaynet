"""DelayNet init."""
from os.path import dirname, join
import logging.config
from ._version import __version__

# Expose most common functions
from .causality import causality
from .causalities import __all_causality_metrics_names_simple__
from .normalisation import normalise
from .norms import __all_norms_names_simple__

# Set package attributes
__author__ = "Carlson Büth"

# Logging configuration using the setup.cfg file
logging.config.fileConfig(join(dirname(__file__), "..", "setup.cfg"))
# Get the logger for this module
logger = logging.getLogger("delaynet")
# numba_logger = logging.getLogger("numba")
# numba_logger.setLevel(logging.WARNING)

logger.info("DelayNet version %s", __version__)
