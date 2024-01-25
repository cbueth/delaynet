"""Logging configuration for DelayNet."""

import logging.config
from os.path import dirname, join

# Logging configuration using the setup.cfg file
logging.config.fileConfig(join(dirname(__file__), "..", "..", "setup.cfg"))
# Get the logger for this module
logger = logging.getLogger("delaynet")
# numba_logger = logging.getLogger("numba")
# numba_logger.setLevel(logging.WARNING)
