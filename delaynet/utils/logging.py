"""Logging configuration for DelayNet."""

import logging

# Get the logger for this module with NullHandler
logging.getLogger("delaynet").addHandler(logging.NullHandler())
logging.basicConfig(
    format="%(asctime)s | %(levelname)8s | %(filename)s:%(lineno)d | %(message)s",
    level=logging.INFO,
)
