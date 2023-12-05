"""General tests for DelayNet module."""

import delaynet


def test_version():
    """Test version.

    Compare delaynet.__version__ to version in delaynet/_version.py.
    """
    # read from relative path - find line '__version__ = "0.0.0"'
    with open("delaynet/_version.py", "r", encoding="utf-8") as f:
        # get line that starts with '__version__ = "'
        line = next(line for line in f if line.startswith('__version__ = "'))
        # get version string
        ver = line.split('"')[1]
    assert delaynet.__version__ == ver


def test_fixed_attributes():
    """Test package attributes set in delaynet/__init__.py."""
    assert delaynet.__name__ == "delaynet"
    assert delaynet.__author__ == "Carlson Büth"
