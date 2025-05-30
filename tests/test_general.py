"""General tests for delaynet module."""

import delaynet


def test_fixed_attributes():
    """Test package attributes set in delaynet/__init__.py."""
    assert delaynet.__name__ == "delaynet"
    assert delaynet.__author__ == "Carlson Büth"
