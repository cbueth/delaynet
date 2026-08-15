"""Regression tests for the benchmark skip logic in ``tests/benchmarks/conftest.py``.

Benchmark tests must be skipped when ``--codspeed`` is not passed, and this must
work even when the ``pytest-codspeed`` plugin (which provides the ``benchmark``
fixture and registers the ``benchmark`` mark) is not installed -- e.g. when a
downstream consumer such as conda-forge runs ``pytest tests/``.
"""

import pytest

from .conftest import _is_benchmark_item, pytest_collection_modifyitems


class _StubItem:
    """Minimal stand-in for a pytest item to exercise the collection hook."""

    def __init__(self, fixturenames):
        self.fixturenames = fixturenames
        self.own_markers = []

    def add_marker(self, marker):
        self.own_markers.append(marker)


def test_is_benchmark_item_detects_benchmark_fixture():
    """A test requesting the ``benchmark`` fixture is identified as a benchmark."""
    # Arrange
    item = _StubItem(["benchmark", "continuous_data"])
    # Act
    is_benchmark = _is_benchmark_item(item)
    # Assert
    assert is_benchmark


def test_is_benchmark_item_false_without_benchmark_fixture():
    """A test without a benchmark fixture is not identified as a benchmark."""
    # Arrange
    item = _StubItem(["continuous_data"])
    # Act
    is_benchmark = _is_benchmark_item(item)
    # Assert
    assert not is_benchmark


def test_benchmark_item_skipped_without_codspeed(pytestconfig, monkeypatch):
    """Without ``--codspeed``, benchmark tests are marked to be skipped."""
    # Arrange
    monkeypatch.setattr(pytestconfig, "getoption", lambda name, default=None: default)
    item = _StubItem(["benchmark", "continuous_data"])
    # Act
    pytest_collection_modifyitems(pytestconfig, [item])
    # Assert
    assert any(m.mark.name == "skip" for m in item.own_markers)


def test_benchmark_item_not_skipped_with_codspeed(pytestconfig, monkeypatch):
    """With ``--codspeed``, benchmark tests are not skipped."""
    # Arrange
    monkeypatch.setattr(
        pytestconfig, "getoption", lambda name, default=None: name == "codspeed"
    )
    item = _StubItem(["benchmark", "continuous_data"])
    # Act
    pytest_collection_modifyitems(pytestconfig, [item])
    # Assert
    assert not any(m.mark.name == "skip" for m in item.own_markers)
