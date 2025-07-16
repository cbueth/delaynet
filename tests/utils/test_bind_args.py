"""Tests for the bind_args function in bind_args.py."""

import pytest
from inspect import signature
from delaynet.utils.bind_args import bind_args


def test_bind_args_basic():
    """Test basic functionality of bind_args."""
    def example_func(a, b, c=3):
        return a + b + c

    # Test with positional args
    bound = bind_args(example_func, [1, 2], {})
    assert len(bound.args) == 3
    assert bound.args[:2] == (1, 2)
    assert bound.args[2] == 3
    assert bound.kwargs == {}

    # Test with keyword args
    bound = bind_args(example_func, [1], {'b': 2, 'c': 4})
    assert len(bound.args) == 3
    assert bound.args[0] == 1
    assert bound.args[1] == 2
    assert bound.args[2] == 4
    assert bound.kwargs == {}


def test_bind_args_check_kwargs_false():
    """Test bind_args with check_kwargs=False."""
    def example_func(a, b, c=3):
        return a + b + c

    # Test with check_kwargs=False
    # This should filter out 'd' which is not in the function signature
    bound = bind_args(example_func, [1, 2], {'d': 4, 'check_kwargs': False})
    assert len(bound.args) == 3
    assert bound.args[:2] == (1, 2)
    assert bound.args[2] == 3
    assert bound.kwargs == {}
    # The 'd' and 'check_kwargs' should have been filtered out
    assert not hasattr(bound, 'd')
    assert not hasattr(bound, 'check_kwargs')


def test_bind_args_check_kwargs_true():
    """Test bind_args with check_kwargs=True."""
    def example_func(a, b, c=3):
        return a + b + c

    # Test with check_kwargs=True
    # This should remove check_kwargs from kwargs but keep other kwargs
    bound = bind_args(example_func, [1, 2], {'c': 4, 'check_kwargs': True})
    assert len(bound.args) == 3
    assert bound.args[:2] == (1, 2)
    assert bound.args[2] == 4  # c is now 4 instead of the default 3
    assert bound.kwargs == {}
    # check_kwargs should have been removed
    assert not hasattr(bound, 'check_kwargs')


def test_bind_args_missing_required_arg():
    """Test bind_args with missing required argument."""
    def example_func(a, b, c=3):
        return a + b + c

    # Missing required argument 'b'
    with pytest.raises(TypeError):
        bind_args(example_func, [1], {})


def test_bind_args_unknown_kwarg():
    """Test bind_args with unknown keyword argument."""
    def example_func(a, b, c=3):
        return a + b + c

    # Unknown keyword argument 'd'
    with pytest.raises(TypeError):
        bind_args(example_func, [1, 2], {'d': 4})
