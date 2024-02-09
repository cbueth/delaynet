"""Function to bind args for the decorators of the connectivity and norm functions."""

from inspect import signature, BoundArguments


def bind_args(func: callable, args: list, kwargs: dict) -> BoundArguments:
    """Bind the arguments to the parameters of the function.

    This will automatically raise a TypeError if a required argument is missing
    or an unknown argument is passed.

    If kwargs have a key ``check_kwargs`` with value ``False``, the kwargs are not
    checked for availability. This is useful if you want to pass unused keyword.

    :param func: The function to bind the arguments to.
    :type func: callable
    :param args: The positional arguments to bind.
    :type args: list
    :param kwargs: The keyword arguments to bind.
    :type kwargs: dict
    :return: The bound arguments.
    :rtype: BoundArguments
    :raises TypeError: If a required argument is missing.
    :raises TypeError: If an unknown kwarg is passed.
    """
    # Get the signature of the function
    sig = signature(func)
    # Bind the arguments to the parameters
    # This will automatically raise a TypeError if a required argument is missing
    # or an unknown argument is passed
    if "check_kwargs" in kwargs:
        if not kwargs["check_kwargs"]:
            # Filter out kwargs that are not in the function's signature
            kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        else:
            kwargs.pop("check_kwargs")
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    return bound_args
