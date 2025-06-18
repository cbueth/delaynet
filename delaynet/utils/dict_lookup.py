"""Dict lookup function, used for metrics and norms."""


def dict_lookup(lookup: dict) -> dict:
    """Dict lookup function.

    :param lookup: The lookup dict.
    :type lookup: dict
    :return: The inverted lookup dict.
    :rtype: dict
    :raises TypeError: If the lookup is not a dict.

    :example:
        >>> dict_lookup({"a": 1, "b": 2, "c": 1})
        {1: ["a", "c"], 2: ["b"]}
        >>> dict_lookup({"a": 1, "b": 2, "c": 1, "d": 2})
        {1: ["a", "c"], 2: ["b", "d"]}
        >>> dict_lookup({})
        {}
        >>> dict_lookup(123)
        Traceback (most recent call last):
            ...
        TypeError: Expected dict, got <class 'int'>.
    """
    # Check if the lookup is a dict
    if not isinstance(lookup, dict):
        raise TypeError(f"Expected dict, got {type(lookup)}.")
    # Create the inverted lookup dict
    inverted_lookup = {}
    # Iterate over the lookup dict
    for key, value in lookup.items():
        # Check if the value is already in the inverted lookup dict
        if value in inverted_lookup:
            # Append the key to the list of keys for the value
            inverted_lookup[value].append(key)
        else:
            # Create a new list for the value
            inverted_lookup[value] = [key]
    # Return the inverted lookup dict
    return inverted_lookup
