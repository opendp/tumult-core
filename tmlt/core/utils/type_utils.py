"""Helpers for type introspection and type-checking."""

# SPDX-License-Identifier: Apache-2.0

from typing import Any, NoReturn, Sequence


def assert_never(x: NoReturn) -> NoReturn:
    """Assertion for statically checking exhaustive pattern matches.

    From https://github.com/python/mypy/issues/5818.
    """
    assert False, "Unhandled type: {}".format(type(x).__name__)


def get_element_type(l: Sequence[Any], allow_none: bool = True) -> type:
    """Return the Python type of the non-``None`` elements of a list.

    If the given list is empty or contains elements with multiple types, raises
    ValueError.

    If ``allow_none`` is true (the default), ``None`` values in the list are
    ignored; if the list contains only ``None`` values, ``NoneType`` is
    returned.  If ``allow_none`` is false, raises ValueError if any element of
    the list is ``None``.
    """
    if len(l) == 0:
        raise ValueError("cannot determine element type of empty list")
    types = {type(e) for e in l}
    if not allow_none and type(None) in types:
        raise ValueError("None is not allowed")
    types -= {type(None)}
    if len(types) == 0:
        return type(None)
    elif len(types) > 1:
        raise ValueError(
            "list contains elements of multiple types "
            f"({','.join(t.__name__ for t in types)})"
        )
    else:
        (list_type,) = types
        return list_type
