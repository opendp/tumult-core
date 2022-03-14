"""Helper functions for mypy related things."""

# <placeholder: boilerplate>

from typing import NoReturn


def assert_never(x: NoReturn) -> NoReturn:
    """Assertion for statically checking exhaustive pattern matches.

    From https://github.com/python/mypy/issues/5818.
    """
    assert False, "Unhandled type: {}".format(type(x).__name__)
