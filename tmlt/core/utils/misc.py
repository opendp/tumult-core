"""Miscellaneous helper functions and classes."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

import copy
from typing import List, TypeVar

import numpy as np
from pyspark.sql import DataFrame

from tmlt.core.utils.type_utils import get_immutable_types


class RNGWrapper:  # pylint: disable=too-few-public-methods
    """Mimics python random interface for discrete gaussian sampling."""

    def __init__(self, rng: np.random.Generator):
        """Constructor.

        Args:
            rng: NumPy random generator.
        """
        self._rng = rng
        self._MAX_INT = int(np.iinfo(np.int64).max)
        assert self._MAX_INT == 2 ** 63 - 1

    def randrange(self, high: int) -> int:
        """Returns a random integer between 0 (inclusive) and `high` (exclusive).

        Args:
            high: upper bound for random integer range.
        """
        # Numpy random.integers only allows high <= MAX_INT
        if high <= self._MAX_INT:
            return int(self._rng.integers(low=0, high=high, endpoint=False))
        # {1} -> 1, {2, 3} -> 2, {4, 5, 6, 7} -> 3, etc
        bits = (high - 1).bit_length()  # only need to represent high - 1, not high
        # Uniformly pick an integer from [0, 2 ** bits - 1].
        random_integer = 0
        while bits >= 63:
            bits -= 63
            random_integer <<= 63
            random_integer += int(
                self._rng.integers(low=0, high=self._MAX_INT, endpoint=True)
            )
        random_integer <<= bits
        random_integer += int(self._rng.integers(low=0, high=2 ** bits, endpoint=False))
        # random_integer may be >= high, but we can try again.
        # Note that this will work at least half of the time.
        if random_integer >= high:
            return self.randrange(high)
        return random_integer


def get_nonconflicting_string(strs: List[str]) -> str:
    """Returns a string distinct from given strings."""
    non_conflicting = []
    for idx, name in enumerate(strs):
        char = name[min(idx, len(name) - 1)]  # Diagonalize
        non_conflicting.append("A" if char != "A" else "B")
    return "".join(non_conflicting)


def print_sdf(sdf: DataFrame) -> None:
    """Prints a spark dataframe in a deterministic way."""
    df = sdf.toPandas()
    # TODO(#2107): Fix typing here
    print(df.sort_values(list(df.columns), ignore_index=True))  # type: ignore


T = TypeVar("T")


def copy_if_mutable(value: T) -> T:
    """Returns a deep copy of argument if it is mutable."""
    if isinstance(value, get_immutable_types()):
        # NOTE: See https://github.com/python/mypy/issues/5720
        return value  # type: ignore
    if isinstance(value, list):
        return [copy_if_mutable(item) for item in value]  # type: ignore
    if isinstance(value, set):
        return {copy_if_mutable(item) for item in value}  # type: ignore
    if isinstance(value, dict):
        return {  # type: ignore
            copy_if_mutable(key): copy_if_mutable(item) for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(copy_if_mutable(item) for item in value)  # type: ignore
    return copy.deepcopy(value)
