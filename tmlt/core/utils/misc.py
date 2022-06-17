"""Miscellaneous helper functions and classes."""

# SPDX-License-Identifier: Apache-2.0

import copy
import math
from typing import List, Optional, TypeVar

import numpy as np
from flint import arb  # pylint: disable=no-name-in-module
from pyspark.sql import DataFrame


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
    print(df.sort_values(list(df.columns), ignore_index=True))


def arb_to_float(x: arb) -> Optional[float]:
    """Returns a float corresponding to `x`.

    If `x.lower()` and `x.upper()` do not round the same float, this returns None.
    """
    if not x.is_nan() and x.is_finite():
        lower_man, lower_exp = x.lower().man_exp()
        upper_man, upper_exp = x.upper().man_exp()
        lower_float = math.ldexp(int(lower_man), int(lower_exp))
        upper_float = math.ldexp(int(upper_man), int(upper_exp))
        if lower_float == upper_float:
            return lower_float
    return None


T = TypeVar("T")


def copy_if_mutable(value: T) -> T:
    # pylint: disable=import-outside-toplevel
    """Returns a deep copy of argument if it is mutable."""
    from tmlt.core.utils.testing import IMMUTABLE_TYPES

    if isinstance(value, IMMUTABLE_TYPES):
        # NOTE: See https://github.com/python/mypy/issues/5720
        return value  # type: ignore
    return copy.deepcopy(value)
