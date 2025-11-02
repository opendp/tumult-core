"""Tumult Core's random number generator."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import os
from typing import Any

import numpy as np
from randomgen.rdrand import RDRAND  # pylint: disable=no-name-in-module
from randomgen.wrapper import UserBitGenerator  # pylint: disable=no-name-in-module

try:
    _core_privacy_prng = np.random.Generator(RDRAND())
except RuntimeError:

    def _random_raw(_: Any) -> int:
        return int.from_bytes(os.urandom(8), "big")

    _core_privacy_prng = np.random.Generator(UserBitGenerator(_random_raw, 64))


def prng() -> np.random.Generator:
    """Getter for prng."""
    return _core_privacy_prng


class RNGWrapper:
    """Mimics Python ``random`` interface for discrete Gaussian sampling."""

    def __init__(self, rng: np.random.Generator):
        """Constructor.

        Args:
            rng: NumPy random generator.
        """
        self._rng = rng
        self._MAX_INT = int(np.iinfo(np.int64).max)
        assert self._MAX_INT == 2**63 - 1

    def randrange(self, stop: int) -> int:
        """Returns a random integer between 0 (inclusive) and ``stop`` (exclusive).

        Args:
            stop: upper bound for random integer range.
        """
        # Numpy random.integers only allows high <= MAX_INT
        if stop <= self._MAX_INT:
            return int(self._rng.integers(low=0, high=stop, endpoint=False))
        # {1} -> 1, {2, 3} -> 2, {4, 5, 6, 7} -> 3, etc
        bits = (stop - 1).bit_length()  # only need to represent high - 1, not high
        # Uniformly pick an integer from [0, 2 ** bits - 1].
        random_integer = 0
        while bits >= 63:
            bits -= 63
            random_integer <<= 63
            random_integer += int(
                self._rng.integers(low=0, high=self._MAX_INT, endpoint=True)
            )
        random_integer <<= bits
        random_integer += int(self._rng.integers(low=0, high=2**bits, endpoint=False))
        # random_integer may be >= high, but we can try again.
        # Note that this will work at least half of the time.
        if random_integer >= stop:
            return self.randrange(stop)
        return random_integer
