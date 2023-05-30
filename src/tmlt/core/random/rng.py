"""Tumult Core's random number generator."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

import os

import numpy as np
from randomgen.rdrand import RDRAND  # pylint: disable=no-name-in-module
from randomgen.wrapper import UserBitGenerator  # pylint: disable=no-name-in-module

try:
    _core_privacy_prng = np.random.Generator(RDRAND())
except RuntimeError:

    def _random_raw(_) -> int:
        return int.from_bytes(os.urandom(8), "big")

    _core_privacy_prng = np.random.Generator(UserBitGenerator(_random_raw, 64))


def prng():
    """Getter for prng."""
    return _core_privacy_prng
