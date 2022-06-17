"""Tumult Core's random number generator."""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
from randomgen.rdrand import RDRAND  # pylint: disable=no-name-in-module

_core_privacy_prng = np.random.Generator(RDRAND())


def prng():
    """Getter for prng."""
    return _core_privacy_prng
