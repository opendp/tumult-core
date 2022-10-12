"""Tests for :mod:`~tmlt.core.random.uniform`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from tmlt.core.random.uniform import uniform_inverse_cdf
from tmlt.core.utils.arb import Arb


def test_uniform_inverse_cdf():
    """Tests for :func:`~.uniform_inverse_cdf`."""
    assert uniform_inverse_cdf(10, 100, Arb.from_float(0.0), 63) == Arb.from_float(10.0)
    assert uniform_inverse_cdf(-100, -10, Arb.from_float(1.0), 63) == Arb.from_float(
        -10.0
    )
    assert uniform_inverse_cdf(10, 100, Arb.from_float(0.5), 63) == Arb.from_float(55.0)
    assert uniform_inverse_cdf(0, 1, Arb.from_float(0.2), 63) == Arb.from_float(0.2)
    assert uniform_inverse_cdf(0, 1, Arb.from_float(0.75), 63) == Arb.from_float(0.75)
