"""Tests for :mod:`~tmlt.core.random.uniform`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

from flint import arb

from tmlt.core.random.uniform import uniform_inverse_cdf


def test_uniform_inverse_cdf():
    """Tests for :func:`~.uniform_inverse_cdf`."""
    assert uniform_inverse_cdf(10, 100, arb(0.0), 63) == arb(10.0)
    assert uniform_inverse_cdf(-100, -10, arb(1.0), 63) == arb(-10.0)
    assert uniform_inverse_cdf(10, 100, arb(0.5), 63) == arb(55.0)
    assert uniform_inverse_cdf(0, 1, arb(0.2), 63) == arb(0.2)
    assert uniform_inverse_cdf(0, 1, arb(0.75), 63) == arb(0.75)
