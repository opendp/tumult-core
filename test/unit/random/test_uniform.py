"""Tests for :mod:`~tmlt.core.random.uniform`."""

# SPDX-License-Identifier: Apache-2.0

from flint import arb  # pylint: disable=no-name-in-module

from tmlt.core.random.uniform import uniform_inverse_cdf


def test_uniform_inverse_cdf():
    """Tests for :func:`~.uniform_inverse_cdf`."""
    assert uniform_inverse_cdf(10, 100, arb(0)) == arb(10)
    assert uniform_inverse_cdf(-100, -10, arb(1)) == arb(-10)
    assert uniform_inverse_cdf(10, 100, arb(0.5)) == arb(55)
    assert uniform_inverse_cdf(0, 1, arb(0.2)) == arb(0.2)
    assert uniform_inverse_cdf(0, 1, arb(0.75)) == arb(0.75)
