"""Tests for :mod:`~tmlt.core.random.uniform`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022-2025, and the Tumult Core Contributors 2025-present

from flint import arb, ctx

from tmlt.core.random.uniform import uniform, uniform_inverse_cdf


def test_uniform_inverse_cdf():
    """Tests for :func:`~.uniform_inverse_cdf`."""
    with ctx.workprec(63):
        assert uniform_inverse_cdf(10, 100, arb(0.0)) == arb(10.0)
        assert uniform_inverse_cdf(-100, -10, arb(1.0)) == arb(-10.0)
        assert uniform_inverse_cdf(10, 100, arb(0.5)) == arb(55.0)
        assert uniform_inverse_cdf(0, 1, arb(0.2)) == arb(0.2)
        assert uniform_inverse_cdf(0, 1, arb(0.75)) == arb(0.75)


def test_uniform_works_with_sampler():
    """Checks that the inverse cdf function works with the inverse sampler.

    We use a parameter range that has caused problems (p not in [0, 1]) before.
    """
    for _ in range(100):
        sample = uniform(0, 1, step_size=1)
        assert 0 <= sample <= 1
