"""Module for sampling uniformly from an interval."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

from flint import arb, ctx

from tmlt.core.random.inverse_cdf import construct_inverse_sampler


def uniform_inverse_cdf(l: float, u: float, p: arb, prec: int) -> arb:
    """Returns the value of inverse CDF of the uniform distribution from ``l`` to ``u``.

    Args:
        l: Lower bound for the uniform distribution.
        u: Upper bound for the uniform distribution.
        p: Probability to compute the inverse CDF at.
        prec: Precision to compute the CDF with.
    """
    assert arb(0) <= p <= arb(1), f"`p` should be in [0,1], not {p}"
    assert l <= u, f"`l` should not be larger than `u`, but {l} > {u}"
    # The following code-block is equivalent to:
    #   return l * (1 - p) + p * u
    with ctx.workprec(prec):
        return (arb(l) * (arb(1) - p)) + (p * arb(u))


def uniform(lower: float, upper: float, step_size: int = 63) -> float:
    """Returns a random floating point number between ``lower`` and ``upper``.

    Args:
        lower: Lower bound of interval to sample from.
        upper: Upper bound of interval to sample from.
        step_size: Number of bits to sampler per iteration.
    """
    assert lower <= upper, f"`l` should not be larger than `u`, but {lower} > {upper}"
    return construct_inverse_sampler(
        inverse_cdf=lambda p, prec: uniform_inverse_cdf(lower, upper, p, prec),
        step_size=step_size,
    )()
