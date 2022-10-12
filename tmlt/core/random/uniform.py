"""Module for sampling uniformly from an interval."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

import tmlt.core.utils.arb as arb
from tmlt.core.random.inverse_cdf import construct_inverse_sampler


def uniform_inverse_cdf(l: float, u: float, p: arb.Arb, prec: int) -> arb.Arb:
    """Returns the value of inverse CDF of the uniform distribution from `l` to `u`.

    Args:
        l: Lower bound for the uniform distribution.
        u: Upper bound for the uniform distribution.
        p: Probability to compute the inverse CDF at.
        prec: Precision to compute the CDF with.
    """
    assert (
        arb.Arb.from_int(0) <= p <= arb.Arb.from_int(1)
    ), f"`p` should be in [0,1], not {p}"
    assert l <= u, f"`l` should not be larger than `u`, but {l} > {u}"
    # The following code-block is equivalent to:
    #   return l * (1 - p) + p * u
    return arb.arb_add(
        arb.arb_mul(
            arb.Arb.from_float(l), arb.arb_sub(arb.Arb.from_int(1), p, prec), prec
        ),
        arb.arb_mul(p, arb.Arb.from_float(u), prec),
        prec,
    )


def uniform(lower: float, upper: float, step_size: int = 63) -> float:
    """Returns a random floating point number between `lower` and `upper`.

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
