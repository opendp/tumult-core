"""Module for sampling from a Laplace distribution."""

# SPDX-License-Identifier: Apache-2.0


import math

from flint import arb  # pylint: disable=no-name-in-module

from tmlt.core.random.inverse_cdf import construct_inverse_sampler


def laplace_inverse_cdf(u: float, b: float, p: arb) -> arb:
    """Returns value of the inverse CDF for Lap(`u`,`b`) at `p`.

    Args:
        u: The mean of the distribution. Must be finite and non-nan.
        b: The scale of the distribution. Must be finite, non-nan and non-negative.
        p: Probability to compute the CDF at.
    """
    if not arb(0) < p < arb(1):
        raise ValueError(f"`p` should be in (0,1), not {p}")
    if math.isnan(u) or math.isinf(u):
        raise ValueError(f"Location `u` should be finite and non-nan, not {u}")
    if math.isnan(b) or math.isinf(b) or b < 0:
        raise ValueError(
            f"Scale `b` should be finite, non-nan and non-negative, not {b}"
        )

    p_minus_half = p - 0.5
    return u - b * arb.sgn(p_minus_half) * arb.log(1 - 2 * abs(p_minus_half))


def laplace(u: float, b: float, step_size: int = 63) -> float:
    """Samples a float from the Laplace distribution.

    Args:
        u: The mean of the distribution. Must be finite and non-nan.
        b: The scale of the distribution. Must be positive, finite and non-nan.
        step_size: How many bits of probability to sample at a time.
    """
    return construct_inverse_sampler(
        inverse_cdf=lambda p: laplace_inverse_cdf(u, b, p), step_size=step_size
    )()
