"""Module for sampling from a Laplace distribution."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022-2025, and the Tumult Core Contributors 2025-present

import math

from flint import arb, ctx

from tmlt.core.random.inverse_cdf import construct_inverse_sampler


def laplace_inverse_cdf(u: float, b: float, p: arb, prec: int) -> arb:
    """Returns inverse CDF for Lap(u,b) at p.

    Args:
        u: The mean of the distribution. Must be finite and non-nan.
        b: The scale of the distribution. Must be finite, non-nan and non-negative.
        p: Probability to compute the CDF at.
        prec: Precision to use for computing CDF.
    """
    if not 0 < p < 1:
        raise ValueError(f"`p` should be in (0,1), not {p}")
    if math.isnan(u) or math.isinf(u):
        raise ValueError(f"Location `u` should be finite and non-nan, not {u}")
    if math.isnan(b) or math.isinf(b) or b < 0:
        raise ValueError(
            f"Scale `b` should be finite, non-nan and non-negative, not {b}"
        )

    # The following code corresponds to:
    #   return u - b * sgn(p-0.5) * log(1 - 2 * abs(p-0.5))
    with ctx.workprec(prec):
        term2 = b * (p - 0.5).sgn() * (1 - 2 * abs(p - 0.5)).log()
        return u - term2


def laplace(u: float, b: float, step_size: int = 63) -> float:
    """Samples a float from the Laplace distribution.

    Args:
        u: The mean of the distribution. Must be finite and non-nan.
        b: The scale of the distribution. Must be positive, finite and non-nan.
        step_size: How many bits of probability to sample at a time.
    """
    return construct_inverse_sampler(
        inverse_cdf=lambda p, prec: laplace_inverse_cdf(u, b, p, prec),
        step_size=step_size,
    )()
