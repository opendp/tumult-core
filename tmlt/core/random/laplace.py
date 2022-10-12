"""Module for sampling from a Laplace distribution."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

import math

import tmlt.core.utils.arb as arb
from tmlt.core.random.inverse_cdf import construct_inverse_sampler


def laplace_inverse_cdf(u: float, b: float, p: arb.Arb, prec: int) -> arb.Arb:
    """Returns inverse CDF for Lap(u,b) at p.

    Args:
        u: The mean of the distribution. Must be finite and non-nan.
        b: The scale of the distribution. Must be finite, non-nan and non-negative.
        p: Probability to compute the CDF at.
        prec: Precision to use for computing CDF.
    """
    if not arb.Arb.from_int(0) < p < arb.Arb.from_int(1):
        raise ValueError(f"`p` should be in (0,1), not {p}")
    if math.isnan(u) or math.isinf(u):
        raise ValueError(f"Location `u` should be finite and non-nan, not {u}")
    if math.isnan(b) or math.isinf(b) or b < 0:
        raise ValueError(
            f"Scale `b` should be finite, non-nan and non-negative, not {b}"
        )

    # The following code corresponds to:
    #   return u - b * sgn(p-0.5) * log(1 - 2 * abs(p-0.5))
    p_minus_half = arb.arb_sub(p, arb.Arb.from_float(0.5), prec)
    arb_u = arb.Arb.from_float(u)
    term2 = arb.arb_mul(
        arb.arb_mul(arb.Arb.from_float(b), arb.arb_sgn(p_minus_half), prec),
        arb.arb_log(
            arb.arb_add(
                arb.Arb.from_int(1),
                arb.arb_mul(arb.Arb.from_int(-2), arb.arb_abs(p_minus_half), prec),
                prec,
            ),
            prec,
        ),
        prec,
    )
    return arb.arb_sub(arb_u, term2, prec)


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
