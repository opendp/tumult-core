"""Module for sampling from a continuous Gaussian distribution."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023-2025, and the Tumult Core Contributors 2025-present

import math
from typing import Union

from flint import arb, ctx

from tmlt.core.random.inverse_cdf import construct_inverse_sampler


def gaussian_inverse_cdf(
    u: Union[int, float, arb],
    sigma_squared: Union[int, float, arb],
    p: arb,
    prec: int,
) -> arb:
    """Returns inverse CDF for N(u,sigma_squared) at p.

    Args:
        u: The mean of the distribution. Must be finite and non-nan.
        sigma_squared: The variance of the distribution.
                       Must be finite, non-nan and non-negative.
        p: Probability to compute the CDF at.
        prec: Precision to use for computing CDF.
    """
    if not arb(0) < p < arb(1):
        raise ValueError(f"`p` should be in (0,1), not {p}")

    if isinstance(u, (float, int)) and (math.isnan(u) or math.isinf(u)):
        raise ValueError(f"Location `u` should be finite and non-nan, not {u}")
    if isinstance(u, arb) and (u.is_nan() or not u.is_finite()):
        raise ValueError(f"Location `u` should be finite and non-nan, not {u}")
    if isinstance(sigma_squared, (float, int)) and (
        math.isnan(sigma_squared) or math.isinf(sigma_squared) or sigma_squared < 0
    ):
        raise ValueError(
            f"Scale should be finite, non-nan and non-negative, not {sigma_squared}"
        )
    if isinstance(sigma_squared, arb) and (
        sigma_squared.is_nan()
        or not sigma_squared.is_finite()
        or sigma_squared < arb(0)
    ):
        raise ValueError(
            f"Scale should be finite, non-nan and non-negative, not {sigma_squared}"
        )

    u_arb = u if isinstance(u, arb) else arb(u)
    sigma_squared_arb = (
        sigma_squared if isinstance(sigma_squared, arb) else arb(sigma_squared)
    )
    # The following code corresponds to:
    #   return u + sigma * sqrt(2) * erfinv(2 * p - 1)
    with ctx.workprec(prec):
        return u_arb + (
            (sigma_squared_arb).sqrt()
            * (arb(2)).sqrt()
            * ((arb(2) * p) - arb(1)).erfinv()
        )


def gaussian(
    sigma_squared: Union[arb, float],
    u: Union[arb, float] = 0,
    step_size: int = 63,
) -> float:
    r"""Samples a float from the Gaussian distribution.

    In particular, this returns a sample from the Gaussian
        :math:`\mathcal{N}_{\mathbb{Z}}(u, sigma\_squared)`

    Args:
        sigma_squared: The variance of the distribution.
                       Must be positive, finite and non-nan.
        u: The mean of the distribution. Must be finite and non-nan. Defaults to 0
        step_size: How many bits of probability to sample at a time.
    """
    return construct_inverse_sampler(
        inverse_cdf=lambda p, prec: gaussian_inverse_cdf(u, sigma_squared, p, prec),
        step_size=step_size,
    )()
