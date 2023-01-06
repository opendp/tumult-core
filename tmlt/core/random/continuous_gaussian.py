"""Module for sampling from a continuous Gaussian distribution."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

import math

from tmlt.core.random.inverse_cdf import construct_inverse_sampler
from tmlt.core.utils import arb


def gaussian_inverse_cdf(
    u: float, sigma_squared: float, p: arb.Arb, prec: int
) -> arb.Arb:
    """Returns inverse CDF for N(u,sigma_squared) at p.

    Args:
        u: The mean of the distribution. Must be finite and non-nan.
        sigma_squared: The variance of the distribution.
                       Must be finite, non-nan and non-negative.
        p: Probability to compute the CDF at.
        prec: Precision to use for computing CDF.
    """
    if not arb.Arb.from_int(0) < p < arb.Arb.from_int(1):
        raise ValueError(f"`p` should be in (0,1), not {p}")
    if math.isnan(u) or math.isinf(u):
        raise ValueError(f"Location `u` should be finite and non-nan, not {u}")
    if math.isnan(sigma_squared) or math.isinf(sigma_squared) or sigma_squared < 0:
        raise ValueError(
            f"Scale should be finite, non-nan and non-negative, not {sigma_squared}"
        )

    from_float = arb.Arb.from_float
    from_int = arb.Arb.from_int
    arb_sub = lambda x, y: arb.arb_sub(x, y, prec)
    arb_mul = lambda x, y: arb.arb_mul(x, y, prec)
    arb_add = lambda x, y: arb.arb_add(x, y, prec)
    arb_sqrt = lambda x: arb.arb_sqrt(x, prec)
    arb_erfinv = lambda x: arb.arb_erfinv(x, prec)

    # The following code corresponds to:
    #   return u + sigma_squared * sqrt(2) * erfinv(2 * p - 1)
    return arb_add(
        from_float(u),
        arb_mul(
            from_float(sigma_squared),
            arb_mul(
                arb_sqrt(from_int(2)),
                arb_erfinv(arb_sub(arb_mul(from_int(2), p), from_int(1))),
            ),
        ),
    )


def gaussian(sigma_squared: float, u: float = 0, step_size: int = 63) -> float:
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
