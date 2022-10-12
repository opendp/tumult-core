"""Module for inverse transform sampling."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from typing import Callable

import tmlt.core.utils.arb as arb
from tmlt.core.random.rng import prng


def construct_inverse_sampler(
    inverse_cdf: Callable[[arb.Arb, int], arb.Arb], step_size: int = 63
) -> Callable[[], float]:
    """Returns a sampler for the distribution corresponding to `inverse_cdf`.

    Args:
       inverse_cdf: The inverse CDF for the distribution to sample from.
       step_size: Number of bits to sample from the prng per iteration.
    """
    if step_size <= 0:
        raise ValueError(f"`step_size` should be positive, not {step_size}")

    def sampler() -> float:
        """Returns a sample from the `inverse_cdf` distribution."""
        n = 0  # used for both the argument to `inverse_cdf`, and the bits of precision
        random_bits = 0  # random bits stored as an integer

        while True:
            n += step_size
            random_bits = (random_bits << step_size) + int(
                prng().integers(pow(2, step_size))
            )
            value = inverse_cdf(
                arb.Arb.from_midpoint_radius(
                    mid=arb.Arb.from_man_exp(2 * random_bits + 1, -n - 1),
                    rad=arb.Arb.from_man_exp(1, -n - 1),
                ),
                n,
            )
            try:
                return value.to_float(n)
            except ValueError:
                pass

    return sampler
