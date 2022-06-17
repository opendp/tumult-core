"""Module for inverse transform sampling."""

# SPDX-License-Identifier: Apache-2.0

from typing import Callable

from flint import arb, ctx  # pylint: disable=no-name-in-module

from tmlt.core.random.rng import prng
from tmlt.core.utils.misc import arb_to_float


def construct_inverse_sampler(
    inverse_cdf: Callable[[arb], arb], step_size: int = 63
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
            ctx.prec = n
            random_bits = (random_bits << step_size) + int(
                prng().integers(pow(2, step_size))
            )
            value = inverse_cdf(arb(mid=(2 * random_bits + 1, -n - 1), rad=(1, -n - 1)))
            float_value = arb_to_float(value)
            if float_value is not None:
                return float_value

    return sampler
