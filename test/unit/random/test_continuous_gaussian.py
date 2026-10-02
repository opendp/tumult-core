"""Tests for :mod:`~tmlt.core.random.continuous_gaussian`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022-2025, and the Tumult Core Contributors 2025-present

import math
from unittest import TestCase

from flint import arb, ctx
from parameterized import parameterized
from scipy.stats import norm

from tmlt.core.random.continuous_gaussian import gaussian, gaussian_inverse_cdf


class TestContinuousGaussianInverseCDF(TestCase):
    """Tests for :func:`~.continuous_gaussian_inverse_cdf`."""

    @parameterized.expand(
        [
            (1, 1, 2.0, r"`p` should be in \(0,1\)"),
            (float("inf"), 1, 0.4, "Location `u` should be finite and non-nan"),
            (float("nan"), 1, 0.4, "Location `u` should be finite and non-nan"),
            (-float("inf"), 1, 0.4, "Location `u` should be finite and non-nan"),
            (1, float("inf"), 0.5, "Scale should be finite, non-nan and non-negative"),
            (1, float("nan"), 0.5, "Scale should be finite, non-nan and non-negative"),
            (1, -1, 0.5, "Scale should be finite, non-nan and non-negative"),
        ]
    )
    def test_bad_arguments(self, u: float, b: float, p: float, error_msg: str):
        """`gaussian_inverse_cdf` raises error when called with bad arguments."""
        with self.assertRaisesRegex(ValueError, error_msg):
            gaussian_inverse_cdf(u, b, arb(p))

    @parameterized.expand(
        [
            (0, 1, 0.5),
            (10, 100, 0.1),
            (10, 100, 0.5),
            (10, 100, 0.9),
            (0, 1, 0.9),
            (0, 5, 0.1),
            (2000, 5, 0.1),
            (-10, 0.5, 0.5),
            (-10, 0.5, 0.09),
        ]
    )
    def test_correctness(self, u: float, sigma_squared: float, p: float):
        """Sanity tests for :func:`gaussian_inverse_cdf`."""
        with ctx.workprec(63):
            actual = float(gaussian_inverse_cdf(u, sigma_squared, arb(p)))
        self.assertAlmostEqual(
            actual,
            norm.ppf(p, loc=u, scale=math.sqrt(sigma_squared)),
        )


def test_gaussian_works_with_sampler():
    """Checks that the inverse cdf function works with the inverse sampler.

    We use a parameter range that has caused problems (p not in [0, 1]) before.
    """
    for _ in range(100):
        sample = gaussian(1, step_size=1)
        assert isinstance(sample, float) and math.isfinite(sample)
