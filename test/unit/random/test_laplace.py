"""Tests for :mod:`~tmlt.core.random.laplace`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from unittest import TestCase

from parameterized import parameterized
from scipy.stats import laplace

from tmlt.core.random.laplace import laplace_inverse_cdf
from tmlt.core.utils.arb import Arb


class TestLaplaceInverseCDF(TestCase):
    """Tests for :func:`~.laplace_inverse_cdf`."""

    @parameterized.expand(
        [
            (1, 1, 2.0, r"`p` should be in \(0,1\)"),
            (float("inf"), 1, 0.4, "Location `u` should be finite and non-nan"),
            (float("nan"), 1, 0.4, "Location `u` should be finite and non-nan"),
            (-float("inf"), 1, 0.4, "Location `u` should be finite and non-nan"),
            (
                1,
                float("inf"),
                0.5,
                "Scale `b` should be finite, non-nan and non-negative",
            ),
            (
                1,
                float("nan"),
                0.5,
                "Scale `b` should be finite, non-nan and non-negative",
            ),
            (1, -1, 0.5, "Scale `b` should be finite, non-nan and non-negative"),
        ]
    )
    def test_bad_arguments(self, u: float, b: float, p: float, error_msg: str):
        """`laplace_inverse_cdf` raises error when called with bad arguments."""
        with self.assertRaisesRegex(ValueError, error_msg):
            laplace_inverse_cdf(u, b, Arb.from_float(p), 63)

    @parameterized.expand(
        [
            (0, 1, 0.5),
            (10, 100, 0.5),
            (10, 100, 0.5),
            (0, 1, 0.9),
            (0, 1, 0.1),
            (-10, 0.5, 0.2),
        ]
    )
    def test_correctness(self, u: float, b: float, p: float):
        """Sanity tests for :func:`laplace_inverse_cdf`."""
        self.assertAlmostEqual(
            float(laplace_inverse_cdf(u, b, Arb.from_float(p), 63)),
            laplace.ppf(p, loc=u, scale=b),
        )
