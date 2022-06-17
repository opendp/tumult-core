"""Tests for :mod:`~tmlt.core.random.laplace`."""

# SPDX-License-Identifier: Apache-2.0

from unittest import TestCase

from flint import arb  # pylint: disable=no-name-in-module
from parameterized import parameterized
from scipy.stats import laplace

from tmlt.core.random.laplace import laplace_inverse_cdf


class TestLaplaceInverseCDF(TestCase):
    """Tests for :func:`~.laplace_inverse_cdf`."""

    @parameterized.expand(
        [
            (1, 1, arb(2), r"`p` should be in \(0,1\)"),
            (float("inf"), 1, arb(0.4), "Location `u` should be finite and non-nan"),
            (float("nan"), 1, arb(0.4), "Location `u` should be finite and non-nan"),
            (-float("inf"), 1, arb(0.4), "Location `u` should be finite and non-nan"),
            (
                1,
                float("inf"),
                arb(0.5),
                "Scale `b` should be finite, non-nan and non-negative",
            ),
            (
                1,
                float("nan"),
                arb(0.5),
                "Scale `b` should be finite, non-nan and non-negative",
            ),
            (1, -1, arb(0.5), "Scale `b` should be finite, non-nan and non-negative"),
        ]
    )
    def test_invalid_step_size(self, u: float, b: float, p: arb, error_msg: str):
        """`laplace_inverse_cdf` raises error when step_size is not valid."""
        with self.assertRaisesRegex(ValueError, error_msg):
            laplace_inverse_cdf(u, b, p)

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
            float(laplace_inverse_cdf(u, b, arb(p))), laplace.ppf(p, loc=u, scale=b)
        )
