"""Tests for :module:`tmlt.core.utils.distributions`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

# pylint: disable=no-self-use

import unittest

import numpy as np
import sympy as sp
from parameterized import parameterized
from scipy.stats import geom, norm

from tmlt.core.utils.distributions import (
    discrete_gaussian_cmf,
    discrete_gaussian_pmf,
    double_sided_geometric_cmf,
    double_sided_geometric_cmf_exact,
    double_sided_geometric_pmf,
)


class TestDoubleSidedGeometric(unittest.TestCase):
    """Test :func:`double_sided_geometric_pmf` and :func:`double_sided_geometric_cmf."""

    @parameterized.expand([(0.3,), (17.3,)])
    def test_double_sided_geometric_pmf_and_cmf(self, alpha: float):
        """Test that the double sided geometric stat functions have the expected values.

        Approximates the answer by calculating the pmf/cdf using scipy.stats.geom.
        This is a very close approximation unless alpha is too small
        (there must be negligible probability mass outside of [min_value, max_value]).
        """
        p = 1 - np.exp(-1 / alpha)
        max_value = 1000
        min_value = -max_value
        one_sided_pmf = geom.pmf(np.arange(1, max_value + 2), p)
        double_sided_pmf = np.zeros((max_value - min_value) + 1)
        for i in range(max_value + 1):
            for j in range(max_value + 1):
                double_sided_pmf[i - j - min_value] += (
                    one_sided_pmf[i] * one_sided_pmf[j]
                )
        double_sided_cmf = np.cumsum(double_sided_pmf)

        def approx_pmf(k: int) -> float:
            """Return the approximate probability mass function at k."""
            return double_sided_pmf[k - min_value]

        def approx_cmf(k: int) -> float:
            """Return the approximate cumulative probability mass function at k."""
            return double_sided_cmf[k - min_value]

        ks_to_test = [-10, -5, -3, -1, 0, 1, 3, 5, 10]
        for k in ks_to_test:
            # Test pmf
            actual = double_sided_geometric_pmf(k, alpha)
            expected = approx_pmf(k)
            np.testing.assert_allclose(actual, expected)

            # Test cmf
            actual = double_sided_geometric_cmf(k, alpha)
            expected = approx_cmf(k)
            np.testing.assert_allclose(actual, expected)

            # Test exact cmf
            actual = double_sided_geometric_cmf_exact(k, sp.Rational(alpha)).to_float(
                round_up=False
            )
            # same expected
            np.testing.assert_allclose(actual, expected)

        # Test passing multiple ks as an np.ndarray
        ks = np.array(ks_to_test)
        # Test pmf
        actual = double_sided_geometric_pmf(ks, alpha)
        expected = approx_pmf(ks)
        np.testing.assert_allclose(actual, expected)

        # Test cmf
        actual = double_sided_geometric_cmf(ks, alpha)
        expected = approx_cmf(ks)
        np.testing.assert_allclose(actual, expected)

    def test_pmf_integrates_to_one(self):
        """The sum of all values of the probability mass function should be 1."""
        actual = np.sum(double_sided_geometric_pmf(np.arange(-1000, 1000), alpha=1.0))
        expected = 1
        np.testing.assert_allclose(actual, expected)

    def test_cmf_monotonically_increases_from_zero_to_one(self):
        """The cmf should monotonically increase from zero to one."""
        actual = double_sided_geometric_cmf(np.arange(-1000, 1000), alpha=1.0)
        np.testing.assert_allclose(actual[0], 0)
        np.testing.assert_allclose(actual[-1], 1)
        for i in range(1, len(actual)):
            self.assertGreaterEqual(actual[i], actual[i - 1])


class TestDiscreteGaussian(unittest.TestCase):
    """Test :func:`discrete_gaussian_pmf` and :func:`discrete_gaussian_cmf."""

    @parameterized.expand([(0.3,), (17.3,)])
    def test_discrete_gaussian_pmf_and_cmf(self, sigma_squared: float):
        """Test that the discrete gaussian stat functions have the expected values.

        For all integers, the pmf of the discrete gaussian distribution should be
        proportional to the pdf of the corresponding continuous gaussian distribution
        at the same value.
        """
        max_value = 1000
        min_value = -1000
        k_values = np.arange(min_value, max_value + 1)
        unnormalized_pmf = norm.pdf(k_values, scale=np.sqrt(sigma_squared))
        pmf = unnormalized_pmf / np.sum(unnormalized_pmf)
        cmf = np.cumsum(pmf)

        def approx_pmf(k: int) -> float:
            """Return the approximate probability mass function at k."""
            return pmf[k - min_value]

        def approx_cmf(k: int) -> float:
            """Return the approximate cumulative probability mass function at k."""
            return cmf[k - min_value]

        ks_to_test = [-10, -5, -3, -1, 0, 1, 3, 5, 10]
        # Test passing multiple ks as an np.ndarray
        ks_to_test.append(np.array(ks_to_test))
        for k in ks_to_test:
            # Test pmf
            actual = discrete_gaussian_pmf(k, sigma_squared)
            expected = approx_pmf(k)
            np.testing.assert_allclose(actual, expected)

            # Test cmf
            actual = discrete_gaussian_cmf(k, sigma_squared)
            expected = approx_cmf(k)
            np.testing.assert_allclose(actual, expected)

    def test_pmf_integrates_to_one(self):
        """The sum of all values of the probability mass function should be 1."""
        actual = np.sum(
            discrete_gaussian_pmf(np.arange(-1000, 1000), sigma_squared=1.0)
        )
        expected = 1
        np.testing.assert_allclose(actual, expected)

    def test_cmf_monotonically_increases_from_zero_to_one(self):
        """The cmf should monotonically increase from zero to one."""
        actual = discrete_gaussian_cmf(np.arange(-1000, 1000), sigma_squared=1.0)
        np.testing.assert_allclose(actual[0], 0)
        np.testing.assert_allclose(actual[-1], 1)
        for i in range(1, len(actual)):
            self.assertGreaterEqual(actual[i], actual[i - 1])
