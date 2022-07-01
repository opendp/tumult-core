"""Tests for :mod:`~tmlt.core.random.discrete_gaussian`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from fractions import Fraction
from typing import Union
from unittest import TestCase

from parameterized import parameterized

from tmlt.core.random.discrete_gaussian import sample_dgauss


class TestDiscreteGaussian(TestCase):
    """Tests functions for sampling from discrete Gaussian distribution."""

    @parameterized.expand(
        [(-1,), (float("nan"),), (float("inf"),), (-0.1,), (Fraction(-1, 100),)]
    )
    def test_sample_dgauss_invalid_scale(
        self, sigma_squared: Union[int, float, Fraction]
    ):
        """Tests that sample_dgauss raises appropriate error with invalid scale."""
        with self.assertRaisesRegex(ValueError, "sigma_squared must be positive"):
            sample_dgauss(sigma_squared=sigma_squared)

    def test_invalid_rng_raises_error(self):
        """Tests that sample_dgauss raises error if rng does not support randrange."""

        class BadRNG:
            """Does not support randrange."""

            ...

        with self.assertRaisesRegex(TypeError, 'type of argument "rng" must be'):
            sample_dgauss(sigma_squared=1, rng=BadRNG())
