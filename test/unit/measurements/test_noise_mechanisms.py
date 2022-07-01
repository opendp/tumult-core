"""Unit tests for :mod:`~tmlt.core.measurements.noise_mechanisms`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

# pylint: disable=no-self-use

import numpy as np
import sympy as sp
from parameterized import parameterized

from tmlt.core.domains.numpy_domains import NumpyFloatDomain, NumpyIntegerDomain
from tmlt.core.measurements.noise_mechanisms import (
    AddDiscreteGaussianNoise,
    AddGeometricNoise,
    AddLaplaceNoise,
)
from tmlt.core.measures import PureDP, RhoZCDP
from tmlt.core.metrics import AbsoluteDifference
from tmlt.core.utils.testing import (
    TestComponent,
    assert_property_immutability,
    get_all_props,
)


class TestAddLaplaceNoise(TestComponent):
    """Tests for :class:`~tmlt.core.measurements.noise_mechanisms.AddLaplaceNoise`."""

    @parameterized.expand(get_all_props(AddLaplaceNoise))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        measurement = AddLaplaceNoise(input_domain=NumpyIntegerDomain(), scale=1)
        assert_property_immutability(measurement, prop_name)

    def test_properties(self):
        """AddLaplaceNoise's properties have the expected values."""
        measurement = AddLaplaceNoise(
            input_domain=NumpyIntegerDomain(), scale=sp.Rational("0.3")
        )
        self.assertEqual(measurement.input_domain, NumpyIntegerDomain())
        self.assertEqual(measurement.input_metric, AbsoluteDifference())
        self.assertEqual(measurement.output_measure, PureDP())
        self.assertEqual(measurement.is_interactive, False)
        self.assertEqual(measurement.scale, sp.Rational("0.3"))

    @parameterized.expand([(-0.4,), (np.nan,), ("invalid",)])
    def test_sigma_validity(self, sigma):
        """Tests that invalid scale values are rejected."""
        with self.assertRaises((ValueError, TypeError)):
            AddLaplaceNoise(input_domain=NumpyIntegerDomain(), scale=sigma)

    def test_no_noise(self):
        """Works correctly with no noise."""
        measurement = AddLaplaceNoise(input_domain=NumpyIntegerDomain(), scale=0)
        self.assertEqual(measurement.privacy_function(1), sp.oo)
        self.assertTrue(measurement.privacy_relation(1, sp.oo))
        self.assertFalse(measurement.privacy_relation(1, sp.Pow(10, 6)))
        self.assertEqual(measurement(5.0), 5.0)

    def test_some_noise(self):
        """Works correctly with some noise."""
        measurement = AddLaplaceNoise(input_domain=NumpyIntegerDomain(), scale=2)
        self.assertEqual(measurement.privacy_function(1), sp.Rational("0.5"))
        self.assertTrue(measurement.privacy_relation(1, sp.Rational("0.5")))
        self.assertFalse(measurement.privacy_relation(1, sp.Rational("0.49999999")))

    def test_infinite_noise(self):
        """Works correctly with infinite noise."""
        measurement = AddLaplaceNoise(input_domain=NumpyIntegerDomain(), scale=sp.oo)
        self.assertEqual(measurement.privacy_function(1), 0)
        self.assertTrue(measurement.privacy_relation(1, 0))
        self.assertTrue(measurement.privacy_relation(1, 1))
        # Equally likely to return -inf or inf
        self.assertTrue(np.isinf(measurement(5.0)))

    @parameterized.expand(
        [(NumpyFloatDomain(allow_inf=True),), (NumpyFloatDomain(allow_nan=True),)]
    )
    def test_input_domain(self, domain: NumpyFloatDomain):
        """Tests that input domain is checked correctly.

        Condition: Must be a *NumpyIntegerDomain* or a
         (non-nullable) *NumpyFloatDomain*
        """
        with self.assertRaisesRegex(
            ValueError, "Input domain must not contain infs or nans"
        ):
            AddLaplaceNoise(input_domain=domain, scale=1)


class TestAddGeometricNoise(TestComponent):
    """Tests for class AddGeometricNoise.

    Tests :class:`~tmlt.core.measurements.noise_mechanisms.AddGeometricNoise`.
    """

    @parameterized.expand(get_all_props(AddGeometricNoise))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        measurement = AddGeometricNoise(alpha=0)
        assert_property_immutability(measurement, prop_name)

    def test_properties(self):
        """AddGeometricNoise's properties have the expected values."""
        measurement = AddGeometricNoise(alpha=sp.Rational("0.5"))
        self.assertEqual(measurement.input_domain, NumpyIntegerDomain())
        self.assertEqual(measurement.input_metric, AbsoluteDifference())
        self.assertEqual(measurement.output_measure, PureDP())
        self.assertEqual(measurement.is_interactive, False)
        self.assertEqual(measurement.alpha, sp.Rational("0.5"))

    @parameterized.expand([(-0.4,), (np.nan,), ("invalid",)])
    def test_alpha_validity(self, alpha):
        """Tests that invalid alpha values are rejected."""
        with self.assertRaises((ValueError, TypeError)):
            AddGeometricNoise(alpha=alpha)

    def test_no_noise(self):
        """Works correctly with no noise."""
        measurement = AddGeometricNoise(alpha=0)
        self.assertTrue(measurement.privacy_function(1), sp.oo)
        self.assertTrue(measurement.privacy_relation(1, sp.oo))
        self.assertFalse(measurement.privacy_relation(1, sp.Pow(10, 6)))
        self.assertEqual(measurement(5), 5)

    def test_some_noise(self):
        """Works correctly with some noise."""
        measurement = AddGeometricNoise(alpha=2)
        self.assertTrue(measurement.privacy_function(1), sp.Rational("0.5"))
        self.assertTrue(measurement.privacy_relation(1, sp.Rational("0.5")))
        self.assertFalse(measurement.privacy_relation(1, sp.Rational("0.49")))

    def test_infinite_noise(self):
        """Raises an error with infinite noise."""
        with self.assertRaisesRegex(
            ValueError, "Invalid alpha: oo is not strictly less than inf"
        ):
            AddGeometricNoise(alpha=sp.oo)


class TestAddDiscreteGaussianNoise(TestComponent):
    """Tests for class AddDiscreteGaussianNoise.

    Tests :class:`~tmlt.core.measurements.noise_mechanisms.AddDiscreteGaussianNoise`.
    """

    @parameterized.expand(get_all_props(AddDiscreteGaussianNoise))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        measurement = AddDiscreteGaussianNoise(sigma_squared=0)
        assert_property_immutability(measurement, prop_name)

    def test_properties(self):
        """AddDiscreteGaussianNoise's properties have the expected values."""
        measurement = AddDiscreteGaussianNoise(sigma_squared=sp.Rational("0.5"))
        self.assertEqual(measurement.input_domain, NumpyIntegerDomain())
        self.assertEqual(measurement.input_metric, AbsoluteDifference())
        self.assertEqual(measurement.output_measure, RhoZCDP())
        self.assertEqual(measurement.is_interactive, False)
        self.assertEqual(measurement.sigma_squared, sp.Rational("0.5"))

    @parameterized.expand([(-0.4,), (np.nan,), ("invalid",)])
    def test_sigma_squared_validity(self, sigma_squared):
        """Tests that invalid sigma_squared values are rejected."""
        with self.assertRaises((ValueError, TypeError)):
            AddDiscreteGaussianNoise(sigma_squared)

    def test_no_noise(self):
        """Works correctly with no noise."""
        measurement = AddDiscreteGaussianNoise(sigma_squared=0)
        self.assertTrue(measurement.privacy_function(1), sp.oo)
        self.assertTrue(measurement.privacy_relation(1, sp.oo))
        self.assertFalse(measurement.privacy_relation(1, sp.Pow(10, 6)))
        self.assertEqual(measurement(5), 5)

    def test_some_noise(self):
        """Works correctly with some noise."""
        measurement = AddDiscreteGaussianNoise(sigma_squared=2)
        self.assertTrue(measurement.privacy_function(1), sp.Rational("0.5"))
        self.assertTrue(measurement.privacy_relation(1, sp.Rational("0.25")))
        self.assertFalse(measurement.privacy_relation(1, sp.Rational("0.2499")))

    def test_infinite_noise(self):
        """Raises an error with infinite noise."""
        with self.assertRaisesRegex(
            ValueError, "Invalid sigma_squared: oo is not strictly less than inf"
        ):
            AddDiscreteGaussianNoise(sigma_squared=sp.oo)

    def test_detailed_fraction(self):
        """Works correctly with fractions that have high numerators/denominators.

        Test for bug #964.
        """
        for _ in range(10):  # Unfortunately, the failure was somewhat flaky.
            AddDiscreteGaussianNoise(sigma_squared=sp.Rational("0.9999999999999999"))(0)
