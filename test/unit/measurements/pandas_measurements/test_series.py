"""Unit tests for :mod:`~tmlt.core.measurements.pandas_measurements.series`."""

# <placeholder: boilerplate>

# pylint: disable=no-self-use

from typing import List, Tuple
from unittest.case import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
import sympy as sp
from parameterized import parameterized
from pyspark.sql.types import DoubleType

from tmlt.core.domains.numpy_domains import NumpyFloatDomain, NumpyIntegerDomain
from tmlt.core.domains.pandas_domains import PandasSeriesDomain
from tmlt.core.measurements.pandas_measurements.series import (
    AddDiscreteGaussianNoise,
    AddGeometricNoise,
    AddLaplaceNoise,
    NoisyQuantile,
    _get_quantile_probabilities,
)
from tmlt.core.measures import PureDP, RhoZCDP
from tmlt.core.metrics import (
    AbsoluteDifference,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.utils.testing import (
    TestComponent,
    assert_property_immutability,
    get_all_props,
)


class TestQuantile(TestCase):
    """Tests for class NoisyQuantile.

    Tests :class:~tmlt.core.measurements.pandas_measurements.series.NoisyQuantile`.
    """

    @parameterized.expand(get_all_props(NoisyQuantile))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        measurement = NoisyQuantile(
            PandasSeriesDomain(NumpyIntegerDomain()),
            output_measure=PureDP(),
            quantile=0.5,
            lower=22,
            upper=29,
            epsilon=10000000,
        )
        assert_property_immutability(measurement, prop_name)

    def test_properties(self):
        """NoisyQuantile's properties have the expected values."""
        measurement = NoisyQuantile(
            PandasSeriesDomain(NumpyIntegerDomain()),
            output_measure=RhoZCDP(),
            quantile=0.5,
            lower=22,
            upper=29,
            epsilon=10000000,
        )
        self.assertEqual(
            measurement.input_domain, PandasSeriesDomain(NumpyIntegerDomain())
        )
        self.assertEqual(measurement.input_metric, SymmetricDifference())
        self.assertEqual(measurement.output_measure, RhoZCDP())
        self.assertEqual(measurement.is_interactive, False)
        self.assertEqual(measurement.output_spark_type, DoubleType())
        self.assertEqual(measurement.quantile, 0.5)
        self.assertEqual(measurement.lower, 22)
        self.assertEqual(measurement.upper, 29)
        self.assertEqual(measurement.epsilon, 10000000)

    @parameterized.expand(
        [
            (pd.Series([28, 26, 27, 29]), 0, (22, 26)),
            (pd.Series([23, 22, 24, 25]), 0, (22, 23)),
            (pd.Series([28, 26, 27, 29]), 0.5, (27, 28)),
            (pd.Series([23, 22, 24, 25]), 0.5, (23, 24)),
            (pd.Series([28, 26, 27, 29]), 1, (28, 29)),
            (pd.Series([23, 22, 24, 25]), 1, (25, 29)),
        ]
    )
    @patch("tmlt.core.random.rng._core_privacy_prng", np.random.default_rng(seed=1))
    def test_correctness(
        self, data: pd.Series, q: float, expected_interval: Tuple[int, int]
    ):
        """Tests that the quantile is correct when epsilon is infinity."""
        measurement = NoisyQuantile(
            PandasSeriesDomain(NumpyIntegerDomain()),
            output_measure=PureDP(),
            quantile=q,
            lower=22,
            upper=29,
            epsilon=sp.oo,
        )
        output = measurement(data)
        (low, high) = expected_interval
        self.assertTrue(low <= output <= high)
        self.assertEqual(measurement.privacy_function(1), sp.oo)
        self.assertTrue(measurement.privacy_relation(1, sp.oo))

    @parameterized.expand(
        [
            (pd.Series([28, 26, 27, 29]), 0),
            (pd.Series([28, 26, 27, 29]), 0.5),
            (pd.Series([28, 26, 27, 29]), 1),
        ]
    )
    def test_clamping(self, data: pd.Series, q: float):
        """Tests that the quantile clamping bounds are applied."""
        output = NoisyQuantile(
            PandasSeriesDomain(NumpyIntegerDomain()),
            output_measure=PureDP(),
            quantile=q,
            lower=16,
            upper=19,
            epsilon=10000000,
        )(data)
        self.assertTrue(16 <= output <= 19)

    def test_equal_clamping_bounds(self):  # TODO(#693)
        """Tests that quantile aggregation works when clamping bounds are equal."""
        actual = NoisyQuantile(
            input_domain=PandasSeriesDomain(NumpyFloatDomain()),
            output_measure=PureDP(),
            quantile=0.5,
            lower=sp.Rational(1, 7),
            upper=sp.Rational(1, 7),
            epsilon=10000000,
        )(pd.Series([10.0, 155.0, -9.0]))
        self.assertAlmostEqual(actual, 1 / 7)  # TODO(#1023)

    def test_privacy_function_and_relation(self):
        """Test that the privacy relation and function are computed correctly."""
        measurement = NoisyQuantile(
            PandasSeriesDomain(NumpyIntegerDomain()),
            output_measure=PureDP(),
            quantile=0.5,
            lower=22,
            upper=29,
            epsilon=2,
        )
        self.assertEqual(measurement.privacy_function(1), 2)
        self.assertTrue(measurement.privacy_relation(1, 2))
        self.assertFalse(measurement.privacy_relation(1, "1.99999"))

        measurement = NoisyQuantile(
            PandasSeriesDomain(NumpyIntegerDomain()),
            output_measure=RhoZCDP(),
            quantile=0.98,
            lower=17,
            upper=42,
            epsilon=3,
        )
        self.assertTrue(measurement.privacy_relation(1, "9/8"))
        self.assertFalse(measurement.privacy_relation(1, "1.124"))

    def test_zero_epsilon(self):
        """Works with zero epsilon."""
        measurement = NoisyQuantile(
            PandasSeriesDomain(NumpyIntegerDomain()),
            output_measure=PureDP(),
            quantile=0.5,
            lower=22,
            upper=29,
            epsilon=0,
        )
        self.assertEqual(measurement.privacy_function(1), 0)
        self.assertTrue(measurement.privacy_relation(1, 0))
        self.assertTrue(measurement.privacy_relation(1, 1))
        self.assertTrue(22 <= measurement(pd.Series([23, 25])) <= 29)

    @parameterized.expand(
        [
            (0.5, [2], 1, 3, 2, [0.5, 0.5]),  # 2 equiwidth intervals median
            (0.5, [2], 2, 3, 1, [0, 1]),  # First interval has width 0.
            (
                0.5,
                [1, 2, 3, 4],
                0,
                5,
                float("inf"),
                [0, 0, 1, 0, 0],
            ),  # eps=inf (n even)
            (0.5, [1, 2, 3], 0, 4, float("inf"), [0, 0.5, 0.5, 0]),  # eps=inf (n odd)
            (0.9, [4, 4, 4], 2, 4, 10, [1, 0, 0, 0]),
            (0.25, [4, 5], 3, 7, 0, [0.25, 0.25, 0.50]),  # eps=0 (samples uniformly)
            (
                0.25,
                [4, 6, 7],
                2,
                9,
                1.5,
                [
                    0.3149490476621088,
                    0.5192631940672673,
                    0.09551312682718223,
                    0.07027463144344179,
                ],
            ),  # hand computed probabilities
            (
                0,
                [1, 5],
                0,
                20,
                2,
                [0.22214585276126372, 0.32689156868946884, 0.4509625785492673],
            ),  # q=0
        ]
    )
    def test_get_quantile_probabilities_correctness(
        self,
        quantile: float,
        data: List,
        lower: float,
        upper: float,
        epsilon: float,
        expected: List,
    ):
        """Tests that `_get_quantile_probabilities`."""
        actual = _get_quantile_probabilities(
            quantile=quantile, data=data, lower=lower, upper=upper, epsilon=epsilon
        )
        np.testing.assert_almost_equal(actual, expected, decimal=9)


class TestAddDiscreteGaussianNoise(TestComponent):
    """Tests for class AddDiscreteGaussianNoise.

    Tests :class:`~tmlt.core.measurements.pandas_measurements.series.
    AddDiscreteGaussiaNoise`.
    """

    @parameterized.expand(get_all_props(AddDiscreteGaussianNoise))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        measurement = AddDiscreteGaussianNoise(
            input_domain=PandasSeriesDomain(element_domain=NumpyIntegerDomain()),
            sigma_squared=10,
        )
        assert_property_immutability(measurement, prop_name)

    def test_properties(self):
        """AddDiscreteGaussianNoise's properties have the expected values."""
        measurement = AddDiscreteGaussianNoise(
            input_domain=PandasSeriesDomain(element_domain=NumpyIntegerDomain()),
            sigma_squared="0.5",
        )
        self.assertEqual(
            measurement.input_domain, PandasSeriesDomain(NumpyIntegerDomain())
        )
        self.assertEqual(
            measurement.input_metric, RootSumOfSquared(AbsoluteDifference())
        )
        self.assertEqual(measurement.output_measure, RhoZCDP())
        self.assertEqual(measurement.is_interactive, False)
        self.assertEqual(measurement.sigma_squared, "0.5")

    @parameterized.expand([(-0.4,), (np.nan,), ("invalid",)])
    def test_sigma_squared_validity(self, sigma_squared):
        """Tests that invalid sigma_squared values are rejected."""
        with self.assertRaises((ValueError, TypeError)):
            AddDiscreteGaussianNoise(
                input_domain=PandasSeriesDomain(element_domain=NumpyIntegerDomain()),
                sigma_squared=sigma_squared,
            )

    def test_no_noise(self):
        """Works correctly with no noise."""
        measurement = AddDiscreteGaussianNoise(
            input_domain=PandasSeriesDomain(element_domain=NumpyIntegerDomain()),
            sigma_squared=0,
        )
        self.assertEqual(measurement.privacy_function(1), sp.oo)
        self.assertTrue(measurement.privacy_relation(1, sp.oo))
        self.assertFalse(measurement.privacy_relation(1, sp.Pow(10, 7)))
        pd.testing.assert_series_equal(measurement(pd.Series([5])), pd.Series([5]))

    def test_some_noise(self):
        """Works correctly with some noise."""
        measurement = AddDiscreteGaussianNoise(
            input_domain=PandasSeriesDomain(element_domain=NumpyIntegerDomain()),
            sigma_squared=2,
        )
        self.assertEqual(measurement.privacy_function(1), "0.25")
        self.assertTrue(measurement.privacy_relation(1, "0.25"))
        self.assertFalse(measurement.privacy_relation(1, "0.2499"))

    def test_infinite_noise(self):
        """Raises an error with infinite noise."""
        with self.assertRaisesRegex(
            ValueError, "Invalid sigma_squared: oo is not strictly less than inf"
        ):
            AddDiscreteGaussianNoise(
                sigma_squared=sp.oo,
                input_domain=PandasSeriesDomain(element_domain=NumpyIntegerDomain()),
            )

    def test_detailed_fraction(self):
        """Works correctly with fractions that have high numerators/denominators.

        Test for bug #964.
        """
        for _ in range(10):  # Unfortunately, the failure was somewhat flaky.
            AddDiscreteGaussianNoise(
                sigma_squared="0.9999999999999999",
                input_domain=PandasSeriesDomain(element_domain=NumpyIntegerDomain()),
            )(pd.Series([0, 0, 0]))


class TestAddLaplaceNoise(TestComponent):
    """Tests for class AddLaplaceNoise.

    Tests :class:`~tmlt.core.measurements.pandas_measurements.series.AddLaplaceNoise`.
    """

    @parameterized.expand(get_all_props(AddLaplaceNoise))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        measurement = AddLaplaceNoise(
            input_domain=PandasSeriesDomain(element_domain=NumpyFloatDomain()), scale=2
        )
        assert_property_immutability(measurement, prop_name)

    def test_properties(self):
        """AddLaplaceNoise's properties have the expected values."""
        measurement = AddLaplaceNoise(
            input_domain=PandasSeriesDomain(element_domain=NumpyIntegerDomain()),
            scale="0.5",
        )
        self.assertEqual(
            measurement.input_domain,
            PandasSeriesDomain(element_domain=NumpyIntegerDomain()),
        )
        self.assertEqual(measurement.input_metric, SumOf(AbsoluteDifference()))
        self.assertEqual(measurement.output_measure, PureDP())
        self.assertEqual(measurement.is_interactive, False)
        self.assertEqual(measurement.scale, sp.Rational(1, 2))

    @parameterized.expand([(-0.4,), (np.nan,), ("invalid",)])
    def test_scale_validity(self, scale):
        """Tests that invalid scale values are rejected."""
        with self.assertRaises((ValueError, TypeError)):
            AddLaplaceNoise(
                input_domain=PandasSeriesDomain(element_domain=NumpyIntegerDomain()),
                scale=scale,
            )

    def test_no_noise(self):
        """Works correctly with no noise."""
        measurement = AddLaplaceNoise(
            input_domain=PandasSeriesDomain(element_domain=NumpyIntegerDomain()),
            scale=0,
        )
        self.assertEqual(measurement.privacy_function(1), sp.oo)
        self.assertTrue(measurement.privacy_relation(1, sp.oo))
        self.assertFalse(measurement.privacy_relation(1, sp.Pow(10, 6)))
        pd.testing.assert_series_equal(measurement(pd.Series([5])), pd.Series([5.0]))

    def test_some_noise(self):
        """Works correctly with some noise."""
        measurement = AddLaplaceNoise(
            input_domain=PandasSeriesDomain(element_domain=NumpyFloatDomain()), scale=2
        )
        self.assertEqual(measurement.privacy_function(1), "0.5")
        self.assertTrue(measurement.privacy_relation(1, "0.5"))
        self.assertFalse(measurement.privacy_relation(1, "0.49"))

    def test_infinite_noise(self):
        """Works correctly with infinite noise."""
        measurement = AddLaplaceNoise(
            input_domain=PandasSeriesDomain(element_domain=NumpyFloatDomain()),
            scale=sp.oo,
        )
        self.assertEqual(measurement.privacy_function(1), 0)
        self.assertTrue(measurement.privacy_relation(1, 0))
        self.assertTrue(measurement.privacy_relation(1, 1))
        # Equally likely to return -inf or inf
        self.assertTrue(np.all(np.isinf(measurement(pd.Series([5.0, 5.0])))))


class TestAddGeometricNoise(TestComponent):
    """Tests for class AddGeometricNoise.

    Tests
    :class:`~tmlt.core.measurements.pandas_measurements.series.AddGeometricNoise`.
    """

    @parameterized.expand(get_all_props(AddGeometricNoise))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        measurement = AddGeometricNoise(
            input_domain=PandasSeriesDomain(element_domain=NumpyIntegerDomain()),
            alpha=2,
        )
        assert_property_immutability(measurement, prop_name)

    def test_properties(self):
        """AddGeometricNoise's properties have the expected values."""
        measurement = AddGeometricNoise(
            input_domain=PandasSeriesDomain(element_domain=NumpyIntegerDomain()),
            alpha="0.5",
        )
        self.assertEqual(
            measurement.input_domain,
            PandasSeriesDomain(element_domain=NumpyIntegerDomain()),
        )
        self.assertEqual(measurement.input_metric, SumOf(AbsoluteDifference()))
        self.assertEqual(measurement.output_measure, PureDP())
        self.assertEqual(measurement.is_interactive, False)
        self.assertEqual(measurement.alpha, "0.5")

    @parameterized.expand([(-0.4,), (np.nan,), ("invalid",)])
    def test_sigma_validity(self, alpha):
        """Tests that invalid alpha values are rejected."""
        with self.assertRaises((ValueError, TypeError)):
            AddGeometricNoise(
                input_domain=PandasSeriesDomain(element_domain=NumpyIntegerDomain()),
                alpha=alpha,
            )

    def test_no_noise(self):
        """Works correctly with no noise."""
        measurement = AddGeometricNoise(
            input_domain=PandasSeriesDomain(element_domain=NumpyIntegerDomain()),
            alpha=0,
        )
        self.assertEqual(measurement.privacy_function(1), sp.oo)
        self.assertTrue(measurement.privacy_relation(1, sp.oo))
        self.assertFalse(measurement.privacy_relation(1, sp.Pow(10, 6)))
        pd.testing.assert_series_equal(measurement(pd.Series([5])), pd.Series([5]))

    def test_some_noise(self):
        """Works correctly with some noise."""
        measurement = AddGeometricNoise(
            input_domain=PandasSeriesDomain(element_domain=NumpyIntegerDomain()),
            alpha=2,
        )
        self.assertEqual(measurement.privacy_function(1), "0.5")
        self.assertTrue(measurement.privacy_relation(1, "0.5"))
        self.assertFalse(measurement.privacy_relation(1, "0.49"))

    def test_infinite_noise(self):
        """Raises an error with infinite noise."""
        with self.assertRaisesRegex(
            ValueError, "Invalid alpha: oo is not strictly less than inf"
        ):
            AddGeometricNoise(
                input_domain=PandasSeriesDomain(element_domain=NumpyIntegerDomain()),
                alpha=sp.oo,
            )
