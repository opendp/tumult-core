"""Unit tests for :mod:`~tmlt.core.measurements.pandas_measurements.series`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

# pylint: disable=no-self-use
import re
from typing import Any, Dict, Tuple, Union
from unittest.case import TestCase
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import sympy as sp
from parameterized import parameterized
from pyspark.sql.types import DoubleType

from tmlt.core.domains.numpy_domains import (
    NumpyFloatDomain,
    NumpyIntegerDomain,
    NumpyStringDomain,
)
from tmlt.core.domains.pandas_domains import PandasSeriesDomain
from tmlt.core.measurements.noise_mechanisms import (
    AddDiscreteGaussianNoise,
    AddGeometricNoise,
    AddLaplaceNoise,
)
from tmlt.core.measurements.pandas_measurements.series import (
    AddNoiseToSeries,
    NoisyQuantile,
)
from tmlt.core.measures import PureDP, RhoZCDP
from tmlt.core.metrics import (
    AbsoluteDifference,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.utils.testing import assert_property_immutability, get_all_props


class TestNoisyQuantile(TestCase):
    """Tests for class :class:`~.NoisyQuantile`."""

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
            ({"quantile": 1.1}, "Quantile must be between 0 and 1."),
            (
                {"lower": float("-inf")},
                "Lower clamping bound must be finite and non-nan",
            ),
            (
                {"upper": float("inf")},
                "Upper clamping bound must be finite and non-nan",
            ),
            (
                {"upper": 0, "lower": 1},
                "Lower bound (1) can not be greater than the upper bound (0).",
            ),
            ({"epsilon": -1}, "Invalid PureDP measure value (epsilon) -1"),
            (
                {"input_domain": PandasSeriesDomain(NumpyStringDomain())},
                "input_domain.element_domain must be NumpyIntegerDomain or "
                "NumpyFloatDomain, not NumpyStringDomain",
            ),
            (
                {"input_domain": PandasSeriesDomain(NumpyFloatDomain(allow_nan=True))},
                "Input domain must disallow NaNs",
            ),
        ]
    )
    def test_bad_inputs(self, bad_kwargs: Dict[str, Any], message: str):
        """Tests that bad inputs are rejected."""
        kwargs = {
            "input_domain": PandasSeriesDomain(NumpyIntegerDomain()),
            "output_measure": PureDP(),
            "quantile": 0.5,
            "lower": 0,
            "upper": 1,
            "epsilon": 1,
        }
        kwargs.update(bad_kwargs)
        with self.assertRaisesRegex(ValueError, re.escape(message)):
            NoisyQuantile(**kwargs)  # type: ignore

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

    def test_equal_clamping_bounds(self):
        """Tests that quantile aggregation works when clamping bounds are equal."""
        actual = NoisyQuantile(
            input_domain=PandasSeriesDomain(NumpyFloatDomain()),
            output_measure=PureDP(),
            quantile=0.5,
            lower=1 / 7,
            upper=1 / 7,
            epsilon=10000000,
        )(pd.Series([10.0, 155.0, -9.0]))
        self.assertEqual(actual, 1 / 7)

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


class TestAddNoiseToSeries(TestCase):
    """Tests for :class:`~.AddNoiseToSeries`."""

    @parameterized.expand(get_all_props(AddNoiseToSeries))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        measurement = AddNoiseToSeries(
            noise_measurement=AddDiscreteGaussianNoise(sigma_squared=1)
        )
        assert_property_immutability(measurement, prop_name)

    @parameterized.expand(
        [
            (AddGeometricNoise(alpha=1), PureDP(), SumOf(AbsoluteDifference())),
            (
                AddDiscreteGaussianNoise(sigma_squared=1),
                RhoZCDP(),
                RootSumOfSquared(AbsoluteDifference()),
            ),
            (
                AddLaplaceNoise(scale=1, input_domain=NumpyFloatDomain()),
                PureDP(),
                SumOf(AbsoluteDifference()),
            ),
        ]
    )
    def test_properties(
        self,
        noise_measurement: Union[
            AddGeometricNoise, AddDiscreteGaussianNoise, AddLaplaceNoise
        ],
        expected_output_measure: Union[PureDP, RhoZCDP],
        expected_input_metric: Union[RootSumOfSquared, SumOf],
    ):
        """AddNoiseToSeries's properties have the expected values."""
        measurement = AddNoiseToSeries(noise_measurement=noise_measurement)
        self.assertEqual(
            measurement.input_domain, PandasSeriesDomain(noise_measurement.input_domain)
        )
        self.assertEqual(measurement.input_metric, expected_input_metric)
        self.assertEqual(measurement.output_measure, expected_output_measure)
        self.assertEqual(measurement.is_interactive, False)
        self.assertEqual(measurement.noise_measurement, noise_measurement)

    def test_correctness(self):
        """AddNoiseToSeries calls noise_measurement as expected."""
        noise_measurement = Mock(AddLaplaceNoise)
        noise_measurement.output_measure = PureDP()
        noise_measurement.input_domain = NumpyIntegerDomain()
        noise_measurement.return_value = 0
        add_noise_to_series = AddNoiseToSeries(noise_measurement)
        actual = add_noise_to_series(pd.Series([1, 2, 3]))
        self.assertEqual(noise_measurement.call_args_list, [call(1), call(2), call(3)])
        self.assertEqual(actual.to_list(), [0, 0, 0])
