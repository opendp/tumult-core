"""Unit tests for :mod:`~tmlt.core.measurements.noise_mechanisms`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import re
from contextlib import nullcontext as does_not_raise
from test.unit.measurements.abstract import MeasurementTests
from typing import Any, Callable, ContextManager, Dict, Type

import numpy as np
import pytest
import sympy as sp
from pyspark.sql.types import DoubleType, LongType
from typeguard import TypeCheckError

from tmlt.core.domains.numpy_domains import (
    NumpyFloatDomain,
    NumpyIntegerDomain,
    NumpyStringDomain,
)
from tmlt.core.measurements.base import Measurement
from tmlt.core.measurements.noise_mechanisms import (
    AddDiscreteGaussianNoise,
    AddGaussianNoise,
    AddGeometricNoise,
    AddLaplaceNoise,
)
from tmlt.core.measures import PureDP, RhoZCDP
from tmlt.core.metrics import AbsoluteDifference
from tmlt.core.utils.exact_number import ExactNumber


class TestAddLaplaceNoise(MeasurementTests):
    """Tests for :class:`~tmlt.core.measurements.noise_mechanisms.AddLaplaceNoise`."""

    @pytest.fixture
    def measurement_type(self) -> Type[Measurement]:
        """Returns the type of the measurement to be tested."""
        return AddLaplaceNoise

    @pytest.mark.parametrize(
        "measurement_args, expectation",
        [
            (
                {"input_domain": NumpyIntegerDomain(), "scale": sp.Rational("0.3")},
                does_not_raise(),
            ),
            (
                {
                    "input_domain": NumpyIntegerDomain(size=32),
                    "scale": sp.Rational("0.3"),
                },
                does_not_raise(),
            ),
            (
                {"input_domain": NumpyStringDomain(), "scale": 1},
                pytest.raises(
                    TypeCheckError,
                    match=re.escape('"input_domain"'),
                ),
            ),
            (
                {"input_domain": NumpyIntegerDomain(), "scale": "x"},
                pytest.raises(
                    ValueError, match="Invalid scale: x contains free symbols"
                ),
            ),
            (
                {"input_domain": NumpyIntegerDomain(), "scale": -1},
                pytest.raises(
                    ValueError,
                    match="Invalid scale: -1 is not greater than or equal to 0",
                ),
            ),
            ({"input_domain": NumpyFloatDomain(), "scale": "52/3"}, does_not_raise()),
            (
                {"input_domain": NumpyFloatDomain(size=32), "scale": "52/3"},
                does_not_raise(),
            ),
            (
                {"input_domain": NumpyFloatDomain(allow_nan=True), "scale": "3 + 2"},
                pytest.raises(
                    ValueError, match="Input domain must not contain infs or nans"
                ),
            ),
            (
                {
                    "input_domain": NumpyFloatDomain(allow_inf=True),
                    "scale": ExactNumber(3),
                },
                pytest.raises(
                    ValueError, match="Input domain must not contain infs or nans"
                ),
            ),
        ],
    )
    def test_construct_component(
        self,
        measurement_type: Type[Measurement],
        measurement_args: Dict[str, Any],
        expectation: ContextManager[None],
    ):
        """Initialization behaves correctly.

        The measurement is constructed correctly and raises exceptions
        when initialized with invalid inputs.

        Args:
            measurement_type: The type of the measurement to be constructed.
            measurement_args: The arguments to the measurement.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
        """
        super().test_construct_component(
            measurement_type, measurement_args, expectation, exception_properties=None
        )

    @pytest.mark.parametrize(
        "measurement, input_data",
        [
            (
                AddLaplaceNoise(input_domain=NumpyIntegerDomain(), scale=float("inf")),
                np.int64(1),
            ),
            (
                AddLaplaceNoise(input_domain=NumpyFloatDomain(), scale=float("inf")),
                np.float64(-1),
            ),
        ],
    )
    def test_infinite_protection(self, measurement: Measurement, input_data: Any):
        """Measurement works with infinite protection.

        Args:
            measurement: The measurement.
            input_data: The input data for the measurement.
        """
        output = measurement(input_data)
        assert output in [float("inf"), -float("inf")]

    @pytest.mark.skip("No arguments to mutate")
    def test_mutable_inputs(
        self,
        measurement_type: Type[Measurement],
        measurement_args: Dict[str, Any],
        key: str,
        mutator: Callable,
    ):
        """Mutable inputs to the measurement are copied.

        Args:
            measurement_type: The type of the measurement to be constructed.
            measurement_args: The arguments to the measurement.
            key: The parameter name to be changed.
            mutator: A lambda function that mutates the parameter.
        """
        super().test_mutable_inputs(measurement_type, measurement_args, key, mutator)

    @pytest.mark.parametrize(
        "measurement, input_data, expected_output",
        [
            (
                AddLaplaceNoise(input_domain=NumpyIntegerDomain(), scale=0),
                np.int64(10),
                10.0,
            ),
            (
                AddLaplaceNoise(input_domain=NumpyFloatDomain(size=32), scale=0),
                np.float32(-0.1234567),
                float(np.float32(-0.1234567)),
            ),
        ],
    )
    def test_no_protection(
        self, measurement: Measurement, input_data: Any, expected_output: Any
    ):
        """Measurement works with no protection.

        Args:
            measurement: The measurement.
            input_data: The input data for the measurement.
            expected_output: Expected unprotected output.
        """
        super().test_no_protection(measurement, input_data, expected_output)

    @pytest.mark.parametrize(
        "measurement, input_data, expected_output_type",
        [
            (
                AddLaplaceNoise(input_domain=NumpyIntegerDomain(), scale=1),
                np.int64(10),
                float,
            ),
            (
                AddLaplaceNoise(input_domain=NumpyFloatDomain(size=32), scale=1),
                np.float32(-0.1234567),
                float,
            ),
        ],
    )
    def test_output(
        self, measurement: Measurement, input_data: Any, expected_output_type: type
    ):
        """Measurement produces an output that has the expected type.

        Args:
            measurement: The measurement.
            input_data: The input data for the measurement.
            expected_output_type: The expected type for the output.
        """
        super().test_output(measurement, input_data, expected_output_type)

    @pytest.mark.parametrize(
        "measurement, d_in, expected_d_out, expectation",
        [
            (
                AddLaplaceNoise(input_domain=NumpyIntegerDomain(), scale=1),
                1,
                1,
                does_not_raise(),
            ),
            (
                AddLaplaceNoise(input_domain=NumpyIntegerDomain(), scale=2),
                1,
                "0.5",
                does_not_raise(),
            ),
            (
                AddLaplaceNoise(input_domain=NumpyFloatDomain(), scale="0.5"),
                1,
                2,
                does_not_raise(),
            ),
            (
                AddLaplaceNoise(input_domain=NumpyFloatDomain(), scale="0.5"),
                3,
                6,
                does_not_raise(),
            ),
            (
                AddLaplaceNoise(input_domain=NumpyFloatDomain(), scale="0.5"),
                -1,
                None,
                pytest.raises(
                    ValueError,
                    match="Invalid value for metric AbsoluteDifference: -1 is not "
                    "greater than or equal to 0",
                ),
            ),
            (
                AddLaplaceNoise(input_domain=NumpyIntegerDomain(), scale=0),
                1,
                float("inf"),
                does_not_raise(),
            ),
            (
                AddLaplaceNoise(input_domain=NumpyIntegerDomain(), scale=0),
                0,
                0,
                does_not_raise(),
            ),
            (
                AddLaplaceNoise(input_domain=NumpyIntegerDomain(), scale=float("inf")),
                1,
                0,
                does_not_raise(),
            ),
        ],
    )
    def test_privacy_function(
        self,
        measurement: Measurement,
        d_in: Any,
        expected_d_out: Any,
        expectation: ContextManager[None],
    ):
        """Testing the measurement's privacy function.

        Args:
            measurement: The measurement to be tested.
            d_in: Distance between inputs.
            expected_d_out: Expected distance between outputs.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
        """
        super().test_privacy_function(
            measurement, d_in, expected_d_out, expectation, exception_properties=None
        )

    @pytest.mark.parametrize(
        "measurement, d_in, d_out, expected, expectation",
        [
            (
                AddLaplaceNoise(input_domain=NumpyIntegerDomain(), scale=1),
                1,
                1,
                True,
                does_not_raise(),
            ),
            (
                AddLaplaceNoise(input_domain=NumpyIntegerDomain(), scale=1),
                1,
                2,
                True,
                does_not_raise(),
            ),
            (
                AddLaplaceNoise(input_domain=NumpyIntegerDomain(), scale=1),
                1,
                "0.5",
                False,
                does_not_raise(),
            ),
            (
                AddLaplaceNoise(input_domain=NumpyFloatDomain(), scale="0.5"),
                3,
                6,
                True,
                does_not_raise(),
            ),
            (
                AddLaplaceNoise(input_domain=NumpyFloatDomain(), scale="0.5"),
                3,
                5,
                False,
                does_not_raise(),
            ),
            (
                AddLaplaceNoise(input_domain=NumpyFloatDomain(), scale="0.5"),
                -1,
                1,
                True,
                pytest.raises(
                    ValueError,
                    match="Invalid value for metric AbsoluteDifference: -1 is not "
                    "greater than or equal to 0",
                ),
            ),
            (
                AddLaplaceNoise(input_domain=NumpyFloatDomain(), scale="0.5"),
                tuple(),
                1,
                True,
                pytest.raises(
                    TypeCheckError,
                    match=re.escape('"d_in"'),
                ),
            ),
            (
                AddLaplaceNoise(input_domain=NumpyFloatDomain(), scale=1),
                1,
                -1,
                True,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "Invalid PureDP measure value (epsilon): -1 is not greater "
                        "than or equal to 0"
                    ),
                ),
            ),
            (
                AddLaplaceNoise(input_domain=NumpyFloatDomain(), scale=1),
                1,
                {},
                True,
                pytest.raises(
                    TypeCheckError,
                    match=re.escape('"value"'),
                ),
            ),
            (
                AddLaplaceNoise(input_domain=NumpyIntegerDomain(), scale=0),
                1,
                float("inf"),
                True,
                does_not_raise(),
            ),
            (
                AddLaplaceNoise(input_domain=NumpyIntegerDomain(), scale=0),
                0,
                0,
                True,
                does_not_raise(),
            ),
            (
                AddLaplaceNoise(input_domain=NumpyIntegerDomain(), scale=float("inf")),
                1,
                0,
                True,
                does_not_raise(),
            ),
        ],
    )
    def test_privacy_relation(
        self,
        measurement: Measurement,
        d_in: Any,
        d_out: Any,
        expected: bool,
        expectation: ContextManager[None],
    ):
        """Testing the measurement's privacy function.

        Args:
            measurement: The measurement to be tested.
            d_in: Distance between inputs.
            d_out: Distance between outputs.
            expected: Whether the d_in, d_out pair is close.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
        """
        super().test_privacy_relation(
            measurement, d_in, d_out, expected, expectation, exception_properties=None
        )

    @pytest.mark.parametrize(
        "measurement, expected_properties",
        [
            (
                AddLaplaceNoise(input_domain=NumpyIntegerDomain(), scale=1),
                {
                    "input_domain": NumpyIntegerDomain(),
                    "scale": 1,
                    "input_metric": AbsoluteDifference(),
                    "output_type": DoubleType(),
                    "output_measure": PureDP(),
                    "is_interactive": False,
                    "adds_no_noise": False,
                },
            ),
            (
                AddLaplaceNoise(input_domain=NumpyFloatDomain(size=32), scale=2),
                {
                    "input_domain": NumpyFloatDomain(size=32),
                    "scale": 2,
                    "input_metric": AbsoluteDifference(),
                    "output_type": DoubleType(),
                    "output_measure": PureDP(),
                    "is_interactive": False,
                    "adds_no_noise": False,
                },
            ),
            (
                AddLaplaceNoise(input_domain=NumpyIntegerDomain(), scale=0),
                {
                    "input_domain": NumpyIntegerDomain(),
                    "scale": 0,
                    "input_metric": AbsoluteDifference(),
                    "output_type": DoubleType(),
                    "output_measure": PureDP(),
                    "is_interactive": False,
                    "adds_no_noise": True,
                },
            ),
        ],
    )
    def test_properties(
        self, measurement: Measurement, expected_properties: Dict[str, Any]
    ):
        """All properties have the expected values.

        Args:
            measurement: The constructed measurement to be tested.
            expected_properties: A dictionary containing all the
                property:value pairs measurement is expected to have.
        """
        super().test_properties(measurement, expected_properties)

    @pytest.mark.parametrize(
        "measurement", [AddLaplaceNoise(input_domain=NumpyIntegerDomain(), scale=1)]
    )
    def test_property_immutability(self, measurement: Measurement):
        """The properties return copies for mutable values.

        Args:
            measurement: The measurement to be tested.
        """
        super().test_property_immutability(measurement)

    @pytest.mark.parametrize(
        "scale, p, expected, expectation",
        [
            (1, 0.5, 0, does_not_raise()),
            (2, 0.5, 0, does_not_raise()),
            (3.1415, 0.5, 0, does_not_raise()),
            (1, 0.75, 0.693147, does_not_raise()),
            (1, 0.25, -0.693147, does_not_raise()),
            (3.5, 0.86, 4.45538, does_not_raise()),
            (3.5, 0.14, -4.45538, does_not_raise()),
            (0, 0.6, 0, does_not_raise()),  # 0 scale is 0
            (  # inf scale is +inf 50% of the time
                float("inf"),
                0.6,
                float("inf"),
                does_not_raise(),
            ),
            (  # inf scale is -inf 50% of the time
                float("inf"),
                0.4,
                float("-inf"),
                does_not_raise(),
            ),
            (1, 0, float("-inf"), does_not_raise()),  # p=0 is -inf
            (1, 1, float("inf"), does_not_raise()),  # p=1 is +inf
            (-1, 0.6, float("nan"), does_not_raise()),  # negative scale is nan
            (float("nan"), 0.6, float("nan"), does_not_raise()),  # nan scale is nan
        ]
        + [  # p values outside [0, 1] raise ValueError
            (
                1,
                bad_p,
                float("nan"),
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "Probabilities input to the inverse CDF must be in [0, 1], "
                        f"but got {bad_p}"
                    ),
                ),
            )
            for bad_p in [-1, 1.1, float("inf"), float("nan")]
        ],
    )
    def test_inverse_cdf(
        self, scale: float, p: float, expected: float, expectation: ContextManager[None]
    ):
        """Testing the inverse_cdf function.

        Args:
            scale: The scale of the Laplace distribution.
            p: The probability.
            expected: The expected value.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
        """
        with expectation:
            actual = AddLaplaceNoise.inverse_cdf(scale=scale, probability=p)
            if np.isnan(expected):
                assert np.isnan(actual), f"Expected {expected}, got {actual}"
            else:
                assert actual == pytest.approx(
                    expected
                ), f"Expected {expected}, got {actual}"


class TestAddGeometricNoise(MeasurementTests):
    """Tests for :class:`~tmlt.core.measurements.noise_mechanisms.AddGeometricNoise`."""

    @pytest.fixture
    def measurement_type(self) -> Type[Measurement]:
        """Returns the type of the measurement to be tested."""
        return AddGeometricNoise

    @pytest.mark.parametrize(
        "measurement_args, expectation",
        [
            ({"alpha": sp.Rational("0.3")}, does_not_raise()),
            (
                {"alpha": "x"},
                pytest.raises(
                    ValueError, match="Invalid alpha: x contains free symbols"
                ),
            ),
            (
                {"alpha": -1},
                pytest.raises(
                    ValueError,
                    match="Invalid alpha: -1 is not greater than or equal to 0",
                ),
            ),
            (
                {"alpha": float("inf")},
                pytest.raises(
                    ValueError, match="Invalid alpha: inf is not strictly less than inf"
                ),
            ),
            ({"alpha": "0"}, does_not_raise()),
        ],
    )
    def test_construct_component(
        self,
        measurement_type: Type[Measurement],
        measurement_args: Dict[str, Any],
        expectation: ContextManager[None],
    ):
        """Initialization behaves correctly.

        The measurement is constructed correctly and raises exceptions
        when initialized with invalid inputs.

        Args:
            measurement_type: The type of the measurement to be constructed.
            measurement_args: The arguments to the measurement.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
        """
        super().test_construct_component(
            measurement_type, measurement_args, expectation, exception_properties=None
        )

    @pytest.mark.skip("Doesn't support infinite protection")
    def test_infinite_protection(
        self, measurement: Measurement, input_data: Any, expected_output: Any
    ):
        """Measurement works with infinite protection.

        Args:
            measurement: The measurement.
            input_data: The input data for the measurement.
            expected_output: The expected output of the measurement.
        """
        super().test_infinite_protection(measurement, input_data, expected_output)

    @pytest.mark.skip("No arguments to mutate")
    def test_mutable_inputs(
        self,
        measurement_type: Type[Measurement],
        measurement_args: Dict[str, Any],
        key: str,
        mutator: Callable,
    ):
        """Mutable inputs to the measurement are copied.

        Args:
            measurement_type: The type of the measurement to be constructed.
            measurement_args: The arguments to the measurement.
            key: The parameter name to be changed.
            mutator: A lambda function that mutates the parameter.
        """
        super().test_mutable_inputs(measurement_type, measurement_args, key, mutator)

    @pytest.mark.parametrize(
        "measurement, input_data, expected_output",
        [
            (AddGeometricNoise(alpha=0), np.int64(10), 10),
            (AddGeometricNoise(alpha=0), np.int64(-10), -10),
        ],
    )
    def test_no_protection(
        self, measurement: Measurement, input_data: Any, expected_output: Any
    ):
        """Measurement works with no protection.

        Args:
            measurement: The measurement.
            input_data: The input data for the measurement.
            expected_output: Expected unprotected output.
        """
        super().test_no_protection(measurement, input_data, expected_output)

    @pytest.mark.parametrize(
        "measurement, input_data, expected_output_type",
        [(AddGeometricNoise(alpha=3), np.int64(0), int)],
    )
    def test_output(
        self, measurement: Measurement, input_data: Any, expected_output_type: type
    ):
        """Measurement produces an output that has the expected type.

        Args:
            measurement: The measurement.
            input_data: The input data for the measurement.
            expected_output_type: The expected type for the output.
        """
        super().test_output(measurement, input_data, expected_output_type)

    @pytest.mark.parametrize(
        "measurement, d_in, expected_d_out, expectation",
        [
            (AddGeometricNoise(alpha=1), 1, 1, does_not_raise()),
            (AddGeometricNoise(alpha=2), 1, "0.5", does_not_raise()),
            (AddGeometricNoise(alpha="0.5"), 1, 2, does_not_raise()),
            (AddGeometricNoise(alpha="0.5"), 3, 6, does_not_raise()),
            (
                AddGeometricNoise(alpha="0.5"),
                -1,
                None,
                pytest.raises(
                    ValueError,
                    match="Invalid value for metric AbsoluteDifference: -1 is not "
                    "greater than or equal to 0",
                ),
            ),
            (AddGeometricNoise(alpha=0), 1, float("inf"), does_not_raise()),
            (AddGeometricNoise(alpha=0), 0, 0, does_not_raise()),
        ],
    )
    def test_privacy_function(
        self,
        measurement: Measurement,
        d_in: Any,
        expected_d_out: Any,
        expectation: ContextManager[None],
    ):
        """Testing the measurement's privacy function.

        Args:
            measurement: The measurement to be tested.
            d_in: Distance between inputs.
            expected_d_out: Expected distance between outputs.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
        """
        super().test_privacy_function(
            measurement, d_in, expected_d_out, expectation, exception_properties=None
        )

    @pytest.mark.parametrize(
        "measurement, d_in, d_out, expected, expectation",
        [
            (AddGeometricNoise(alpha=1), 1, 1, True, does_not_raise()),
            (AddGeometricNoise(alpha=1), 1, 2, True, does_not_raise()),
            (AddGeometricNoise(alpha=1), 1, "0.5", False, does_not_raise()),
            (AddGeometricNoise(alpha="0.5"), 3, 6, True, does_not_raise()),
            (AddGeometricNoise(alpha="0.5"), 3, 5, False, does_not_raise()),
            (
                AddGeometricNoise(alpha="0.5"),
                -1,
                1,
                True,
                pytest.raises(
                    ValueError,
                    match="Invalid value for metric AbsoluteDifference: -1 is not "
                    "greater than or equal to 0",
                ),
            ),
            (
                AddGeometricNoise(alpha="0.5"),
                tuple(),
                1,
                True,
                pytest.raises(
                    TypeCheckError,
                    match=re.escape('"d_in"'),
                ),
            ),
            (
                AddGeometricNoise(alpha="0.5"),
                1,
                -1,
                True,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "Invalid PureDP measure value (epsilon): -1 is not greater "
                        "than or equal to 0"
                    ),
                ),
            ),
            (
                AddGeometricNoise(alpha="0.5"),
                1,
                {},
                True,
                pytest.raises(
                    TypeCheckError,
                    match=re.escape('"value"'),
                ),
            ),
            (AddGeometricNoise(alpha=0), 1, float("inf"), True, does_not_raise()),
            (AddGeometricNoise(alpha=0), 0, 0, True, does_not_raise()),
        ],
    )
    def test_privacy_relation(
        self,
        measurement: Measurement,
        d_in: Any,
        d_out: Any,
        expected: bool,
        expectation: ContextManager[None],
    ):
        """Testing the measurement's privacy function.

        Args:
            measurement: The measurement to be tested.
            d_in: Distance between inputs.
            d_out: Distance between outputs.
            expected: Whether the d_in, d_out pair is close.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
        """
        super().test_privacy_relation(
            measurement, d_in, d_out, expected, expectation, exception_properties=None
        )

    @pytest.mark.parametrize(
        "measurement, expected_properties",
        [
            (
                AddGeometricNoise(alpha=1),
                {
                    "input_domain": NumpyIntegerDomain(),
                    "alpha": 1,
                    "input_metric": AbsoluteDifference(),
                    "output_type": LongType(),
                    "output_measure": PureDP(),
                    "is_interactive": False,
                    "adds_no_noise": False,
                },
            ),
            (
                AddGeometricNoise(alpha=2),
                {
                    "input_domain": NumpyIntegerDomain(),
                    "alpha": 2,
                    "input_metric": AbsoluteDifference(),
                    "output_type": LongType(),
                    "output_measure": PureDP(),
                    "is_interactive": False,
                    "adds_no_noise": False,
                },
            ),
            (
                AddGeometricNoise(alpha=0),
                {
                    "input_domain": NumpyIntegerDomain(),
                    "alpha": 0,
                    "input_metric": AbsoluteDifference(),
                    "output_type": LongType(),
                    "output_measure": PureDP(),
                    "is_interactive": False,
                    "adds_no_noise": True,
                },
            ),
        ],
    )
    def test_properties(
        self, measurement: Measurement, expected_properties: Dict[str, Any]
    ):
        """All properties have the expected values.

        Args:
            measurement: The constructed measurement to be tested.
            expected_properties: A dictionary containing all the
                property:value pairs measurement is expected to have.
        """
        super().test_properties(measurement, expected_properties)

    @pytest.mark.parametrize("measurement", [AddGeometricNoise(alpha=1)])
    def test_property_immutability(self, measurement: Measurement):
        """The properties return copies for mutable values.

        Args:
            measurement: The measurement to be tested.
        """
        super().test_property_immutability(measurement)

    @pytest.mark.parametrize(
        "alpha, p, expected, expectation",
        [
            (1, 0.5, 0, does_not_raise()),
            (2, 0.5, 0, does_not_raise()),
            (3.1415, 0.5, 0, does_not_raise()),
            (1, 0.75, 1, does_not_raise()),
            (1, 0.25, -1, does_not_raise()),
            (3.5, 0.86, 4, does_not_raise()),
            (3.5, 0.14, -4, does_not_raise()),
            (0, 0.6, 0, does_not_raise()),  # 0 scale is 0
            (  # inf scale is +inf 50% of the time
                float("inf"),
                0.6,
                float("inf"),
                does_not_raise(),
            ),
            (  # inf scale is -inf 50% of the time
                float("inf"),
                0.4,
                float("-inf"),
                does_not_raise(),
            ),
            (1, 0, float("-inf"), does_not_raise()),  # p=0 is -inf
            (1, 1, float("inf"), does_not_raise()),  # p=1 is +inf
            (-1, 0.6, float("nan"), does_not_raise()),  # negative scale is nan
            (float("nan"), 0.6, float("nan"), does_not_raise()),  # nan scale is nan
        ]
        + [  # p values outside [0, 1] raise ValueError
            (
                1,
                bad_p,
                float("nan"),
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "Probabilities input to the inverse CDF must be in [0, 1], "
                        f"but got {bad_p}"
                    ),
                ),
            )
            for bad_p in [-1, 1.1, float("inf"), float("nan")]
        ],
    )
    def test_inverse_cdf(
        self, alpha: float, p: float, expected: float, expectation: ContextManager[None]
    ):
        """Testing the inverse_cdf function.

        Args:
            alpha: The alpha of the Geometric distribution.
            p: The probability.
            expected: The expected value.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
        """
        with expectation:
            actual = AddGeometricNoise.inverse_cdf(alpha=alpha, probability=p)
            if np.isnan(expected):
                assert np.isnan(actual), f"Expected {expected}, got {actual}"
            else:
                assert actual == pytest.approx(
                    expected
                ), f"Expected {expected}, got {actual}"


class TestAddGaussianNoise(MeasurementTests):
    """Tests for :class:`~tmlt.core.measurements.noise_mechanisms.AddGaussianNoise`."""

    @pytest.fixture
    def measurement_type(self) -> Type[Measurement]:
        """Returns the type of the measurement to be tested."""
        return AddGaussianNoise

    @pytest.mark.parametrize(
        "measurement_args, expectation",
        [
            (
                {
                    "input_domain": NumpyIntegerDomain(),
                    "sigma_squared": sp.Rational("0.3"),
                },
                does_not_raise(),
            ),
            (
                {
                    "input_domain": NumpyIntegerDomain(size=32),
                    "sigma_squared": sp.Rational("0.3"),
                },
                does_not_raise(),
            ),
            (
                {"input_domain": NumpyStringDomain(), "sigma_squared": 1},
                pytest.raises(
                    TypeCheckError,
                    match=re.escape('"input_domain"'),
                ),
            ),
            (
                {"input_domain": NumpyIntegerDomain(), "sigma_squared": "x"},
                pytest.raises(
                    ValueError, match="Invalid sigma_squared: x contains free symbols"
                ),
            ),
            (
                {"input_domain": NumpyIntegerDomain(), "sigma_squared": -1},
                pytest.raises(
                    ValueError,
                    match="Invalid sigma_squared: -1 is not greater than or equal to 0",
                ),
            ),
            (
                {"input_domain": NumpyFloatDomain(), "sigma_squared": "52/3"},
                does_not_raise(),
            ),
            (
                {"input_domain": NumpyFloatDomain(size=32), "sigma_squared": "52/3"},
                does_not_raise(),
            ),
            (
                {
                    "input_domain": NumpyFloatDomain(allow_nan=True),
                    "sigma_squared": "3 + 2",
                },
                pytest.raises(
                    ValueError, match="Input domain must not contain infs or nans"
                ),
            ),
            (
                {
                    "input_domain": NumpyFloatDomain(allow_inf=True),
                    "sigma_squared": ExactNumber(3),
                },
                pytest.raises(
                    ValueError, match="Input domain must not contain infs or nans"
                ),
            ),
        ],
    )
    def test_construct_component(
        self,
        measurement_type: Type[Measurement],
        measurement_args: Dict[str, Any],
        expectation: ContextManager[None],
    ):
        """Initialization behaves correctly.

        The measurement is constructed correctly and raises exceptions
        when initialized with invalid inputs.

        Args:
            measurement_type: The type of the measurement to be constructed.
            measurement_args: The arguments to the measurement.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
        """
        super().test_construct_component(
            measurement_type, measurement_args, expectation, exception_properties=None
        )

    @pytest.mark.parametrize(
        "measurement, input_data",
        [
            (
                AddGaussianNoise(
                    input_domain=NumpyIntegerDomain(), sigma_squared=float("inf")
                ),
                np.int64(1),
            ),
            (
                AddGaussianNoise(
                    input_domain=NumpyFloatDomain(), sigma_squared=float("inf")
                ),
                np.float64(-1),
            ),
        ],
    )
    def test_infinite_protection(self, measurement: Measurement, input_data: Any):
        """Measurement works with infinite protection.

        Args:
            measurement: The measurement.
            input_data: The input data for the measurement.
        """
        output = measurement(input_data)
        assert output in [float("inf"), -float("inf")]

    @pytest.mark.skip("No arguments to mutate")
    def test_mutable_inputs(
        self,
        measurement_type: Type[Measurement],
        measurement_args: Dict[str, Any],
        key: str,
        mutator: Callable,
    ):
        """Mutable inputs to the measurement are copied.

        Args:
            measurement_type: The type of the measurement to be constructed.
            measurement_args: The arguments to the measurement.
            key: The parameter name to be changed.
            mutator: A lambda function that mutates the parameter.
        """
        super().test_mutable_inputs(measurement_type, measurement_args, key, mutator)

    @pytest.mark.parametrize(
        "measurement, input_data, expected_output",
        [
            (
                AddGaussianNoise(input_domain=NumpyIntegerDomain(), sigma_squared=0),
                np.int64(10),
                10.0,
            ),
            (
                AddGaussianNoise(
                    input_domain=NumpyFloatDomain(size=32), sigma_squared=0
                ),
                np.float32(-0.1234567),
                float(np.float32(-0.1234567)),
            ),
        ],
    )
    def test_no_protection(
        self, measurement: Measurement, input_data: Any, expected_output: Any
    ):
        """Measurement works with no protection.

        Args:
            measurement: The measurement.
            input_data: The input data for the measurement.
            expected_output: Expected unprotected output.
        """
        super().test_no_protection(measurement, input_data, expected_output)

    @pytest.mark.parametrize(
        "measurement, input_data, expected_output_type",
        [
            (
                AddGaussianNoise(input_domain=NumpyIntegerDomain(), sigma_squared=1),
                np.int64(10),
                float,
            ),
            (
                AddGaussianNoise(
                    input_domain=NumpyFloatDomain(size=32), sigma_squared=1
                ),
                np.float32(-0.1234567),
                float,
            ),
        ],
    )
    def test_output(
        self, measurement: Measurement, input_data: Any, expected_output_type: type
    ):
        """Measurement produces an output that has the expected type.

        Args:
            measurement: The measurement.
            input_data: The input data for the measurement.
            expected_output_type: The expected type for the output.
        """
        super().test_output(measurement, input_data, expected_output_type)

    @pytest.mark.parametrize(
        "measurement, d_in, expected_d_out, expectation",
        [
            (
                AddGaussianNoise(input_domain=NumpyIntegerDomain(), sigma_squared=1),
                1,
                "0.5",
                does_not_raise(),
            ),
            (
                AddGaussianNoise(input_domain=NumpyIntegerDomain(), sigma_squared=2),
                1,
                "0.25",
                does_not_raise(),
            ),
            (
                AddGaussianNoise(input_domain=NumpyFloatDomain(), sigma_squared="0.5"),
                1,
                1,
                does_not_raise(),
            ),
            (
                AddGaussianNoise(input_domain=NumpyFloatDomain(), sigma_squared="0.5"),
                3,
                9,
                does_not_raise(),
            ),
            (
                AddGaussianNoise(input_domain=NumpyFloatDomain(), sigma_squared="0.5"),
                -1,
                None,
                pytest.raises(
                    ValueError,
                    match="Invalid value for metric AbsoluteDifference: -1 is not "
                    "greater than or equal to 0",
                ),
            ),
            (
                AddGaussianNoise(input_domain=NumpyIntegerDomain(), sigma_squared=0),
                1,
                float("inf"),
                does_not_raise(),
            ),
            (
                AddGaussianNoise(input_domain=NumpyIntegerDomain(), sigma_squared=0),
                0,
                0,
                does_not_raise(),
            ),
            (
                AddGaussianNoise(
                    input_domain=NumpyIntegerDomain(), sigma_squared=float("inf")
                ),
                1,
                0,
                does_not_raise(),
            ),
        ],
    )
    def test_privacy_function(
        self,
        measurement: Measurement,
        d_in: Any,
        expected_d_out: Any,
        expectation: ContextManager[None],
    ):
        """Testing the measurement's privacy function.

        Args:
            measurement: The measurement to be tested.
            d_in: Distance between inputs.
            expected_d_out: Expected distance between outputs.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
        """
        super().test_privacy_function(
            measurement, d_in, expected_d_out, expectation, exception_properties=None
        )

    @pytest.mark.parametrize(
        "measurement, d_in, d_out, expected, expectation",
        [
            (
                AddGaussianNoise(input_domain=NumpyIntegerDomain(), sigma_squared=1),
                1,
                "1/2",
                True,
                does_not_raise(),
            ),
            (
                AddGaussianNoise(input_domain=NumpyIntegerDomain(), sigma_squared=1),
                1,
                2,
                True,
                does_not_raise(),
            ),
            (
                AddGaussianNoise(input_domain=NumpyIntegerDomain(), sigma_squared=1),
                1,
                "0.3",
                False,
                does_not_raise(),
            ),
            (
                AddGaussianNoise(input_domain=NumpyFloatDomain(), sigma_squared="0.5"),
                3,
                10,
                True,
                does_not_raise(),
            ),
            (
                AddGaussianNoise(input_domain=NumpyFloatDomain(), sigma_squared="0.5"),
                3,
                5,
                False,
                does_not_raise(),
            ),
            (
                AddGaussianNoise(input_domain=NumpyFloatDomain(), sigma_squared="0.5"),
                -1,
                1,
                True,
                pytest.raises(
                    ValueError,
                    match="Invalid value for metric AbsoluteDifference: -1 is not "
                    "greater than or equal to 0",
                ),
            ),
            (
                AddGaussianNoise(input_domain=NumpyFloatDomain(), sigma_squared="0.5"),
                tuple(),
                1,
                True,
                pytest.raises(
                    TypeCheckError,
                    match=re.escape('"d_in"'),
                ),
            ),
            (
                AddGaussianNoise(input_domain=NumpyFloatDomain(), sigma_squared="0.5"),
                1,
                -1,
                True,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "Invalid RhoZCDP measure value (rho): -1 is not greater than "
                        "or equal to 0"
                    ),
                ),
            ),
            (
                AddGaussianNoise(input_domain=NumpyFloatDomain(), sigma_squared="0.5"),
                1,
                {},
                True,
                pytest.raises(
                    TypeCheckError,
                    match=re.escape('"value"'),
                ),
            ),
            (
                AddGaussianNoise(input_domain=NumpyIntegerDomain(), sigma_squared=0),
                1,
                float("inf"),
                True,
                does_not_raise(),
            ),
            (
                AddGaussianNoise(input_domain=NumpyIntegerDomain(), sigma_squared=0),
                0,
                0,
                True,
                does_not_raise(),
            ),
            (
                AddGaussianNoise(
                    input_domain=NumpyIntegerDomain(), sigma_squared=float("inf")
                ),
                1,
                0,
                True,
                does_not_raise(),
            ),
        ],
    )
    def test_privacy_relation(
        self,
        measurement: Measurement,
        d_in: Any,
        d_out: Any,
        expected: bool,
        expectation: ContextManager[None],
    ):
        """Testing the measurement's privacy function.

        Args:
            measurement: The measurement to be tested.
            d_in: Distance between inputs.
            d_out: Distance between outputs.
            expected: Whether the d_in, d_out pair is close.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
        """
        super().test_privacy_relation(
            measurement, d_in, d_out, expected, expectation, exception_properties=None
        )

    @pytest.mark.parametrize(
        "measurement, expected_properties",
        [
            (
                AddGaussianNoise(input_domain=NumpyIntegerDomain(), sigma_squared=1),
                {
                    "input_domain": NumpyIntegerDomain(),
                    "sigma_squared": 1,
                    "input_metric": AbsoluteDifference(),
                    "output_type": DoubleType(),
                    "output_measure": RhoZCDP(),
                    "is_interactive": False,
                    "adds_no_noise": False,
                },
            ),
            (
                AddGaussianNoise(
                    input_domain=NumpyFloatDomain(size=32), sigma_squared=2
                ),
                {
                    "input_domain": NumpyFloatDomain(size=32),
                    "sigma_squared": 2,
                    "input_metric": AbsoluteDifference(),
                    "output_type": DoubleType(),
                    "output_measure": RhoZCDP(),
                    "is_interactive": False,
                    "adds_no_noise": False,
                },
            ),
            (
                AddGaussianNoise(input_domain=NumpyIntegerDomain(), sigma_squared=0),
                {
                    "input_domain": NumpyIntegerDomain(),
                    "sigma_squared": 0,
                    "input_metric": AbsoluteDifference(),
                    "output_type": DoubleType(),
                    "output_measure": RhoZCDP(),
                    "is_interactive": False,
                    "adds_no_noise": True,
                },
            ),
        ],
    )
    def test_properties(
        self, measurement: Measurement, expected_properties: Dict[str, Any]
    ):
        """All properties have the expected values.

        Args:
            measurement: The constructed measurement to be tested.
            expected_properties: A dictionary containing all the
                property:value pairs measurement is expected to have.
        """
        super().test_properties(measurement, expected_properties)

    @pytest.mark.parametrize(
        "measurement",
        [AddGaussianNoise(input_domain=NumpyIntegerDomain(), sigma_squared=1)],
    )
    def test_property_immutability(self, measurement: Measurement):
        """The properties return copies for mutable values.

        Args:
            measurement: The measurement to be tested.
        """
        super().test_property_immutability(measurement)


class TestAddDiscreteGaussianNoise(MeasurementTests):
    """:class:`~tmlt.core.measurements.noise_mechanisms.AddDiscreteGaussianNoise`."""

    @pytest.fixture
    def measurement_type(self) -> Type[Measurement]:
        """Returns the type of the measurement to be tested."""
        return AddDiscreteGaussianNoise

    @pytest.mark.parametrize(
        "measurement_args, expectation",
        [
            ({"sigma_squared": sp.Rational("0.3")}, does_not_raise()),
            (
                {"sigma_squared": "x"},
                pytest.raises(
                    ValueError, match="Invalid sigma_squared: x contains free symbols"
                ),
            ),
            (
                {"sigma_squared": -1},
                pytest.raises(
                    ValueError,
                    match="Invalid sigma_squared: -1 is not greater than or equal to 0",
                ),
            ),
            (
                {"sigma_squared": float("inf")},
                pytest.raises(
                    ValueError,
                    match="Invalid sigma_squared: inf is not strictly less than inf",
                ),
            ),
            ({"sigma_squared": "0"}, does_not_raise()),
        ],
    )
    def test_construct_component(
        self,
        measurement_type: Type[Measurement],
        measurement_args: Dict[str, Any],
        expectation: ContextManager[None],
    ):
        """Initialization behaves correctly.

        The measurement is constructed correctly and raises exceptions
        when initialized with invalid inputs.

        Args:
            measurement_type: The type of the measurement to be constructed.
            measurement_args: The arguments to the measurement.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
        """
        super().test_construct_component(
            measurement_type, measurement_args, expectation, exception_properties=None
        )

    @pytest.mark.skip("Doesn't support infinite protection")
    def test_infinite_protection(
        self, measurement: Measurement, input_data: Any, expected_output: Any
    ):
        """Measurement works with infinite protection.

        Args:
            measurement: The measurement.
            input_data: The input data for the measurement.
            expected_output: The expected output of the measurement.
        """
        super().test_infinite_protection(measurement, input_data, expected_output)

    @pytest.mark.skip("No arguments to mutate")
    def test_mutable_inputs(
        self,
        measurement_type: Type[Measurement],
        measurement_args: Dict[str, Any],
        key: str,
        mutator: Callable,
    ):
        """Mutable inputs to the measurement are copied.

        Args:
            measurement_type: The type of the measurement to be constructed.
            measurement_args: The arguments to the measurement.
            key: The parameter name to be changed.
            mutator: A lambda function that mutates the parameter.
        """
        super().test_mutable_inputs(measurement_type, measurement_args, key, mutator)

    @pytest.mark.parametrize(
        "measurement, input_data, expected_output",
        [
            (AddDiscreteGaussianNoise(sigma_squared=0), np.int64(10), 10),
            (AddDiscreteGaussianNoise(sigma_squared=0), np.int64(-10), -10),
        ],
    )
    def test_no_protection(
        self, measurement: Measurement, input_data: Any, expected_output: Any
    ):
        """Measurement works with no protection.

        Args:
            measurement: The measurement.
            input_data: The input data for the measurement.
            expected_output: Expected unprotected output.
        """
        super().test_no_protection(measurement, input_data, expected_output)

    @pytest.mark.parametrize(
        "measurement, input_data, expected_output_type",
        [(AddDiscreteGaussianNoise(sigma_squared=3), np.int64(0), int)],
    )
    def test_output(
        self, measurement: Measurement, input_data: Any, expected_output_type: type
    ):
        """Measurement produces an output that has the expected type.

        Args:
            measurement: The measurement.
            input_data: The input data for the measurement.
            expected_output_type: The expected type for the output.
        """
        super().test_output(measurement, input_data, expected_output_type)

    @pytest.mark.parametrize(
        "measurement, d_in, expected_d_out, expectation",
        [
            (AddDiscreteGaussianNoise(sigma_squared=1), 1, "1/2", does_not_raise()),
            (AddDiscreteGaussianNoise(sigma_squared=2), 1, "1/4", does_not_raise()),
            (AddDiscreteGaussianNoise(sigma_squared="0.5"), 1, 1, does_not_raise()),
            (AddDiscreteGaussianNoise(sigma_squared="0.5"), 3, 9, does_not_raise()),
            (
                AddDiscreteGaussianNoise(sigma_squared="0.5"),
                -1,
                None,
                pytest.raises(
                    ValueError,
                    match="Invalid value for metric AbsoluteDifference: -1 is not "
                    "greater than or equal to 0",
                ),
            ),
            (
                AddDiscreteGaussianNoise(sigma_squared=0),
                1,
                float("inf"),
                does_not_raise(),
            ),
            (AddDiscreteGaussianNoise(sigma_squared=0), 0, 0, does_not_raise()),
        ],
    )
    def test_privacy_function(
        self,
        measurement: Measurement,
        d_in: Any,
        expected_d_out: Any,
        expectation: ContextManager[None],
    ):
        """Testing the measurement's privacy function.

        Args:
            measurement: The measurement to be tested.
            d_in: Distance between inputs.
            expected_d_out: Expected distance between outputs.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
        """
        super().test_privacy_function(
            measurement, d_in, expected_d_out, expectation, exception_properties=None
        )

    @pytest.mark.parametrize(
        "measurement, d_in, d_out, expected, expectation",
        [
            (
                AddDiscreteGaussianNoise(sigma_squared=1),
                1,
                "1/2",
                True,
                does_not_raise(),
            ),
            (AddDiscreteGaussianNoise(sigma_squared=1), 1, 1, True, does_not_raise()),
            (
                AddDiscreteGaussianNoise(sigma_squared=1),
                1,
                "0.3",
                False,
                does_not_raise(),
            ),
            (
                AddDiscreteGaussianNoise(sigma_squared="0.5"),
                3,
                9,
                True,
                does_not_raise(),
            ),
            (
                AddDiscreteGaussianNoise(sigma_squared="0.5"),
                3,
                7,
                False,
                does_not_raise(),
            ),
            (
                AddDiscreteGaussianNoise(sigma_squared="0.5"),
                -1,
                1,
                True,
                pytest.raises(
                    ValueError,
                    match="Invalid value for metric AbsoluteDifference: -1 is not "
                    "greater than or equal to 0",
                ),
            ),
            (
                AddDiscreteGaussianNoise(sigma_squared="0.5"),
                tuple(),
                1,
                True,
                pytest.raises(
                    TypeCheckError,
                    match=re.escape('"d_in"'),
                ),
            ),
            (
                AddDiscreteGaussianNoise(sigma_squared="0.5"),
                1,
                -1,
                True,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "Invalid RhoZCDP measure value (rho): -1 is not greater than "
                        "or equal to 0"
                    ),
                ),
            ),
            (
                AddDiscreteGaussianNoise(sigma_squared="0.5"),
                1,
                {},
                True,
                pytest.raises(
                    TypeCheckError,
                    match=re.escape('"value"'),
                ),
            ),
            (
                AddDiscreteGaussianNoise(sigma_squared=0),
                1,
                float("inf"),
                True,
                does_not_raise(),
            ),
            (AddDiscreteGaussianNoise(sigma_squared=0), 0, 0, True, does_not_raise()),
        ],
    )
    def test_privacy_relation(
        self,
        measurement: Measurement,
        d_in: Any,
        d_out: Any,
        expected: bool,
        expectation: ContextManager[None],
    ):
        """Testing the measurement's privacy function.

        Args:
            measurement: The measurement to be tested.
            d_in: Distance between inputs.
            d_out: Distance between outputs.
            expected: Whether the d_in, d_out pair is close.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
        """
        super().test_privacy_relation(
            measurement, d_in, d_out, expected, expectation, exception_properties=None
        )

    @pytest.mark.parametrize(
        "measurement, expected_properties",
        [
            (
                AddDiscreteGaussianNoise(sigma_squared=1),
                {
                    "input_domain": NumpyIntegerDomain(),
                    "sigma_squared": 1,
                    "input_metric": AbsoluteDifference(),
                    "output_type": LongType(),
                    "output_measure": RhoZCDP(),
                    "is_interactive": False,
                    "adds_no_noise": False,
                },
            ),
            (
                AddDiscreteGaussianNoise(sigma_squared=2),
                {
                    "input_domain": NumpyIntegerDomain(),
                    "sigma_squared": 2,
                    "input_metric": AbsoluteDifference(),
                    "output_type": LongType(),
                    "output_measure": RhoZCDP(),
                    "is_interactive": False,
                    "adds_no_noise": False,
                },
            ),
            (
                AddDiscreteGaussianNoise(sigma_squared=0),
                {
                    "input_domain": NumpyIntegerDomain(),
                    "sigma_squared": 0,
                    "input_metric": AbsoluteDifference(),
                    "output_type": LongType(),
                    "output_measure": RhoZCDP(),
                    "is_interactive": False,
                    "adds_no_noise": True,
                },
            ),
        ],
    )
    def test_properties(
        self, measurement: Measurement, expected_properties: Dict[str, Any]
    ):
        """All properties have the expected values.

        Args:
            measurement: The constructed measurement to be tested.
            expected_properties: A dictionary containing all the
                property:value pairs measurement is expected to have.
        """
        super().test_properties(measurement, expected_properties)

    @pytest.mark.parametrize("measurement", [AddDiscreteGaussianNoise(sigma_squared=1)])
    def test_property_immutability(self, measurement: Measurement):
        """The properties return copies for mutable values.

        Args:
            measurement: The measurement to be tested.
        """
        super().test_property_immutability(measurement)

    @pytest.mark.parametrize(
        "sigma_squared, p, expected, expectation",
        [
            (1, 0.5, 0, does_not_raise()),
            (2, 0.5, 0, does_not_raise()),
            (3.1415, 0.5, 0, does_not_raise()),
            (1, 0.75, 1, does_not_raise()),
            (1, 0.25, -1, does_not_raise()),
            (3.5, 0.86, 2, does_not_raise()),
            (3.5, 0.14, -2, does_not_raise()),
            (0, 0.6, 0, does_not_raise()),  # 0 scale is 0
            (  # inf scale is +inf 50% of the time
                float("inf"),
                0.6,
                float("inf"),
                does_not_raise(),
            ),
            (  # inf scale is -inf 50% of the time
                float("inf"),
                0.4,
                float("-inf"),
                does_not_raise(),
            ),
            (1, 0, float("-inf"), does_not_raise()),  # p=0 is -inf
            (1, 1, float("inf"), does_not_raise()),  # p=1 is +inf
            (-1, 0.6, float("nan"), does_not_raise()),  # negative scale is nan
            (float("nan"), 0.6, float("nan"), does_not_raise()),  # nan scale is nan
        ]
        + [  # p values outside [0, 1] raise ValueError
            (
                1,
                bad_p,
                float("nan"),
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "Probabilities input to the inverse CDF must be in [0, 1], "
                        f"but got {bad_p}"
                    ),
                ),
            )
            for bad_p in [-1, 1.1, float("inf"), float("nan")]
        ],
    )
    def test_inverse_cdf(
        self,
        sigma_squared: float,
        p: float,
        expected: float,
        expectation: ContextManager[None],
    ):
        """Testing the inverse_cdf function.

        Args:
            sigma_squared: The sigma_squared of the DiscreteGaussian distribution.
            p: The probability.
            expected: The expected value.
            expectation: A context manager that captures the correct expected type of
        """
        with expectation:
            actual = AddDiscreteGaussianNoise.inverse_cdf(
                sigma_squared=sigma_squared, probability=p
            )
            if np.isnan(expected):
                assert np.isnan(actual), f"Expected {expected}, got {actual}"
            else:
                assert actual == pytest.approx(
                    expected
                ), f"Expected {expected}, got {actual}"
