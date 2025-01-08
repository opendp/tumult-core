"""Abstract class for testing input/output measurements."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import copy

# pylint: disable=unused-import
from abc import ABC, abstractmethod
from contextlib import nullcontext as does_not_raise
from typing import Any, Callable, ContextManager, Dict, Optional, Type

import pytest

from tmlt.core.measurements.base import Measurement
from tmlt.core.utils.testing import assert_property_immutability, get_all_props


class MeasurementTests(ABC):
    """Test classes for Measurements should inherit from this class."""

    @abstractmethod
    @pytest.fixture
    def measurement_type(self) -> Type[Measurement]:
        """Returns the type of the measurement to be tested."""
        return Measurement

    @abstractmethod
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
        measurement = measurement_type(**measurement_args)
        assert hasattr(measurement, key), f"{key} not in {measurement}"
        expected = copy.deepcopy(getattr(measurement, key))
        mutator(measurement_args[key])
        actual_value = getattr(measurement, key)
        assert (
            getattr(measurement, key) == expected
        ), f"Expected {key} to be {expected}, got {actual_value}"

    @abstractmethod
    def test_property_immutability(self, measurement: Measurement):
        """The properties return copies for mutable values.

        Args:
            measurement: The measurement to be tested.
        """
        props = [prop[0] for prop in get_all_props(type(measurement))]
        for prop in props:
            assert_property_immutability(measurement, prop)

    @abstractmethod
    def test_properties(
        self, measurement: Measurement, expected_properties: Dict[str, Any]
    ):
        """All properties have the expected values.

        Args:
            measurement: The constructed measurement to be tested.
            expected_properties: A dictionary containing all the
                property:value pairs measurement is expected to have.
        """
        actual_props = [prop[0] for prop in get_all_props(type(measurement))]
        assert set(expected_properties.keys()) == set(actual_props), (
            "Keys of expected properties don't match actual properties. Expected "
            f"{set(expected_properties.keys())}; got {set(actual_props)}"
        )
        for prop, expected_val in expected_properties.items():
            assert hasattr(measurement, prop), f"{prop} not in {measurement}"
            actual_value = getattr(measurement, prop)
            assert (
                getattr(measurement, prop) == expected_val
            ), f"Expected {prop} to be {expected_val}; got {actual_value}"

    @abstractmethod
    def test_construct_component(
        self,
        measurement_type: Type[Measurement],
        measurement_args: Dict[str, Any],
        expectation: ContextManager[None],
        exception_properties: Optional[Dict[str, Any]],
    ):
        """Initialization behaves correctly.

        The measurement is constructed correctly and raises exceptions
        when initialized with invalid inputs.

        Args:
            measurement_type: The type of the measurement to be constructed.
            measurement_args: The arguments to the measurement.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
            exception_properties: A dictionary containing all the property:value pairs
                the exception is expected to have. Mostly used for testing the custom
                exceptions.
        """
        with expectation as exception:
            measurement_type(**measurement_args)
        if exception_properties is None or len(exception_properties) == 0:
            return
        # Help out mypy
        assert isinstance(exception, pytest.ExceptionInfo)
        for prop, expected_value in exception_properties.items():
            assert hasattr(exception.value, prop), f"{prop} not in {exception.value}"
            actual_value = getattr(exception.value, prop)
            assert (
                actual_value == expected_value
            ), f"Expected {prop} to be {expected_value}, got {actual_value}"

    @abstractmethod
    def test_privacy_function(
        self,
        measurement: Measurement,
        d_in: Any,
        expected_d_out: Any,
        expectation: ContextManager[None],
        exception_properties: Optional[Dict[str, Any]],
    ):
        """Testing the measurement's privacy function.

        Args:
            measurement: The measurement to be tested.
            d_in: Distance between inputs.
            expected_d_out: Expected distance between outputs.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
            exception_properties: A dictionary containing all the property:value pairs
                the exception is expected to have. Mostly used for testing the custom
                exceptions.
        """
        with expectation as exception:
            d_out = measurement.privacy_function(d_in)
            assert d_out == expected_d_out, f"Expected {expected_d_out}, got {d_out}"
        if exception_properties is None or len(exception_properties) == 0:
            return
        for prop, expected_value in exception_properties.items():
            assert hasattr(exception, prop), f"{prop} not in {exception}"
            actual_value = getattr(exception, prop)
            assert (
                getattr(exception, prop) == expected_value
            ), f"Expected {prop} to be {expected_value}, got {actual_value}"

    @abstractmethod
    def test_privacy_relation(
        self,
        measurement: Measurement,
        d_in: Any,
        d_out: Any,
        expected: bool,
        expectation: ContextManager[None],
        exception_properties: Optional[Dict[str, Any]],
    ):
        """Testing the measurement's privacy function.

        Args:
            measurement: The measurement to be tested.
            d_in: Distance between inputs.
            d_out: Distance between outputs.
            expected: Whether the d_in, d_out pair is close.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
            exception_properties: A dictionary containing all the property:value pairs
                the exception is expected to have. Mostly used for testing the custom
                exceptions.
        """
        with expectation as exception:
            assert measurement.privacy_relation(d_in, d_out) == expected
        if exception_properties is None or len(exception_properties) == 0:
            return
        for prop, expected_value in exception_properties.items():
            assert hasattr(exception, prop), f"{prop} not in {exception}"
            actual_value = getattr(exception, prop)
            assert (
                getattr(exception, prop) == expected_value
            ), f"Expected {prop} to be {expected_value}, got {actual_value}"

    @abstractmethod
    def test_output(
        self, measurement: Measurement, input_data: Any, expected_output_type: type
    ):
        """Measurement produces an output that has the expected type.

        Args:
            measurement: The measurement.
            input_data: The input data for the measurement.
            expected_output_type: The expected type for the output.
        """
        output = measurement(input_data)
        assert isinstance(output, expected_output_type)

    @abstractmethod
    def test_no_protection(
        self, measurement: Measurement, input_data: Any, expected_output: Any
    ):
        """Measurement works with no protection.

        Args:
            measurement: The measurement.
            input_data: The input data for the measurement.
            expected_output: Expected unprotected output.
        """
        output = measurement(input_data)
        assert output == expected_output

    @abstractmethod
    def test_infinite_protection(
        self, measurement: Measurement, input_data: Any, expected_output: Any
    ):
        """Measurement works with infinite protection.

        Args:
            measurement: The measurement.
            input_data: The input data for the measurement.
            expected_output: Expected infinitely protected output.
        """
        output = measurement(input_data)
        assert output == expected_output
