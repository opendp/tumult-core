"""Abstract class for testing measures."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import copy

# pylint: disable=unused-import
from abc import ABC, abstractmethod
from contextlib import nullcontext as does_not_raise
from typing import Any, Callable, ContextManager, Dict, Optional, Type

import pytest

from tmlt.core.measures import Measure
from tmlt.core.utils.exact_number import ExactNumberInput
from tmlt.core.utils.testing import assert_property_immutability, get_all_props


class MeasureTests(ABC):
    """Test classes for Measures should inherit from this class."""

    @abstractmethod
    @pytest.fixture
    def measure_type(self) -> Type[Measure]:
        """Returns the type of the measure to be tested."""
        return Measure

    @abstractmethod
    def test_validate(
        self,
        measure: Measure,
        candidate: Any,
        expectation: ContextManager[None],
        exception_properties: Optional[Dict[str, Any]],
    ):
        """Validate works correctly.

        Args:
            measure: The measure to test.
            candidate: The value to validate using measure.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
            exception_properties: A dictionary containing all the property:value pairs
                the exception is expected to have. Mostly used for testing the custom
                exceptions.
        """
        with expectation as exception:
            measure.validate(candidate)
        if exception_properties is None or len(exception_properties) == 0:
            return
        # Help out mypy
        assert isinstance(exception, pytest.ExceptionInfo)
        for prop, expected_value in exception_properties.items():
            assert hasattr(exception.value, prop), f"Expected prop was missing: {prop}"
            actual_value = getattr(exception.value, prop)
            assert (
                actual_value == expected_value
            ), f"Expected {prop} to be {expected_value}, got {actual_value}"

    @abstractmethod
    def test_mutable_inputs(
        self,
        measure_type: Type[Measure],
        measure_args: Dict[str, Any],
        key: str,
        mutator: Callable,
    ):
        """The mutable inputs to the measure are copied.

        Args:
            measure_type: The type of measure to be constructed.
            measure_args: The arguments to the measure.
            key: The parameter name to be changed.
            mutator: A lambda function that mutates the parameter.
        """
        measure = measure_type(**measure_args)
        assert hasattr(measure, key)
        expected = copy.deepcopy(getattr(measure, key))
        mutator(measure_args[key])
        actual_value = getattr(measure, key)
        assert (
            getattr(measure, key) == expected
        ), f"Expected {key} to be {expected}, got {actual_value}"

    @abstractmethod
    def test_property_immutability(self, measure: Measure):
        """The properties return copies for mutable values.

        Args:
            measure: The measure to be tested.
        """
        props = [prop[0] for prop in get_all_props(type(measure))]
        for prop in props:
            assert_property_immutability(measure, prop)

    @abstractmethod
    def test_properties(self, measure: Measure, expected_properties: Dict[str, Any]):
        """All properties have the expected values.

        Args:
            measure: The constructed measure to be tested.
            expected_properties: A dictionary containing all the
                property:value pairs measure is expected to have.
        """
        actual_props = [prop[0] for prop in get_all_props(type(measure))]
        assert set(expected_properties.keys()) == set(actual_props)
        for prop, expected_val in expected_properties.items():
            assert hasattr(measure, prop), f"{prop} not in {measure}"
            actual_value = getattr(measure, prop)
            assert (
                getattr(measure, prop) == expected_val
            ), f"Expected {prop} to be {expected_val}; got {actual_value}"

    @abstractmethod
    def test_compare(
        self,
        measure: Measure,
        value1: ExactNumberInput,
        value2: ExactNumberInput,
        expected: bool,
        expectation: ContextManager[None],
        exception_properties: Optional[Dict[str, Any]],
    ):
        """Compare returns or raises the expected result.

        Args:
            measure: The measure doing the comparison.
            value1: A distance between two distributions under this measure.
            value2: A distance between two distributions under this measure.
            expected: Expected equivalence.
            expectation: A context manager that captures the
                correct expected type of error that is raised.
            exception_properties: A dictionary containing all the property:value pairs
                the exception is expected to have. Mostly used for testing the custom
                exceptions.
        """
        with expectation as exception:
            assert measure.compare(value1, value2) == expected
        if exception_properties is None or len(exception_properties) == 0:
            return
        for prop, expected_value in exception_properties.items():
            assert hasattr(exception, prop), f"{prop} not in {exception}"
            actual_value = getattr(exception, prop)
            assert (
                getattr(exception, prop) == expected_value
            ), f"Expected {prop} to be {expected_value}; got {actual_value}"

    @abstractmethod
    def test_repr(self, measure: Measure, representation: str):
        """`repr` returns the expected result.

        Args:
            measure: The measure to be represented.
            representation: The expected string representation of the
            measure.
        """
        assert repr(measure) == representation
