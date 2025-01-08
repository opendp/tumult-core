"""Abstract class for testing transformations."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import copy

# pylint: disable=unused-import
from abc import ABC, abstractmethod
from contextlib import nullcontext as does_not_raise
from test.conftest import assert_frame_equal_with_sort
from typing import Any, Callable, ContextManager, Dict, Optional, Type

import pandas as pd
import pytest
from pyspark.sql import DataFrame

from tmlt.core.transformations.base import Transformation
from tmlt.core.utils.testing import assert_property_immutability, get_all_props


class TransformationTests(ABC):
    """Test classes for Transformations should inherit from this class."""

    @abstractmethod
    @pytest.fixture
    def transformation_type(
        self,
    ) -> Type[Transformation]:
        """Returns the type of the transformation to be tested."""
        return Transformation

    @abstractmethod
    def test_mutable_inputs(
        self,
        transformation_type: Type[Transformation],
        transformation_args: Dict[str, Any],
        key: str,
        mutator: Callable,
    ):
        """Mutable inputs to the transformation are copied.

        Args:
            transformation_type: The type of the transformation to be constructed.
            transformation_args: The arguments to the transformation.
            key: The parameter name to be changed.
            mutator: A lambda function that mutates the parameter.
        """
        transformation = transformation_type(**transformation_args)
        assert hasattr(transformation, key)
        expected = copy.deepcopy(getattr(transformation, key))
        mutator(transformation_args[key])
        actual_value = getattr(transformation, key)
        assert (
            getattr(transformation, key) == expected
        ), f"Expected {key} to be {expected}, got {actual_value}"

    @abstractmethod
    def test_property_immutability(self, transformation: Transformation):
        """The properties return copies for mutable values.

        Args:
            transformation: The transformation to be tested.
        """
        props = [prop[0] for prop in get_all_props(type(transformation))]
        for prop in props:
            assert_property_immutability(transformation, prop)

    @abstractmethod
    def test_properties(
        self, transformation: Transformation, expected_properties: Dict[str, Any]
    ):
        """All properties have the expected values.

        Args:
            transformation: The constructed transformation to be tested.
            expected_properties: A dictionary containing all the
                property:value pairs transformation is expected to have.
        """
        actual_props = [prop[0] for prop in get_all_props(type(transformation))]
        assert set(expected_properties.keys()) == set(actual_props)
        for prop, expected_val in expected_properties.items():
            assert hasattr(transformation, prop), f"{prop} not in {transformation}"
            actual_value = getattr(transformation, prop)
            assert (
                getattr(transformation, prop) == expected_val
            ), f"Expected {prop} to be {expected_val}, got {actual_value}"

    @abstractmethod
    def test_construct_component(
        self,
        transformation_type: Type[Transformation],
        transformation_args: Dict[str, Any],
        expectation: ContextManager[None],
        exception_properties: Optional[Dict[str, Any]],
    ):
        """Initialization behaves correctly.

        The transformation is constructed correctly and raises exceptions
        when initialized with invalid inputs.

        Args:
            transformation_type: The type of the transformation to be constructed.
            transformation_args: The arguments to the transformation.
            expectation: A context manager that captures the
                correct expected type of error that is raised.
            exception_properties: A dictionary containing all the property:value pairs
                the exception is expected to have. Mostly used for testing the custom
                exceptions.
        """
        with expectation as exception:
            transformation_type(**transformation_args)
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
    def test_stability_function(
        self,
        transformation: Transformation,
        d_in: Any,
        expected_d_out: Any,
        expectation: ContextManager[None],
        exception_properties: Optional[Dict[str, Any]],
    ):
        """Testing the transformation's stability function.

        Args:
            transformation: The transformation to be tested.
            d_in: Distance between inputs.
            expected_d_out: Expected distance between outputs.
            expectation: A context manager that captures the
                correct expected type of error that is raised.
            exception_properties: A dictionary containing all the property:value pairs
                the exception is expected to have. Mostly used for testing the custom
                exceptions.
        """
        with expectation as exception:
            assert transformation.stability_function(d_in) == expected_d_out
        if exception_properties is None or len(exception_properties) == 0:
            return
        for prop, expected_value in exception_properties.items():
            assert hasattr(exception, prop), f"{prop} not in {exception}"
            actual_value = getattr(exception, prop)
            assert (
                getattr(exception, prop) == expected_value
            ), f"Expected {prop} to be {expected_value}, got {actual_value}"

    @abstractmethod
    def test_stability_relation(
        self,
        transformation: Transformation,
        d_in: Any,
        d_out: Any,
        expected: bool,
        expectation: ContextManager[None],
        exception_properties: Optional[Dict[str, Any]],
    ):
        """Testing the transformation's stability relation.

        Args:
            transformation: The transformation to be tested.
            d_in: Distance between inputs.
            d_out: Distance between outputs.
            expected: Whether the d_in, d_out pair is close.
            expectation: A context manager that captures the
                correct expected type of error that is raised.
            exception_properties: A dictionary containing all the property:value pairs
                the exception is expected to have. Mostly used for testing the custom
                exceptions.
        """
        with expectation as exception:
            assert transformation.stability_relation(d_in, d_out) == expected
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
        self, transformation: Transformation, input_data: Any, expected_output: Any
    ):
        """Output is still within the expected domain.

        Args:
            transformation: The transformation.
            input_data: The input data for the transformation.
            expected_output: The expected output to match.
        """
        output = transformation(input_data)
        transformation.output_domain.validate(output)
        if isinstance(expected_output, (pd.DataFrame, DataFrame)):
            assert_frame_equal_with_sort(output, expected_output)
        else:
            assert output == expected_output
