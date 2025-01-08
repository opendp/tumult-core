"""Abstract class for testing metrics."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import copy

# pylint: disable=unused-import
from abc import ABC, abstractmethod
from contextlib import nullcontext as does_not_raise
from typing import Any, Callable, ContextManager, Dict, Optional, Type

import pytest

from tmlt.core.domains.base import Domain
from tmlt.core.metrics import Metric
from tmlt.core.utils.exact_number import ExactNumberInput
from tmlt.core.utils.testing import assert_property_immutability, get_all_props


class MetricTests(ABC):
    """Test classes for Metrics should inherit from this class."""

    @abstractmethod
    @pytest.fixture
    def metric_type(self) -> Type[Metric]:
        """Returns the type of the metric to be tested."""
        return Metric

    @abstractmethod
    def test_mutable_inputs(
        self,
        metric_type: Type[Metric],
        metric_args: Dict[str, Any],
        key: str,
        mutator: Callable,
    ):
        """The mutable inputs to the metric are copied.

        Args:
            metric_type: The type of metric to be constructed.
            metric_args: The arguments to the metric.
            key: The parameter name to be changed.
            mutator: A lambda function that mutates the parameter.
        """
        metric = metric_type(**metric_args)
        assert hasattr(metric, key)
        expected = copy.deepcopy(getattr(metric, key))
        mutator(metric_args[key])
        actual_value = getattr(metric, key)
        assert (
            getattr(metric, key) == expected
        ), f"Expected {key} to be {expected}, got {actual_value}"

    @abstractmethod
    def test_property_immutability(self, metric: Metric):
        """The properties return copies for mutable values.

        Args:
            metric: The measure to be tested.
        """
        props = [prop[0] for prop in get_all_props(type(metric))]
        for prop in props:
            assert_property_immutability(metric, prop)

    @abstractmethod
    def test_properties(self, metric_type: type, metric_args: Dict[str, Any]):
        """All properties have the expected values.

        Args:
            metric_type: The type of metric to be constructed.
            metric_args: The arguments to the metric.
        """
        metric = metric_type(**metric_args)
        for arg_name, value in metric_args.items():
            if not hasattr(metric, arg_name):
                continue
            assert getattr(metric, arg_name) == value

    @abstractmethod
    def test_construct_component(
        self,
        metric_type: Type[Metric],
        metric_args: Dict[str, Any],
        expectation: ContextManager[None],
        exception_properties: Optional[Dict[str, Any]],
    ):
        """Initialization behaves correctly.

        The metric is constructed correctly and raises exceptions
        when initialized with invalid inputs.

        Args:
            metric_type: The type of metric to be constructed.
            metric_args: The arguments to the metric.
            expectation: A context manager that captures the
                correct expected type of error that is raised.
            exception_properties: A dictionary containing all the property:value pairs
                the exception is expected to have. Mostly used for testing the custom
                exceptions.
        """
        with expectation as exception:
            metric_type(**metric_args)
        if exception_properties is None or len(exception_properties) == 0:
            return
        for prop, expected_value in exception_properties.items():
            assert hasattr(exception, prop), f"{prop} not in {exception}"
            actual_value = getattr(exception, prop)
            assert (
                getattr(exception, prop) == expected_value
            ), f"Expected {prop} to be {expected_value}, got {actual_value}"

    @abstractmethod
    def test_eq(self, metric: Metric, other_metric: Metric, expected: bool):
        """Metrics can be compared correctly.

        Args:
            metric: The metric to compare.
            other_metric: The metric to compare with.
            expected: Expected equivalence.
        """
        assert (metric == other_metric) == expected

    @abstractmethod
    def test_validate(
        self,
        metric: Metric,
        candidate: Any,
        expectation: ContextManager[None],
        exception_properties: Optional[Dict[str, Any]],
    ):
        """Validate works correctly.

        Args:
            metric: The metric to test.
            candidate: The value to validate using metric.
            expectation: A context manager that captures the
                correct expected type of error that is raised.
            exception_properties: A dictionary containing all the property:value pairs
                the exception is expected to have. Mostly used for testing the custom
                exceptions.
        """
        with expectation as exception:
            metric.validate(candidate)
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
    def test_compare(
        self,
        metric: Metric,
        value1: ExactNumberInput,
        value2: ExactNumberInput,
        expected: bool,
    ):
        """Compare returns the expected result.

        Args:
            metric: The metric doing the comparison.
            value1: A distance between two datasets under this metric.
            value2: A distance between two datasets under this metric.
            expected: Expected equivalence.
        """
        assert metric.compare(value1, value2) == expected

    @abstractmethod
    def test_repr(self, metric: Metric, representation: str):
        """`repr` creates the correct representation of the metric.

        Args:
            metric: The metric to be represented.
            representation: The expected string representation of the
            metric.
        """
        assert repr(metric) == representation

    @abstractmethod
    def test_distance(
        self,
        metric: Metric,
        domain: Domain,
        value1: Any,
        value2: Any,
        expectation: ContextManager[None],
        exception_properties: Optional[Dict[str, Any]],
    ):
        """Distances raises an exception for invalid inputs.

        Args:
            metric: The metric performing the comparison.
            domain: The domain under which the metric performs
                the comparison under.
            value1: A distance between two datasets under this metric.
            value2: A distance between two datasets under this metric.
            expectation: A context manager that captures the
                correct expected type of error that is raised.
            exception_properties: A dictionary containing all the property:value pairs
                the exception is expected to have. Mostly used for testing the custom
                exceptions.
        """
        with expectation as exception:
            metric.distance(value1, value2, domain)
        if exception_properties is None or len(exception_properties) == 0:
            return
        for prop, expected_value in exception_properties.items():
            assert hasattr(exception, prop), f"{prop} not in {exception}"
            actual_value = getattr(exception, prop)
            assert (
                getattr(exception, prop) == expected_value
            ), f"Expected {prop} to be {expected_value}, got {actual_value}"

    @abstractmethod
    def test_supports_domain(self, metric: Metric, domain: Domain, expected: bool):
        """Metric supports the correct domains.

        Args:
            metric: The metric to test.
            domain: The domain to test.
            expected: The expected result.
        """
        assert metric.supports_domain(domain) == expected
