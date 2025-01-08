"""Abstract class for testing input/output domains."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import copy

# pylint: disable=unused-import
from abc import ABC, abstractmethod
from contextlib import nullcontext as does_not_raise
from typing import Any, Callable, ContextManager, Dict, Optional, Type

import pytest

from tmlt.core.domains.base import Domain
from tmlt.core.utils.testing import assert_property_immutability, get_all_props


class DomainTests(ABC):
    """Test classes for Domains should inherit from this class."""

    @abstractmethod
    @pytest.fixture
    def domain_type(self) -> Type[Domain]:
        """Returns the type of the domain to be tested."""
        return Domain

    @abstractmethod
    def test_eq(self, domain: Domain, other_domain: Domain, expected: bool):
        """__eq__ works correctly.

        Args:
            domain: The domain to test.
            other_domain: The domain to compare to.
            expected: The expected result of the comparison.
        """
        assert (domain == other_domain) == expected

    @abstractmethod
    def test_validate(
        self,
        domain: Domain,
        candidate: Any,
        expectation: ContextManager[None],
        exception_properties: Optional[Dict[str, Any]],
    ):
        """Validate works correctly.

        Args:
            domain: The domain to test.
            candidate: The value to validate using domain.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
            exception_properties: A dictionary containing all the property:value pairs
                the exception is expected to have. Mostly used for testing the custom
                exceptions.
        """
        with expectation as exception:
            domain.validate(candidate)
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
        domain_type: Type[Domain],
        domain_args: Dict[str, Any],
        key: str,
        mutator: Callable[[Any], Any],
    ):
        """The mutable inputs to the domain are copied.

        Args:
            domain_type: The type of domain to be constructed.
            domain_args: The arguments to the domain.
            key: The parameter name to be changed.
            mutator: A lambda function that mutates the parameter.
        """
        domain = domain_type(**domain_args)
        assert hasattr(domain, key), f"{key} not in {domain}"
        expected = copy.deepcopy(getattr(domain, key))
        mutator(domain_args[key])
        actual_value = getattr(domain, key)
        assert (
            getattr(domain, key) == expected
        ), f"Expected {key} to be {expected}, got {actual_value}"

    @abstractmethod
    def test_property_immutability(self, domain: Domain):
        """The properties return copies for mutable values.

        Args:
            domain: The domain to be tested.
        """
        props = [prop[0] for prop in get_all_props(type(domain))]
        for prop in props:
            assert_property_immutability(domain, prop)

    @abstractmethod
    def test_properties(self, domain: Domain, expected_properties: Dict[str, Any]):
        """All properties have the expected values.

        Args:
            domain: The constructed domain to be tested.
            expected_properties: A dictionary containing all the property:value pairs
                domain is expected to have.
        """
        actual_props = [prop[0] for prop in get_all_props(type(domain))]
        assert set(expected_properties.keys()) == set(actual_props)
        for prop, expected_val in expected_properties.items():
            assert hasattr(domain, prop), f"{prop} not in {domain}"
            actual_value = getattr(domain, prop)
            assert (
                actual_value == expected_val
            ), f"Expected {prop} to be {expected_val}, got {actual_value}"

    @abstractmethod
    def test_construct_component(
        self,
        domain_type: Type[Domain],
        domain_args: Dict[str, Any],
        expectation: ContextManager[None],
        exception_properties: Optional[Dict[str, Any]],
    ):
        """Initialization behaves correctly.

        The domain is constructed correctly and raises exceptions when initialized with
        invalid inputs.

        Args:
            domain_type: The type of domain to be constructed.
            domain_args: The arguments to the domain.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
            exception_properties: A dictionary containing all the property:value pairs
                the exception is expected to have. Mostly used for testing the custom
                exceptions.
        """
        with expectation as exception:
            domain_type(**domain_args)
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
