"""Abstract class for testing budgets."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import copy

# pylint: disable=unused-import
from abc import ABC, abstractmethod
from contextlib import nullcontext as does_not_raise
from typing import Any, Callable, ContextManager, Dict, Optional, Type

import pytest

from tmlt.core.measures import PrivacyBudget
from tmlt.core.utils.exact_number import ExactNumberInput
from tmlt.core.utils.testing import assert_property_immutability, get_all_props


class BudgetTests(ABC):
    """Test classes for PrivacyBudgets should inherit from this class."""

    @abstractmethod
    @pytest.fixture
    def budget_type(self) -> Type[PrivacyBudget]:
        """Returns the type of the budget to be tested."""
        return PrivacyBudget

    @abstractmethod
    def test_mutable_inputs(
        self,
        budget_type: Type[PrivacyBudget],
        budget_args: Dict[str, Any],
        key: str,
        mutator: Callable,
    ):
        """The mutable inputs to the budget are copied.

        Args:
            budget_type: The type of budget to be constructed.
            budget_args: The arguments to the budget.
            key: The parameter name to be changed.
            mutator: A lambda function that mutates the parameter.
        """
        budget = budget_type(**budget_args)
        assert hasattr(budget, key)
        expected = copy.deepcopy(getattr(budget, key))
        mutator(budget_args[key])
        assert getattr(budget, key) == expected

    @abstractmethod
    def test_property_immutability(self, budget: PrivacyBudget):
        """The properties return copies for mutable values.

        Args:
            budget: The budget to be tested.
        """
        props = [prop[0] for prop in get_all_props(type(budget))]
        for prop in props:
            assert_property_immutability(budget, prop)

    @abstractmethod
    def test_properties(
        self, budget: PrivacyBudget, expected_properties: Dict[str, Any]
    ):
        """All properties have the expected values.

        Args:
            budget: The constructed budget to be tested.
            expected_properties: A dictionary containing all the
                property:value pairs budget is expected to have.
        """
        actual_props = [prop[0] for prop in get_all_props(type(budget))]
        assert set(expected_properties.keys()) == set(actual_props)
        for prop, expected_val in expected_properties.items():
            assert hasattr(budget, prop)
            assert getattr(budget, prop) == expected_val

    @abstractmethod
    def test_construct_component(
        self,
        budget_type: Type[PrivacyBudget],
        budget_args: Dict[str, Any],
        expectation: ContextManager[None],
        exception_properties: Optional[Dict[str, Any]],
    ):
        """Initialization behaves correctly.

        The budget is constructed correctly and raises exceptions
        when initialized with invalid inputs.

        Args:
            budget_type: The type of budget to be constructed.
            budget_args: The arguments to the budget.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
            exception_properties: A dictionary containing all the property:value pairs
                the exception is expected to have. Mostly used for testing the custom
                exceptions.
        """
        with expectation as exception:
            budget_type(**budget_args)
        if exception_properties is None or len(exception_properties) == 0:
            return
        for prop, expected_value in exception_properties.items():
            assert hasattr(exception, prop)
            assert getattr(exception, prop) == expected_value

    @abstractmethod
    def test_is_finite(
        self, budget_type: Type[PrivacyBudget], value: Any, expected: bool
    ):
        """Test that is_finite returns the expected value."""
        assert budget_type(value).is_finite() == expected

    @abstractmethod
    def test_assert_can_spend_budget(
        self,
        budget_type: Type[PrivacyBudget],
        initial_budget: Any,
        spending_budget: Any,
        expected: bool,
        expectation: ContextManager[None],
        exception_properties: Optional[Dict[str, Any]],
    ):
        """Tests that initial_budget can spend spending_budget.

        Args:
            budget_type: The type of budget to be constructed.
            initial_budget: The initial budget.
            spending_budget: The budget to be spent.
            expected: Whether the initial budget can spend the spending budget.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
            exception_properties: A dictionary containing all the property:value pairs
                the exception is expected to have. Mostly used for testing the custom
                exceptions.
        """
        with expectation as exception:
            assert (
                budget_type(initial_budget).assert_can_spend_budget(spending_budget)
                == expected
            )
        if exception_properties is None or len(exception_properties) == 0:
            return
        for prop, expected_value in exception_properties.items():
            assert hasattr(exception, prop)
            assert getattr(exception, prop) == expected_value

    @abstractmethod
    def test_subtract(
        self,
        budget_type: Type[PrivacyBudget],
        initial_budget: Any,
        spending_budget: Any,
        expected: bool,
        expectation: ContextManager[None],
        exception_properties: Optional[Dict[str, Any]],
    ):
        """Tests that subtract works correctly.

        Args:
            budget_type: The type of budget to be constructed.
            initial_budget: The initial budget.
            spending_budget: The budget to be spent.
            expected: Whether the initial budget subtract the spending budget.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
            exception_properties: A dictionary containing all the property:value pairs
                the exception is expected to have. Mostly used for testing the custom
                exceptions.
        """
        with expectation as exception:
            assert budget_type(initial_budget).subtract(spending_budget) == expected
        if exception_properties is None or len(exception_properties) == 0:
            return
        for prop, expected_value in exception_properties.items():
            assert hasattr(exception, prop)
            assert getattr(exception, prop) == expected_value

    @abstractmethod
    def test_eq(
        self, budget: PrivacyBudget, other_budget: PrivacyBudget, expected: bool
    ):
        """`PrivacyBudget` can be compared correctly.

        Args:
            budget: The budget to compare.
            other_budget: The budget to compare with.
            expected: Expected equivalence.
        """
        assert (budget == other_budget) == expected

    @abstractmethod
    def test_repr(self, budget: PrivacyBudget, representation: str):
        """`repr` returns the expected result.

        Args:
            budget: The budget to be represented.
            representation: The expected string representation of the
            budget.
        """
        assert repr(budget) == representation
