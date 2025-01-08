"""Unit tests for :mod:`~tmlt.core.domains.collections`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import re
from contextlib import nullcontext as does_not_raise
from dataclasses import dataclass
from itertools import combinations_with_replacement
from test.unit.domains.abstract import DomainTests
from typing import Any, Callable, ContextManager, Dict, Optional, Type

import numpy as np
import pytest
from pyspark.sql.types import StringType
from typeguard import TypeCheckError

from tmlt.core.domains.base import Domain, OutOfDomainError
from tmlt.core.domains.collections import DictDomain, ListDomain
from tmlt.core.domains.numpy_domains import NumpyFloatDomain, NumpyIntegerDomain
from tmlt.core.utils.misc import get_fullname


class TestListDomain(DomainTests):
    """Tests for :class:`~tmlt.core.domains.collections.ListDomain`."""

    @pytest.fixture
    def domain_type(self) -> Type[Domain]:
        """Returns the type of the domain to be tested."""
        return ListDomain

    @pytest.fixture(scope="class")
    def domain(self) -> ListDomain:
        """Get a base ListDomain."""
        return ListDomain(NumpyIntegerDomain())

    @pytest.mark.parametrize(
        "domain_args, expectation, exception_properties",
        [
            (
                {"element_domain": invalid_type},
                pytest.raises(
                    TypeCheckError,
                ),
                None,
            )
            for invalid_type in [None, "not a domain", StringType, np.int64(1)]
        ]
        + [
            (
                {"element_domain": NumpyIntegerDomain(), "length": -1},
                pytest.raises(ValueError, match="length must be non-negative"),
                None,
            ),
            (
                {"element_domain": NumpyFloatDomain(), "length": 1.5},
                pytest.raises(
                    TypeCheckError,
                ),
                None,
            ),
            (
                {"element_domain": NumpyIntegerDomain(), "length": 1},
                does_not_raise(),
                None,
            ),
            (
                {"element_domain": NumpyIntegerDomain(), "length": np.int64(5)},
                pytest.raises(
                    TypeCheckError,
                ),
                None,
            ),
            ({"element_domain": NumpyIntegerDomain()}, does_not_raise(), None),
            (
                {"element_domain": ListDomain(NumpyIntegerDomain())},
                does_not_raise(),
                None,
            ),
        ],
    )
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
        super().test_construct_component(
            domain_type, domain_args, expectation, exception_properties
        )

    @pytest.mark.parametrize(
        "domain, other_domain, expected",
        [
            (ListDomain(NumpyIntegerDomain()), NumpyFloatDomain(), False),
            (ListDomain(NumpyIntegerDomain()), ListDomain(NumpyFloatDomain()), False),
            (ListDomain(NumpyIntegerDomain()), ListDomain(NumpyIntegerDomain()), True),
        ]
        + [
            # Lengths should be equal as well
            (
                ListDomain(NumpyIntegerDomain(), length=len1),
                ListDomain(NumpyIntegerDomain(), length=len2),
                len1 == len2,
            )
            for len1, len2 in combinations_with_replacement([1, 2, 3], 2)
        ],
    )
    def test_eq(self, domain: Domain, other_domain: Domain, expected: bool):
        """__eq__ works correctly.

        Args:
            domain: The domain to test.
            other_domain: The domain to compare to.
            expected: The expected result of the comparison.
        """
        super().test_eq(domain, other_domain, expected)

    @pytest.mark.skip("No arguments to mutate")
    @pytest.mark.parametrize("domain_args, key, mutator", [])
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
        super().test_mutable_inputs(domain_type, domain_args, key, mutator)

    @pytest.mark.parametrize(
        "domain, expected_properties",
        [
            (
                ListDomain(NumpyIntegerDomain()),
                {
                    "element_domain": NumpyIntegerDomain(),
                    "carrier_type": list,
                    "length": None,
                },
            ),
            (
                ListDomain(NumpyFloatDomain()),
                {
                    "element_domain": NumpyFloatDomain(),
                    "carrier_type": list,
                    "length": None,
                },
            ),
            (
                ListDomain(NumpyIntegerDomain(), length=5),
                {
                    "element_domain": NumpyIntegerDomain(),
                    "carrier_type": list,
                    "length": 5,
                },
            ),
        ],
    )
    def test_properties(self, domain: Domain, expected_properties: Dict[str, Any]):
        """All properties have the expected values.

        Args:
            domain: The constructed domain to be tested.
            expected_properties: A dictionary containing all the property:value pairs
                domain is expected to have.
        """
        super().test_properties(domain, expected_properties)

    @pytest.mark.parametrize(
        "domain", [ListDomain(NumpyIntegerDomain()), ListDomain(NumpyFloatDomain())]
    )
    def test_property_immutability(self, domain: Domain):
        """The properties return copies for mutable values.

        Args:
            domain: The domain to be tested.
        """
        super().test_property_immutability(domain)

    @pytest.mark.parametrize(
        "domain, candidate, expectation, exception_properties",
        [
            (ListDomain(NumpyIntegerDomain()), [np.int64(1)], does_not_raise(), None),
            (
                ListDomain(NumpyIntegerDomain()),
                [np.float64(1.0)],
                pytest.raises(
                    OutOfDomainError,
                    match="Found invalid value in list: Value must be "
                    f"{get_fullname(np.int64)}, "
                    f"instead it is {get_fullname(np.float64)}",
                ),
                {
                    "domain": ListDomain(NumpyIntegerDomain()),
                    "value": [np.float64(1.0)],
                },
            ),
            (
                ListDomain(NumpyIntegerDomain()),
                "not a list",
                pytest.raises(
                    OutOfDomainError,
                    match=f"Value must be {get_fullname(list)}, instead it is "
                    f"{get_fullname(str)}.",
                ),
                {"domain": ListDomain(NumpyIntegerDomain()), "value": "not a list"},
            ),
            (
                ListDomain(NumpyIntegerDomain(), length=3),
                [np.int64(i) for i in range(10)],
                pytest.raises(
                    OutOfDomainError,
                    match=f"Expected list of length {3}, found list of length {10}",
                ),
                {
                    "domain": ListDomain(NumpyIntegerDomain(), length=3),
                    "value": [np.int64(i) for i in range(10)],
                },
            ),
            (
                ListDomain(NumpyIntegerDomain(), length=10),
                [np.int64(i) for i in range(10)],
                does_not_raise(),
                None,
            ),
        ],
    )
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
        super().test_validate(domain, candidate, expectation, exception_properties)


class TestDictDomain(DomainTests):
    """Tests for :class:`~tmlt.core.domains.collections.DictDomain`."""

    @pytest.fixture
    def domain_type(self) -> Type[Domain]:
        """Returns the type of the domain to be tested."""
        return DictDomain

    @pytest.fixture(scope="class")
    def domain(self) -> DictDomain:
        """Get a base DictDomain."""
        return DictDomain({"A": NumpyIntegerDomain(), "B": NumpyFloatDomain()})

    @pytest.mark.parametrize(
        "domain_args, expectation, exception_properties",
        [
            (
                {"key_to_domain": []},
                pytest.raises(
                    TypeCheckError,
                ),
                None,
            ),
            (
                {"key_to_domain": (1, 2, 3)},
                pytest.raises(
                    TypeCheckError,
                ),
                None,
            ),
            (
                {"key_to_domain": "not a domain"},
                pytest.raises(TypeCheckError, match='"key_to_domain"'),
                None,
            ),
            (
                {"key_to_domain": {"A": np.int64(1)}},
                pytest.raises(TypeCheckError, match="'A'"),
                None,
            ),
            (
                {"key_to_domain": {"A": 1}},
                pytest.raises(TypeCheckError, match="'A'"),
                None,
            ),
            ({"key_to_domain": {1: NumpyIntegerDomain()}}, does_not_raise(), None),
            (
                {"key_to_domain": {"A": ListDomain(NumpyIntegerDomain())}},
                does_not_raise(),
                None,
            ),
            (
                {
                    "key_to_domain": {
                        "A": DictDomain({"B": DictDomain({"C": NumpyIntegerDomain()})})
                    }
                },
                does_not_raise(),
                None,
            ),
        ]
        + [
            # Testing complex keys
            ({"key_to_domain": {key: NumpyIntegerDomain()}}, does_not_raise(), None)
            for key in [("A",), "3", 1.0, np.int64(-3)]
        ],
    )
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
        super().test_construct_component(
            domain_type, domain_args, expectation, exception_properties
        )

    @pytest.mark.parametrize(
        "other_domain, expected",
        [
            (DictDomain({"A": NumpyIntegerDomain(), "B": NumpyFloatDomain()}), True),
            (DictDomain({"B": NumpyFloatDomain(), "A": NumpyIntegerDomain()}), True),
            (
                DictDomain(
                    {"A": NumpyIntegerDomain(), "B": NumpyFloatDomain(allow_nan=True)}
                ),
                False,
            ),
            (DictDomain({"A": NumpyIntegerDomain(), "C": NumpyFloatDomain()}), False),
            (
                DictDomain(
                    {
                        "A": NumpyIntegerDomain(),
                        "B": NumpyFloatDomain(),
                        "C": NumpyFloatDomain(),
                    }
                ),
                False,
            ),
            (DictDomain({"A": NumpyIntegerDomain()}), False),
            (NumpyIntegerDomain(), False),
        ],
    )
    def test_eq(self, domain: Domain, other_domain: Domain, expected: bool):
        """__eq__ works correctly.

        Args:
            domain: The domain to test.
            other_domain: The domain to compare to.
            expected: The expected result of the comparison.
        """
        super().test_eq(domain, other_domain, expected)

    @pytest.mark.parametrize(
        "domain_args, key, mutator",
        [
            (
                {"key_to_domain": {"A": NumpyIntegerDomain()}},
                "key_to_domain",
                lambda x: x.update({"A": NumpyFloatDomain()}),
            )
        ],
    )
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
        super().test_mutable_inputs(domain_type, domain_args, key, mutator)

    @pytest.mark.parametrize(
        "domain, expected_properties",
        [
            (DictDomain({}), {"key_to_domain": {}, "length": 0, "carrier_type": dict}),
            (
                DictDomain({"A": NumpyIntegerDomain()}),
                {
                    "key_to_domain": {"A": NumpyIntegerDomain()},
                    "length": 1,
                    "carrier_type": dict,
                },
            ),
            (
                DictDomain({"A": NumpyIntegerDomain(), "B": NumpyFloatDomain()}),
                {
                    "key_to_domain": {
                        "A": NumpyIntegerDomain(),
                        "B": NumpyFloatDomain(),
                    },
                    "length": 2,
                    "carrier_type": dict,
                },
            ),
        ],
    )
    def test_properties(self, domain: Domain, expected_properties: Dict[str, Any]):
        """All properties have the expected values.

        Args:
            domain: The constructed domain to be tested.
            expected_properties: A dictionary containing all the property:value pairs
                domain is expected to have.
        """
        super().test_properties(domain, expected_properties)

    @pytest.mark.parametrize(
        "domain", [(DictDomain({"A": NumpyIntegerDomain(), "B": NumpyFloatDomain()}))]
    )
    def test_property_immutability(self, domain: Domain):
        """The properties return copies for mutable values.

        Args:
            domain: The domain to be tested.
        """
        super().test_property_immutability(domain)

    @pytest.mark.parametrize(
        "domain, candidate, expectation, exception_properties",
        [
            (
                DictDomain({"A": NumpyIntegerDomain()}),
                NumpyIntegerDomain(),
                pytest.raises(
                    OutOfDomainError,
                    match=f"Value must be {get_fullname(dict)}, instead it is "
                    f"{get_fullname(NumpyIntegerDomain)}.",
                ),
                {"domain": DictDomain({"A": NumpyIntegerDomain()})},
            ),
            (
                DictDomain({"A": NumpyIntegerDomain()}),
                {"A": np.int64(1)},
                does_not_raise(),
                None,
            ),
            (
                DictDomain({"A": NumpyIntegerDomain()}),
                {"A": np.int64(1), "B": np.int64(2)},
                pytest.raises(
                    OutOfDomainError,
                    match=re.escape(
                        "Keys are not as expected, value must match domain.\n"
                        "Value keys: {"
                    )
                    + "'.', '.'"
                    + re.escape("}\nDomain keys: {'A'}"),
                ),
                {
                    "domain": DictDomain({"A": NumpyIntegerDomain()}),
                    "value": {"A", "B"},
                },
            ),
            (
                DictDomain({"A": NumpyIntegerDomain(), "B": NumpyFloatDomain()}),
                {"A": np.int64(1)},
                pytest.raises(
                    OutOfDomainError,
                    match=re.escape(
                        "Keys are not as expected, value must match domain.\n"
                        "Value keys: {'A'}\nDomain keys: {"
                    )
                    + "'.', '.'"
                    + re.escape("}"),
                ),
                {
                    "domain": DictDomain(
                        {"A": NumpyIntegerDomain(), "B": NumpyFloatDomain()}
                    ),
                    "value": {"A"},
                },
            ),
            (
                DictDomain({"A": NumpyIntegerDomain(), "B": NumpyFloatDomain()}),
                {"A": np.float64(1.0), "B": np.float64(2.0)},
                pytest.raises(
                    OutOfDomainError,
                    match=f"Found invalid value at 'A': Value must be "
                    f"{get_fullname(np.int64)}, instead it is "
                    f"{get_fullname(np.float64)}.",
                ),
                {
                    "domain": DictDomain(
                        {"A": NumpyIntegerDomain(), "B": NumpyFloatDomain()}
                    ),
                    "value": {"A": np.float64(1.0), "B": np.float64(2.0)},
                },
            ),
            (DictDomain({}), {}, does_not_raise(), None),
            (
                DictDomain({}),
                {"A": np.int64(1), "B": np.float64(2.0)},
                pytest.raises(
                    OutOfDomainError,
                    match=re.escape(
                        "Keys are not as expected, value must match domain.\n"
                        "Value keys: {"
                    )
                    + "'.', '.'"
                    + re.escape("}\nDomain keys: set()"),
                ),
                {"domain": DictDomain({}), "value": {"A", "B"}},
            ),
        ],
    )
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
        super().test_validate(domain, candidate, expectation, exception_properties)

    def test_validate_with_unsortable_types(self):
        """Test that DictDomain.validate works with keys that don't support sorting."""

        @dataclass(frozen=True)
        class UnsortableKey:
            """Class to use as DictDomain key; cannot be sorted."""

            s: str

        key_a = UnsortableKey("A")
        key_b = UnsortableKey("B")

        domain = DictDomain({key_a: NumpyIntegerDomain(), key_b: NumpyFloatDomain()})

        in_domain = {key_a: np.int64(1), key_b: np.float64(2.5)}

        # This should run without raising an error
        domain.validate(in_domain)

        with pytest.raises(
            OutOfDomainError,
            match=re.escape(
                f"Keys are not as expected, value must match domain.\n"
                f"Value keys: {set([key_a])}\n"
                f"Domain keys: {set([key_a, key_b])}"
            ),
        ):
            domain.validate({key_a: np.int64(1)})

        with pytest.raises(
            OutOfDomainError,
            match=re.escape(
                "Keys are not as expected, value must match domain.\n"
                f"Value keys: {set([key_b])}\n"
                f"Domain keys: {set([key_a, key_b])}"
            ),
        ):
            domain.validate({key_b: np.int64(1)})

        with pytest.raises(
            OutOfDomainError, match=re.escape(f"Found invalid value at '{key_a}'")
        ):
            # right keys, but they're both floats
            domain.validate({key_a: np.float64(1.5), key_b: np.float64(1.5)})
