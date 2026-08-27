"""Unit tests for :class:`~tmlt.core.domains.pandas_domains.PandasRowDomain`.

The other domains in that module are covered by ``test_pandas_domains.py``
(the element-domain family) and ``test_pandas_table_domains.py`` (the column
descriptors and :class:`~tmlt.core.domains.pandas_domains.PandasTableDomain`).
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

from contextlib import nullcontext as does_not_raise
from test.unit.domains.abstract import DomainTests
from typing import Any, Callable, ContextManager, Dict, Optional, Type

import pytest
from typeguard import TypeCheckError

from tmlt.core.domains.base import Domain
from tmlt.core.domains.pandas_domains import (
    PandasFloatColumnDescriptor,
    PandasIntegerColumnDescriptor,
    PandasRowDomain,
    PandasStringColumnDescriptor,
    PandasTableDomain,
)
from tmlt.core.domains.spark_domains import (
    SparkIntegerColumnDescriptor,
    SparkRowDomain,
    SparkStringColumnDescriptor,
)


class TestPandasRowDomain(DomainTests):
    """Tests for :class:`~tmlt.core.domains.pandas_domains.PandasRowDomain`."""

    @pytest.fixture
    def domain_type(self) -> Type[Domain]:
        """Returns the type of the domain to be tested."""
        return PandasRowDomain

    @pytest.fixture
    def domain(self) -> PandasRowDomain:
        """Get a base PandasRowDomain."""
        return PandasRowDomain(
            schema={
                "A": PandasIntegerColumnDescriptor(),
                "B": PandasStringColumnDescriptor(),
            }
        )

    @pytest.mark.parametrize(
        "domain_args, expectation, exception_properties",
        [
            (
                {
                    "schema": {
                        "A": PandasIntegerColumnDescriptor(),
                        "B": PandasStringColumnDescriptor(),
                    }
                },
                does_not_raise(),
                None,
            ),
            ({"schema": {}}, does_not_raise(), None),
            (
                {"schema": int},
                pytest.raises(TypeCheckError, match='"schema"'),
                None,
            ),
            (
                # A Spark descriptor is not a pandas one.
                {"schema": {"A": SparkIntegerColumnDescriptor()}},
                pytest.raises(
                    TypeCheckError,
                    match=(
                        r"value of key 'A' of argument \"schema\" .* is not an "
                        r"instance of "
                        r"tmlt\.core\.domains\.pandas_domains\.PandasColumnDescriptor"
                    ),
                ),
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

        Args:
            domain_type: The type of domain to be constructed.
            domain_args: The arguments to the domain.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
            exception_properties: A dictionary containing all the property:value pairs
                the exception is expected to have.
        """
        super().test_construct_component(
            domain_type, domain_args, expectation, exception_properties
        )

    @pytest.mark.parametrize(
        "domain, other_domain, expected",
        [
            (
                PandasRowDomain(
                    schema={
                        "A": PandasIntegerColumnDescriptor(),
                        "B": PandasStringColumnDescriptor(),
                    }
                ),
                PandasRowDomain(
                    schema={
                        "A": PandasIntegerColumnDescriptor(),
                        "B": PandasStringColumnDescriptor(),
                    }
                ),
                True,
            ),
            (
                # testing that order does matter
                PandasRowDomain(
                    schema={
                        "B": PandasStringColumnDescriptor(),
                        "A": PandasIntegerColumnDescriptor(),
                    }
                ),
                PandasRowDomain(
                    schema={
                        "A": PandasIntegerColumnDescriptor(),
                        "B": PandasStringColumnDescriptor(),
                    }
                ),
                False,
            ),
            (
                PandasRowDomain(schema={"A": PandasIntegerColumnDescriptor()}),
                PandasRowDomain(schema={"B": PandasStringColumnDescriptor()}),
                False,
            ),
            (
                # A row domain is not the table domain over the same schema,
                # and it is not the Spark row domain over the same columns.
                PandasRowDomain(schema={"A": PandasIntegerColumnDescriptor()}),
                PandasTableDomain(schema={"A": PandasIntegerColumnDescriptor()}),
                False,
            ),
            (
                PandasRowDomain(schema={"A": PandasStringColumnDescriptor()}),
                SparkRowDomain(schema={"A": SparkStringColumnDescriptor()}),
                False,
            ),
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
                {
                    "schema": {
                        "A": PandasIntegerColumnDescriptor(),
                        "B": PandasStringColumnDescriptor(),
                    }
                },
                "schema",
                lambda x: x.update({"A": PandasFloatColumnDescriptor()}),
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
            (
                PandasRowDomain(
                    schema={
                        "A": PandasIntegerColumnDescriptor(),
                        "B": PandasStringColumnDescriptor(),
                    }
                ),
                {
                    "schema": {
                        "A": PandasIntegerColumnDescriptor(),
                        "B": PandasStringColumnDescriptor(),
                    },
                    "carrier_type": dict,
                },
            )
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

    def test_property_immutability(self, domain: Domain):
        """The properties return copies for mutable values.

        Args:
            domain: The domain to be tested.
        """
        super().test_property_immutability(domain)

    @pytest.mark.skip(reason="PandasRowDomain does not implement validate.")
    @pytest.mark.parametrize("domain, candidate, expectation, exception_properties", [])
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
                the exception is expected to have.
        """
        super().test_validate(domain, candidate, expectation, exception_properties)

    def test_repr(self, domain: Domain):
        """Tests that __repr__ works correctly."""
        expected = (
            "PandasRowDomain(schema={'A': PandasIntegerColumnDescriptor("
            "allow_null=False, size=64), 'B': PandasStringColumnDescriptor("
            "allow_null=False)})"
        )
        assert repr(domain) == expected


def test_validate_and_contains_are_not_implemented():
    """PandasRowDomain does not implement validate or __contains__.

    This mirrors :class:`~tmlt.core.domains.spark_domains.SparkRowDomain`
    exactly; a row's values are checked with
    :meth:`~tmlt.core.domains.pandas_domains.PandasColumnDescriptor.valid_py_value`
    instead.
    """
    domain = PandasRowDomain({"A": PandasIntegerColumnDescriptor()})
    with pytest.raises(NotImplementedError):
        domain.validate({"A": 1})
    with pytest.raises(NotImplementedError):
        _ = {"A": 1} in domain


def test_format():
    """PandasRowDomain formats its schema as labeled children."""
    domain = PandasRowDomain(
        {
            "A": PandasIntegerColumnDescriptor(),
            "B": PandasStringColumnDescriptor(allow_null=True),
        }
    )
    assert domain.format() == (
        "PandasRowDomain\n"
        "* A: PandasIntegerColumnDescriptor allow_null=False size=64\n"
        "* B: PandasStringColumnDescriptor allow_null=True"
    )


def test_empty_schema_formats_without_children():
    """An empty PandasRowDomain formats as just its name."""
    assert PandasRowDomain({}).format() == "PandasRowDomain"


def test_mirrors_the_spark_row_domain_api():
    """PandasRowDomain has the same public surface as SparkRowDomain."""

    def public_names(domain_type: type) -> set:
        return {
            name
            for name in dir(domain_type)
            if not name.startswith("_") and name.isupper() is False
        }

    assert public_names(PandasRowDomain) == public_names(SparkRowDomain)
