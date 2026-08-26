"""Unit tests for :class:`~tmlt.core.domains.pandas_domains.PandasGroupedTableDomain`.

These mirror ``TestSparkGroupedDataFrameDomain`` in ``test_spark_domains.py``:
the two domains have the same constructor guards -- including the refusal to
group by a floating point column -- the same split of validation into the inner
table and the group keys, and the same total-aggregation branch.

Nothing here needs a Spark session.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

from contextlib import nullcontext as does_not_raise
from test.unit.domains.abstract import DomainTests
from typing import Any, Callable, ContextManager, Dict, List, Optional, Type

import numpy as np
import pandas as pd
import pytest
from typeguard import TypeCheckError

from tmlt.core.domains.base import Domain, OutOfDomainError
from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.pandas_domains import (
    PandasFloatColumnDescriptor,
    PandasGroupedTableDomain,
    PandasIntegerColumnDescriptor,
    PandasStringColumnDescriptor,
    PandasTableDomain,
)
from tmlt.core.domains.spark_domains import (
    SparkGroupedDataFrameDomain,
    SparkStringColumnDescriptor,
)
from tmlt.core.utils.pandas_grouped_table import PandasGroupedTable
from tmlt.core.utils.testing import get_all_props

_base_schema: Dict[str, Any] = {
    "A": PandasIntegerColumnDescriptor(allow_null=False),
    "B": PandasStringColumnDescriptor(allow_null=False),
    "C": PandasIntegerColumnDescriptor(allow_null=False),
}

_base_groupby_columns: List[str] = ["A", "B"]


def _frame(**columns: Any) -> pd.DataFrame:
    """Returns a DataFrame holding the given columns.

    Args:
        columns: One entry per column, mapping its name to its values.
    """
    return pd.DataFrame(columns)


def _strings(*values: Any) -> pd.Series:
    """Returns an object-dtype Series, which is what a string column is.

    Args:
        values: The column's values.
    """
    return pd.Series(list(values), dtype=object)


def _base_table(
    group_keys: Optional[pd.DataFrame] = None,
) -> PandasGroupedTable:
    """Returns a table in the base domain, with the given group keys.

    Args:
        group_keys: The group keys, or None for the base ones.
    """
    if group_keys is None:
        group_keys = _frame(A=[1, 2, 3], B=_strings("W", "X", "Y"))
    return PandasGroupedTable(
        dataframe=_frame(A=[1, 2, 3], B=_strings("W", "X", "Y"), C=[10, 12, 13]),
        group_keys=group_keys,
    )


class TestPandasGroupedTableDomain(DomainTests):
    """Tests for :class:`~tmlt.core.domains.pandas_domains.PandasGroupedTableDomain`."""

    @pytest.fixture
    def domain_type(self) -> Type[Domain]:
        """Returns the type of the domain to be tested."""
        return PandasGroupedTableDomain

    @pytest.mark.parametrize(
        "domain_args, expectation, exception_properties",
        [
            (
                {"schema": _base_schema, "groupby_columns": _base_groupby_columns},
                does_not_raise(),
                None,
            ),
            (
                {"schema": _base_schema, "groupby_columns": []},
                does_not_raise(),
                None,
            ),
            # _base_schema does not have column "D"
            (
                {"schema": _base_schema, "groupby_columns": ["D"]},
                pytest.raises(ValueError, match=r"Invalid groupby columns: \{'D'\}"),
                None,
            ),
            (
                {"schema": _base_schema, "groupby_columns": ["A", "A"]},
                pytest.raises(
                    ValueError,
                    match=r"groupby_columns contains duplicate column names\.",
                ),
                None,
            ),
            # A floating point column cannot be grouped by.
            (
                {
                    "schema": {**_base_schema, "D": PandasFloatColumnDescriptor()},
                    "groupby_columns": ["D"],
                },
                pytest.raises(
                    ValueError, match="Can not group by a floating point column: D"
                ),
                None,
            ),
            # Invalid schemas.
            (
                {"schema": "not a schema", "groupby_columns": []},
                pytest.raises(TypeCheckError, match='"schema"'),
                None,
            ),
            (
                {
                    "schema": {
                        "A": PandasStringColumnDescriptor(),
                        "B": DictDomain({}),
                    },
                    "groupby_columns": ["A"],
                },
                pytest.raises(TypeCheckError, match="'B'"),
                None,
            ),
            (  # A Spark descriptor is not a pandas one.
                {
                    "schema": {"A": SparkStringColumnDescriptor()},
                    "groupby_columns": ["A"],
                },
                pytest.raises(TypeCheckError, match="'A'"),
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
            (  # eq with same schema and groupby_columns
                PandasGroupedTableDomain(_base_schema, _base_groupby_columns),
                PandasGroupedTableDomain(_base_schema, _base_groupby_columns),
                True,
            ),
            (  # eq with groupby columns in another order
                PandasGroupedTableDomain(_base_schema, ["A", "B"]),
                PandasGroupedTableDomain(_base_schema, ["B", "A"]),
                True,
            ),
            (  # not eq with different groupby columns
                PandasGroupedTableDomain(_base_schema, _base_groupby_columns),
                PandasGroupedTableDomain(_base_schema, ["A"]),
                False,
            ),
            (  # not eq with different schemas
                PandasGroupedTableDomain(_base_schema, ["A"]),
                PandasGroupedTableDomain(
                    {**_base_schema, "D": PandasStringColumnDescriptor()}, ["A"]
                ),
                False,
            ),
            (  # not eq with a shuffled schema
                PandasGroupedTableDomain(_base_schema, ["A"]),
                PandasGroupedTableDomain(
                    {
                        "C": _base_schema["C"],
                        "B": _base_schema["B"],
                        "A": _base_schema["A"],
                    },
                    ["A"],
                ),
                False,
            ),
            (  # not eq with the ungrouped domain
                PandasGroupedTableDomain(_base_schema, ["A"]),
                PandasTableDomain(_base_schema),
                False,
            ),
            (  # not eq with the Spark domain describing the same table
                PandasGroupedTableDomain({"A": PandasStringColumnDescriptor()}, ["A"]),
                SparkGroupedDataFrameDomain(
                    {"A": SparkStringColumnDescriptor()}, ["A"]
                ),
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
                    "schema": dict(_base_schema),
                    "groupby_columns": list(_base_groupby_columns),
                },
                "schema",
                lambda schema: schema.pop("A"),
            ),
            (
                {
                    "schema": dict(_base_schema),
                    "groupby_columns": list(_base_groupby_columns),
                },
                "groupby_columns",
                lambda columns: columns.pop(),
            ),
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
                PandasGroupedTableDomain(_base_schema, _base_groupby_columns),
                {
                    "schema": _base_schema,
                    "carrier_type": PandasGroupedTable,
                    "pandas_dtypes": {
                        "A": np.dtype("int64"),
                        "B": np.dtype(object),
                        "C": np.dtype("int64"),
                    },
                    "groupby_columns": frozenset(_base_groupby_columns),
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
        actual_props = [prop[0] for prop in get_all_props(type(domain))]
        assert set(expected_properties.keys()) == set(actual_props)
        for prop, expected_val in expected_properties.items():
            assert hasattr(domain, prop) and getattr(domain, prop) == expected_val

    @pytest.mark.parametrize(
        "domain",
        [PandasGroupedTableDomain(_base_schema, _base_groupby_columns)],
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
            (  # Normal
                PandasGroupedTableDomain(_base_schema, _base_groupby_columns),
                _base_table(),
                does_not_raise(),
                None,
            ),
            (  # A group key that is not in the table is still a group key
                PandasGroupedTableDomain(_base_schema, _base_groupby_columns),
                _base_table(_frame(A=[9], B=_strings("Z"))),
                does_not_raise(),
                None,
            ),
            (  # Nulls, where the schema allows them
                PandasGroupedTableDomain(
                    {
                        **_base_schema,
                        "B": PandasStringColumnDescriptor(allow_null=True),
                    },
                    _base_groupby_columns,
                ),
                PandasGroupedTable(
                    dataframe=_frame(
                        A=[1, 2, 3], B=_strings("W", "X", None), C=[10, 12, 13]
                    ),
                    group_keys=_frame(A=[1, 2, 3], B=_strings("W", "X", None)),
                ),
                does_not_raise(),
                None,
            ),
            (  # Unexpected nulls in the table
                PandasGroupedTableDomain(_base_schema, _base_groupby_columns),
                PandasGroupedTable(
                    dataframe=_frame(
                        A=[1, 2, 3], B=_strings("W", "X", None), C=[10, 12, 13]
                    ),
                    group_keys=_frame(A=[1, 2, 3], B=_strings("W", "X", "Y")),
                ),
                pytest.raises(
                    OutOfDomainError,
                    match=r"Invalid inner DataFrame: .*Column contains null values",
                ),
                None,
            ),
            (  # Unexpected nulls in the group keys
                PandasGroupedTableDomain(_base_schema, _base_groupby_columns),
                _base_table(_frame(A=[1, 2, 3], B=_strings("W", "X", None))),
                pytest.raises(
                    OutOfDomainError,
                    match=r"Invalid group keys: .*Column contains null values",
                ),
                None,
            ),
            (  # Missing column in the table
                PandasGroupedTableDomain(_base_schema, ["A"]),
                PandasGroupedTable(
                    dataframe=_frame(A=[1, 2, 3], B=_strings("W", "X", "Y")),
                    group_keys=_frame(A=[1, 2, 3]),
                ),
                pytest.raises(
                    OutOfDomainError,
                    match=r"Invalid inner DataFrame: .*Columns are not as expected",
                ),
                None,
            ),
            (  # Group keys in an order the domain does not have
                PandasGroupedTableDomain(_base_schema, _base_groupby_columns),
                _base_table(_frame(B=_strings("W", "X", "Y"), A=[1, 2, 3])),
                pytest.raises(
                    OutOfDomainError,
                    match=r"Invalid group keys: .*Columns are not as expected",
                ),
                None,
            ),
            (  # No columns in the group keys, where groups were expected
                PandasGroupedTableDomain(_base_schema, _base_groupby_columns),
                _base_table(pd.DataFrame()),
                pytest.raises(
                    OutOfDomainError,
                    match=(
                        "Invalid group keys: expected groups, but got total aggregation"
                    ),
                ),
                None,
            ),
            (  # A total aggregation, where the domain has no groupby columns
                PandasGroupedTableDomain(_base_schema, []),
                _base_table(pd.DataFrame()),
                does_not_raise(),
                None,
            ),
            (  # Not a grouped table at all
                PandasGroupedTableDomain(_base_schema, _base_groupby_columns),
                _frame(A=[1], B=_strings("W"), C=[10]),
                pytest.raises(OutOfDomainError, match="Value must be"),
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
                the exception is expected to have.
        """
        super().test_validate(domain, candidate, expectation, exception_properties)


@pytest.mark.parametrize(
    "groupby_columns, expected",
    [
        (
            ["A", "B"],
            PandasTableDomain({"C": _base_schema["C"]}),
        ),
        (
            ["B"],
            PandasTableDomain({"A": _base_schema["A"], "C": _base_schema["C"]}),
        ),
        ([], PandasTableDomain(_base_schema)),
    ],
)
def test_get_group_domain(
    groupby_columns: List[str], expected: PandasTableDomain
) -> None:
    """get_group_domain drops the groupby columns, keeping the schema's order."""
    domain = PandasGroupedTableDomain(_base_schema, groupby_columns)
    assert domain.get_group_domain() == expected


def test_getitem() -> None:
    """__getitem__ returns the descriptor of a column."""
    domain = PandasGroupedTableDomain(_base_schema, _base_groupby_columns)
    assert domain["B"] == PandasStringColumnDescriptor()
    with pytest.raises(KeyError):
        domain["D"]


def test_repr() -> None:
    """The repr names the schema and the groupby columns."""
    domain = PandasGroupedTableDomain({"A": PandasStringColumnDescriptor()}, ["A"])
    assert repr(domain) == (
        "PandasGroupedTableDomain(schema={'A': "
        "PandasStringColumnDescriptor(allow_null=False)}, groupby_columns={'A'})"
    )


def test_format() -> None:
    """The formatted domain shows the groupby columns and the schema."""
    domain = PandasGroupedTableDomain(_base_schema, ["A"])
    assert domain.format().splitlines()[0] == (
        "PandasGroupedTableDomain groupby_columns={'A'}"
    )
