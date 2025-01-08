"""Unit tests for :mod:`~tmlt.core.domains.spark_domains`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

import copy
import datetime
from contextlib import nullcontext as does_not_raise
from itertools import combinations_with_replacement, product
from test.conftest import assert_frame_equal_with_sort
from test.unit.domains.abstract import DomainTests
from typing import Any, Callable, ContextManager, Dict, List, Optional, Type

import pandas as pd
import pytest
from pyspark import Row
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.session import SparkSession
from pyspark.sql.types import (
    DataType,
    DateType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from typeguard import TypeCheckError

from tmlt.core.domains.base import Domain, OutOfDomainError
from tmlt.core.domains.collections import DictDomain, ListDomain
from tmlt.core.domains.numpy_domains import (
    NumpyDomain,
    NumpyFloatDomain,
    NumpyIntegerDomain,
    NumpyStringDomain,
)
from tmlt.core.domains.pandas_domains import PandasDataFrameDomain, PandasSeriesDomain
from tmlt.core.domains.spark_domains import (
    SparkColumnDescriptor,
    SparkColumnsDescriptor,
    SparkDataFrameDomain,
    SparkDateColumnDescriptor,
    SparkFloatColumnDescriptor,
    SparkGroupedDataFrameDomain,
    SparkIntegerColumnDescriptor,
    SparkRowDomain,
    SparkStringColumnDescriptor,
    SparkTimestampColumnDescriptor,
    convert_numpy_domain,
    convert_pandas_domain,
    convert_spark_schema,
)
from tmlt.core.utils.grouped_dataframe import GroupedDataFrame
from tmlt.core.utils.misc import get_fullname
from tmlt.core.utils.testing import get_all_props


@pytest.mark.usefixtures("class_spark")
class TestSparkDataFrameDomain(DomainTests):
    """Tests for :class:`~tmlt.core.domains.spark_domains.SparkDataFrameDomain`."""

    spark: SparkSession

    @pytest.fixture
    def domain_type(self) -> Type[Domain]:
        """Returns the type of the domain to be tested."""
        return SparkDataFrameDomain

    @pytest.fixture(scope="class")
    def domain(self) -> SparkDataFrameDomain:
        """Get a base SparkDataFrameDomain."""
        return SparkDataFrameDomain(
            schema={
                "A": SparkStringColumnDescriptor(),
                "B": SparkStringColumnDescriptor(),
                "C": SparkFloatColumnDescriptor(),
            }
        )

    @pytest.mark.parametrize(
        "domain_args, expectation, exception_properties",
        [
            (
                {"schema": invalid_schema},
                pytest.raises(TypeCheckError, match='"schema"'),
                None,
            )
            for invalid_schema in [StringType, ListDomain(NumpyIntegerDomain())]
        ]
        + [
            (
                {"schema": {"A": SparkStringColumnDescriptor(), "B": DictDomain({})}},
                pytest.raises(TypeCheckError, match="'B'"),
                None,
            ),
            (
                {"schema": {"A": "B"}},
                pytest.raises(TypeCheckError, match="'A'"),
                None,
            ),
        ]
        + [
            ({"schema": valid_schema}, does_not_raise(), None)
            for valid_schema in [
                {
                    "A": SparkStringColumnDescriptor(),
                    "B": SparkStringColumnDescriptor(),
                },
                {},
                {"A": SparkStringColumnDescriptor()},
            ]
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
            (  # matching
                SparkDataFrameDomain(
                    {
                        "A": SparkStringColumnDescriptor(),
                        "B": SparkStringColumnDescriptor(),
                        "C": SparkFloatColumnDescriptor(),
                    }
                ),
                True,
            ),
            (  # shuffled
                SparkDataFrameDomain(
                    {
                        "B": SparkStringColumnDescriptor(),
                        "C": SparkFloatColumnDescriptor(),
                        "A": SparkStringColumnDescriptor(),
                    }
                ),
                False,
            ),
            (  # Mismatching Types
                SparkDataFrameDomain(
                    {
                        "A": SparkStringColumnDescriptor(),
                        "B": SparkStringColumnDescriptor(),
                        "C": SparkFloatColumnDescriptor(size=32),
                    }
                ),
                False,
            ),
            (  # Extra attribute
                SparkDataFrameDomain(
                    {
                        "A": SparkStringColumnDescriptor(),
                        "B": SparkStringColumnDescriptor(),
                        "C": SparkFloatColumnDescriptor(),
                        "D": SparkFloatColumnDescriptor(),
                    }
                ),
                False,
            ),
            (  # Missing attribute
                SparkDataFrameDomain(
                    {
                        "A": SparkStringColumnDescriptor(),
                        "B": SparkStringColumnDescriptor(),
                    }
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
                    "schema": {
                        "A": SparkIntegerColumnDescriptor(),
                        "B": SparkStringColumnDescriptor(),
                        "C": SparkFloatColumnDescriptor(),
                    }
                },
                "schema",
                mutator,
            )
            for mutator in [
                lambda x: x.update({"A": SparkFloatColumnDescriptor()}),
                lambda x: x.pop("A"),
                lambda x: x.clear(),
            ]
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
                SparkDataFrameDomain(
                    schema={
                        "A": SparkIntegerColumnDescriptor(),
                        "B": SparkStringColumnDescriptor(),
                        "C": SparkFloatColumnDescriptor(),
                    }
                ),
                {
                    "schema": {
                        "A": SparkIntegerColumnDescriptor(),
                        "B": SparkStringColumnDescriptor(),
                        "C": SparkFloatColumnDescriptor(),
                    },
                    "carrier_type": DataFrame,
                    "spark_schema": StructType(
                        [
                            StructField("A", LongType(), False),
                            StructField("B", StringType(), False),
                            StructField("C", DoubleType(), False),
                        ]
                    ),
                },
            ),
            (
                SparkDataFrameDomain(
                    schema={
                        "A": SparkIntegerColumnDescriptor(allow_null=True),
                        "B": SparkStringColumnDescriptor(allow_null=True),
                        "C": SparkFloatColumnDescriptor(
                            allow_inf=True, allow_nan=True, allow_null=True
                        ),
                    }
                ),
                {
                    "schema": {
                        "A": SparkIntegerColumnDescriptor(allow_null=True),
                        "B": SparkStringColumnDescriptor(allow_null=True),
                        "C": SparkFloatColumnDescriptor(
                            allow_inf=True, allow_nan=True, allow_null=True
                        ),
                    },
                    "carrier_type": DataFrame,
                    "spark_schema": StructType(
                        [
                            StructField("A", LongType(), True),
                            StructField("B", StringType(), True),
                            StructField("C", DoubleType(), True),
                        ]
                    ),
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
        "domain",
        [
            SparkDataFrameDomain(
                schema={
                    "A": SparkIntegerColumnDescriptor(),
                    "B": SparkStringColumnDescriptor(),
                    "C": SparkFloatColumnDescriptor(),
                }
            )
        ],
    )
    def test_property_immutability(self, domain: Domain):
        """The properties return copies for mutable values.

        Args:
            domain: The domain to be tested.
        """
        super().test_property_immutability(domain)

    @pytest.mark.parametrize(
        "candidate, expectation, exception_properties",
        [
            (  # LongType() instead of DoubleType()
                pd.DataFrame(
                    [["A", "B", 10], ["V", "E", 12], ["A", "V", 13]],
                    columns=["A", "B", "C"],
                ),
                pytest.raises(
                    OutOfDomainError,
                    match="Found invalid value in column 'C': Column must be "
                    f"{get_fullname(DoubleType)}; got {get_fullname(LongType)} "
                    "instead",
                ),
                {
                    "domain": SparkDataFrameDomain(
                        schema={
                            "A": SparkStringColumnDescriptor(),
                            "B": SparkStringColumnDescriptor(),
                            "C": SparkFloatColumnDescriptor(),
                        }
                    ),
                    "value": pd.DataFrame(
                        [["A", "B", 10], ["V", "E", 12], ["A", "V", 13]],
                        columns=["A", "B", "C"],
                    ),
                },
            ),
            (  # Missing Columns
                pd.DataFrame([["A", "B"], ["V", "E"], ["A", "V"]], columns=["A", "B"]),
                pytest.raises(
                    OutOfDomainError,
                    match="Columns are not as expected. DataFrame and Domain "
                    "must contain the same columns in the same order.\n"
                    r"DataFrame columns: \['A', 'B'\]"
                    "\n"
                    r"Domain columns: \['A', 'B', 'C'\]",
                ),
                {
                    "domain": SparkDataFrameDomain(
                        schema={
                            "A": SparkStringColumnDescriptor(),
                            "B": SparkStringColumnDescriptor(),
                            "C": SparkFloatColumnDescriptor(),
                        }
                    ),
                    "value": pd.DataFrame(
                        [["A", "B"], ["V", "E"], ["A", "V"]], columns=["A", "B"]
                    ),
                },
            ),
            (
                pd.DataFrame(
                    [["A", "B", 1.1], ["V", "E", 1.2], ["A", "V", None]],
                    columns=["A", "B", "C"],
                ),
                pytest.raises(
                    OutOfDomainError,
                    match="Found invalid value in column 'C': Column contains null "
                    "values.",
                ),
                {
                    "domain": SparkDataFrameDomain(
                        schema={
                            "A": SparkStringColumnDescriptor(),
                            "B": SparkStringColumnDescriptor(),
                            "C": SparkFloatColumnDescriptor(),
                        }
                    ),
                    "value": pd.DataFrame(
                        [["A", "B", 1.1], ["V", "E", 1.2], ["A", "V", None]],
                        columns=["A", "B", "C"],
                    ),
                },
            ),
            (
                pd.DataFrame(
                    [["A", "B", 1.1], ["V", "E", 1.2], ["A", "V", 1.3]],
                    columns=["A", "B", "C"],
                ),
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
        if isinstance(candidate, pd.DataFrame):
            candidate = self.spark.createDataFrame(candidate)
        with expectation as exception:
            domain.validate(candidate)
        if exception_properties is None or len(exception_properties) == 0:
            return
        # Help out mypy
        assert isinstance(exception, pytest.ExceptionInfo)
        for prop, expected_value in exception_properties.items():
            assert hasattr(exception.value, prop), f"Expected prop was missing: {prop}"
            actual_value = getattr(exception.value, prop)
            if isinstance(actual_value, DataFrame):
                assert_frame_equal_with_sort(actual_value, expected_value)
                continue
            assert (
                actual_value == expected_value
            ), f"Expected {prop} to be {expected_value}, got {actual_value}"

    @pytest.mark.parametrize(
        "spark_schema, expected, expectation",
        [
            (StructType([]), SparkDataFrameDomain(schema={}), does_not_raise()),
            (
                StructType(
                    [
                        StructField("A", LongType(), False),
                        StructField("B", StringType(), False),
                        StructField("C", DoubleType(), False),
                    ]
                ),
                SparkDataFrameDomain(
                    schema={
                        "A": SparkIntegerColumnDescriptor(),
                        "B": SparkStringColumnDescriptor(),
                        "C": SparkFloatColumnDescriptor(allow_inf=True, allow_nan=True),
                    }
                ),
                does_not_raise(),
            ),
        ],
    )
    def test_from_spark_schema(
        self,
        spark_schema: StructType,
        expected: SparkDataFrameDomain,
        expectation: ContextManager[None],
    ):
        """from_spark_schema constructs the correct domain.

        Args:
            spark_schema: The spark schema to construct the domain from.
            expected: The expected domain.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
        """
        with expectation:
            assert SparkDataFrameDomain.from_spark_schema(spark_schema) == expected


_base_schema: Dict[str, SparkColumnDescriptor] = {
    "A": SparkIntegerColumnDescriptor(allow_null=True),
    "B": SparkStringColumnDescriptor(allow_null=True),
    "C": SparkIntegerColumnDescriptor(allow_null=True),
}
_schema_without_nulls: Dict[str, SparkColumnDescriptor] = {
    "A": SparkIntegerColumnDescriptor(allow_null=False),
    "B": SparkStringColumnDescriptor(allow_null=False),
    "C": SparkIntegerColumnDescriptor(allow_null=False),
}

_base_groupby_columns: List[str] = ["A", "B"]
_base_group_key_args: Dict[str, Any] = {
    "data": [(1, "W"), (2, "X"), (3, "Y")],
    "schema": ["A", "B"],
}
_group_key_with_nulls_args: Dict[str, Any] = {
    "data": [(1, "W"), (2, "X"), (3, None)],
    "schema": ["A", "B"],
}
_empty_group_key_args: Dict[str, Any] = {"data": [], "schema": StructType([])}


# Helper functions to help mypy out by widening the type of a context manager.
def _WidenContextManager(cm: Any) -> ContextManager:
    """Widen the type of a context manager."""
    return cm


@pytest.mark.usefixtures("class_spark")
class TestSparkGroupedDataFrameDomain(DomainTests):
    """Testing :class:`~tmlt.core.domains.spark_domains.SparkGroupedDataFrameDomain`."""

    spark: SparkSession

    @pytest.fixture
    def domain_type(self) -> Type[Domain]:
        """Returns the type of the domain to be tested."""
        return SparkGroupedDataFrameDomain

    @pytest.mark.parametrize(
        "domain_args, expectation, exception_properties",
        [
            (
                {"schema": _base_schema, "groupby_columns": _base_groupby_columns},
                does_not_raise(),
                None,
            ),
            # _base_schema does not have column "D"
            (
                {"schema": _base_schema, "groupby_columns": ["D"]},
                pytest.raises(ValueError, match="Invalid groupby columns: {'D'}"),
                None,
            ),
            # Invalid schema
            (
                {"schema": "not a schema", "groupby_columns": ["C"]},
                pytest.raises(TypeCheckError, match='"schema"'),
                None,
            ),
            # Invalid Column "C"
            (
                {
                    "schema": {
                        "A": SparkIntegerColumnDescriptor(allow_null=True),
                        "B": SparkStringColumnDescriptor(allow_null=True),
                        "C": ListDomain(NumpyIntegerDomain()),
                    },
                    "groupby_columns": ["A"],
                },
                pytest.raises(TypeCheckError, match="'C'"),
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
            (
                # eq with same schema and groupby_columns
                SparkGroupedDataFrameDomain(
                    schema=_base_schema, groupby_columns=_base_groupby_columns
                ),
                SparkGroupedDataFrameDomain(
                    schema=_base_schema, groupby_columns=_base_groupby_columns
                ),
                True,
            ),
            (
                # eq with no groupby columns
                SparkGroupedDataFrameDomain(schema=_base_schema, groupby_columns=[]),
                SparkGroupedDataFrameDomain(schema=_base_schema, groupby_columns=[]),
                True,
            ),
            (
                # not eq with different schemas
                SparkGroupedDataFrameDomain(
                    schema=_base_schema, groupby_columns=_base_groupby_columns
                ),
                SparkGroupedDataFrameDomain(
                    schema=_schema_without_nulls, groupby_columns=_base_groupby_columns
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
                    "schema": copy.deepcopy(_base_schema),
                    "groupby_columns": copy.deepcopy(_base_groupby_columns),
                },
                "schema",
                lambda x: x.update({"A": SparkFloatColumnDescriptor()}),
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
                SparkGroupedDataFrameDomain(_base_schema, _base_groupby_columns),
                {
                    "schema": _base_schema,
                    "carrier_type": GroupedDataFrame,
                    "spark_schema": StructType(
                        [
                            StructField("A", LongType(), True),
                            StructField("B", StringType(), True),
                            StructField("C", LongType(), True),
                        ]
                    ),
                    "groupby_columns": _base_groupby_columns,
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
        "domain", [SparkGroupedDataFrameDomain(_base_schema, _base_groupby_columns)]
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
                # Normal
                SparkGroupedDataFrameDomain(_base_schema, _base_groupby_columns),
                {
                    "dataframe": {
                        "data": [(1, "W", 10), (2, "X", 12), (3, "Y", 13)],
                        "schema": ["A", "B", "C"],
                    },
                    "group_keys": _base_group_key_args,
                },
                does_not_raise(),
                None,
            ),
            (
                # Nulls
                SparkGroupedDataFrameDomain(_base_schema, _base_groupby_columns),
                {
                    "dataframe": {
                        "data": [(1, "W", 10), (2, "X", 12), (3, "Y", None)],
                        "schema": ["A", "B", "C"],
                    },
                    "group_keys": _group_key_with_nulls_args,
                },
                does_not_raise(),
                None,
            ),
            (
                # Unexpected nulls in dataframe
                SparkGroupedDataFrameDomain(
                    _schema_without_nulls, _base_groupby_columns
                ),
                {
                    "dataframe": {
                        "data": [(1, "W", 10), (2, "X", 12), (3, "Y", None)],
                        "schema": ["A", "B", "C"],
                    },
                    "group_keys": _base_group_key_args,
                },
                pytest.raises(
                    OutOfDomainError,
                    match=(
                        "Invalid inner DataFrame: Found invalid value in column"
                        " 'C': Column contains null values."
                    ),
                ),
                {
                    "domain": SparkGroupedDataFrameDomain(
                        _schema_without_nulls, _base_groupby_columns
                    ),
                    "value": {
                        "dataframe": {
                            "data": [(1, "W", 10), (2, "X", 12), (3, "Y", None)],
                            "schema": ["A", "B", "C"],
                        },
                        "group_keys": _base_group_key_args,
                    },
                },
            ),
            (  # Unexpected nulls in group_keys
                SparkGroupedDataFrameDomain(
                    _schema_without_nulls, _base_groupby_columns
                ),
                {
                    "dataframe": {
                        "data": [(1, "W", 10), (2, "X", 12), (3, "Y", 13)],
                        "schema": ["A", "B", "C"],
                    },
                    "group_keys": _group_key_with_nulls_args,
                },
                pytest.raises(
                    OutOfDomainError,
                    match=(
                        "Invalid group keys: Found invalid value in column 'B': "
                        "Column contains null values."
                    ),
                ),
                {
                    "domain": SparkGroupedDataFrameDomain(
                        _schema_without_nulls, _base_groupby_columns
                    ),
                    "value": {
                        "dataframe": {
                            "data": [(1, "W", 10), (2, "X", 12), (3, "Y", 13)],
                            "schema": ["A", "B", "C"],
                        },
                        "group_keys": _group_key_with_nulls_args,
                    },
                },
            ),
            (  # Missing column in dataframe
                SparkGroupedDataFrameDomain(_base_schema, _base_groupby_columns),
                {
                    "dataframe": {
                        "data": [(1, "W"), (2, "X"), (3, "Y")],
                        "schema": ["A", "B"],
                    },
                    "group_keys": _base_group_key_args,
                },
                pytest.raises(
                    OutOfDomainError,
                    match=(
                        "Invalid inner DataFrame: Columns are not as expected. "
                        "DataFrame and Domain must contain the same columns in the "
                        "same order.\nDataFrame columns: \\['A', 'B'\\]\nDomain "
                        "columns: \\['A', 'B', 'C'\\]"
                    ),
                ),
                {
                    "domain": SparkGroupedDataFrameDomain(
                        _base_schema, _base_groupby_columns
                    ),
                    "value": {
                        "dataframe": {
                            "data": [(1, "W"), (2, "X"), (3, "Y")],
                            "schema": ["A", "B"],
                        },
                        "group_keys": _base_group_key_args,
                    },
                },
            ),
            (  # Missing column in group_keys
                SparkGroupedDataFrameDomain(_base_schema, _base_groupby_columns),
                {
                    "dataframe": {
                        "data": [(1, "W", 10), (2, "X", 12), (3, "Y", 13)],
                        "schema": ["A", "B", "C"],
                    },
                    "group_keys": _empty_group_key_args,
                },
                pytest.raises(
                    OutOfDomainError,
                    match=(
                        "Invalid group keys: Columns are not as expected. "
                        "DataFrame and Domain must contain the same columns in the "
                        "same order.\nDataFrame columns: \\[\\]\nDomain "
                        "columns: \\['A', 'B'\\]"
                    ),
                ),
                {
                    "domain": SparkGroupedDataFrameDomain(
                        _base_schema, _base_groupby_columns
                    ),
                    "value": {
                        "dataframe": {
                            "data": [(1, "W", 10), (2, "X", 12), (3, "Y", 13)],
                            "schema": ["A", "B", "C"],
                        },
                        "group_keys": _empty_group_key_args,
                    },
                },
            ),
            (  # empty group_keys
                SparkGroupedDataFrameDomain(_base_schema, []),
                {
                    "dataframe": {
                        "data": [(1, "W", 10), (2, "X", 12), (3, "Y", 13)],
                        "schema": ["A", "B", "C"],
                    },
                    "group_keys": _empty_group_key_args,
                },
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
        candidate["dataframe"] = self.spark.createDataFrame(**candidate["dataframe"])
        candidate["group_keys"] = self.spark.createDataFrame(**candidate["group_keys"])
        candidate = GroupedDataFrame(**candidate)
        if exception_properties is not None:
            exception_properties["value"]["dataframe"] = self.spark.createDataFrame(
                **exception_properties["value"]["dataframe"]
            )
            exception_properties["value"]["group_keys"] = self.spark.createDataFrame(
                **exception_properties["value"]["group_keys"]
            )
            exception_properties["value"] = GroupedDataFrame(
                **exception_properties["value"]
            )
        with expectation as exception:
            domain.validate(candidate)
        if exception_properties is None or len(exception_properties) == 0:
            return
        # Help out mypy
        assert isinstance(exception, pytest.ExceptionInfo)
        core_exception = exception.value
        assert hasattr(core_exception, "domain"), "Exception has no domain attribute"
        assert (
            exception_properties["domain"] == core_exception.domain
        ), "Exception domain is not as expected"
        assert hasattr(core_exception, "value"), "Exception has no value attribute"
        # Separate asserts for the frames are required since GroupedDataFrame does not
        # implement __eq__.
        assert_frame_equal_with_sort(
            exception_properties[  # pylint: disable=protected-access
                "value"
            ]._dataframe,
            core_exception.value._dataframe,  # pylint: disable=protected-access
        )
        assert_frame_equal_with_sort(
            exception_properties["value"].group_keys, core_exception.value.group_keys
        )

    @pytest.mark.parametrize(
        "domain", [SparkGroupedDataFrameDomain(_base_schema, _base_groupby_columns)]
    )
    def test_repr(self, domain: Domain):
        """Tests that __repr__ works correctly."""
        expected = (
            "SparkGroupedDataFrameDomain(schema={'A': SparkIntegerColumnDescriptor("
            "allow_null=True, size=64), 'B': SparkStringColumnDescriptor(allow_null="
            "True), 'C': SparkIntegerColumnDescriptor(allow_null=True, size=64)},"
            " groupby_columns=['A', 'B'])"
        )
        assert repr(domain) == expected

    @pytest.mark.parametrize(
        "domain, expected, expectation, exception_properties",
        [
            (
                SparkGroupedDataFrameDomain(
                    schema={
                        "A": SparkIntegerColumnDescriptor(),
                        "B": SparkStringColumnDescriptor(),
                        "C": SparkFloatColumnDescriptor(),
                    },
                    groupby_columns=["A", "B"],
                ),
                SparkDataFrameDomain(schema={"C": SparkFloatColumnDescriptor()}),
                does_not_raise(),
                None,
            )
        ],
    )
    def test_get_group_domain(
        self,
        domain: SparkGroupedDataFrameDomain,
        expected: SparkDataFrameDomain,
        expectation: ContextManager[None],
        exception_properties: Optional[Dict[str, Any]],
    ):
        """get_group_domain returns the correct group domain."""
        with expectation as exception:
            assert domain.get_group_domain() == expected
        if exception_properties is None:
            return
        for key, value in exception_properties.items():
            assert hasattr(exception, key)
            assert getattr(exception, key) == value


class TestSparkRowDomain(DomainTests):
    """Tests for :class:`~tmlt.core.domains.spark_domains.SparkRowDomain`."""

    @pytest.fixture
    def domain_type(self) -> Type[Domain]:
        """Returns the type of the domain to be tested."""
        return SparkRowDomain

    @pytest.fixture
    def domain(self) -> SparkRowDomain:
        """Get a base SparkRowDomain."""
        return SparkRowDomain(
            schema={
                "A": SparkIntegerColumnDescriptor(),
                "B": SparkStringColumnDescriptor(),
            }
        )

    @pytest.mark.parametrize(
        "domain_args, expectation, exception_properties",
        [
            (
                {
                    "schema": {
                        "A": SparkIntegerColumnDescriptor(),
                        "B": SparkStringColumnDescriptor(),
                    }
                },
                does_not_raise(),
                None,
            ),
            (
                {"schema": int},
                pytest.raises(TypeCheckError, match='"schema"'),
                None,
            ),
            (
                {"schema": StringType},
                pytest.raises(TypeCheckError, match='"schema"'),
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
            (
                SparkRowDomain(
                    schema={
                        "A": SparkIntegerColumnDescriptor(),
                        "B": SparkStringColumnDescriptor(),
                    }
                ),
                SparkRowDomain(
                    schema={
                        "A": SparkIntegerColumnDescriptor(),
                        "B": SparkStringColumnDescriptor(),
                    }
                ),
                True,
            ),
            (
                # testing that order does matter
                SparkRowDomain(
                    schema={
                        "B": SparkStringColumnDescriptor(),
                        "A": SparkIntegerColumnDescriptor(),
                    }
                ),
                SparkRowDomain(
                    schema={
                        "A": SparkIntegerColumnDescriptor(),
                        "B": SparkStringColumnDescriptor(),
                    }
                ),
                False,
            ),
            (
                SparkRowDomain(schema={"A": SparkIntegerColumnDescriptor()}),
                SparkRowDomain(schema={"B": SparkStringColumnDescriptor()}),
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
                        "A": SparkIntegerColumnDescriptor(),
                        "B": SparkStringColumnDescriptor(),
                    }
                },
                "schema",
                lambda x: x.update({"A": SparkFloatColumnDescriptor()}),
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
                SparkRowDomain(
                    schema={
                        "A": SparkIntegerColumnDescriptor(),
                        "B": SparkStringColumnDescriptor(),
                    }
                ),
                {
                    "schema": {
                        "A": SparkIntegerColumnDescriptor(),
                        "B": SparkStringColumnDescriptor(),
                    },
                    "carrier_type": Row,
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

    @pytest.mark.skip(reason="SparkRowDomain does not implement validate.")
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
                the exception is expected to have. Mostly used for testing the custom
                exceptions.
        """
        super().test_validate(domain, candidate, expectation, exception_properties)

    def test_repr(self, domain: Domain):
        """Tests that __repr__ works correctly."""
        expected = (
            "SparkRowDomain(schema={'A': SparkIntegerColumnDescriptor(allow_null=False,"
            " size=64), 'B': SparkStringColumnDescriptor(allow_null=False)})"
        )
        assert repr(domain) == expected


_column_descriptors = {
    "int32": SparkIntegerColumnDescriptor(size=32),
    "int64": SparkIntegerColumnDescriptor(size=64),
    "float32": SparkFloatColumnDescriptor(size=32),
    "str": SparkStringColumnDescriptor(allow_null=True),
    "date": SparkDateColumnDescriptor(),
    "timestamp": SparkTimestampColumnDescriptor(),
}
_col_name_to_type: Dict[str, str] = {
    "A": "int32",
    "B": "int64",
    "C": "float32",
    "D": "str",
    "E": "date",
    "F": "timestamp",
}
_type_to_spark_type: Dict[str, DataType] = {
    "int32": IntegerType(),
    "int64": LongType(),
    "float32": FloatType(),
    "str": StringType(),
    "date": DateType(),
    "timestamp": TimestampType(),
}


@pytest.mark.usefixtures("class_spark")
class TestSparkColumnDescriptors:
    r"""Tests for subclasses of class SparkColumnDescriptor.

    See subclasses of
    :class:`~tmlt.core.domains.spark_domains.SparkColumnDescriptor`\ s."""

    spark: SparkSession

    @pytest.fixture
    def test_df(self) -> DataFrame:
        """Get a base DataFrame"""
        return self.spark.createDataFrame(
            [
                (
                    1,
                    2,
                    1.0,
                    "X",
                    datetime.date.fromisoformat("1970-01-01"),
                    datetime.datetime.fromisoformat("1970-01-01 00:00:00.000+00:00"),
                ),
                (
                    11,
                    239,
                    2.0,
                    None,
                    datetime.date.fromisoformat("2022-01-01"),
                    datetime.datetime.fromisoformat("2022-01-01 08:30:00.000+00:00"),
                ),
            ],
            schema=StructType(
                [
                    StructField("A", IntegerType(), False),
                    StructField("B", LongType(), False),
                    StructField("C", FloatType(), False),
                    StructField("D", StringType(), True),
                    StructField("E", DateType(), True),
                    StructField("F", TimestampType(), True),
                ]
            ),
        )

    @pytest.mark.parametrize(
        "descriptor, expected_domain",
        [
            (SparkIntegerColumnDescriptor(size=32), NumpyIntegerDomain(size=32)),
            (SparkIntegerColumnDescriptor(size=64), NumpyIntegerDomain(size=64)),
            (SparkFloatColumnDescriptor(size=64), NumpyFloatDomain(size=64)),
            (
                SparkFloatColumnDescriptor(size=64, allow_inf=True),
                NumpyFloatDomain(size=64, allow_inf=True),
            ),
            (
                SparkFloatColumnDescriptor(size=64, allow_nan=True),
                NumpyFloatDomain(size=64, allow_nan=True),
            ),
            (
                SparkStringColumnDescriptor(allow_null=True),
                NumpyStringDomain(allow_null=True),
            ),
            (
                SparkStringColumnDescriptor(allow_null=False),
                NumpyStringDomain(allow_null=False),
            ),
        ],
    )
    def test_to_numpy_domain(
        self, descriptor: SparkColumnDescriptor, expected_domain: Domain
    ):
        """Tests that to_numpy_domain works correctly."""
        assert descriptor.to_numpy_domain() == expected_domain

    @pytest.mark.parametrize(
        "descriptor, expectation",
        [
            (
                SparkIntegerColumnDescriptor(allow_null=True),
                pytest.raises(
                    RuntimeError,
                    match="Nullable column does not have corresponding NumPy domain.",
                ),
            ),
            (
                SparkDateColumnDescriptor(),
                pytest.raises(
                    RuntimeError, match="NumPy does not have support for date types."
                ),
            ),
            (
                SparkTimestampColumnDescriptor(),
                pytest.raises(
                    RuntimeError,
                    match="NumPy does not have support for timestamp types.",
                ),
            ),
        ],
    )
    def test_to_numpy_domain_invalid(
        self, descriptor: SparkColumnDescriptor, expectation: ContextManager[None]
    ):
        """Tests that to_numpy_domain raises appropriate exceptions."""
        with expectation:
            descriptor.to_numpy_domain()

    @pytest.mark.parametrize(
        "descriptor, col_name, expectation",
        [
            (
                _column_descriptors[col_type],
                col_name,
                (
                    does_not_raise()
                    if col_type == _col_name_to_type[col_name]
                    else pytest.raises(
                        ValueError,
                        match="Column must be "
                        f"{get_fullname(_type_to_spark_type[col_type])}; got "
                        f"{get_fullname(_type_to_spark_type[_col_name_to_type[col_name]])} "  # pylint: disable=line-too-long
                        "instead",
                    )
                ),
            )
            for col_name, col_type in product(
                _col_name_to_type.keys(), _col_name_to_type.values()
            )
        ],
    )
    def test_validate_column(
        self,
        test_df: DataFrame,
        descriptor: SparkColumnDescriptor,
        col_name: str,
        expectation: ContextManager[None],
    ):
        """Tests that validate_column works correctly."""
        with expectation:
            descriptor.validate_column(test_df, col_name)

    @pytest.mark.parametrize(
        "domain, other_domain, expected",
        [
            (
                _column_descriptors[base_key],
                _column_descriptors[other_key],
                base_key == other_key,
            )
            for base_key, other_key in combinations_with_replacement(
                _col_name_to_type.values(), 2
            )
        ]
        + [
            (
                SparkIntegerColumnDescriptor(size=32),
                SparkIntegerColumnDescriptor(size=32, allow_null=True),
                False,
            )
        ],
    )
    def test_eq(
        self,
        domain: SparkColumnDescriptor,
        other_domain: SparkColumnDescriptor,
        expected: bool,
    ):
        """Tests that __eq__ works correctly."""
        assert (domain == other_domain) == expected


class TestSparkUtilityFunctions:
    """Tests for utility functions for creating Spark domains."""

    @pytest.mark.parametrize(
        "spark_schema, expected, expectation",
        [
            (StructType(), {}, does_not_raise()),
            (
                StructType([StructField("A", IntegerType(), False)]),
                {"A": SparkIntegerColumnDescriptor(size=32)},
                does_not_raise(),
            ),
            (
                StructType(
                    [
                        StructField("A", IntegerType(), False),
                        StructField("B", LongType(), False),
                        StructField("C", FloatType(), False),
                        StructField("D", StringType(), True),
                        StructField("E", DateType(), True),
                        StructField("F", TimestampType(), True),
                    ]
                ),
                {
                    "A": SparkIntegerColumnDescriptor(size=32),
                    "B": SparkIntegerColumnDescriptor(size=64),
                    "C": SparkFloatColumnDescriptor(
                        allow_nan=True, allow_inf=True, size=32
                    ),
                    "D": SparkStringColumnDescriptor(allow_null=True),
                    "E": SparkDateColumnDescriptor(allow_null=True),
                    "F": SparkTimestampColumnDescriptor(allow_null=True),
                },
                does_not_raise(),
            ),
        ],
    )
    def test_convert_spark_schema(
        self,
        spark_schema: StructType,
        expected: SparkColumnsDescriptor,
        expectation: ContextManager[None],
    ):
        """Tests that convert_spark_schema works correctly.

        Args:
            spark_schema: The Spark schema to convert.
            expected: The expected result of the conversion.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
        """
        with expectation:
            assert convert_spark_schema(spark_schema) == expected

    @pytest.mark.parametrize(
        "pandas_domain, expected, expectation",
        [
            (PandasDataFrameDomain({}), {}, does_not_raise()),
            (
                PandasDataFrameDomain(
                    {
                        "A": PandasSeriesDomain(NumpyIntegerDomain(size=32)),
                        "B": PandasSeriesDomain(NumpyIntegerDomain(size=64)),
                        "C": PandasSeriesDomain(
                            NumpyFloatDomain(allow_nan=True, allow_inf=True, size=32)
                        ),
                        "D": PandasSeriesDomain(NumpyStringDomain()),
                    }
                ),
                {
                    "A": SparkIntegerColumnDescriptor(size=32),
                    "B": SparkIntegerColumnDescriptor(size=64),
                    "C": SparkFloatColumnDescriptor(
                        allow_nan=True, allow_inf=True, size=32
                    ),
                    "D": SparkStringColumnDescriptor(),
                },
                does_not_raise(),
            ),
        ],
    )
    def test_convert_pandas_domain(
        self,
        pandas_domain: PandasDataFrameDomain,
        expected: SparkColumnsDescriptor,
        expectation: ContextManager[None],
    ):
        """Tests that convert_pandas_domain works correctly.

        Args:
            pandas_domain: The Pandas domain to convert.
            expected: The expected result of the conversion.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
        """
        with expectation:
            assert convert_pandas_domain(pandas_domain) == expected

    @pytest.mark.parametrize(
        "numpy_domain, expected, expectation",
        [
            (
                NumpyIntegerDomain(size=32),
                SparkIntegerColumnDescriptor(size=32),
                does_not_raise(),
            ),
            (
                NumpyFloatDomain(),
                SparkFloatColumnDescriptor(allow_nan=False, allow_inf=False, size=64),
                does_not_raise(),
            ),
            (
                NumpyFloatDomain(allow_nan=True, allow_inf=True, size=32),
                SparkFloatColumnDescriptor(allow_nan=True, allow_inf=True, size=32),
                does_not_raise(),
            ),
            (NumpyStringDomain(), SparkStringColumnDescriptor(), does_not_raise()),
            (
                NumpyStringDomain(allow_null=True),
                SparkStringColumnDescriptor(allow_null=True),
                does_not_raise(),
            ),
            ("Not a numpy domain", None, pytest.raises(NotImplementedError)),
        ],
    )
    def test_convert_numpy_domain(
        self,
        numpy_domain: NumpyDomain,
        expected: SparkColumnDescriptor,
        expectation: ContextManager[None],
    ):
        """Tests that convert_numpy_domain works correctly.

        Args:
            numpy_domain: The Numpy domain to convert.
            expected: The expected result of the conversion.
            expectation: A context manager that captures the correct expected type of
                error that is raised.
        """
        with expectation:
            assert convert_numpy_domain(numpy_domain) == expected
