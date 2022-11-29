"""Unit tests for :mod:`~tmlt.core.domains.spark_domains`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

import datetime
from typing import Any, Optional

import pandas as pd
from parameterized import parameterized
from pyspark.sql.types import (
    DateType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from tmlt.core.domains.base import Domain, OutOfDomainError
from tmlt.core.domains.collections import ListDomain
from tmlt.core.domains.numpy_domains import (
    NumpyFloatDomain,
    NumpyIntegerDomain,
    NumpyStringDomain,
)
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
)
from tmlt.core.utils.grouped_dataframe import GroupedDataFrame
from tmlt.core.utils.testing import (
    PySparkTest,
    assert_property_immutability,
    get_all_props,
)


class TestSparkDataFrameDomain(PySparkTest):
    """Tests for :class:`SparkDataFrameDomain`."""

    def setUp(self):
        """Setup."""
        self.domain = SparkDataFrameDomain(
            schema={
                "A": SparkIntegerColumnDescriptor(),
                "B": SparkStringColumnDescriptor(),
            }
        )

    def test_constructor_mutable_arguments(self):
        """Tests that mutable constructor arguments are copied."""
        schema = {
            "A": SparkIntegerColumnDescriptor(),
            "B": SparkStringColumnDescriptor(),
        }
        domain = SparkDataFrameDomain(schema=schema)
        schema["A"] = NumpyFloatDomain()
        self.assertDictEqual(
            domain.schema,
            {"A": SparkIntegerColumnDescriptor(), "B": SparkStringColumnDescriptor()},
        )

    @parameterized.expand(get_all_props(SparkDataFrameDomain))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        assert_property_immutability(self.domain, prop_name)

    def test_repr(self):
        """Tests that __repr__ works correctly."""
        expected = (
            "SparkDataFrameDomain(schema={'A':"
            " SparkIntegerColumnDescriptor(allow_null=False, size=64), 'B':"
            " SparkStringColumnDescriptor(allow_null=False)})"
        )
        print(repr(self.domain))
        self.assertEqual(repr(self.domain), expected)


class TestSparkRowDomain(PySparkTest):
    """Tests for :class:`SparkRowDomain`."""

    def setUp(self):
        """Setup."""
        self.domain = SparkRowDomain(
            schema={
                "A": SparkIntegerColumnDescriptor(),
                "B": SparkStringColumnDescriptor(),
            }
        )

    def test_constructor_mutable_arguments(self):
        """Tests that mutable constructor arguments are copied."""
        schema = {
            "A": SparkIntegerColumnDescriptor(),
            "B": SparkStringColumnDescriptor(),
        }
        domain = SparkRowDomain(schema=schema)
        schema["A"] = NumpyFloatDomain()
        self.assertDictEqual(
            domain.schema,
            {"A": SparkIntegerColumnDescriptor(), "B": SparkStringColumnDescriptor()},
        )

    @parameterized.expand(get_all_props(SparkRowDomain))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""

        assert_property_immutability(self.domain, prop_name)

    def test_repr(self):
        """Tests that __repr__ works correctly."""
        expected = (
            "SparkRowDomain(schema={'A': SparkIntegerColumnDescriptor(allow_null=False,"
            " size=64), 'B': SparkStringColumnDescriptor(allow_null=False)})"
        )
        self.assertEqual(repr(self.domain), expected)


class TestSparkBasedDomains(PySparkTest):
    """Tests for Spark-based Domains.

    In particular, the following domains are tested:
        1. SparkDataFrameDomain
        2. SparkRowDomain
        3. ListDomain[SparkRowDomain]
    """

    def setUp(self):
        """Setup Schema."""

        self.schema = {
            "A": SparkStringColumnDescriptor(),
            "B": SparkStringColumnDescriptor(),
            "C": SparkFloatColumnDescriptor(),
        }

    @parameterized.expand(
        [
            (SparkDataFrameDomain, StringType),
            (SparkRowDomain, int),
            (SparkRowDomain, StringType),
        ]
    )
    def test_invalid_spark_domain_inputs(self, SparkDomain: type, invalid_input: type):
        """Test Spark-based domains with invalid inputs."""
        with self.assertRaises(TypeError):
            SparkDomain(invalid_input)

    @parameterized.expand(
        [
            (  # LongType() instead of DoubleType()
                SparkDataFrameDomain,
                pd.DataFrame(
                    [["A", "B", 10], ["V", "E", 12], ["A", "V", 13]],
                    columns=["A", "B", "C"],
                ),
                "Found invalid value in column 'C': Column must be"
                r" DoubleType(\(\))?, instead it is LongType(\(\))?.",
            ),
            (  # Missing Columns
                SparkDataFrameDomain,
                pd.DataFrame([["A", "B"], ["V", "E"], ["A", "V"]], columns=["A", "B"]),
                "Columns are not as expected. DataFrame and Domain must contain the "
                "same columns in the same order.\n"
                r"DataFrame columns: \['A', 'B'\]"
                "\n"
                r"Domain columns: \['A', 'B', 'C'\]",
            ),
            (
                SparkDataFrameDomain,
                pd.DataFrame(
                    [["A", "B", 1.1], ["V", "E", 1.2], ["A", "V", 1.3]],
                    columns=["A", "B", "C"],
                ),
                None,
            ),
        ]
    )
    def test_validate(
        self, SparkDomain: type, candidate: Any, exception: Optional[str]
    ):
        """Tests that validate works as expected.

        Args:
            SparkDomain: Domain type to be checked.
            candidate: Object to be checked for membership.
            exception: Expected exception if validation fails.
        """
        domain = (
            SparkDomain(self.schema)
            if SparkDomain != ListDomain
            else SparkDomain(SparkRowDomain(self.schema))
        )
        if isinstance(candidate, pd.DataFrame):
            candidate = self.spark.createDataFrame(candidate)

        if exception is not None:
            with self.assertRaisesRegex(OutOfDomainError, exception):
                domain.validate(candidate)
        else:
            self.assertEqual(domain.validate(candidate), exception)

    @parameterized.expand(
        [
            (  # matching
                {
                    "A": SparkStringColumnDescriptor(),
                    "B": SparkStringColumnDescriptor(),
                    "C": SparkFloatColumnDescriptor(),
                },
                True,
            ),
            (  # shuffled
                {
                    "B": SparkStringColumnDescriptor(),
                    "C": SparkFloatColumnDescriptor(),
                    "A": SparkStringColumnDescriptor(),
                },
                False,
            ),
            (  # Mismatching Types
                {
                    "A": SparkStringColumnDescriptor(),
                    "B": SparkStringColumnDescriptor(),
                    "C": SparkFloatColumnDescriptor(size=32),
                },
                False,
            ),
            (  # Extra attribute
                {
                    "A": SparkStringColumnDescriptor(),
                    "B": SparkStringColumnDescriptor(),
                    "C": SparkFloatColumnDescriptor(),
                    "D": SparkFloatColumnDescriptor(),
                },
                False,
            ),
            (  # Missing attribute
                {
                    "A": SparkStringColumnDescriptor(),
                    "B": SparkStringColumnDescriptor(),
                },
                False,
            ),
        ]
    )
    def test_eq(self, other_schema: SparkColumnsDescriptor, is_equal: bool):
        """Tests that __eq__ works as expected for SparkDataFrameDomain."""
        self.assertEqual(
            SparkDataFrameDomain(other_schema) == SparkDataFrameDomain(self.schema),
            is_equal,
        )
        self.assertEqual(
            SparkRowDomain(other_schema) == SparkRowDomain(self.schema), is_equal
        )
        self.assertEqual(
            ListDomain(SparkRowDomain(other_schema))
            == ListDomain(SparkRowDomain(self.schema)),
            is_equal,
        )


class TestSparkGroupedDataFrameDomain(PySparkTest):
    """Tests for SparkGroupedDataFrameDomain."""

    def setUp(self):
        """Setup test."""
        self.group_keys = self.spark.createDataFrame(
            [(1, "W"), (2, "X"), (3, "Y")], schema=["A", "B"]
        )
        self.schema = {
            "A": SparkIntegerColumnDescriptor(allow_null=True),
            "B": SparkStringColumnDescriptor(allow_null=True),
            "C": SparkIntegerColumnDescriptor(allow_null=True),
        }
        self.domain = SparkGroupedDataFrameDomain(
            schema=self.schema, group_keys=self.group_keys
        )

    def test_constructor_mutable_arguments(self):
        """Tests that mutable constructor arguments are copied."""
        schema = {
            "A": SparkIntegerColumnDescriptor(),
            "B": SparkStringColumnDescriptor(),
        }
        domain = SparkGroupedDataFrameDomain(schema=schema, group_keys=self.group_keys)
        schema["A"] = NumpyFloatDomain()
        self.assertDictEqual(
            domain.schema,
            {"A": SparkIntegerColumnDescriptor(), "B": SparkStringColumnDescriptor()},
        )

    @parameterized.expand(get_all_props(SparkGroupedDataFrameDomain))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        assert_property_immutability(self.domain, prop_name)

    def test_carrier_type(self):
        """Tests that SparkGroupedDataFrameDomain has expected carrier type."""
        self.assertEqual(self.domain.carrier_type, GroupedDataFrame)

    @parameterized.expand(
        [
            (
                pd.DataFrame({"A": [1, 2], "B": ["W", "W"], "C": [4, 5]}),
                pd.DataFrame({"A": [1, 2, 3], "B": ["W", "X", "Y"]}),
                True,
            ),
            (  # Mismatching DataFrame domain (extra column D)
                pd.DataFrame({"A": [1, 2], "B": ["W", "W"], "C": [4, 5], "D": [4, 5]}),
                pd.DataFrame({"A": [1, 2, 3], "B": ["W", "X", "Y"]}),
                False,
            ),
            (  # Mismatching group keys
                pd.DataFrame({"A": [1, 2], "B": ["W", "W"], "C": [4, 5]}),
                pd.DataFrame({"A": [2, 3, 1], "B": ["W", "X", "Y"]}),
                False,
            ),
        ]
    )
    def test_contains(
        self, dataframe: pd.DataFrame, group_keys: pd.DataFrame, expected: bool
    ):
        """Tests that __contains__ works correctly."""
        grouped_data = GroupedDataFrame(
            dataframe=self.spark.createDataFrame(dataframe),
            group_keys=self.spark.createDataFrame(group_keys),
        )
        self.assertEqual(grouped_data in self.domain, expected)

    def test_eq_positive(self):
        """Tests that __eq__ returns True correctly."""
        other_domain = SparkGroupedDataFrameDomain(
            schema=self.schema, group_keys=self.group_keys
        )
        self.assertTrue(self.domain == other_domain)

    def test_eq_negative(self):
        """Tests that __eq__ returns False correctly."""
        mismatching_schema_domain = SparkGroupedDataFrameDomain(
            schema={
                "A": SparkIntegerColumnDescriptor(),
                "B": SparkStringColumnDescriptor(),
                "C": SparkIntegerColumnDescriptor(),
                "D": SparkIntegerColumnDescriptor(),
            },
            group_keys=self.group_keys,
        )
        mismatching_group_keys_domain = SparkGroupedDataFrameDomain(
            schema=self.schema,
            group_keys=self.spark.createDataFrame(
                [(1, "W"), (2, "X"), (3, "Y"), (4, "Z")], schema=["A", "B"]
            ),
        )
        self.assertFalse(self.domain == mismatching_schema_domain)
        self.assertFalse(self.domain == mismatching_group_keys_domain)

    @parameterized.expand(
        [
            (
                {
                    "A": SparkIntegerColumnDescriptor(),
                    "B": SparkStringColumnDescriptor(),
                },
                pd.DataFrame({"C": [1, 2]}),
                "Invalid groupby column: {'C'}",
            ),
            (
                {
                    "A": SparkIntegerColumnDescriptor(),
                    "B": SparkStringColumnDescriptor(),
                },
                pd.DataFrame({"B": [1, 2]}),
                "Column must be StringType",
                OutOfDomainError,
            ),
        ]
    )
    def test_post_init(
        self,
        schema: SparkColumnsDescriptor,
        group_keys: pd.DataFrame,
        error_msg: str,
        error_type: type = ValueError,
    ):
        """Tests that __post_init__ correctly rejects invalid inputs."""
        with self.assertRaisesRegex(error_type, error_msg):
            SparkGroupedDataFrameDomain(
                schema=schema, group_keys=self.spark.createDataFrame(group_keys)
            )

    def test_post_init_removes_duplicate_keys(self):
        """Tests that __post_init__ removes duplicate group keys."""
        domain = SparkGroupedDataFrameDomain(
            schema=self.schema,
            group_keys=self.spark.createDataFrame(
                [(1, "W"), (1, "W")], schema=["A", "B"]
            ),
        )
        expected = pd.DataFrame({"A": [1], "B": ["W"]})
        actual = domain.group_keys.toPandas()
        self.assert_frame_equal_with_sort(expected, actual)

    def test_eq_no_group_keys(self):
        """Tests that __eq__ works for empty group_keys."""
        domain1 = SparkGroupedDataFrameDomain(
            schema=self.schema,
            group_keys=self.spark.createDataFrame([], schema=StructType([])),
        )
        domain2 = SparkGroupedDataFrameDomain(
            schema=self.schema,
            group_keys=self.spark.createDataFrame([], schema=StructType([])),
        )
        self.assertTrue(domain1 == domain2)

    def test_validate_with_nulls(self):
        """Test that validate works correctly with nulls."""
        group_keys_with_nulls = self.spark.createDataFrame(
            [(1, "W"), (2, "X"), (3, None)], schema=["A", "B"]
        )
        group_keys_without_nulls = self.spark.createDataFrame(
            [(1, "W"), (1, "X")], schema=["A", "B"]
        )
        dataframe_with_nulls = self.spark.createDataFrame(
            [(1, "W", 0), (None, "X", 1), (None, "X", 2)], schema=["A", "B", "C"]
        )
        domain_with_nulls = SparkGroupedDataFrameDomain(
            schema=self.schema, group_keys=group_keys_with_nulls
        )
        # everything matches, no exception should be raised
        domain_with_nulls.validate(
            GroupedDataFrame(
                dataframe=dataframe_with_nulls, group_keys=group_keys_with_nulls
            )
        )
        # group keys don't match, should raise exception
        with self.assertRaises(OutOfDomainError):
            domain_with_nulls.validate(
                GroupedDataFrame(
                    dataframe=dataframe_with_nulls, group_keys=group_keys_without_nulls
                )
            )
        domain_without_nulls = SparkGroupedDataFrameDomain(
            schema=self.schema, group_keys=group_keys_without_nulls
        )
        # group keys match, nulls should be fine
        domain_without_nulls.validate(
            GroupedDataFrame(
                dataframe=dataframe_with_nulls, group_keys=group_keys_without_nulls
            )
        )
        # group keys don't match, should raise exception
        with self.assertRaises(OutOfDomainError):
            domain_without_nulls.validate(
                GroupedDataFrame(
                    dataframe=dataframe_with_nulls, group_keys=group_keys_with_nulls
                )
            )

    def test_eq_with_nulls(self):
        """Tests that eq works correctly with null values."""
        domain_with_nulls1 = SparkGroupedDataFrameDomain(
            schema=self.schema,
            group_keys=self.spark.createDataFrame(
                [(1, "W"), (1, "X"), (None, "X")], schema=["A", "B"]
            ),
        )
        domain_with_nulls2 = SparkGroupedDataFrameDomain(
            schema=self.schema,
            group_keys=self.spark.createDataFrame(
                [(1, "W"), (1, "X"), (None, "X")], schema=["A", "B"]
            ),
        )
        self.assertEqual(domain_with_nulls1, domain_with_nulls2)
        domain_without_nulls = SparkGroupedDataFrameDomain(
            schema=self.schema,
            group_keys=self.spark.createDataFrame(
                [(1, "W"), (1, "X")], schema=["A", "B"]
            ),
        )
        self.assertNotEqual(domain_with_nulls1, domain_without_nulls)

    def test_repr(self):
        """Tests that __repr__ works correctly."""
        expected = (
            "SparkGroupedDataFrameDomain(schema={'A': SparkIntegerColumnDescriptor("
            "allow_null=True, size=64), 'B': SparkStringColumnDescriptor(allow_null="
            "True), 'C': SparkIntegerColumnDescriptor(allow_null=True, size=64)},"
            " group_keys=DataFrame[A: bigint, B: string])"
        )
        self.assertEqual(repr(self.domain), expected)


class TestSparkColumnDescriptors(PySparkTest):
    r"""Tests for subclasses of class SparkColumnDescriptor.

    See subclasses of
    :class:`~tmlt.core.domains.spark_domains.SparkColumnDescriptor`\ s."""

    def setUp(self):
        """Setup"""
        self.int32_column_descriptor = SparkIntegerColumnDescriptor(size=32)
        self.int64_column_descriptor = SparkIntegerColumnDescriptor(size=64)
        self.float32_column_descriptor = SparkFloatColumnDescriptor(size=32)
        self.str_column_descriptor = SparkStringColumnDescriptor(allow_null=False)
        self.date_column_descriptor = SparkDateColumnDescriptor()
        self.timestamp_column_descriptor = SparkTimestampColumnDescriptor()
        self.test_df = self.spark.createDataFrame(
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

    @parameterized.expand(
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
        ]
    )
    def test_to_numpy_domain(
        self, descriptor: SparkColumnDescriptor, expected_domain: Domain
    ):
        """Tests that to_numpy_domain works correctly."""
        self.assertEqual(descriptor.to_numpy_domain(), expected_domain)

    @parameterized.expand(
        [
            (
                SparkIntegerColumnDescriptor(allow_null=True),
                "Nullable column does not have corresponding NumPy domain.",
            ),
            (
                SparkDateColumnDescriptor(),
                "NumPy does not have support for date types.",
            ),
            (
                SparkTimestampColumnDescriptor(),
                "NumPy does not have support for timestamp types.",
            ),
        ]
    )
    def test_to_numpy_domain_invalid(
        self, descriptor: SparkColumnDescriptor, expected_error: str
    ):
        """Tests that to_numpy_domain raises appropriate exceptions."""
        with self.assertRaisesRegex(RuntimeError, expected_error):
            descriptor.to_numpy_domain()

    @parameterized.expand(
        [
            ("A", "int32"),
            ("B", "int64"),
            ("C", "float32"),
            ("D", "str"),
            ("E", "date"),
            ("F", "timestamp"),
        ]
    )
    def test_validate_column(self, col_name: str, col_type: str):
        """Tests that validate_column works correctly."""
        if col_type == "int32":
            self.int32_column_descriptor.validate_column(self.test_df, col_name)
        else:
            with self.assertRaisesRegex(
                OutOfDomainError,
                r"Column must be IntegerType(\(\))?, instead it is "
                f"{self.test_df.schema[col_name].dataType}.",
            ):
                self.int32_column_descriptor.validate_column(self.test_df, col_name)

        if col_type == "int64":
            self.int64_column_descriptor.validate_column(self.test_df, col_name)
        else:
            with self.assertRaisesRegex(
                OutOfDomainError,
                r"Column must be LongType(\(\))?, instead it is "
                f"{self.test_df.schema[col_name].dataType}.",
            ):
                self.int64_column_descriptor.validate_column(self.test_df, col_name)

        if col_type == "float32":
            self.float32_column_descriptor.validate_column(self.test_df, col_name)
        else:
            with self.assertRaisesRegex(
                OutOfDomainError,
                r"Column must be FloatType(\(\))?, instead it is "
                f"{self.test_df.schema[col_name].dataType}.",
            ):
                self.float32_column_descriptor.validate_column(self.test_df, col_name)

        if col_type == "str":
            with self.assertRaisesRegex(
                OutOfDomainError, "Column contains null values."
            ):
                self.str_column_descriptor.validate_column(self.test_df, col_name)
        else:
            with self.assertRaisesRegex(
                OutOfDomainError,
                r"Column must be StringType(\(\))?, instead it is "
                f"{self.test_df.schema[col_name].dataType}.",
            ):
                self.str_column_descriptor.validate_column(self.test_df, col_name)

        if col_type == "date":
            self.date_column_descriptor.validate_column(self.test_df, col_name)
        else:
            with self.assertRaisesRegex(
                OutOfDomainError,
                r"Column must be DateType(\(\))?, instead it is "
                f"{self.test_df.schema[col_name].dataType}.",
            ):
                self.date_column_descriptor.validate_column(self.test_df, col_name)

        if col_type == "timestamp":
            self.timestamp_column_descriptor.validate_column(self.test_df, col_name)
        else:
            with self.assertRaisesRegex(
                OutOfDomainError,
                r"Column must be TimestampType(\(\))?, instead it is "
                f"{self.test_df.schema[col_name].dataType}.",
            ):
                self.timestamp_column_descriptor.validate_column(self.test_df, col_name)

    @parameterized.expand(
        [
            (SparkIntegerColumnDescriptor(size=32), "int32"),
            (SparkIntegerColumnDescriptor(allow_null=True, size=32), "int32_null"),
            (SparkIntegerColumnDescriptor(), "int64"),
            (SparkIntegerColumnDescriptor(allow_null=True), "int64_null"),
            (SparkFloatColumnDescriptor(), "float64"),
            (SparkFloatColumnDescriptor(size=32), "float32"),
            (SparkFloatColumnDescriptor(size=32, allow_nan=True), "float32_nan"),
            (SparkStringColumnDescriptor(), "str"),
            (SparkStringColumnDescriptor(allow_null=True), "str_null"),
            (SparkDateColumnDescriptor(), "date"),
            (SparkDateColumnDescriptor(allow_null=True), "date_null"),
            (SparkTimestampColumnDescriptor(), "timestamp"),
            (SparkTimestampColumnDescriptor(allow_null=True), "timestamp_null"),
        ]
    )
    def test_eq(self, candidate: Any, col_type: str):
        """Tests that __eq__ works correctly."""
        self.assertEqual(self.int32_column_descriptor == candidate, col_type == "int32")
        self.assertEqual(self.int64_column_descriptor == candidate, col_type == "int64")
        self.assertEqual(
            self.float32_column_descriptor == candidate, col_type == "float32"
        )
        self.assertEqual(self.str_column_descriptor == candidate, col_type == "str")
        self.assertEqual(self.date_column_descriptor == candidate, col_type == "date")
        self.assertEqual(
            self.timestamp_column_descriptor == candidate, col_type == "timestamp"
        )
