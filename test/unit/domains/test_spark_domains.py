"""Unit tests for :mod:`~tmlt.core.domains.spark_domains`."""

# <placeholder: boilerplate>
from typing import Any, Optional

import pandas as pd
from parameterized import parameterized
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
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
    SparkFloatColumnDescriptor,
    SparkGroupedDataFrameDomain,
    SparkIntegerColumnDescriptor,
    SparkRowDomain,
    SparkStringColumnDescriptor,
)
from tmlt.core.utils.grouped_dataframe import GroupedDataFrame
from tmlt.core.utils.testing import PySparkTest


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
                "Found invalid value in column 'C': Column must be "
                "DoubleType, instead it is LongType.",
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
            "A": SparkIntegerColumnDescriptor(),
            "B": SparkStringColumnDescriptor(),
            "C": SparkIntegerColumnDescriptor(),
        }
        self.domain = SparkGroupedDataFrameDomain(
            schema=self.schema, group_keys=self.group_keys
        )

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
        self.test_df = self.spark.createDataFrame(
            [(1, 2, 1.0, "X"), (11, 239, 2.0, None)],
            schema=StructType(
                [
                    StructField("A", IntegerType(), False),
                    StructField("B", LongType(), False),
                    StructField("C", FloatType(), False),
                    StructField("D", StringType(), True),
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
            ("A", True, False, False, False),
            ("B", False, True, False, False),
            ("C", False, False, True, False),
            ("D", False, False, False, False),
        ]
    )
    def test_validate_column(
        self,
        col_name: str,
        int32_col: bool,
        int64_col: bool,
        float32_col: bool,
        str_col: bool,
    ):
        """Tests that validate_column works correctly."""
        if not int32_col:
            with self.assertRaisesRegex(
                OutOfDomainError,
                "Column must be IntegerType, instead it is "
                f"{self.test_df.schema[col_name].dataType}.",
            ):
                self.int32_column_descriptor.validate_column(self.test_df, col_name)
        else:
            self.assertEqual(
                self.int32_column_descriptor.validate_column(self.test_df, col_name),
                None,
            )

        if not int64_col:
            with self.assertRaisesRegex(
                OutOfDomainError,
                "Column must be LongType, instead it is "
                f"{self.test_df.schema[col_name].dataType}.",
            ):
                self.int64_column_descriptor.validate_column(self.test_df, col_name)
        else:
            self.assertEqual(
                self.int64_column_descriptor.validate_column(self.test_df, col_name),
                None,
            )

        if not float32_col:
            with self.assertRaisesRegex(
                OutOfDomainError,
                "Column must be FloatType, instead it is "
                f"{self.test_df.schema[col_name].dataType}.",
            ):
                self.float32_column_descriptor.validate_column(self.test_df, col_name)
        else:
            self.assertEqual(
                self.float32_column_descriptor.validate_column(self.test_df, col_name),
                None,
            )

        if not str_col:
            if col_name != "D":
                e = (
                    "Column must be StringType, instead it is "
                    f"{self.test_df.schema[col_name].dataType}."
                )
            else:
                e = "Column contains null values."
                with self.assertRaisesRegex(OutOfDomainError, e):
                    self.str_column_descriptor.validate_column(self.test_df, col_name)
        else:
            self.assertEqual(
                self.str_column_descriptor.validate_column(self.test_df, col_name), None
            )

    @parameterized.expand(
        [
            (SparkIntegerColumnDescriptor(size=32), True, False, False, False),
            (
                SparkIntegerColumnDescriptor(allow_null=True, size=32),
                False,
                False,
                False,
                False,
            ),
            (SparkIntegerColumnDescriptor(), False, True, False, False),
            (SparkIntegerColumnDescriptor(allow_null=True), False, False, False, False),
            (SparkFloatColumnDescriptor(), False, False, False, False),
            (SparkFloatColumnDescriptor(size=32), False, False, True, False),
            (
                SparkFloatColumnDescriptor(size=32, allow_nan=True),
                False,
                False,
                False,
                False,
            ),
            (SparkStringColumnDescriptor(), False, False, False, True),
            (SparkStringColumnDescriptor(allow_null=True), False, False, False, False),
        ]
    )
    def test_eq(
        self,
        candidate: Any,
        int32_col: bool,
        int64_col: bool,
        float32_col: bool,
        str_col: bool,
    ):
        """Tests that __eq__ works correctly."""
        self.assertEqual(self.int32_column_descriptor == candidate, int32_col)
        self.assertEqual(self.int64_column_descriptor == candidate, int64_col)
        self.assertEqual(self.float32_column_descriptor == candidate, float32_col)
        self.assertEqual(self.str_column_descriptor == candidate, str_col)
