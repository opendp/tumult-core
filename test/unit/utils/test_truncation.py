"""Tests for :mod:`~tmlt.core.utils.truncation`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import copy
import datetime
import itertools
from typing import Any, List, Tuple
from unittest.mock import patch

import pandas as pd
import pytest
from parameterized import parameterized
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import (
    BinaryType,
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
from pyspark.testing import assertSchemaEqual

from tmlt.core.utils.testing import PySparkTest, assert_dataframe_equal
from tmlt.core.utils.truncation import (
    _hash_column,
    drop_large_groups,
    limit_keys_per_group,
    truncate_large_groups,
)


class TestTruncateLargeGroups(PySparkTest):
    """Tests for :meth:`~tmlt.core.utils.truncation.truncate_large_groups`."""

    @parameterized.expand(
        [
            (2, [(1, "x"), (1, "y"), (1, "z"), (1, "w")], 2),
            (2, [(1, "x")], 1),
            (0, [(1, "x"), (1, "y"), (1, "z"), (1, "w")], 0),
        ]
    )
    def test_correctness(
        self, threshold: int, rows: List[Tuple], expected_count: int
    ) -> None:
        """Tests that truncate_large_groups works correctly."""
        df = self.spark.createDataFrame(rows, schema=["A", "B"])
        self.assertEqual(
            truncate_large_groups(df, ["A"], threshold).count(), expected_count
        )

    def test_consistency(self) -> None:
        """Tests that truncate_large_groups does not truncate randomly across calls."""
        df = self.spark.createDataFrame([(i,) for i in range(1000)], schema=["A"])

        expected_output = truncate_large_groups(df, ["A"], 5)
        for _ in range(5):
            assert_dataframe_equal(truncate_large_groups(df, ["A"], 5), expected_output)

    def test_rows_dropped_consistently(self) -> None:
        """Tests that truncate_large_groups drops that same rows for unchanged keys."""
        df1 = self.spark.createDataFrame(
            [("A", 1), ("B", 2), ("B", 3)], schema=["W", "X"]
        )
        df2 = self.spark.createDataFrame(
            [("A", 0), ("A", 1), ("B", 2), ("B", 3)], schema=["W", "X"]
        )

        df1_truncated = truncate_large_groups(df1, ["W"], 1)
        df2_truncated = truncate_large_groups(df2, ["W"], 1)
        assert_dataframe_equal(
            df1_truncated.filter("W='B'"),
            df2_truncated.filter("W='B'"),
        )

    def test_hash_truncation_order_agnostic(self) -> None:
        """Tests that truncate_large_groups doesn't depend on row order."""
        df_rows = [(1, 2, "A"), (3, 4, "A"), (5, 6, "A"), (7, 8, "B")]

        truncated_dfs: List[pd.DataFrame] = []
        for permutation in itertools.permutations(df_rows, 4):
            df = self.spark.createDataFrame(list(permutation), schema=["W", "X", "Y"])
            truncated_dfs.append(truncate_large_groups(df, ["Y"], 1).toPandas())
        for df in truncated_dfs[1:]:
            assert_dataframe_equal(truncated_dfs[0], df)

    def test_hash_truncation_duplicate_rows_not_clumped(self) -> None:
        """Tests that truncate_large_groups doesn't clump duplicate rows together."""
        df = self.spark.createDataFrame(
            [
                (1, 2, "A"),
                (1, 2, "A"),
                (1, 2, "A"),
                (1, 2, "A"),
                (1, 2, "A"),
                (2, 4, "A"),
                (2, 4, "A"),
                (2, 4, "A"),
                (2, 4, "A"),
                (2, 4, "A"),
            ],
            schema=["X", "Y", "Z"],
        )
        actual = truncate_large_groups(df, ["Z"], threshold=5).toPandas()
        assert isinstance(actual, pd.DataFrame)
        assert len(actual.drop_duplicates()) == 2


class TestDropLargeGroups(PySparkTest):
    """Tests for :meth:`~tmlt.core.utils.truncation.drop_large_groups`."""

    @parameterized.expand(
        [
            (1, [(1, "A"), (1, "B"), (2, "C")], [(2, "C")]),
            (1, [(1, "A"), (2, "C")], [(1, "A"), (2, "C")]),
            (2, [(1, "A"), (2, "C"), (2, "D"), (2, "E")], [(1, "A")]),
            (1, [(1, "A"), (1, "B"), (2, "C"), (2, "D"), (2, "E")], []),
            (0, [(1, "x"), (2, "y"), (3, "z"), (3, "w")], []),
        ]
    )
    def test_correctness(
        self, threshold: int, input_rows: List[Tuple], expected: List[Tuple]
    ) -> None:
        """Tests that drop_large_groups works correctly."""
        df = self.spark.createDataFrame(input_rows, schema=["A", "B"])
        actual = drop_large_groups(df, ["A"], threshold)
        expected = pd.DataFrame.from_records(expected, columns=["A", "B"])
        assert_dataframe_equal(actual, expected)


class TestLimitKeysPerGroup(PySparkTest):
    """Tests for :func:`~tmlt.core.utils.truncation.limit_keys_per_group`."""

    def test_hash_collisions(self):
        """Test :func:`~.limit_keys_per_group` works when there are hash collisions.

        This test fails for a previous, incorrect version of
        :func:`~.limit_keys_per_group`. See
        https://gitlab.com/tumult-labs/tumult/-/issues/2455 for more details.
        """
        df = self.spark.createDataFrame(
            pd.DataFrame({"A": [1, 1, 1, 1, 2, 2, 2, 2], "B": [1, 1, 2, 2, 1, 2, 3, 4]})
        )
        # replace the hash function with one that always returns 1
        hash_collision_mock = udf(lambda _, __: 1, IntegerType())
        with patch("pyspark.sql.functions.hash", hash_collision_mock):
            actual = limit_keys_per_group(df, ["A"], ["B"], 1)
        self.assertEqual(actual.count(), 3)


# Note: The values in these tests are arbitrary and not meaningful.
@pytest.mark.parametrize(
    "test_rows,schema",
    [
        # Int, Long, Float, Double Types Checked
        (
            [(1, 1, 1.0, 1.0)],
            StructType(
                [
                    StructField("A", IntegerType(), True),
                    StructField("B", LongType(), True),
                    StructField("C", FloatType(), True),
                    StructField("D", DoubleType(), True),
                ]
            ),
        ),
        # Binary and String Types Checked
        (
            [("String", bytes("String", "utf-8"))],
            StructType(
                [
                    StructField("A", StringType(), True),
                    StructField("B", BinaryType(), True),
                ]
            ),
        ),
        # Date and Timestamp Types Checked
        (
            [
                (
                    datetime.date.fromisoformat("2022-01-01"),
                    datetime.datetime.fromisoformat("2022-01-01T12:30:00"),
                )
            ],
            StructType(
                [
                    StructField("A", DateType(), True),
                    StructField("B", TimestampType(), True),
                ]
            ),
        ),
    ],
)
def test_hash_column(test_rows: List[Any], schema: StructType):
    """Smoke test to ensure that expected datatypes are hashed correctly."""
    # Initialize Spark Session
    spark = SparkSession.builder.getOrCreate()

    # Create a DataFrame with the specific data types from a schema
    test_df = spark.createDataFrame(test_rows, schema)

    for column in test_df.columns:
        end_df, new_col_name = _hash_column(test_df, column)
        # Triggers Spark's lazy evaluation
        end_df.count()

        # Construct schema to compare to:
        original_schema_copy = copy.deepcopy(schema)
        result_schema = original_schema_copy.add(
            StructField(new_col_name, StringType(), nullable=True)
        )
        # Check that the end dtype is correct.
        assertSchemaEqual(end_df.schema, result_schema)
