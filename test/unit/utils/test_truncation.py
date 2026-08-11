"""Tests for :mod:`~tmlt.core.utils.truncation`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import copy
import datetime
from typing import Any, List
from unittest.mock import patch

import pandas as pd
import pytest
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

from tmlt.core.utils.testing import PySparkTest
from tmlt.core.utils.truncation import _hash_column, limit_keys_per_group

# The behavioral tests that used to live here (correctness, consistency and
# order-agnosticism of the three truncation functions) are now the
# backend-parametrized tests of test.unit.utils.test_truncation_backends,
# which run the same cases against this module's Spark implementations and
# their pandas counterparts alike. This module keeps only the tests that are
# specific to the Spark implementation.


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
        # Float and Double Edge Types Checked
        (
            [
                (
                    float("nan"),
                    float("inf"),
                    float("-inf"),
                    float("nan"),
                    float("inf"),
                    float("-inf"),
                )
            ],
            StructType(
                [
                    StructField("A", DoubleType(), True),
                    StructField("B", DoubleType(), True),
                    StructField("C", DoubleType(), True),
                    StructField("D", FloatType(), True),
                    StructField("E", FloatType(), True),
                    StructField("F", FloatType(), True),
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
        result_df, new_col_name = _hash_column(test_df, column)
        # Triggers Spark's lazy evaluation
        result_df.count()

        # Construct schema to compare to:
        expected_schema = copy.deepcopy(schema).add(
            StructField(new_col_name, StringType(), nullable=True)
        )
        # Check that the end dtype is correct.
        for result, expectation in zip(result_df.schema.fields, expected_schema.fields):
            assert result.name == expectation.name, (
                f"Result field name ({result.name}) didn't match "
                f"expected field name ({expectation.name})."
            )

            assert result.dataType == expectation.dataType, (
                f"Result field type ({result.dataType}) didn't match "
                f"expected field type ({expectation.dataType})."
            )
