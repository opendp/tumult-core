"""Tests for :mod:`~tmlt.core.util.grouped_dataframe`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

import pandas as pd
from parameterized import parameterized
from pyspark.sql import Row
from pyspark.sql import functions as sf
from pyspark.sql.types import IntegerType, StructField, StructType

from tmlt.core.utils.grouped_dataframe import GroupedDataFrame
from tmlt.core.utils.testing import PySparkTest

# pylint: disable=no-member


class TestGroupedDataFrame(PySparkTest):
    """Tests for GroupedDataFrame."""

    @parameterized.expand(
        [
            (
                pd.DataFrame([(1, 2)], columns=["A", "Z"]),
                pd.DataFrame([(1,)], columns=["B"]),
                "Invalid groupby columns",
            )
        ]
    )
    def test_constructor_invalid_inputs(
        self, dataframe: pd.DataFrame, group_keys: pd.DataFrame, error_msg: str
    ):
        """Tests that error is raised when constructor called with invalid inputs."""
        with self.assertRaisesRegex(ValueError, error_msg):
            GroupedDataFrame(
                dataframe=self.spark.createDataFrame(dataframe),
                group_keys=self.spark.createDataFrame(group_keys),
            )

    def test_constructor_drops_duplicate_group_keys(self):
        """Tests that duplicate group keys are silently dropped."""
        grouped_dataframe = GroupedDataFrame(
            dataframe=self.spark.createDataFrame([(1, 2)], schema=["A", "B"]),
            group_keys=self.spark.createDataFrame([(1,), (1,)], schema=["A"]),
        )
        expected_group_keys = pd.DataFrame({"A": [1]})
        self.assert_frame_equal_with_sort(
            expected_group_keys, grouped_dataframe.group_keys.toPandas()
        )

    def test_select_works_correctly(self):
        """Tests that select works correctly."""
        grouped_dataframe = GroupedDataFrame(
            dataframe=self.spark.createDataFrame([(1, 2, 3)], schema=["A", "B", "C"]),
            group_keys=self.spark.createDataFrame([(1,), (1,)], schema=["A"]),
        )
        expected = pd.DataFrame({"A": [1], "B": [2]})
        actual = grouped_dataframe.select(  # pylint:disable=protected-access
            ["A", "B"]
        )._dataframe.toPandas()
        self.assert_frame_equal_with_sort(actual, expected)

    def test_null_safe_join(self) -> None:
        """Test that _null_safe_join works correctly."""
        data = pd.DataFrame({"A": [None, "a0", "a1"], "B": [1, 2, 3]})
        grouped_dataframe = GroupedDataFrame(
            dataframe=self.spark.createDataFrame(data),
            group_keys=self.spark.createDataFrame(pd.DataFrame({"A": [None, "a999"]})),
        )
        expected_join_result = pd.DataFrame({"A": [None, "a999"], "B": [1, None]})
        # pylint: disable=protected-access
        self.assert_frame_equal_with_sort(
            grouped_dataframe._null_safe_join(
                self.spark.createDataFrame(data)
            ).toPandas(),
            expected_join_result,
        )
        # pylint: enable=protected-access

    def test_group_keys_no_rows_one_column(self):
        """Tests that group keys must have no columns it is empty."""

        with self.assertRaisesRegex(
            ValueError, "Group keys cannot have no rows, unless it also has no columns"
        ):
            GroupedDataFrame(
                dataframe=self.spark.createDataFrame(
                    pd.DataFrame([(1, 2)], columns=["A", "Z"])
                ),
                group_keys=self.spark.createDataFrame(
                    [], StructType([StructField("A", IntegerType())])
                ),
            )

    @parameterized.expand(
        [
            (
                pd.DataFrame([("A", 1), ("B", 1), ("B", 2)], columns=["X", "Y"]),
                pd.DataFrame([("A",), ("B",)], columns=["X"]),
                pd.DataFrame([("A", 1), ("B", 2)], columns=["X", "count"]),
            ),
            (
                pd.DataFrame([("A", 1), ("B", 1), ("B", 2)], columns=["X", "Y"]),
                pd.DataFrame([("A", 1), ("B", 1)], columns=["X", "Y"]),
                pd.DataFrame([("A", 1, 1), ("B", 1, 1)], columns=["X", "Y", "count"]),
            ),
        ]
    )
    def test_count_agg(
        self, df: pd.DataFrame, group_keys: pd.DataFrame, expected: pd.DataFrame
    ):
        """Tests that count aggregation works on GroupedDataFrames."""
        actual = (
            GroupedDataFrame(
                dataframe=self.spark.createDataFrame(df),
                group_keys=self.spark.createDataFrame(group_keys),
            )
            .agg(sf.count("*").alias("count"), fill_value=0)
            .toPandas()
        )
        self.assert_frame_equal_with_sort(actual, expected)

    @parameterized.expand(
        [
            (
                pd.DataFrame([("A", 1), ("B", 4), ("B", 5)], columns=["X", "Y"]),
                pd.DataFrame([("A",), ("B",)], columns=["X"]),
                pd.DataFrame([("A", 1), ("B", 9)], columns=["X", "sum(Y)"]),
            ),
            (
                pd.DataFrame([("A", 1), ("B", 4), ("C", 5)], columns=["X", "Y"]),
                pd.DataFrame([("A",), ("B",)], columns=["X"]),
                pd.DataFrame([("A", 1), ("B", 4)], columns=["X", "sum(Y)"]),
            ),
            (
                pd.DataFrame([("A", 1), ("B", 4)], columns=["X", "Y"]),
                pd.DataFrame([("A",), ("B",), ("C",)], columns=["X"]),
                pd.DataFrame([("A", 1), ("B", 4), ("C", 0)], columns=["X", "sum(Y)"]),
            ),
        ]
    )
    def test_sum_agg(
        self, df: pd.DataFrame, group_keys: pd.DataFrame, expected: pd.DataFrame
    ):
        """Tests that agg works as expected."""
        sum_func = sf.sum(sf.col("Y")).alias("sum(Y)")
        self.assert_frame_equal_with_sort(
            GroupedDataFrame(
                dataframe=self.spark.createDataFrame(df),
                group_keys=self.spark.createDataFrame(group_keys),
            )
            .agg(sum_func, fill_value=0)
            .toPandas(),
            expected,
        )

    def test_empty_agg(self):
        """Tests that agg works for empty group keys."""
        sum_func = sf.sum(sf.col("Y")).alias("sum(Y)")
        self.assert_frame_equal_with_sort(
            GroupedDataFrame(
                dataframe=self.spark.createDataFrame(
                    pd.DataFrame([("A", 1), ("B", 4)], columns=["X", "Y"])
                ),
                group_keys=self.spark.createDataFrame([], schema=StructType()),
            )
            .agg(sum_func, fill_value=0)
            .toPandas(),
            pd.DataFrame({"sum(Y)": [5]}),
        )

    def test_agg_fill_value(self):
        """Tests that agg fills correct value for missing keys."""
        expected = pd.DataFrame({"X": ["A", "B"], "sum(Y)": [1, 10]})
        actual = (
            GroupedDataFrame(
                dataframe=self.spark.createDataFrame([("A", 1)], schema=["X", "Y"]),
                group_keys=self.spark.createDataFrame([("A",), ("B",)], schema=["X"]),
            )
            .agg(func=sf.sum(sf.col("Y")).alias("sum(Y)"), fill_value=10)
            .toPandas()
        )
        self.assert_frame_equal_with_sort(expected, actual)

    def test_agg_does_not_override_valid_nulls(self):
        """Tests that agg does not replace nulls associated with existing keys."""
        expected = pd.DataFrame({"X": ["A", "B", "C"], "sum(Y)": [1, None, 0]})
        actual = (
            GroupedDataFrame(
                dataframe=self.spark.createDataFrame(
                    [("A", 1), ("B", None)], schema=["X", "Y"]
                ),
                group_keys=self.spark.createDataFrame(
                    [("A",), ("B",), ("C",)], schema=["X"]
                ),
            )
            .agg(func=sf.sum(sf.col("Y")).alias("sum(Y)"), fill_value=0)
            .toPandas()
        )
        self.assert_frame_equal_with_sort(expected, actual)

    def test_get_groups(self):
        """Tests `get_groups` returns correct groups."""
        actual_groups = GroupedDataFrame(
            dataframe=self.spark.createDataFrame(
                [("A", 1), ("B", 2), ("B", 4), (None, 2), (None, 3)], schema=["X", "Y"]
            ),
            group_keys=self.spark.createDataFrame(
                [("A",), ("B",), ("C",), (None,)], schema=["X"]
            ),
        ).get_groups()
        expected_groups = {
            "A": pd.DataFrame({"Y": [1]}),
            "B": pd.DataFrame({"Y": [2, 4]}),
            "C": pd.DataFrame({"Y": []}),
            None: pd.DataFrame({"Y": [2, 3]}),
        }
        for key, expected_group in expected_groups.items():
            self.assert_frame_equal_with_sort(
                expected_group, actual_groups[Row(A=key)].toPandas()
            )
