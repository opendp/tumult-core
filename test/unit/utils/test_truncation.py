"""Tests for :mod:`~tmlt.core.utils.truncation`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

import itertools
from typing import List, Tuple

import pandas as pd
from parameterized import parameterized

from tmlt.core.utils.testing import PySparkTest
from tmlt.core.utils.truncation import drop_large_groups, truncate_large_groups


class TestTruncateLargeGroups(PySparkTest):
    """Tests for :meth:`~tmlt.core.utils.truncation.truncate_large_groups`."""

    @parameterized.expand(
        [
            (2, [(1, "x"), (1, "y"), (1, "z"), (1, "w")], 2),
            (2, [(1, "x")], 1),
            (0, [(1, "x"), (1, "y"), (1, "z"), (1, "w")], 0),
        ]
    )
    def test_correctness(self, threshold: int, rows: List[Tuple], expected_count: int):
        """Tests that truncate_large_groups works correctly."""
        df = self.spark.createDataFrame(rows, schema=["A", "B"])
        self.assertEqual(
            truncate_large_groups(df, ["A"], threshold).count(), expected_count
        )

    def test_consistency(self):
        """Tests that truncate_large_groups does not truncate randomly across calls."""
        df = self.spark.createDataFrame([(i,) for i in range(1000)], schema=["A"])

        expected_output = truncate_large_groups(df, ["A"], 5).toPandas()
        for _ in range(5):
            self.assert_frame_equal_with_sort(
                truncate_large_groups(df, ["A"], 5).toPandas(), expected_output
            )

    def test_rows_dropped_consistently(self):
        """Tests that truncate_large_groups drops that same rows for unchanged keys."""
        df1 = self.spark.createDataFrame(
            [("A", 1), ("B", 2), ("B", 3)], schema=["W", "X"]
        )
        df2 = self.spark.createDataFrame(
            [("A", 0), ("A", 1), ("B", 2), ("B", 3)], schema=["W", "X"]
        )

        df1_truncated = truncate_large_groups(df1, ["W"], 1)
        df2_truncated = truncate_large_groups(df2, ["W"], 1)
        self.assert_frame_equal_with_sort(
            df1_truncated.filter("W='B'").toPandas(),
            df2_truncated.filter("W='B'").toPandas(),
        )

    def test_hash_truncation_order_agnostic(self):
        """Tests that truncate_large_groups doesn't depend on row order."""
        df_rows = [(1, 2, "A"), (3, 4, "A"), (5, 6, "A"), (7, 8, "B")]

        truncated_dfs: List[pd.DataFrame] = []
        for permutation in itertools.permutations(df_rows, 4):
            df = self.spark.createDataFrame(list(permutation), schema=["W", "X", "Y"])
            truncated_dfs.append(truncate_large_groups(df, ["Y"], 1).toPandas())
        for df in truncated_dfs[1:]:
            self.assert_frame_equal_with_sort(first_df=truncated_dfs[0], second_df=df)


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
    ):
        """Tests that drop_large_groups works correctly."""
        df = self.spark.createDataFrame(input_rows, schema=["A", "B"])
        actual = drop_large_groups(df, ["A"], threshold).toPandas()
        expected = pd.DataFrame.from_records(expected, columns=["A", "B"])
        self.assert_frame_equal_with_sort(actual, expected)
