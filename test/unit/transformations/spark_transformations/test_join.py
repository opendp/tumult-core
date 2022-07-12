"""Unit tests for :mod:`~tmlt.core.transformations.spark_transformations.join`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from typing import List, Optional, Union, cast

import pandas as pd
from parameterized import parameterized
from pyspark.sql import types as st

from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
)
from tmlt.core.metrics import (
    DictMetric,
    HammingDistance,
    IfGroupedBy,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.spark_transformations.join import (
    PrivateJoin,
    PublicJoin,
    TruncationStrategy,
)
from tmlt.core.utils.testing import (
    PySparkTest,
    TestComponent,
    assert_property_immutability,
    get_all_props,
)


class TestPublicJoin(TestComponent):
    """Tests for class PublicJoin.

    Tests :class:`~tmlt.core.transformations.spark_transformations.join.PublicJoin`.
    """

    def setUp(self):
        """Test setup."""
        self.input_domain = SparkDataFrameDomain(
            {"A": SparkFloatColumnDescriptor(), "B": SparkStringColumnDescriptor()}
        )
        self.public_df = self.spark.createDataFrame(
            [("X", 10.0), ("X", 11.0)],
            schema=st.StructType(
                [
                    st.StructField("B", st.StringType(), nullable=False),
                    st.StructField("C", st.DoubleType(), nullable=False),
                ]
            ),
        )
        self.private_df = self.spark.createDataFrame(
            [(1.2, "X")],
            schema=st.StructType(
                [
                    st.StructField("A", st.DoubleType(), nullable=False),
                    st.StructField("B", st.StringType(), nullable=False),
                ]
            ),
        )

    def test_constructor_mutable_arguments(self):
        """Tests that mutable constructor arguments are copied."""
        join_cols = ["B"]
        transformation = PublicJoin(
            input_domain=self.input_domain,
            metric=SymmetricDifference(),
            public_df=self.public_df,
            join_cols=join_cols,
        )
        join_cols.append("C")
        self.assertListEqual(transformation.join_cols, ["B"])

    @parameterized.expand(get_all_props(PublicJoin))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        transformation = PublicJoin(
            input_domain=self.input_domain,
            metric=SymmetricDifference(),
            public_df=self.public_df,
            join_cols=["B"],
        )
        assert_property_immutability(transformation, prop_name)

    def test_properties(self):
        """PublicJoin's properties have the expected values."""
        transformation = PublicJoin(
            input_domain=self.input_domain,
            metric=SymmetricDifference(),
            public_df=self.public_df,
            join_cols=["B"],
        )
        self.assertEqual(transformation.input_domain, self.input_domain)
        self.assertEqual(transformation.input_metric, SymmetricDifference())
        self.assertEqual(
            transformation.output_domain,
            SparkDataFrameDomain(
                {
                    "B": SparkStringColumnDescriptor(),
                    "A": SparkFloatColumnDescriptor(),
                    "C": SparkFloatColumnDescriptor(allow_nan=True, allow_inf=True),
                }
            ),
        )
        self.assertEqual(transformation.output_metric, SymmetricDifference())
        self.assertEqual(transformation.join_cols, ["B"])
        pd.testing.assert_frame_equal(
            transformation.public_df.toPandas(), self.public_df.toPandas()
        )
        self.assertEqual(transformation.stability, 2)

    @parameterized.expand(
        [
            (SymmetricDifference(), 2),
            (IfGroupedBy("B", SumOf(SymmetricDifference())), 2),
            (IfGroupedBy("B", RootSumOfSquared(SymmetricDifference())), 2),
            (IfGroupedBy("B", SymmetricDifference()), 1),
        ]
    )
    def test_public_join_correctness(
        self, metric: Union[SymmetricDifference, IfGroupedBy], d_out: int
    ):
        """Tests that public join works correctly."""
        public_join_transformation = PublicJoin(
            input_domain=self.input_domain,
            public_df=self.public_df,
            metric=metric,
            join_cols=["B"],
        )
        self.assertTrue(
            public_join_transformation.output_metric
            == metric
            == public_join_transformation.input_metric
        )
        self.assertEqual(public_join_transformation.stability_function(1), d_out)
        self.assertTrue(public_join_transformation.stability_relation(1, d_out))
        joined_df = public_join_transformation(self.private_df)
        self.assertEqual(
            joined_df.schema,
            cast(
                SparkDataFrameDomain, public_join_transformation.output_domain
            ).spark_schema,
        )
        actual = joined_df.toPandas()
        expected = pd.DataFrame(
            [[1.2, "X", 10.0], [1.2, "X", 11.0]], columns=["A", "B", "C"]
        )
        self.assert_frame_equal_with_sort(actual, expected)

    def test_public_join_overlapping_columns(self):
        """Tests that public join works when columns not used in join overlap."""
        public_df = self.spark.createDataFrame(
            pd.DataFrame(
                [["ABC", "X", 10.0], ["DEF", "X", 11.0]], columns=["A", "B", "C"]
            )
        )
        public_join_transformation = PublicJoin(
            input_domain=self.input_domain,
            metric=SymmetricDifference(),
            public_df=public_df,
            join_cols=["B"],
        )
        expected_df = pd.DataFrame(
            [[1.2, "ABC", "X", 10.0], [1.2, "DEF", "X", 11.0]],
            columns=["A_left", "A_right", "B", "C"],
        )
        actual_df = public_join_transformation(self.private_df).toPandas()
        self.assert_frame_equal_with_sort(actual_df, expected_df)

    @parameterized.expand(
        [
            (
                ["B", "C"],
                ["B"],
                "C",
                "C is an overlapping column but not a join key",
                SymmetricDifference(),
            ),
            (
                ["A", "B"],
                ["B"],
                "D",
                "Input metric .* and input domain .* are not compatible",
                SymmetricDifference(),
            ),
            (["A", "B"], ["B"], "A", "must be SymmetricDifference", HammingDistance()),
        ]
    )
    def test_if_grouped_by_metric_invalid_parameters(
        self,
        private_cols: List[str],
        join_cols: List[str],
        groupby_col: str,
        error_msg: str,
        inner_metric: Union[SymmetricDifference, HammingDistance],
    ):
        """Tests that PublicJoin raises appropriate errors with invalid params."""
        with self.assertRaisesRegex(ValueError, error_msg):
            PublicJoin(
                input_domain=SparkDataFrameDomain(
                    {col: SparkStringColumnDescriptor() for col in private_cols}
                ),
                public_df=self.spark.createDataFrame(
                    pd.DataFrame(
                        {"X": ["a1", "a2"], "C": ["z1", "z2"], "B": ["1", "2"]}
                    )
                ),
                metric=IfGroupedBy(groupby_col, SumOf(inner_metric)),
                join_cols=join_cols,
            )

    def test_join_with_mismatching_public_df_and_domain(self):
        """Tests that error is raised if public_df spark schema and domain mismatch."""
        with self.assertRaisesRegex(
            ValueError, "public_df's Spark schema does not match public_df_domain"
        ):
            PublicJoin(
                input_domain=self.input_domain,
                metric=SymmetricDifference(),
                public_df=self.spark.createDataFrame(
                    [("X", 10.0), ("X", 11.0)],
                    schema=st.StructType(
                        [
                            st.StructField("B", st.StringType(), nullable=True),
                            st.StructField("C", st.DoubleType(), nullable=True),
                        ]
                    ),
                ),
                public_df_domain=SparkDataFrameDomain(
                    {
                        "B": SparkStringColumnDescriptor(),
                        "C": SparkFloatColumnDescriptor(),
                    }
                ),
            )

    def test_join_with_public_df_domain(self):
        """Tests that join output domain is correctly inferred from public DF domain."""
        public_join = PublicJoin(
            input_domain=self.input_domain,
            metric=SymmetricDifference(),
            public_df=self.public_df,
            public_df_domain=SparkDataFrameDomain(
                {"B": SparkStringColumnDescriptor(), "C": SparkFloatColumnDescriptor()}
            ),
        )
        actual = public_join.output_domain
        expected = SparkDataFrameDomain(
            {
                "B": SparkStringColumnDescriptor(),
                "A": SparkFloatColumnDescriptor(),
                "C": SparkFloatColumnDescriptor(),
            }
        )
        self.assertEqual(actual, expected)

    def test_join_drops_invalid_rows_from_public_df(self):
        """ "Tests that nans/infs are dropped from public DataFrame when disallowed."""
        public_join = PublicJoin(
            input_domain=self.input_domain,
            metric=SymmetricDifference(),
            public_df=self.spark.createDataFrame(
                [("X", float("nan")), ("X", float("inf")), ("X", 1.1)],
                schema=st.StructType(
                    [
                        st.StructField("B", st.StringType(), nullable=False),
                        st.StructField("C", st.DoubleType(), nullable=False),
                    ]
                ),
            ),
            public_df_domain=SparkDataFrameDomain(
                {"B": SparkStringColumnDescriptor(), "C": SparkFloatColumnDescriptor()}
            ),
        )
        actual = public_join.public_df.toPandas()
        expected = pd.DataFrame({"B": ["X"], "C": [1.1]})
        self.assert_frame_equal_with_sort(actual, expected)

    @parameterized.expand(
        [
            (
                True,
                pd.DataFrame(
                    [["X", 1.2, 1.1], [None, 0.1, 1.2], [None, 0.1, 2.1]],
                    columns=["B", "A", "C"],
                ),
            ),
            (False, pd.DataFrame([["X", 1.2, 1.1]], columns=["B", "A", "C"])),
        ]
    )
    def test_join_null_behavior(self, join_on_nulls: bool, expected: pd.DataFrame):
        """Tests that PublicJoin deals with null values on join columns correctly."""
        public_join = PublicJoin(
            input_domain=SparkDataFrameDomain(
                {
                    "A": SparkFloatColumnDescriptor(),
                    "B": SparkStringColumnDescriptor(allow_null=True),
                }
            ),
            metric=SymmetricDifference(),
            public_df=self.spark.createDataFrame(
                [(None, 2.1), (None, 1.2), ("X", 1.1)], schema=["B", "C"]
            ),
            public_df_domain=SparkDataFrameDomain(
                {
                    "B": SparkStringColumnDescriptor(allow_null=True),
                    "C": SparkFloatColumnDescriptor(allow_null=True),
                }
            ),
            join_on_nulls=join_on_nulls,
        )
        private_df = self.spark.createDataFrame(
            [(1.2, "X"), (0.1, None)], schema=["A", "B"]
        )
        actual = public_join(private_df).toPandas()
        self.assert_frame_equal_with_sort(actual, expected)

    def test_join_on_nulls_stability(self):
        """Tests that PublicJoin computes stability correctly when joining on nulls."""
        public_join = PublicJoin(
            input_domain=self.input_domain,
            metric=SymmetricDifference(),
            public_df=self.spark.createDataFrame(
                [(None, 2.1), (None, 1.2), ("X", 1.1)],
                schema=st.StructType(
                    [
                        st.StructField("B", st.StringType()),
                        st.StructField("C", st.DoubleType(), nullable=False),
                    ]
                ),
            ),
            public_df_domain=SparkDataFrameDomain(
                {
                    "B": SparkStringColumnDescriptor(allow_null=True),
                    "C": SparkFloatColumnDescriptor(),
                }
            ),
            join_on_nulls=True,
        )
        self.assertTrue(public_join.stability == 2)

    def test_join_stability_ignores_nulls(self):
        """Tests that stability is correct when join_on_nulls is False."""
        public_join = PublicJoin(
            input_domain=self.input_domain,
            metric=SymmetricDifference(),
            public_df=self.spark.createDataFrame(
                [(None, 2.1), (None, 1.2), ("X", 1.1)],
                schema=st.StructType(
                    [
                        st.StructField("B", st.StringType()),
                        st.StructField("C", st.DoubleType(), nullable=False),
                    ]
                ),
            ),
            public_df_domain=SparkDataFrameDomain(
                {
                    "B": SparkStringColumnDescriptor(allow_null=True),
                    "C": SparkFloatColumnDescriptor(),
                }
            ),
            join_on_nulls=False,
        )
        self.assertTrue(public_join.stability == 1)

    def test_empty_public_dataframe(self):
        """Tests that PublicJoin works with empty public DataFrame."""
        public_join_transformation = PublicJoin(
            input_domain=self.input_domain,
            metric=SymmetricDifference(),
            public_df=self.spark.createDataFrame([], schema=self.public_df.schema),
            join_cols=["B"],
        )
        actual = public_join_transformation(self.private_df).toPandas()
        expected = pd.DataFrame({"B": [], "A": [], "C": []})
        self.assert_frame_equal_with_sort(actual, expected)


class TestPrivateJoin(PySparkTest):
    """Tests for class PrivateJoin.

    Tests :class:`~tmlt.core.transformations.spark_transformations.join.PrivateJoin`.
    """

    def setUp(self):
        """Setup."""
        self.left_domain = SparkDataFrameDomain(
            {"A": SparkIntegerColumnDescriptor(), "B": SparkStringColumnDescriptor()}
        )
        self.right_domain = SparkDataFrameDomain(
            {"B": SparkStringColumnDescriptor(), "C": SparkStringColumnDescriptor()}
        )

    def test_constructor_mutable_arguments(self):
        """Tests that mutable constructor arguments are copied."""
        join_cols = ["B"]
        transformation = PrivateJoin(
            input_domain=DictDomain(
                {"l": self.left_domain, ("r", "i", "g", "h", "t"): self.right_domain}
            ),
            left_key="l",
            right_key=("r", "i", "g", "h", "t"),
            left_truncation_strategy=TruncationStrategy.TRUNCATE,
            right_truncation_strategy=TruncationStrategy.TRUNCATE,
            left_truncation_threshold=1,
            right_truncation_threshold=1,
            join_cols=join_cols,
        )
        join_cols.append("C")
        self.assertListEqual(transformation.join_cols, ["B"])

    @parameterized.expand(get_all_props(PrivateJoin))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        transformation = PrivateJoin(
            input_domain=DictDomain(
                {"l": self.left_domain, ("r", "i", "g", "h", "t"): self.right_domain}
            ),
            left_key="l",
            right_key=("r", "i", "g", "h", "t"),
            left_truncation_strategy=TruncationStrategy.TRUNCATE,
            right_truncation_strategy=TruncationStrategy.TRUNCATE,
            left_truncation_threshold=1,
            right_truncation_threshold=1,
            join_cols=["B"],
        )
        assert_property_immutability(transformation, prop_name)

    @parameterized.expand([(["B"],), (None,)])
    def test_properties(self, join_cols: Optional[List[str]]):
        """Tests that PrivateJoin's properties have expected values."""
        input_domain = DictDomain(
            {"l": self.left_domain, ("r", "i", "g", "h", "t"): self.right_domain}
        )
        transformation = PrivateJoin(
            input_domain=input_domain,
            left_key="l",
            right_key=("r", "i", "g", "h", "t"),
            left_truncation_strategy=TruncationStrategy.TRUNCATE,
            right_truncation_strategy=TruncationStrategy.TRUNCATE,
            left_truncation_threshold=1,
            right_truncation_threshold=2,
            join_cols=join_cols,
        )

        expected_output_metric = DictMetric(
            {
                "l": SymmetricDifference(),
                ("r", "i", "g", "h", "t"): SymmetricDifference(),
            }
        )
        expected_output_domain = SparkDataFrameDomain(
            {
                "B": SparkStringColumnDescriptor(),
                "A": SparkIntegerColumnDescriptor(),
                "C": SparkStringColumnDescriptor(),
            }
        )

        self.assertEqual(transformation.input_domain, input_domain)
        self.assertEqual(transformation.input_metric, expected_output_metric)
        self.assertEqual(transformation.output_domain, expected_output_domain)
        self.assertEqual(transformation.output_metric, SymmetricDifference())
        self.assertEqual(transformation.left_key, "l")
        self.assertEqual(transformation.right_key, ("r", "i", "g", "h", "t"))
        self.assertEqual(
            transformation.left_truncation_strategy, TruncationStrategy.TRUNCATE
        )
        self.assertEqual(
            transformation.right_truncation_strategy, TruncationStrategy.TRUNCATE
        )
        self.assertEqual(transformation.left_truncation_threshold, 1)
        self.assertEqual(transformation.right_truncation_threshold, 2)
        self.assertEqual(transformation.join_cols, ["B"])

    @parameterized.expand(
        [
            (left_cols, right_cols, join_cols, expected_ordering, join_on_nulls)
            for (left_cols, right_cols, join_cols, expected_ordering) in [
                (["A", "B", "C"], ["B", "D"], ["B"], ["B", "A", "C", "D"]),
                (
                    ["A", "B", "C"],
                    ["B", "D", "C"],
                    ["B"],
                    ["B", "A", "C_left", "D", "C_right"],
                ),
                (
                    ["A", "B", "C"],
                    ["B", "D", "C"],
                    ["B"],
                    ["B", "A", "C_left", "D", "C_right"],
                ),
                (["A", "B", "C"], ["B", "C", "D"], ["C", "B"], ["C", "B", "A", "D"]),
                (["A", "B"], ["B", "C"], ["B"], ["B", "A", "C"]),
            ]
            for join_on_nulls in [True, False]
        ]
    )
    def test_columns_ordering(
        self,
        left_cols: List[str],
        right_cols: List[str],
        join_cols: List[str],
        expected_ordering: List[str],
        join_on_nulls: bool,
    ):
        """Tests that the output columns of join are in expected order.

        This checks:
            - Join columns (in the order given by the user) appear first.
            - Columns of left table (with _left appended as required) appear
             next in the input order. (excluding join columns)
            - Columns of the right table (with _right appended as required) appear
             last in the input order. (excluding join columns)
        """
        left_domain = SparkDataFrameDomain(
            {col: SparkStringColumnDescriptor() for col in left_cols}
        )
        right_domain = SparkDataFrameDomain(
            {col: SparkStringColumnDescriptor() for col in right_cols}
        )

        left_df = self.spark.createDataFrame(
            [("x",) * len(left_cols)], schema=left_cols
        )
        right_df = self.spark.createDataFrame(
            [("x",) * len(right_cols)], schema=right_cols
        )

        private_join = PrivateJoin(
            input_domain=DictDomain({"left": left_domain, "right": right_domain}),
            left_key="left",
            right_key="right",
            left_truncation_strategy=TruncationStrategy.TRUNCATE,
            right_truncation_strategy=TruncationStrategy.TRUNCATE,
            left_truncation_threshold=1,
            right_truncation_threshold=1,
            join_cols=join_cols,
            join_on_nulls=join_on_nulls,
        )

        answer = private_join({"left": left_df, "right": right_df})
        self.assertTrue(answer in private_join.output_domain)
        self.assertEqual(answer.columns, expected_ordering)

    @parameterized.expand(
        [
            (1, 10, 22, TruncationStrategy.TRUNCATE),
            (5, 5, 20, TruncationStrategy.TRUNCATE),
            (1, 10, 20, TruncationStrategy.DROP),
            (5, 5, 50, TruncationStrategy.DROP),
        ]
    )
    def test_stability_relation(
        self,
        threshold_left: int,
        threshold_right: int,
        d_out: int,
        truncation_strategy: TruncationStrategy,
    ):
        """Tests that PrivateJoin's stability relation is correct."""
        join_transformation = PrivateJoin(
            input_domain=DictDomain(
                {"left": self.left_domain, "right": self.right_domain}
            ),
            left_key="left",
            right_key="right",
            left_truncation_strategy=truncation_strategy,
            right_truncation_strategy=truncation_strategy,
            left_truncation_threshold=threshold_left,
            right_truncation_threshold=threshold_right,
            join_cols=["B"],
        )
        self.assertEqual(
            join_transformation.stability_function({"left": 1, "right": 1}), d_out
        )

    @parameterized.expand(
        [
            (
                pd.DataFrame([(1, 2), (1, 3), (2, 4)], columns=["A", "B"]),
                pd.DataFrame([(2, 5), (1, 6)], columns=["A", "B"]),
                TruncationStrategy.TRUNCATE,
                2,
                ["A"],
                pd.DataFrame(
                    [(1, 2, 6), (1, 3, 6), (2, 4, 5)],
                    columns=["A", "B_left", "B_right"],
                ),
            ),
            (
                pd.DataFrame([(1, 2), (1, 3), (2, 4)], columns=["A", "B"]),
                pd.DataFrame([(2, 5), (1, 6)], columns=["A", "B"]),
                TruncationStrategy.DROP,
                1,
                ["A"],
                pd.DataFrame([(2, 4, 5)], columns=["A", "B_left", "B_right"]),
            ),
            (
                pd.DataFrame([(1, 2), (1, 3), (2, 4)], columns=["A", "B"]),
                pd.DataFrame([(2, 5), (2, 2), (1, 6)], columns=["A", "B"]),
                TruncationStrategy.DROP,
                1,
                ["A"],
                pd.DataFrame([], columns=["A", "B_left", "B_right"]),
            ),
        ]
    )
    def test_correctness(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        truncation_strategy: TruncationStrategy,
        threshold: int,
        join_cols: List[str],
        expected: pd.DataFrame,
    ):
        """Tests that join is computed correctly."""
        left_domain = SparkDataFrameDomain(
            {col: SparkIntegerColumnDescriptor() for col in left.columns}
        )
        right_domain = SparkDataFrameDomain(
            {col: SparkIntegerColumnDescriptor() for col in right.columns}
        )
        private_join = PrivateJoin(
            input_domain=DictDomain({"left": left_domain, "right": right_domain}),
            left_key="left",
            right_key="right",
            left_truncation_strategy=truncation_strategy,
            right_truncation_strategy=truncation_strategy,
            left_truncation_threshold=threshold,
            right_truncation_threshold=threshold,
            join_cols=join_cols,
        )
        left_sdf = self.spark.createDataFrame(left)
        right_sdf = self.spark.createDataFrame(right)
        actual = private_join({"left": left_sdf, "right": right_sdf}).toPandas()
        self.assert_frame_equal_with_sort(actual, expected)

    @parameterized.expand(
        [
            (  # Domain contains > 2 keys
                DictDomain(
                    {
                        "df1": SparkDataFrameDomain(
                            {"A": SparkIntegerColumnDescriptor()}
                        ),
                        "df2": SparkDataFrameDomain(
                            {"A": SparkIntegerColumnDescriptor()}
                        ),
                        "df3": SparkDataFrameDomain(
                            {"A": SparkIntegerColumnDescriptor()}
                        ),
                    }
                ),
                "df1",
                "df2",
                ["A"],
                "must be a DictDomain with 2 keys",
            ),
            (  # Invalid key
                DictDomain(
                    {
                        "df1": SparkDataFrameDomain(
                            {"A": SparkIntegerColumnDescriptor()}
                        ),
                        "df2": SparkDataFrameDomain(
                            {"A": SparkIntegerColumnDescriptor()}
                        ),
                    }
                ),
                "df3",
                "df1",
                ["A"],
                "Key 'df3' not in input domain",
            ),
            (  # Identical left and right
                DictDomain(
                    {
                        "df1": SparkDataFrameDomain(
                            {"A": SparkIntegerColumnDescriptor()}
                        ),
                        "df2": SparkDataFrameDomain(
                            {"A": SparkIntegerColumnDescriptor()}
                        ),
                    }
                ),
                "df1",
                "df1",
                ["A"],
                "Left and right keys must be distinct",
            ),
            (  # No common columns
                DictDomain(
                    {
                        "df1": SparkDataFrameDomain(
                            {"B": SparkStringColumnDescriptor()}
                        ),
                        "df2": SparkDataFrameDomain(
                            {"A": SparkIntegerColumnDescriptor()}
                        ),
                    }
                ),
                "df1",
                "df2",
                None,
                "No common columns",
            ),
            (  # Mismatching column types
                DictDomain(
                    {
                        "df1": SparkDataFrameDomain(
                            {"A": SparkStringColumnDescriptor()}
                        ),
                        "df2": SparkDataFrameDomain(
                            {"A": SparkIntegerColumnDescriptor()}
                        ),
                    }
                ),
                "df1",
                "df2",
                ["A"],
                "mismatching types on join column A",
            ),
            (  # _right column already exists
                DictDomain(
                    {
                        "df1": SparkDataFrameDomain(
                            {
                                "A": SparkStringColumnDescriptor(),
                                "B": SparkStringColumnDescriptor(),
                                "B_right": SparkStringColumnDescriptor(),
                            }
                        ),
                        "df2": SparkDataFrameDomain(
                            {
                                "A": SparkStringColumnDescriptor(),
                                "B": SparkStringColumnDescriptor(),
                            }
                        ),
                    }
                ),
                "df1",
                "df2",
                ["A"],
                "Join would rename overlapping column 'B' to an existing column name",
            ),
        ]
    )
    def test_invalid_arguments_rejected(
        self,
        input_domain: DictDomain,
        left: str,
        right: str,
        join_cols: Optional[List[str]],
        error_msg: str,
    ):
        """Tests that PrivateJoin cannot be constructed with invalid arguments."""
        with self.assertRaisesRegex(ValueError, error_msg):
            PrivateJoin(
                input_domain=input_domain,
                left_key=left,
                right_key=right,
                left_truncation_strategy=TruncationStrategy.TRUNCATE,
                right_truncation_strategy=TruncationStrategy.TRUNCATE,
                left_truncation_threshold=1,
                right_truncation_threshold=1,
                join_cols=join_cols,
            )

    def test_join_drops_invalid_rows(self):
        """Tests nans are dropped from the right-hand DataFrame when disallowed.

        Infinity should still be allowed.
        """
        left = pd.DataFrame([(1, 2), (1, 3), (2, 4)], columns=["A", "B"])
        right_sdf = self.spark.createDataFrame(
            [(1, 2, float("nan")), (1, 3, float("inf")), (2, 4, float(0))],
            schema=st.StructType(
                [
                    st.StructField("A", st.IntegerType(), nullable=False),
                    st.StructField("B", st.IntegerType(), nullable=False),
                    st.StructField("C", st.DoubleType(), nullable=False),
                ]
            ),
        )
        left_domain = SparkDataFrameDomain(
            {col: SparkIntegerColumnDescriptor() for col in left.columns}
        )
        right_domain = SparkDataFrameDomain(
            {
                "A": SparkIntegerColumnDescriptor(),
                "B": SparkIntegerColumnDescriptor(),
                "C": SparkFloatColumnDescriptor(allow_inf=True),
            }
        )
        join_cols = ["A", "B"]
        private_join = PrivateJoin(
            input_domain=DictDomain({"left": left_domain, "right": right_domain}),
            left_key="left",
            right_key="right",
            left_truncation_strategy=TruncationStrategy.TRUNCATE,
            right_truncation_strategy=TruncationStrategy.TRUNCATE,
            left_truncation_threshold=1,
            right_truncation_threshold=1,
            join_cols=join_cols,
            join_on_nulls=False,
        )
        left_sdf = self.spark.createDataFrame(left)
        actual = private_join({"left": left_sdf, "right": right_sdf}).toPandas()
        expected = pd.DataFrame(
            [[2, 4, float(0)], [1, 3, float("inf")]], columns=["A", "B", "C"]
        )
        self.assert_frame_equal_with_sort(actual, expected)

    def test_join_without_nulls_changes_domain(self):
        """Test that when join_on_null=False, output domain does not allow null."""
        left_domain = SparkDataFrameDomain(
            {
                "A": SparkFloatColumnDescriptor(),
                "B": SparkStringColumnDescriptor(allow_null=True),
            }
        )
        right_domain = SparkDataFrameDomain(
            {
                "B": SparkStringColumnDescriptor(allow_null=True),
                "C": SparkFloatColumnDescriptor(allow_null=True),
            }
        )
        private_join = PrivateJoin(
            input_domain=DictDomain({"left": left_domain, "right": right_domain}),
            left_key="left",
            right_key="right",
            left_truncation_strategy=TruncationStrategy.TRUNCATE,
            right_truncation_strategy=TruncationStrategy.TRUNCATE,
            left_truncation_threshold=10,
            right_truncation_threshold=10,
            join_on_nulls=False,
        )
        expected_output_domain = SparkDataFrameDomain(
            {
                "A": SparkFloatColumnDescriptor(),
                "B": SparkStringColumnDescriptor(allow_null=False),
                "C": SparkFloatColumnDescriptor(allow_null=True),
            }
        )
        actual = private_join.output_domain
        self.assertEqual(expected_output_domain["A"], actual["A"])
        self.assertEqual(expected_output_domain["B"], actual["B"])
        self.assertEqual(expected_output_domain["C"], actual["C"])

    @parameterized.expand(
        [
            (
                True,
                pd.DataFrame(
                    [["X", 1.2, 1.1], [None, 0.1, 1.2], [None, 0.1, 2.1]],
                    columns=["B", "A", "C"],
                ),
            ),
            (False, pd.DataFrame([["X", 1.2, 1.1]], columns=["B", "A", "C"])),
        ]
    )
    def test_join_on_nulls_behavior(self, join_on_nulls: bool, expected: pd.DataFrame):
        """Test that PrivateJoin deals with null values on join columns correctly."""
        left = self.spark.createDataFrame([(1.2, "X"), (0.1, None)], schema=["A", "B"])
        right = self.spark.createDataFrame(
            [(None, 2.1), (None, 1.2), ("X", 1.1)], schema=["B", "C"]
        )
        left_domain = SparkDataFrameDomain(
            {
                "A": SparkFloatColumnDescriptor(),
                "B": SparkStringColumnDescriptor(allow_null=True),
            }
        )
        right_domain = SparkDataFrameDomain(
            {
                "B": SparkStringColumnDescriptor(allow_null=True),
                "C": SparkFloatColumnDescriptor(allow_null=True),
            }
        )
        private_join = PrivateJoin(
            input_domain=DictDomain({"left": left_domain, "right": right_domain}),
            left_key="left",
            right_key="right",
            left_truncation_strategy=TruncationStrategy.DROP,
            right_truncation_strategy=TruncationStrategy.DROP,
            left_truncation_threshold=10,
            right_truncation_threshold=10,
            join_on_nulls=join_on_nulls,
        )
        actual = private_join({"left": left, "right": right}).toPandas()
        self.assert_frame_equal_with_sort(actual, expected)
