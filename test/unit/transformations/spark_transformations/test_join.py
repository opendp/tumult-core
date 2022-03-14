"""Unit tests for :mod:`~tmlt.core.transformations.spark_transformations.join`."""

# <placeholder: boilerplate>

import itertools
from typing import List, Optional, Tuple, Union, cast

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
    DropAllTruncation,
    HashTopKTruncation,
    PrivateJoin,
    PublicJoin,
    Truncation,
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
            (SymmetricDifference(),),
            (IfGroupedBy("B", SumOf(SymmetricDifference())),),
            (IfGroupedBy("B", RootSumOfSquared(SymmetricDifference())),),
        ]
    )
    def test_public_join_correctness(
        self, metric: Union[SymmetricDifference, IfGroupedBy]
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
        self.assertEqual(public_join_transformation.stability_function(1), 2)
        self.assertTrue(public_join_transformation.stability_relation(1, 2))
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
            (["A", "B"], ["B"], "D", "D not in input domain", SymmetricDifference()),
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

    @parameterized.expand(get_all_props(PrivateJoin))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        transformation = PrivateJoin(
            input_domain=DictDomain(
                {"l": self.left_domain, ("r", "i", "g", "h", "t"): self.right_domain}
            ),
            left="l",
            right=("r", "i", "g", "h", "t"),
            left_truncator=HashTopKTruncation(
                domain=self.left_domain, keys=["B"], threshold=1
            ),
            right_truncator=HashTopKTruncation(
                domain=self.right_domain, keys=["B"], threshold=1
            ),
            join_cols=["B"],
        )
        assert_property_immutability(transformation, prop_name)

    @parameterized.expand([(["B"],), (None,)])
    def test_properties(self, join_cols: Optional[List[str]]):
        """Tests that PrivateJoin's properties have expected values."""
        input_domain = DictDomain(
            {"l": self.left_domain, ("r", "i", "g", "h", "t"): self.right_domain}
        )
        left_truncator = HashTopKTruncation(
            domain=self.left_domain, keys=["B"], threshold=1
        )
        right_truncator = HashTopKTruncation(
            domain=self.right_domain, keys=["B"], threshold=2
        )
        transformation = PrivateJoin(
            input_domain=input_domain,
            left="l",
            right=("r", "i", "g", "h", "t"),
            left_truncator=left_truncator,
            right_truncator=right_truncator,
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
        self.assertEqual(transformation.left, "l")
        self.assertEqual(transformation.right, ("r", "i", "g", "h", "t"))
        self.assertEqual(transformation.left_truncator, left_truncator)
        self.assertEqual(transformation.right_truncator, right_truncator)
        self.assertEqual(transformation.join_cols, ["B"])

    @parameterized.expand(
        [
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
        ]
    )
    def test_columns_ordering(
        self,
        left_cols: List[str],
        right_cols: List[str],
        join_cols: List[str],
        expected_ordering: List[str],
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
            left="left",
            right="right",
            left_truncator=HashTopKTruncation(
                domain=left_domain, keys=join_cols, threshold=1
            ),
            right_truncator=HashTopKTruncation(
                domain=right_domain, keys=join_cols, threshold=1
            ),
            join_cols=join_cols,
        )

        answer = private_join({"left": left_df, "right": right_df})
        self.assertTrue(answer in private_join.output_domain)
        self.assertEqual(answer.columns, expected_ordering)

    @parameterized.expand(
        [
            (
                pd.DataFrame([(1, 2), (1, 3), (2, 4)], columns=["A", "B"]),
                pd.DataFrame([(2, 5), (1, 6)], columns=["A", "B"]),
                2,
                HashTopKTruncation,
                ["A"],
                pd.DataFrame(
                    [(1, 2, 6), (1, 3, 6), (2, 4, 5)],
                    columns=["A", "B_left", "B_right"],
                ),
            ),
            (
                pd.DataFrame([(1, 2), (1, 3), (2, 4)], columns=["A", "B"]),
                pd.DataFrame([(2, 5), (1, 6)], columns=["A", "B"]),
                1,
                DropAllTruncation,
                ["A"],
                pd.DataFrame([(2, 4, 5)], columns=["A", "B_left", "B_right"]),
            ),
            (
                pd.DataFrame([(1, 2), (1, 3), (2, 4)], columns=["A", "B"]),
                pd.DataFrame([(2, 5), (2, 2), (1, 6)], columns=["A", "B"]),
                1,
                DropAllTruncation,
                ["A"],
                pd.DataFrame([], columns=["A", "B_left", "B_right"]),
            ),
        ]
    )
    def test_correctness(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        threshold: int,
        TruncationType: type,
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
            left="left",
            right="right",
            left_truncator=TruncationType(
                domain=left_domain, keys=join_cols, threshold=threshold
            ),
            right_truncator=TruncationType(
                domain=right_domain, keys=join_cols, threshold=threshold
            ),
            join_cols=join_cols,
        )
        left_sdf = self.spark.createDataFrame(left)
        right_sdf = self.spark.createDataFrame(right)
        actual = private_join({"left": left_sdf, "right": right_sdf}).toPandas()
        self.assert_frame_equal_with_sort(actual, expected)

    @parameterized.expand(
        [
            (1, 10, 22, HashTopKTruncation),
            (5, 5, 20, HashTopKTruncation),
            (1, 10, 20, DropAllTruncation),
            (5, 5, 50, DropAllTruncation),
        ]
    )
    def test_stability_relation(
        self,
        threshold_left: int,
        threshold_right: int,
        d_out: int,
        TruncationType: type,
    ):
        """Tests that PrivateJoin's stability relation is correct."""
        join_transformation = PrivateJoin(
            input_domain=DictDomain(
                {"left": self.left_domain, "right": self.right_domain}
            ),
            left="left",
            right="right",
            left_truncator=TruncationType(
                domain=self.left_domain, keys=["B"], threshold=threshold_left
            ),
            right_truncator=TruncationType(
                domain=self.right_domain, keys=["B"], threshold=threshold_right
            ),
            join_cols=["B"],
        )
        self.assertTrue(
            join_transformation.stability_relation({"left": 1, "right": 1}, d_out)
        )
        self.assertFalse(
            join_transformation.stability_relation({"left": 1, "right": 1}, d_out - 1)
        )

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
                HashTopKTruncation(
                    domain=SparkDataFrameDomain({"A": SparkIntegerColumnDescriptor()}),
                    keys=["A"],
                    threshold=1,
                ),
                HashTopKTruncation(
                    domain=SparkDataFrameDomain({"A": SparkIntegerColumnDescriptor()}),
                    keys=["A"],
                    threshold=1,
                ),
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
                HashTopKTruncation(
                    domain=SparkDataFrameDomain({"A": SparkIntegerColumnDescriptor()}),
                    keys=["A"],
                    threshold=1,
                ),
                HashTopKTruncation(
                    domain=SparkDataFrameDomain({"A": SparkIntegerColumnDescriptor()}),
                    keys=["A"],
                    threshold=1,
                ),
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
                HashTopKTruncation(
                    domain=SparkDataFrameDomain({"A": SparkIntegerColumnDescriptor()}),
                    keys=["A"],
                    threshold=1,
                ),
                HashTopKTruncation(
                    domain=SparkDataFrameDomain({"A": SparkIntegerColumnDescriptor()}),
                    keys=["A"],
                    threshold=1,
                ),
                ["A"],
                "Left and right keys must be distinct",
            ),
            (  # Left Truncator has mismatching domain
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
                "df2",
                HashTopKTruncation(
                    domain=SparkDataFrameDomain({"A": SparkStringColumnDescriptor()}),
                    keys=["A"],
                    threshold=1,
                ),
                HashTopKTruncation(
                    domain=SparkDataFrameDomain({"A": SparkIntegerColumnDescriptor()}),
                    keys=["A"],
                    threshold=1,
                ),
                ["A"],
                "Input domain for left_truncator does not match left key",
            ),
            (  # Truncation key different from join key
                DictDomain(
                    {
                        "df1": SparkDataFrameDomain(
                            {
                                "A": SparkIntegerColumnDescriptor(),
                                "B": SparkStringColumnDescriptor(),
                            }
                        ),
                        "df2": SparkDataFrameDomain(
                            {"A": SparkIntegerColumnDescriptor()}
                        ),
                    }
                ),
                "df1",
                "df2",
                HashTopKTruncation(
                    domain=SparkDataFrameDomain(
                        {
                            "A": SparkIntegerColumnDescriptor(),
                            "B": SparkStringColumnDescriptor(),
                        }
                    ),
                    keys=["A", "B"],
                    threshold=1,
                ),
                HashTopKTruncation(
                    domain=SparkDataFrameDomain({"A": SparkIntegerColumnDescriptor()}),
                    keys=["A"],
                    threshold=1,
                ),
                ["A"],
                "Truncation keys must match join columns",
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
                HashTopKTruncation(
                    domain=SparkDataFrameDomain({"B": SparkStringColumnDescriptor()}),
                    keys=["B"],
                    threshold=1,
                ),
                HashTopKTruncation(
                    domain=SparkDataFrameDomain({"A": SparkIntegerColumnDescriptor()}),
                    keys=["A"],
                    threshold=1,
                ),
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
                HashTopKTruncation(
                    domain=SparkDataFrameDomain({"A": SparkStringColumnDescriptor()}),
                    keys=["A"],
                    threshold=1,
                ),
                HashTopKTruncation(
                    domain=SparkDataFrameDomain({"A": SparkIntegerColumnDescriptor()}),
                    keys=["A"],
                    threshold=1,
                ),
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
                HashTopKTruncation(
                    domain=SparkDataFrameDomain(
                        {
                            "A": SparkStringColumnDescriptor(),
                            "B": SparkStringColumnDescriptor(),
                            "B_right": SparkStringColumnDescriptor(),
                        }
                    ),
                    keys=["A"],
                    threshold=1,
                ),
                HashTopKTruncation(
                    domain=SparkDataFrameDomain(
                        {
                            "A": SparkStringColumnDescriptor(),
                            "B": SparkStringColumnDescriptor(),
                        }
                    ),
                    keys=["A"],
                    threshold=1,
                ),
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
        left_truncator: Truncation,
        right_truncator: Truncation,
        join_cols: Optional[List[str]],
        error_msg: str,
    ):
        """Tests that PrivateJoin cannot be constructed with invalid arguments."""
        with self.assertRaisesRegex(ValueError, error_msg):
            PrivateJoin(
                input_domain=input_domain,
                left=left,
                right=right,
                left_truncator=left_truncator,
                right_truncator=right_truncator,
                join_cols=join_cols,
            )


class TestHashTopKTruncation(PySparkTest):
    """Tests for class HashTopKTruncation.

    Tests
    :class:`~tmlt.core.transformations.spark_transformations.join.HashTopKTruncation`.
    """

    def setUp(self):
        """Setup for tests."""
        self.domain = SparkDataFrameDomain(
            {"A": SparkIntegerColumnDescriptor(), "B": SparkStringColumnDescriptor()}
        )

    @parameterized.expand(get_all_props(HashTopKTruncation))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        transformation = HashTopKTruncation(
            domain=self.domain, keys=["A"], threshold=10
        )
        assert_property_immutability(transformation, prop_name)

    def test_properties(self):
        """Tests that HashTopKTruncation's properties have expected values."""
        transformation = HashTopKTruncation(domain=self.domain, keys=["B"], threshold=1)
        self.assertTrue(
            transformation.input_domain == self.domain == transformation.output_domain
        )
        self.assertTrue(
            transformation.input_metric
            == SymmetricDifference()
            == transformation.output_metric
        )

        self.assertEqual(transformation.threshold, 1)
        self.assertEqual(transformation.keys, ["B"])
        self.assertEqual(transformation.stability, 2)

    @parameterized.expand(
        [(2, [(1, "x"), (1, "y"), (1, "z"), (1, "w")], 2), (2, [(1, "x")], 1)]
    )
    def test_correctness(self, threshold: int, rows: List[Tuple], expected_count: int):
        """Tests that HashTopKTruncation works correctly."""
        truncator = HashTopKTruncation(
            domain=self.domain, keys=["A"], threshold=threshold
        )
        df = self.spark.createDataFrame(rows, schema=["A", "B"])
        self.assertEqual(truncator(df).count(), expected_count)

    def test_consistency(self):
        """Tests that HashTopKTruncation does not truncate randomly across calls."""
        df = self.spark.createDataFrame([(i,) for i in range(1000)], schema=["A"])
        truncator = HashTopKTruncation(
            domain=SparkDataFrameDomain({"A": SparkIntegerColumnDescriptor()}),
            keys=["A"],
            threshold=5,
        )
        expected_output = truncator(df).toPandas()
        for _ in range(5):
            self.assert_frame_equal_with_sort(truncator(df).toPandas(), expected_output)

    def test_rows_dropped_consistently(self):
        """Tests that HashTopKTruncation drops that same rows for unchanged keys."""
        df1 = [("A", 1), ("B", 2), ("B", 3)]
        df2 = [("A", 0), ("A", 1), ("B", 2), ("B", 3)]
        dom = SparkDataFrameDomain(
            {"W": SparkStringColumnDescriptor(), "X": SparkIntegerColumnDescriptor()}
        )
        truncator = HashTopKTruncation(domain=dom, keys=["W"], threshold=1)
        df1_truncated = truncator(self.spark.createDataFrame(df1, schema=["W", "X"]))
        df2_truncated = truncator(self.spark.createDataFrame(df2, schema=["W", "X"]))
        self.assert_frame_equal_with_sort(
            df1_truncated.filter("W='B'").toPandas(),
            df2_truncated.filter("W='B'").toPandas(),
        )

    def test_hash_truncation_order_agnostic(self):
        """Tests that HashTopKTruncation drops consistently regardless of row order."""
        df_rows = [(1, 2, "A"), (3, 4, "A"), (5, 6, "A"), (7, 8, "B")]
        dom = SparkDataFrameDomain(
            {
                "W": SparkIntegerColumnDescriptor(),
                "X": SparkIntegerColumnDescriptor(),
                "Y": SparkStringColumnDescriptor(),
            }
        )
        truncator = HashTopKTruncation(domain=dom, keys=["Y"], threshold=1)
        truncated_dfs: List[pd.DataFrame] = []
        for permutation in itertools.permutations(df_rows, 4):
            df = self.spark.createDataFrame(list(permutation), schema=["W", "X", "Y"])
            truncated_dfs.append(truncator(df).toPandas())
        for df in truncated_dfs[1:]:
            self.assert_frame_equal_with_sort(first_df=truncated_dfs[0], second_df=df)

    def test_stability_relation(self):
        """Tests that HashTopKTruncation's stability relation is correct."""
        truncator = HashTopKTruncation(domain=self.domain, keys=["A"], threshold=5)
        self.assertTrue(truncator.stability_relation(1, 2))
        self.assertFalse(truncator.stability_relation(1, 1))

    @parameterized.expand(
        [
            (-1, ["B"], "threshold must be a positive integer"),
            (1, [], "No key provided"),
            (1, ["A", "A"], "must be distinct"),
            (1, ["X"], "not in domain: {'X'}"),
        ]
    )
    def test_invalid_arguments_rejected(
        self, threshold: int, keys: List[str], error_msg: str
    ):
        """Tests that a HashTopKTruncation cannot be constructed with invalid arguments.

        In particular, these conditions should be checked:
            - `threshold` is a positive integer.
            - Columns in `keys` are in the given domain.
        """
        with self.assertRaisesRegex(ValueError, error_msg):
            HashTopKTruncation(domain=self.domain, keys=keys, threshold=threshold)


class TestDropAllTruncation(TestComponent):
    """Tests for class DropAllTruncation.

    Tests
    :class:`~tmlt.core.transformations.spark_transformations.join.DropAllTruncation`.
    """

    def setUp(self):
        """Setup for tests."""
        self.domain = SparkDataFrameDomain(
            {"A": SparkIntegerColumnDescriptor(), "B": SparkStringColumnDescriptor()}
        )

    @parameterized.expand(get_all_props(DropAllTruncation))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        transformation = DropAllTruncation(domain=self.domain, keys=["A"], threshold=10)
        assert_property_immutability(transformation, prop_name)

    def test_properties(self):
        """Tests that DropAllTruncation's properties have expected values."""
        transformation = DropAllTruncation(domain=self.domain, keys=["B"], threshold=14)
        self.assertTrue(
            transformation.input_domain == self.domain == transformation.output_domain
        )
        self.assertTrue(
            transformation.input_metric
            == SymmetricDifference()
            == transformation.output_metric
        )

        self.assertEqual(transformation.threshold, 14)
        self.assertEqual(transformation.keys, ["B"])
        self.assertEqual(transformation.stability, 14)

    @parameterized.expand(
        [
            (1, [(1, "A"), (1, "B"), (2, "C")], [(2, "C")]),
            (1, [(1, "A"), (2, "C")], [(1, "A"), (2, "C")]),
            (2, [(1, "A"), (2, "C"), (2, "D"), (2, "E")], [(1, "A")]),
            (1, [(1, "A"), (1, "B"), (2, "C"), (2, "D"), (2, "E")], []),
        ]
    )
    def test_correctness(
        self, threshold: int, input_rows: List[Tuple], expected: List[Tuple]
    ):
        """Tests that DropAllTruncation works correctly."""
        truncator = DropAllTruncation(
            domain=self.domain, keys=["A"], threshold=threshold
        )
        df = self.spark.createDataFrame(input_rows, schema=["A", "B"])
        actual = truncator(df).toPandas()
        expected = pd.DataFrame.from_records(expected, columns=["A", "B"])
        self.assert_frame_equal_with_sort(actual, expected)

    def test_stability_relation(self):
        """Tests that DropAllTruncation's stability relation is correct."""
        truncator = DropAllTruncation(domain=self.domain, keys=["A"], threshold=5)
        self.assertTrue(truncator.stability_relation(1, 5))
        self.assertFalse(truncator.stability_relation(1, 4))

    @parameterized.expand(
        [
            (-1, ["B"], "threshold must be a positive integer"),
            (1, [], "No key provided"),
            (1, ["A", "A"], "must be distinct"),
            (1, ["X"], "not in domain: {'X'}"),
        ]
    )
    def test_invalid_arguments_rejected(
        self, threshold: int, keys: List[str], error_msg: str
    ):
        """Tests that a DropAllTruncation cannot be constructed with invalid arguments.

        In particular, these conditions should be checked:
            - `threshold` is a positive integer.
            - Columns in `keys` are in the given domain.
        """
        with self.assertRaisesRegex(ValueError, error_msg):
            DropAllTruncation(domain=self.domain, keys=keys, threshold=threshold)
