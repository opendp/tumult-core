"""Unit tests for :mod:`~tmlt.core.transformations.spark_transformations.filter`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022
from typing import Union

import pandas as pd
from parameterized import parameterized

from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.metrics import (
    HammingDistance,
    IfGroupedBy,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.spark_transformations.filter import Filter
from tmlt.core.utils.testing import (
    TestComponent,
    assert_property_immutability,
    get_all_props,
)


class TestFilter(TestComponent):
    """Tests for class Filter.

    Tests :class:`~tmlt.core.transformations.spark_transformations.filter.Filter`.
    """

    @parameterized.expand(get_all_props(Filter))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        transformation = Filter(
            domain=SparkDataFrameDomain(self.schema_a),
            metric=SymmetricDifference(),
            filter_expr="A >= 1.0",
        )
        assert_property_immutability(transformation, prop_name)

    def test_properties(self):
        """Filter's properties have the expected values."""
        domain = SparkDataFrameDomain(self.schema_a)
        transformation = Filter(
            domain=domain, metric=SymmetricDifference(), filter_expr="A >= 1.0"
        )
        self.assertEqual(transformation.input_domain, domain)
        self.assertEqual(transformation.input_metric, SymmetricDifference())
        self.assertEqual(transformation.output_domain, domain)
        self.assertEqual(transformation.output_metric, SymmetricDifference())
        self.assertEqual(transformation.filter_expr, "A >= 1.0")

    @parameterized.expand(
        [
            ("A >= 1.0", pd.DataFrame([[1.2, "X"]], columns=["A", "B"])),
            ("A <= 1.0", pd.DataFrame([[0.9, "Y"]], columns=["A", "B"])),
        ]
    )
    def test_filter_works_correctly(self, filter_expr: str, expected_df: pd.DataFrame):
        """Tests that filter works correctly."""
        df = self.spark.createDataFrame(
            pd.DataFrame([[1.2, "X"], [0.9, "Y"]], columns=["A", "B"])
        )
        geq_1_filter = Filter(
            domain=SparkDataFrameDomain(self.schema_a),
            metric=SymmetricDifference(),
            filter_expr=filter_expr,
        )
        self.assertEqual(geq_1_filter.stability_function(1), 1)
        self.assertTrue(geq_1_filter.stability_relation(1, 1))
        actual_df = geq_1_filter(df).toPandas()
        self.assert_frame_equal_with_sort(actual_df, expected_df)

    @parameterized.expand(["NONEXISTENT>1", "A+1"])
    def test_invalid_filter_exprs_rejected(self, filter_expr: str):
        """Tests that invalid filter expressions are rejected."""
        with self.assertRaises(ValueError):
            Filter(
                domain=SparkDataFrameDomain(self.schema_a),
                metric=SymmetricDifference(),
                filter_expr=filter_expr,
            )

    @parameterized.expand(
        [
            (SymmetricDifference(),),
            (IfGroupedBy("B", SumOf(SymmetricDifference())),),
            (IfGroupedBy("B", RootSumOfSquared(SymmetricDifference())),),
            (IfGroupedBy("B", SymmetricDifference()),),
        ]
    )
    def test_metrics(self, metric: Union[SymmetricDifference, IfGroupedBy]):
        """Tests that Filter works correctly with supported metrics."""
        negative_filter = Filter(
            filter_expr="A < 0",
            domain=SparkDataFrameDomain(self.schema_a),
            metric=metric,
        )
        self.assertEqual(negative_filter.stability_function(1), 1)
        self.assertTrue(
            negative_filter.input_metric == metric == negative_filter.output_metric
        )
        actual = negative_filter(self.df_a).toPandas()
        expected = pd.DataFrame([], columns=["A", "B"])
        self.assert_frame_equal_with_sort(actual, expected)

    @parameterized.expand(
        [
            ("B", HammingDistance(), "must be SymmetricDifference"),
            ("C", SymmetricDifference(), "C not in domain"),
        ]
    )
    def test_if_grouped_by_invalid_parameters(
        self,
        groupby_col: str,
        inner_metric: Union[HammingDistance, SymmetricDifference],
        error_msg: str,
    ):
        """Tests that Filter raises appropriate error with invalid parameters."""
        with self.assertRaisesRegex(ValueError, error_msg):
            Filter(
                domain=SparkDataFrameDomain(self.schema_a),
                metric=IfGroupedBy(groupby_col, SumOf(inner_metric)),
                filter_expr="A < 0",
            )
