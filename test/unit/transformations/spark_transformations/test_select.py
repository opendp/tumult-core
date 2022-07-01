"""Unit tests for :mod:`~tmlt.core.transformations.spark_transformations.select`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from typing import List, Union

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
from tmlt.core.transformations.spark_transformations.select import Select
from tmlt.core.utils.testing import (
    TestComponent,
    assert_property_immutability,
    get_all_props,
)


class TestSelect(TestComponent):
    """Tests for class Select.

    Tests :class:`~tmlt.core.transformations.spark_transformations.select.Select`.
    """

    def test_constructor_mutable_arguments(self):
        """Tests that mutable constructor arguments are copied."""
        columns = ["A", "B"]
        transformation = Select(
            input_domain=SparkDataFrameDomain(self.schema_a),
            metric=SymmetricDifference(),
            columns=columns,
        )
        columns.append("C")
        self.assertListEqual(transformation.columns, ["A", "B"])

    @parameterized.expand(get_all_props(Select))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        select_transformation = Select(
            input_domain=SparkDataFrameDomain(self.schema_a),
            metric=SymmetricDifference(),
            columns=["A", "B"],
        )
        assert_property_immutability(select_transformation, prop_name)

    def test_properties(self):
        """Select's properties have the expected values."""
        columns = ["A"]
        transformation = Select(
            input_domain=SparkDataFrameDomain(self.schema_a),
            metric=SymmetricDifference(),
            columns=columns,
        )
        self.assertEqual(
            transformation.input_domain, SparkDataFrameDomain(self.schema_a)
        )
        self.assertEqual(transformation.input_metric, SymmetricDifference())
        self.assertEqual(
            transformation.output_domain,
            SparkDataFrameDomain({"A": self.schema_a["A"]}),
        )
        self.assertEqual(transformation.output_metric, SymmetricDifference())
        self.assertEqual(transformation.columns, columns)

    @parameterized.expand(
        [
            (SymmetricDifference(),),
            (IfGroupedBy("B", SumOf(SymmetricDifference())),),
            (IfGroupedBy("B", SymmetricDifference()),),
        ]
    )
    def test_select_works_correctly(
        self, metric: Union[SymmetricDifference, IfGroupedBy]
    ):
        """Tests that Select works correctly."""
        select_transformation = Select(
            input_domain=SparkDataFrameDomain(self.schema_a),
            metric=metric,
            columns=["B"],
        )
        self.assertTrue(
            select_transformation.input_metric
            == metric
            == select_transformation.output_metric
        )
        self.assertEqual(select_transformation.stability_function(1), 1)
        self.assertTrue(select_transformation.stability_relation(1, 1))
        actual_df = select_transformation(self.df_a).toPandas()
        expected_df = pd.DataFrame({"B": ["X"]})
        self.assert_frame_equal_with_sort(actual_df, expected_df)

    @parameterized.expand([(["A", "D"],), (["D"],), (["A", "A", "B"],)])
    def test_select_fails_on_bad_columns(self, columns: List[str]):
        """Tests that rename transformation fails when columns are invalid."""
        with self.assertRaises(ValueError):
            Select(
                input_domain=SparkDataFrameDomain(self.schema_a),
                metric=SymmetricDifference(),
                columns=columns,
            )

    @parameterized.expand(
        [
            (
                ["D"],
                "D",
                SumOf(SymmetricDifference()),
                "Non existent columns in select columns : {'D'}",
            ),
            (["A"], "B", SumOf(SymmetricDifference()), "must be selected: B"),
            (["B"], "B", SumOf(HammingDistance()), "must be SymmetricDifference"),
        ]
    )
    def test_if_grouped_by_metric_invalid_parameters(
        self,
        select_columns: List[str],
        groupby_col: str,
        inner_metric: Union[SumOf, RootSumOfSquared, SymmetricDifference],
        error_msg: str,
    ):
        """Tests that Select raises appropriate errors with invalid params."""
        with self.assertRaisesRegex(ValueError, error_msg):
            Select(
                input_domain=SparkDataFrameDomain(self.schema_a),
                metric=IfGroupedBy(groupby_col, inner_metric),
                columns=select_columns,
            )
