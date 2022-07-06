"""Unit tests for :mod:`~tmlt.core.transformations.spark_transformations.rename`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from typing import Dict, Union

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
from tmlt.core.transformations.spark_transformations.rename import Rename
from tmlt.core.utils.testing import (
    TestComponent,
    assert_property_immutability,
    get_all_props,
)


class TestRename(TestComponent):
    """Tests for class Rename.

    Tests :class:`~tmlt.core.transformations.spark_transformations.rename.Rename`.
    """

    def test_constructor_mutable_arguments(self):
        """Tests that mutable constructor arguments are copied."""
        rename_mapping = {"A": "AA"}
        transformation = Rename(
            input_domain=SparkDataFrameDomain(self.schema_a),
            metric=SymmetricDifference(),
            rename_mapping=rename_mapping,
        )
        rename_mapping["A"] = "BB"
        rename_mapping["C"] = "A"
        self.assertDictEqual(transformation.rename_mapping, {"A": "AA"})

    @parameterized.expand(get_all_props(Rename))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        transformation = Rename(
            input_domain=SparkDataFrameDomain(self.schema_a),
            metric=SymmetricDifference(),
            rename_mapping={"A": "AA", "B": "BB"},
        )
        assert_property_immutability(transformation, prop_name)

    def test_properties(self):
        """Rename's properties have the expected values."""
        rename_mapping = {"A": "AA", "B": "BB"}
        input_domain = SparkDataFrameDomain(self.schema_a)
        transformation = Rename(
            input_domain=input_domain,
            metric=SymmetricDifference(),
            rename_mapping=rename_mapping,
        )
        self.assertEqual(
            transformation.input_domain, SparkDataFrameDomain(self.schema_a)
        )
        self.assertEqual(transformation.input_metric, SymmetricDifference())
        self.assertEqual(
            transformation.output_domain,
            SparkDataFrameDomain(
                {
                    rename_mapping.get(column, column): descriptor
                    for column, descriptor in self.schema_a.items()
                }
            ),
        )
        self.assertEqual(transformation.output_metric, SymmetricDifference())
        self.assertEqual(transformation.rename_mapping, rename_mapping)

    @parameterized.expand(
        [
            (SymmetricDifference(), SymmetricDifference(), {"A": "AA"}),
            (
                IfGroupedBy("B", SumOf(SymmetricDifference())),
                IfGroupedBy("B", SumOf(SymmetricDifference())),
                {"A": "AA"},
            ),
            (
                IfGroupedBy("B", RootSumOfSquared(SymmetricDifference())),
                IfGroupedBy("BB", RootSumOfSquared(SymmetricDifference())),
                {"B": "BB"},
            ),
            (
                IfGroupedBy("B", SymmetricDifference()),
                IfGroupedBy("BB", SymmetricDifference()),
                {"B": "BB"},
            ),
        ]
    )
    def test_rename_works_correctly(
        self,
        metric: Union[SymmetricDifference, IfGroupedBy],
        expected_output_metric: Union[SymmetricDifference, IfGroupedBy],
        rename_mapping: Dict[str, str],
    ):
        """Tests that rename transformation works correctly."""
        rename_transformation = Rename(
            input_domain=SparkDataFrameDomain(self.schema_a),
            metric=metric,
            rename_mapping=rename_mapping,
        )
        self.assertEqual(rename_transformation.input_metric, metric)
        self.assertEqual(rename_transformation.output_metric, expected_output_metric)
        self.assertEqual(rename_transformation.stability_function(1), 1)
        self.assertTrue(rename_transformation.stability_relation(1, 1))
        actual_df = rename_transformation(self.df_a).toPandas()
        expected_df = pd.DataFrame(
            [[1.2, "X"]],
            columns=[rename_mapping.get("A", "A"), rename_mapping.get("B", "B")],
        )
        self.assert_frame_equal_with_sort(actual_df, expected_df)

    @parameterized.expand([({"D": "E"},), ({"A": "B"},)])
    def test_rename_fails_on_bad_columns(self, rename_mapping: Dict[str, str]):
        """Tests that rename transformation fails when column doesn not exist."""
        with self.assertRaises(ValueError):
            Rename(
                input_domain=SparkDataFrameDomain(self.schema_a),
                metric=SymmetricDifference(),
                rename_mapping=rename_mapping,
            )

    @parameterized.expand(
        [
            (
                {"A": "AA"},
                "D",
                SumOf(SymmetricDifference()),
                "Input metric .* and input domain .* are not compatible",
            ),
            (
                {"A": "AA"},
                "D",
                RootSumOfSquared(SymmetricDifference()),
                "Input metric .* and input domain .* are not compatible",
            ),
            ({"A": "AA"}, "A", SumOf(HammingDistance()), "must be SymmetricDifference"),
        ]
    )
    def test_if_grouped_by_metric_invalid_parameters(
        self,
        rename_mapping: Dict[str, str],
        groupby_col: str,
        inner_metric: Union[SumOf, RootSumOfSquared, SymmetricDifference],
        error_msg: str,
    ):
        """Tests that Rename raises appropriate errors with invalid params."""
        with self.assertRaisesRegex(ValueError, error_msg):
            Rename(
                input_domain=SparkDataFrameDomain(self.schema_a),
                metric=IfGroupedBy(groupby_col, inner_metric),
                rename_mapping=rename_mapping,
            )
