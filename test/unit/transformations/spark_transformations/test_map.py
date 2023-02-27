"""Unit tests for :mod:`~tmlt.core.transformations.spark_transformations.map`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from typing import Callable, List, Optional, Union

import pandas as pd
import sympy as sp
from parameterized import parameterized
from pyspark import Row

from tmlt.core.domains.base import Domain
from tmlt.core.domains.collections import ListDomain
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
    SparkRowDomain,
    SparkStringColumnDescriptor,
)
from tmlt.core.metrics import (
    HammingDistance,
    IfGroupedBy,
    NullMetric,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.spark_transformations.map import (
    FlatMap,
    GroupingFlatMap,
    Map,
    RowToRowsTransformation,
    RowToRowTransformation,
)
from tmlt.core.utils.testing import (
    TestComponent,
    assert_property_immutability,
    get_all_props,
)


class TestRowToRowsTransformer(TestComponent):
    """Tests for class RowToRowsTransformation.

    Tests
    :class:`~tmlt.core.transformations.spark_transformations.map.
    RowToRowsTransformation`.
    """

    @parameterized.expand(get_all_props(RowToRowsTransformation))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        transformer = RowToRowsTransformation(
            SparkRowDomain(self.schema_a),
            ListDomain(SparkRowDomain(self.schema_a)),
            lambda x: [x],
            False,
        )
        assert_property_immutability(transformer, prop_name)

    @parameterized.expand([(True,), (False,)])
    def test_properties(self, augment: bool):
        """RowToRowsTransformation's properties have the expected values."""
        input_domain = SparkRowDomain(self.schema_a)
        transformer = RowToRowsTransformation(
            input_domain, ListDomain(input_domain), lambda x: [x], augment
        )
        self.assertEqual(transformer.input_domain, input_domain)
        self.assertEqual(transformer.input_metric, NullMetric())
        self.assertEqual(transformer.output_domain, ListDomain(input_domain))
        self.assertEqual(transformer.output_metric, NullMetric())
        self.assertEqual(transformer.trusted_f(Row(x=5)), [Row(x=5)])
        self.assertEqual(transformer.augment, augment)

    @parameterized.expand(
        [
            (
                lambda x: [x, x],
                False,
                [Row(A=4.2, B="X"), Row(A=4.2, B="X")],
                ListDomain(
                    SparkRowDomain(
                        {
                            "A": SparkFloatColumnDescriptor(),
                            "B": SparkStringColumnDescriptor(),
                        }
                    )
                ),
            ),
            (
                lambda x: [{"A2": 2 * x["A"]}],
                False,
                [Row(A2=8.4)],
                ListDomain(SparkRowDomain({"A2": SparkFloatColumnDescriptor()})),
            ),
            (
                lambda x: [{"A2": 2 * x["A"]}],
                True,
                [Row(A=4.2, B="X", A2=8.4)],
                ListDomain(
                    SparkRowDomain(
                        {
                            "A": SparkFloatColumnDescriptor(),
                            "B": SparkStringColumnDescriptor(),
                            "A2": SparkStringColumnDescriptor(),
                        }
                    )
                ),
            ),
        ]
    )
    def test_transformer_works_correctly(
        self, f: Callable, augment: bool, expected: List, output_domain: ListDomain
    ):
        """Tests that row transformer works correctly."""
        transformer = RowToRowsTransformation(
            SparkRowDomain(self.schema_a), output_domain, f, augment
        )

        self.assertEqual(transformer.input_domain, SparkRowDomain(self.schema_a))
        self.assertEqual(transformer.output_domain, output_domain)
        row = Row(A=4.2, B="X")
        result_rows = transformer(row)
        self.assertEqual(result_rows, expected)


class TestMap(TestComponent):
    """Tests for :class:`~tmlt.core.transformations.spark_transformations.map.Map`."""

    @parameterized.expand(get_all_props(Map))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        t = Map(
            metric=SymmetricDifference(),
            row_transformer=RowToRowTransformation(
                input_domain=SparkRowDomain(self.schema_a),
                output_domain=SparkRowDomain(self.schema_a),
                trusted_f=lambda row: row,
                augment=False,
            ),
        )
        assert_property_immutability(t, prop_name)

    def test_properties(self):
        """Map's properties have the expected values."""
        times_two = lambda row: {"A": row["A"] * 2, "B": row["B"]}
        row_transformer = RowToRowTransformation(
            input_domain=SparkRowDomain(self.schema_a),
            output_domain=SparkRowDomain(self.schema_a),
            trusted_f=times_two,
            augment=False,
        )
        transformation = Map(
            metric=SymmetricDifference(), row_transformer=row_transformer
        )
        self.assertEqual(
            transformation.input_domain, SparkDataFrameDomain(self.schema_a)
        )
        self.assertEqual(transformation.input_metric, SymmetricDifference())
        self.assertEqual(
            transformation.output_domain, SparkDataFrameDomain(self.schema_a)
        )
        self.assertEqual(transformation.output_metric, SymmetricDifference())
        self.assertEqual(transformation.row_transformer, row_transformer)

    def test_map(self):
        """Tests that map transformation works correctly."""
        times_two = lambda row: {"A": row["A"] * 2, "B": row["B"]}
        times_two_transformation = Map(
            metric=SymmetricDifference(),
            row_transformer=RowToRowTransformation(
                input_domain=SparkRowDomain(self.schema_a),
                output_domain=SparkRowDomain(self.schema_a),
                trusted_f=times_two,
                augment=False,
            ),
        )
        self.assertEqual(times_two_transformation.stability_function(1), 1)
        self.assertTrue(times_two_transformation.stability_relation(1, 1))
        actual_df = times_two_transformation(self.df_a).toPandas()
        expected_df = pd.DataFrame([[2.4, "X"]], columns=["A", "B"])
        self.assert_frame_equal_with_sort(actual_df, expected_df)

    def test_map_empty(self):
        """Tests that map transformation works correctly."""
        times_two = lambda row: {"A": row["A"] * 2, "B": row["B"]}
        times_two_transformation = Map(
            metric=SymmetricDifference(),
            row_transformer=RowToRowTransformation(
                input_domain=SparkRowDomain(self.schema_a),
                output_domain=SparkRowDomain(self.schema_a),
                trusted_f=times_two,
                augment=False,
            ),
        )
        empty_df = pd.DataFrame([], columns=["A", "B"])
        actual_df = times_two_transformation(
            self.spark.createDataFrame(empty_df, self.df_a.schema)
        ).toPandas()
        expected_df = empty_df
        self.assert_frame_equal_with_sort(actual_df, expected_df)

    @parameterized.expand(
        [
            (SymmetricDifference(),),
            (HammingDistance(),),
            (IfGroupedBy("B", SumOf(SymmetricDifference())),),
            (IfGroupedBy("B", RootSumOfSquared(SymmetricDifference())),),
            (IfGroupedBy("B", SymmetricDifference()),),
        ]
    )
    def test_metrics(self, metric: Union[SymmetricDifference, IfGroupedBy]):
        """Tests that Map works correctly with supported metrics."""
        id_map = Map(
            metric=metric,
            row_transformer=RowToRowTransformation(
                input_domain=SparkRowDomain(self.schema_a),
                output_domain=SparkRowDomain(self.schema_a),
                trusted_f=lambda row: row,
                augment=True,
            ),
        )
        self.assertTrue(id_map.input_metric == metric == id_map.output_metric)
        self.assertTrue(id_map.stability_function(1), 1)
        self.assertTrue(id_map.stability_relation(1, 1))
        actual = id_map(self.df_a).toPandas()
        self.assert_frame_equal_with_sort(actual, self.df_a.toPandas())

    @parameterized.expand(
        [
            (
                "C",
                RootSumOfSquared(SymmetricDifference()),
                True,
                "Input metric .* and input domain .* are not compatible",
            ),  # Invalid column
            (
                "B",
                RootSumOfSquared(SymmetricDifference()),
                False,
                "Transformer must be augmenting",
            ),  # Does not augment
            (
                "B",
                SumOf(HammingDistance()),
                True,
                "must be SymmetricDifference",
            ),  # Unsupported inner metric
        ]
    )
    def test_if_grouped_by_metric_invalid_parameters(
        self,
        groupby_column: str,
        inner_metric: Union[SumOf, RootSumOfSquared, SymmetricDifference],
        augment: bool,
        error_msg: str,
    ):
        """Tests that Map raises appropriate error with invalid parameters."""
        with self.assertRaisesRegex(ValueError, error_msg):
            Map(
                metric=IfGroupedBy(groupby_column, inner_metric),
                row_transformer=RowToRowTransformation(
                    input_domain=SparkRowDomain(self.schema_a),
                    output_domain=SparkRowDomain(self.schema_a),
                    trusted_f=lambda row: row,
                    augment=augment,
                ),
            )


class TestFlatMap(TestComponent):
    """Tests for class FlatMap.

    Tests :class:`~tmlt.core.transformations.spark_transformations.map.FlatMap`.
    """

    @parameterized.expand(get_all_props(FlatMap))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        t = FlatMap(
            metric=SymmetricDifference(),
            row_transformer=self.duplicate_transformer,
            max_num_rows=1,
        )
        assert_property_immutability(t, prop_name)

    def test_properties(self):
        """FlatMap's properties have the expected values."""
        duplicate = lambda row: [row, row]
        row_transformer = RowToRowsTransformation(
            input_domain=SparkRowDomain(self.schema_a),
            output_domain=ListDomain(SparkRowDomain(self.schema_a)),
            trusted_f=duplicate,
            augment=False,
        )
        transformation = FlatMap(
            metric=SymmetricDifference(),
            row_transformer=row_transformer,
            max_num_rows=2,
        )
        self.assertEqual(
            transformation.input_domain, SparkDataFrameDomain(self.schema_a)
        )
        self.assertEqual(transformation.input_metric, SymmetricDifference())
        self.assertEqual(
            transformation.output_domain, SparkDataFrameDomain(self.schema_a)
        )
        self.assertEqual(transformation.output_metric, SymmetricDifference())
        self.assertEqual(transformation.row_transformer, row_transformer)
        self.assertEqual(transformation.max_num_rows, 2)

    @parameterized.expand(
        [
            (1, pd.DataFrame([[1.2, "X"]], columns=["A", "B"])),
            (2, pd.DataFrame([[1.2, "X"], [1.2, "X"]], columns=["A", "B"])),
            (3, pd.DataFrame([[1.2, "X"], [1.2, "X"]], columns=["A", "B"])),
            (0, pd.DataFrame([], columns=["A", "B"])),
        ]
    )
    def test_flat_map(self, max_num_rows: int, expected_df: pd.DataFrame):
        """Tests that flat_map transformation works correctly."""
        flat_map_transformation = FlatMap(
            metric=SymmetricDifference(),
            row_transformer=self.duplicate_transformer,
            max_num_rows=max_num_rows,
        )
        self.assertEqual(flat_map_transformation.stability_function(1), max_num_rows)
        self.assertTrue(flat_map_transformation.stability_relation(1, max_num_rows))
        actual_df = flat_map_transformation(self.df_a).toPandas()
        self.assert_frame_equal_with_sort(actual_df, expected_df)

    @parameterized.expand(
        [
            (
                IfGroupedBy("B", SumOf(SymmetricDifference())),
                1,
                pd.DataFrame([[1.2, "X"]], columns=["A", "B"]),
                1,
            ),
            (
                IfGroupedBy("B", SumOf(SymmetricDifference())),
                2,
                pd.DataFrame([[1.2, "X"], [1.2, "X"]], columns=["A", "B"]),
                2,
            ),
            (
                IfGroupedBy("B", SymmetricDifference()),
                0,
                pd.DataFrame([], columns=["A", "B"]),
                1,
            ),
            (
                IfGroupedBy("B", SymmetricDifference()),
                None,
                pd.DataFrame([[1.2, "X"], [1.2, "X"]], columns=["A", "B"]),
                1,
            ),
        ]
    )
    def test_flat_map_augmenting(
        self,
        metric: IfGroupedBy,
        max_num_rows: int,
        expected_df: pd.DataFrame,
        stability: int,
    ):
        """Tests that flat_map transformation works correctly on IfGroupedBy metrics."""

        flat_map_transformation = FlatMap(
            metric=metric,
            row_transformer=self.augmenting_duplicate_transformer,
            max_num_rows=max_num_rows,
        )
        self.assertEqual(flat_map_transformation.stability_function(1), stability)
        self.assertTrue(flat_map_transformation.stability_relation(1, stability))
        actual_df = flat_map_transformation(self.df_a).toPandas()
        self.assert_frame_equal_with_sort(actual_df, expected_df)

    @parameterized.expand(
        [
            (  # Bad input_domain
                ListDomain(SparkRowDomain({})),
                ListDomain(SparkRowDomain({})),
            ),
            (  # Bad output_domain.element_domain
                SparkRowDomain({}),
                ListDomain(SparkDataFrameDomain({})),
            ),
            (SparkRowDomain({}), SparkDataFrameDomain({})),  # Bad output_domain
        ]
    )
    def test_domains(self, input_domain: Domain, output_domain: Domain):
        """Tests that the constructor checks domains correctly.

        RowToRowsTransformation must meet the following conditions
            - input_domain is a *SparkRowDomain*
            - output_domain is a *ListDomain*
            - output_domain.element_domain is a *SparkRowDomain*
        """
        with self.assertRaises((TypeError, ValueError)):
            FlatMap(
                metric=SymmetricDifference(),
                row_transformer=RowToRowsTransformation(
                    input_domain,  # type: ignore
                    output_domain,  # type: ignore
                    lambda x: [x],
                    augment=False,
                ),
                max_num_rows=1,
            )

    @parameterized.expand(
        [
            (SymmetricDifference(), 2, 2),
            (IfGroupedBy("B", SumOf(SymmetricDifference())), 2, 2),
            (IfGroupedBy("B", RootSumOfSquared(SymmetricDifference())), 2, 2),
            (IfGroupedBy("B", SymmetricDifference()), 1, 2),
            (IfGroupedBy("B", SymmetricDifference()), 1, None),
        ]
    )
    def test_metrics(
        self,
        metric: Union[SymmetricDifference, IfGroupedBy],
        d_out: int,
        max_num_rows: Optional[int],
    ):
        """Tests that FlatMap works correctly with supported metrics."""
        split_map = FlatMap(
            metric=metric,
            row_transformer=RowToRowsTransformation(
                input_domain=SparkRowDomain(self.schema_a),
                output_domain=ListDomain(
                    SparkRowDomain(
                        {**self.schema_a, "C": SparkIntegerColumnDescriptor()}
                    )
                ),
                trusted_f=lambda row: [
                    {**row.asDict(), "C": 1},
                    {**row.asDict(), "C": 2},
                ],
                augment=True,
            ),
            max_num_rows=max_num_rows,
        )
        self.assertTrue(split_map.input_metric == metric == split_map.output_metric)
        self.assertTrue(split_map.stability_function(1), d_out)
        self.assertTrue(split_map.stability_relation(1, d_out))
        actual = split_map(self.df_a).toPandas()
        expected = pd.DataFrame({"A": [1.2, 1.2], "B": ["X", "X"], "C": [1, 2]})
        self.assert_frame_equal_with_sort(actual, expected)

    @parameterized.expand(
        [
            (
                "C",
                True,
                "Input metric .* and input domain .* are not compatible",
                SymmetricDifference(),
            ),  # Invalid column
            (
                "B",
                False,
                "Transformer must be augmenting",
                SymmetricDifference(),
            ),  # Does not augment
            (
                "B",
                True,
                "must be SymmetricDifference",
                HammingDistance(),
            ),  # Does not augment
        ]
    )
    def test_if_grouped_by_metric_invalid_parameters(
        self,
        groupby_column: str,
        augment: bool,
        error_msg: str,
        inner_metric: Union[SymmetricDifference, HammingDistance],
    ):
        """Tests that Map raises appropriate error with invalid parameters."""
        with self.assertRaisesRegex(ValueError, error_msg):
            FlatMap(
                metric=IfGroupedBy(groupby_column, SumOf(inner_metric)),
                row_transformer=RowToRowsTransformation(
                    input_domain=SparkRowDomain(self.schema_a),
                    output_domain=ListDomain(SparkRowDomain(self.schema_a)),
                    trusted_f=lambda row: [row],
                    augment=augment,
                ),
                max_num_rows=2,
            )

    @parameterized.expand(
        [
            (SymmetricDifference(),),
            (IfGroupedBy("B", SumOf(SymmetricDifference())),),
            (IfGroupedBy("B", RootSumOfSquared(SymmetricDifference())),),
        ]
    )
    def test_none_stability(self, metric: Union[SymmetricDifference, IfGroupedBy]):
        """Tests that FlatMap raises exaception when None stability not allowed."""
        with self.assertRaisesRegex(
            ValueError, "max_num_rows must be specified if metric is not"
        ):
            FlatMap(
                metric=metric,
                row_transformer=RowToRowsTransformation(
                    input_domain=SparkRowDomain(self.schema_a),
                    output_domain=ListDomain(SparkRowDomain(self.schema_a)),
                    trusted_f=lambda row: [row],
                    augment=True,
                ),
                max_num_rows=None,
            )


class TestGroupingFlatMap(TestComponent):
    """Tests for class GroupingFlatMap.

    Tests
    :class:`~tmlt.core.transformations.spark_transformations.map.GroupingFlatMap`.
    """

    @parameterized.expand(get_all_props(GroupingFlatMap))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        duplicate = lambda row: [{**row.asDict(), "G": 1}, {**row.asDict(), "G": 2}]
        row_transformer = RowToRowsTransformation(
            input_domain=SparkRowDomain(self.schema_a),
            output_domain=ListDomain(
                SparkRowDomain({**self.schema_a, "G": SparkIntegerColumnDescriptor()})
            ),
            trusted_f=duplicate,
            augment=True,
        )
        t = GroupingFlatMap(
            output_metric=RootSumOfSquared(SymmetricDifference()),
            row_transformer=row_transformer,
            max_num_rows=2,
        )
        assert_property_immutability(t, prop_name)

    def test_properties(self):
        """GroupingFlatMap's properties have the expected values."""
        duplicate = lambda row: [{**row.asDict(), "G": 1}, {**row.asDict(), "G": 2}]
        output_schema = {**self.schema_a, "G": SparkIntegerColumnDescriptor()}
        row_transformer = RowToRowsTransformation(
            input_domain=SparkRowDomain(self.schema_a),
            output_domain=ListDomain(SparkRowDomain(output_schema)),
            trusted_f=duplicate,
            augment=True,
        )
        transformation = GroupingFlatMap(
            output_metric=RootSumOfSquared(SymmetricDifference()),
            row_transformer=row_transformer,
            max_num_rows=2,
        )
        self.assertEqual(
            transformation.input_domain, SparkDataFrameDomain(self.schema_a)
        )
        self.assertEqual(transformation.input_metric, SymmetricDifference())
        self.assertEqual(
            transformation.output_domain, SparkDataFrameDomain(output_schema)
        )
        self.assertEqual(
            transformation.output_metric,
            IfGroupedBy("G", RootSumOfSquared(SymmetricDifference())),
        )
        self.assertEqual(transformation.row_transformer, row_transformer)
        self.assertEqual(transformation.max_num_rows, 2)

    @parameterized.expand(
        [
            (
                lambda row: [{**row.asDict(), "NEW": 1}, {**row.asDict(), "NEW": 2}],
                2,
                pd.DataFrame(
                    [("A", 1), ("A", 2), ("B", 1), ("B", 2)], columns=["X", "NEW"]
                ),
            ),
            (  # Duplicate group value
                lambda row: [{**row.asDict(), "NEW": 1}, {**row.asDict(), "NEW": 1}],
                2,
                pd.DataFrame([("A", 1), ("B", 1)], columns=["X", "NEW"]),
            ),
            (  # Truncates correctly
                lambda row: [{**row.asDict(), "NEW": 1}, {**row.asDict(), "NEW": 2}],
                1,
                pd.DataFrame([("A", 1), ("B", 1)], columns=["X", "NEW"]),
            ),
            (  # Does not overwrite
                lambda row: [{"X": "C", "NEW": 1}, {"X": "C", "NEW": 2}],
                2,
                pd.DataFrame(
                    [("A", 1), ("A", 2), ("B", 1), ("B", 2)], columns=["X", "NEW"]
                ),
            ),
        ]
    )
    def test_grouping_flatmap(
        self, trusted_f: Callable, max_num_rows: int, expected_df
    ):
        """Tests that GroupingFlatMap works correctly."""
        df = self.spark.createDataFrame(pd.DataFrame({"X": ["A", "B"]}))
        transformation = GroupingFlatMap(
            output_metric=RootSumOfSquared(SymmetricDifference()),
            row_transformer=RowToRowsTransformation(
                input_domain=SparkRowDomain({"X": SparkStringColumnDescriptor()}),
                output_domain=ListDomain(
                    SparkRowDomain(
                        {
                            "X": SparkStringColumnDescriptor(),
                            "NEW": SparkIntegerColumnDescriptor(),
                        }
                    )
                ),
                trusted_f=trusted_f,
                augment=True,
            ),
            max_num_rows=max_num_rows,
        )
        self.assert_frame_equal_with_sort(transformation(df).toPandas(), expected_df)

    @parameterized.expand([(1,), (2,), (4,), (9,)])
    def test_stability_function_and_relation(self, max_num_rows: int):
        """Tests that stability relation works correctly."""
        transformation = GroupingFlatMap(
            output_metric=RootSumOfSquared(SymmetricDifference()),
            row_transformer=RowToRowsTransformation(
                input_domain=SparkRowDomain(
                    {
                        "A": SparkStringColumnDescriptor(),
                        "B": SparkStringColumnDescriptor(),
                    }
                ),
                output_domain=ListDomain(
                    SparkRowDomain(
                        {
                            "A": SparkStringColumnDescriptor(),
                            "B": SparkStringColumnDescriptor(),
                            "C": SparkStringColumnDescriptor(),
                        }
                    )
                ),
                trusted_f=lambda row: [{**row.asDict(), "C": "1"}],
                augment=True,
            ),
            max_num_rows=max_num_rows,
        )
        self.assertEqual(transformation.stability_function(1), sp.sqrt(max_num_rows))
        self.assertTrue(transformation.stability_relation(1, sp.sqrt(max_num_rows)))

    @parameterized.expand(
        [
            (
                RowToRowsTransformation(
                    input_domain=SparkRowDomain({"A": SparkIntegerColumnDescriptor()}),
                    output_domain=ListDomain(
                        SparkRowDomain(
                            {
                                "A": SparkIntegerColumnDescriptor(),
                                "B": SparkIntegerColumnDescriptor(),
                            }
                        )
                    ),
                    trusted_f=lambda row: [row],
                    augment=False,
                ),
                "Transformer must be augmenting",
            ),
            (
                RowToRowsTransformation(
                    input_domain=SparkRowDomain({"A": SparkIntegerColumnDescriptor()}),
                    output_domain=ListDomain(
                        SparkRowDomain({"A": SparkIntegerColumnDescriptor()})
                    ),
                    trusted_f=lambda row: [row],
                    augment=True,
                ),
                "No grouping column provided",
            ),
            (
                RowToRowsTransformation(
                    input_domain=SparkRowDomain({"A": SparkIntegerColumnDescriptor()}),
                    output_domain=ListDomain(
                        SparkRowDomain(
                            {
                                "A": SparkIntegerColumnDescriptor(),
                                "B": SparkIntegerColumnDescriptor(),
                                "C": SparkIntegerColumnDescriptor(),
                            }
                        )
                    ),
                    trusted_f=lambda row: [row],
                    augment=True,
                ),
                "Only one grouping column allowed",
            ),
        ]
    )
    def test_invalid_transformers_fail(
        self, transformer: RowToRowsTransformation, error_msg: str
    ):
        """Tests that invalid (nonaugmenting, no extra, more than 1 extra)."""
        with self.assertRaisesRegex(ValueError, error_msg):
            GroupingFlatMap(
                output_metric=RootSumOfSquared(SymmetricDifference()),
                row_transformer=transformer,
                max_num_rows=1,
            )
