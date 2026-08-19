"""Unit tests for :mod:`~tmlt.core.transformations.spark_transformations.truncation`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026
import re
from typing import Dict, List, Sequence, Type, Union

import pyspark.sql.functions as sf
from parameterized import parameterized

from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkStringColumnDescriptor,
)
from tmlt.core.metrics import IfGroupedBy, RootSumOfSquared, SumOf, SymmetricDifference
from tmlt.core.transformations.spark_transformations.truncation import (
    LimitGroupsPerID,
    LimitRowsPerGroupPerID,
    LimitRowsPerID,
)
from tmlt.core.utils.testing import (
    PySparkTest,
    assert_dataframe_equal,
    assert_property_immutability,
    get_all_props,
)
from tmlt.core.utils.truncation import limit_groups_per_id, truncate_large_groups


class TestLimitRowsPerID(PySparkTest):
    """Tests for class LimitRowsPerID."""

    def setUp(self):
        """Setup."""
        self.schema = {
            "A": SparkStringColumnDescriptor(),
            "B": SparkStringColumnDescriptor(),
        }
        self.df = self.spark.createDataFrame(
            [("x1", "y1"), ("x2", "y2")], schema=["A", "B"]
        )

    @parameterized.expand(get_all_props(LimitRowsPerID))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        truncate = LimitRowsPerID(
            input_domain=SparkDataFrameDomain(self.schema),
            output_metric=SymmetricDifference(),
            id_columns=["A"],
            threshold=2,
        )
        assert_property_immutability(truncate, prop_name)

    def test_properties(self):
        """LimitRowsPerID's properties have the expected values."""
        transformation = LimitRowsPerID(
            input_domain=SparkDataFrameDomain(self.schema),
            output_metric=SymmetricDifference(),
            id_columns=["A", "B"],
            threshold=2,
        )
        self.assertEqual(transformation.input_domain, SparkDataFrameDomain(self.schema))
        self.assertEqual(
            transformation.input_metric, IfGroupedBy(["A", "B"], SymmetricDifference())
        )
        self.assertEqual(
            transformation.output_domain, SparkDataFrameDomain(self.schema)
        )
        self.assertEqual(transformation.output_metric, SymmetricDifference())
        self.assertEqual(transformation.id_columns, frozenset({"A", "B"}))
        self.assertEqual(transformation.threshold, 2)

    @parameterized.expand(
        [
            (grouping_column, threshold, output_metric)
            for grouping_column in [["A"], ["B"], ["A", "B"], ("A",)]
            for threshold in [0, 1, 2]
            for output_metric in [
                SymmetricDifference(),
                IfGroupedBy(grouping_column, SymmetricDifference()),
            ]
        ]
    )
    def test_correctness(
        self,
        id_columns: Sequence[str],
        threshold: int,
        output_metric: Union[SymmetricDifference, IfGroupedBy],
    ):
        """Tests that LimitRowsPerID works correctly."""
        transformation = LimitRowsPerID(
            input_domain=SparkDataFrameDomain(self.schema),
            output_metric=output_metric,
            id_columns=id_columns,
            threshold=threshold,
        )
        actual_df = transformation(self.df)
        expected_df = truncate_large_groups(self.df, id_columns, threshold)
        assert_dataframe_equal(actual_df, expected_df)
        rows_per_group = actual_df.groupby(list(id_columns)).count()
        self.assertTrue(
            all([row["count"] <= threshold for row in rows_per_group.collect()])
        )

    @parameterized.expand(
        [
            (3, 1, SymmetricDifference(), 3),
            (2, 2, SymmetricDifference(), 4),
            (0, 1, SymmetricDifference(), 0),
            (3, 3, IfGroupedBy(["A"], SymmetricDifference()), 3),
        ]
    )
    def test_stability_function(
        self,
        threshold: int,
        d_in: int,
        output_metric: Union[SymmetricDifference, IfGroupedBy],
        expected_d_out: int,
    ):
        """Tests that supported metrics have the correct stability functions."""
        transformation = LimitRowsPerID(
            input_domain=SparkDataFrameDomain(self.schema),
            output_metric=output_metric,
            id_columns=["A"],
            threshold=threshold,
        )
        self.assertEqual(transformation.stability_function(d_in), expected_d_out)
        self.assertTrue(transformation.stability_relation(d_in, expected_d_out))

    @parameterized.expand(
        [
            ({"threshold": -1}, ValueError, "Threshold must be nonnegative"),
            (
                {"id_columns": ["invalid"]},
                ValueError,
                "Input metric .* and input domain .* are not compatible.",
            ),
            (
                {"output_metric": IfGroupedBy(["notA"], SymmetricDifference())},
                ValueError,
                re.escape(
                    "Output metric must be `SymmetricDifference()` or "
                    "`IfGroupedBy(['A'], SymmetricDifference())`"
                ),
            ),
            (
                {"output_metric": IfGroupedBy(["A"], SumOf(SymmetricDifference()))},
                ValueError,
                re.escape(
                    "Output metric must be `SymmetricDifference()` or "
                    "`IfGroupedBy(['A'], SymmetricDifference())`"
                ),
            ),
        ]
    )
    def test_invalid_parameters(
        self, updated_args: Dict, error_type: Type[Exception], error_msg: str
    ):
        """Tests that appropriate errors are raised for invalid params."""
        args = {
            "input_domain": SparkDataFrameDomain(self.schema),
            "id_columns": ["A"],
            "threshold": 1,
            "output_metric": SymmetricDifference(),
        }
        args.update(updated_args)
        with self.assertRaisesRegex(error_type, error_msg):
            LimitRowsPerID(**args)  # type: ignore

    def test_format(self):
        """Tests that format returns the expected string."""
        transformation = LimitRowsPerID(
            input_domain=SparkDataFrameDomain(
                {"A": SparkStringColumnDescriptor(), "B": SparkStringColumnDescriptor()}
            ),
            output_metric=SymmetricDifference(),
            id_columns=["A"],
            threshold=2,
        )
        assert transformation.format() == "LimitRowsPerID id_columns={'A'} threshold=2"


class TestLimitGroupsPerID(PySparkTest):
    """Tests for class LimitGroupsPerID."""

    def setUp(self):
        """Setup."""
        self.schema = {
            "A": SparkStringColumnDescriptor(),
            "B": SparkStringColumnDescriptor(),
            "C": SparkStringColumnDescriptor(),
        }
        self.df = self.spark.createDataFrame(
            [("x1", "y1", "z1"), ("x2", "y2", "z2")], schema=["A", "B", "C"]
        )

    @parameterized.expand(get_all_props(LimitGroupsPerID))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        truncate = LimitGroupsPerID(
            input_domain=SparkDataFrameDomain(self.schema),
            output_metric=IfGroupedBy(
                ["B"], SumOf(IfGroupedBy(["A"], SymmetricDifference()))
            ),
            id_columns=["A"],
            grouping_column="B",
            threshold=2,
        )
        assert_property_immutability(truncate, prop_name)

    def test_properties(self):
        """LimitGroupsPerID's properties have the expected values."""
        transformation = LimitGroupsPerID(
            input_domain=SparkDataFrameDomain(self.schema),
            output_metric=IfGroupedBy(
                ["C"], SumOf(IfGroupedBy(["A", "B"], SymmetricDifference()))
            ),
            id_columns=["A", "B"],
            grouping_column="C",
            threshold=2,
        )
        self.assertEqual(transformation.input_domain, SparkDataFrameDomain(self.schema))
        self.assertEqual(
            transformation.input_metric, IfGroupedBy(["A", "B"], SymmetricDifference())
        )
        self.assertEqual(
            transformation.output_domain, SparkDataFrameDomain(self.schema)
        )

        self.assertEqual(
            transformation.output_metric,
            IfGroupedBy(["C"], SumOf(IfGroupedBy(["A", "B"], SymmetricDifference()))),
        )
        self.assertEqual(transformation.id_columns, frozenset({"A", "B"}))
        self.assertEqual(transformation.grouping_column, "C")
        self.assertEqual(transformation.threshold, 2)

    @parameterized.expand(
        [
            (id_columns, threshold)
            for id_columns in [["A"], ["B"], ["A", "B"]]
            for threshold in [0, 1, 2]
        ]
    )
    def test_correctness(self, id_columns: List[str], threshold: int):
        """Tests that LimitGroupsPerID works correctly."""
        df = self.spark.createDataFrame(
            [
                ("x1", "y1", "z1"),
                ("x1", "y2", "z2"),
                ("x1", "y3", "z3"),
                ("x2", "y1", "z4"),
                ("x2", "y2", "z5"),
                ("x2", "y3", "z6"),
                ("x3", "y1", "z7"),
                ("x3", "y2", "z8"),
                ("x3", "y3", "z9"),
            ],
            schema=["A", "B", "C"],
        )
        transformation = LimitGroupsPerID(
            input_domain=SparkDataFrameDomain(self.schema),
            output_metric=IfGroupedBy(
                ["C"],
                SumOf(IfGroupedBy(id_columns, SymmetricDifference())),
            ),
            id_columns=id_columns,
            grouping_column="C",
            threshold=threshold,
        )
        actual_df = transformation(df)
        expected_df = limit_groups_per_id(df, id_columns, ["C"], threshold)
        assert_dataframe_equal(actual_df, expected_df)
        groups_by_id = actual_df.groupby(id_columns).agg(
            sf.count_distinct("C").alias("count")
        )
        self.assertTrue(
            all([row["count"] <= threshold for row in keys_per_group.collect()])
        )

    @parameterized.expand(
        [
            (
                3,
                1,
                3,
                IfGroupedBy(["B"], SumOf(IfGroupedBy(["A"], SymmetricDifference()))),
            ),
            (
                2,
                2,
                4,
                IfGroupedBy(["B"], SumOf(IfGroupedBy(["A"], SymmetricDifference()))),
            ),
            (
                0,
                1,
                0,
                IfGroupedBy(["B"], SumOf(IfGroupedBy(["A"], SymmetricDifference()))),
            ),
            (
                9,
                1,
                3,
                IfGroupedBy(
                    ["B"], RootSumOfSquared(IfGroupedBy(["A"], SymmetricDifference()))
                ),
            ),
            (
                4,
                2,
                4,
                IfGroupedBy(
                    ["B"], RootSumOfSquared(IfGroupedBy(["A"], SymmetricDifference()))
                ),
            ),
            (
                0,
                1,
                0,
                IfGroupedBy(
                    ["B"], RootSumOfSquared(IfGroupedBy(["A"], SymmetricDifference()))
                ),
            ),
            (5, 2, 2, IfGroupedBy(["A"], SymmetricDifference())),
            (0, 4, 4, IfGroupedBy(["A"], SymmetricDifference())),
        ]
    )
    def test_stability_function(
        self, threshold: int, d_in: int, expected_d_out: int, output_metric: IfGroupedBy
    ):
        """Tests that supported metrics have the correct stability functions."""
        transformation = LimitGroupsPerID(
            input_domain=SparkDataFrameDomain(self.schema),
            output_metric=output_metric,
            id_columns=["A"],
            grouping_column="B",
            threshold=threshold,
        )
        self.assertEqual(transformation.stability_function(d_in), expected_d_out)
        self.assertTrue(transformation.stability_relation(d_in, expected_d_out))

    @parameterized.expand(
        [
            ({"threshold": -1}, ValueError, "Threshold must be nonnegative"),
            (
                {
                    "id_columns": ["invalid"],
                    "output_metric": IfGroupedBy(
                        ["B"], SumOf(IfGroupedBy(["invalid"], SymmetricDifference()))
                    ),
                },
                ValueError,
                "Input metric .* and input domain .* are not compatible.",
            ),
            (
                {
                    "grouping_column": "invalid",
                    "output_metric": IfGroupedBy(
                        ["invalid"], SumOf(IfGroupedBy(["A"], SymmetricDifference()))
                    ),
                },
                ValueError,
                "Output metric .* and output domain .* are not compatible.",
            ),
            (
                {"output_metric": IfGroupedBy(["B"], SymmetricDifference())},
                ValueError,
                r"Output metric must be one of `IfGroupedBy\(\['B'\], "
                r"SumOf\(IfGroupedBy\(\['A'\], SymmetricDifference\(\)\)\)\)` "
                r"or `IfGroupedBy\(\['B'\], RootSumOfSquared\(IfGroupedBy\(\['A'\],"
                r" SymmetricDifference\(\)\)\)\)` "
                r"or `IfGroupedBy\(\['A'\], SymmetricDifference\(\)\)",
            ),
            (
                {"id_columns": ["A", "B"], "grouping_column": "B"},
                ValueError,
                "ID column cannot be a grouping column",
            ),
        ]
    )
    def test_invalid_parameters(
        self, updated_args: Dict, error_type: Type[Exception], error_msg: str
    ):
        """Tests that appropriate errors are raised for invalid params."""
        args = {
            "input_domain": SparkDataFrameDomain(self.schema),
            "output_metric": IfGroupedBy(
                ["B"], SumOf(IfGroupedBy(["A"], SymmetricDifference()))
            ),
            "id_columns": ["A"],
            "grouping_column": "B",
            "threshold": 1,
        }
        args.update(updated_args)
        with self.assertRaisesRegex(error_type, error_msg):
            LimitGroupsPerID(**args)  # type: ignore

    def test_format(self):
        """Tests that format returns the expected string."""
        transformation = LimitGroupsPerID(
            input_domain=SparkDataFrameDomain(
                {
                    "A": SparkStringColumnDescriptor(),
                    "B": SparkStringColumnDescriptor(),
                    "C": SparkStringColumnDescriptor(),
                }
            ),
            output_metric=IfGroupedBy(
                ["B"], SumOf(IfGroupedBy(["A"], SymmetricDifference()))
            ),
            id_columns=["A"],
            grouping_column="B",
            threshold=2,
        )
        assert (
            transformation.format()
            == "LimitGroupsPerID id_columns={'A'} grouping_column='B' threshold=2"
        )


class TestLimitRowsPerGroupPerID(PySparkTest):
    """Tests for class LimitRowsPerGroupPerID."""

    def setUp(self):
        """Setup."""
        self.schema = {
            "A": SparkStringColumnDescriptor(),
            "B": SparkStringColumnDescriptor(),
            "C": SparkStringColumnDescriptor(),
        }
        self.df = self.spark.createDataFrame(
            [("x1", "y1", "z1"), ("x2", "y2", "z2")], schema=["A", "B", "C"]
        )

    @parameterized.expand(get_all_props(LimitRowsPerGroupPerID))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        truncate = LimitRowsPerGroupPerID(
            input_domain=SparkDataFrameDomain(self.schema),
            input_metric=IfGroupedBy(
                ["B"], SumOf(IfGroupedBy(["A"], SymmetricDifference()))
            ),
            id_columns=["A"],
            grouping_column="B",
            threshold=2,
        )
        assert_property_immutability(truncate, prop_name)

    def test_properties(self):
        """LimitRowsPerGroupPerID's properties have the expected values."""
        transformation = LimitRowsPerGroupPerID(
            input_domain=SparkDataFrameDomain(self.schema),
            input_metric=IfGroupedBy(
                ["C"], SumOf(IfGroupedBy(["A", "B"], SymmetricDifference()))
            ),
            id_columns=["A", "B"],
            grouping_column="C",
            threshold=2,
        )
        self.assertEqual(transformation.input_domain, SparkDataFrameDomain(self.schema))
        self.assertEqual(
            transformation.input_metric,
            IfGroupedBy(["C"], SumOf(IfGroupedBy(["A", "B"], SymmetricDifference()))),
        )
        self.assertEqual(
            transformation.output_domain, SparkDataFrameDomain(self.schema)
        )
        self.assertEqual(transformation.output_metric, SymmetricDifference())
        self.assertEqual(transformation.id_columns, frozenset({"A", "B"}))
        self.assertEqual(transformation.grouping_column, "C")
        self.assertEqual(transformation.threshold, 2)

    @parameterized.expand(
        [
            (grouping_column, threshold)
            for grouping_column in [["A"], ["B"], ["A", "B"]]
            for threshold in [0, 1, 2]
        ]
    )
    def test_correctness(self, id_columns: List[str], threshold: int):
        """Tests that LimitRowsPerGroupPerID works correctly."""
        df = self.spark.createDataFrame(
            [
                ("x1", "y1", "z1", "d1"),
                ("x1", "y1", "z1", "d2"),
                ("x1", "y1", "z1", "d3"),
                ("x1", "y2", "z1", "d4"),
                ("x1", "y2", "z1", "d5"),
                ("x1", "y2", "z1", "d6"),
                ("x1", "y1", "z3", "d7"),
                ("x1", "y1", "z3", "d8"),
                ("x1", "y1", "z3", "d9"),
            ],
            schema=["A", "B", "C", "D"],
        )
        transformation = LimitRowsPerGroupPerID(
            input_domain=SparkDataFrameDomain(self.schema),
            input_metric=IfGroupedBy(
                ["C"],
                SumOf(IfGroupedBy(id_columns, SymmetricDifference())),
            ),
            id_columns=id_columns,
            grouping_column="C",
            threshold=threshold,
        )
        actual_df = transformation(df)
        expected_df = truncate_large_groups(df, [*id_columns, "C"], threshold)
        assert_dataframe_equal(actual_df, expected_df)
        rows_per_group_per_id = actual_df.groupby([*id_columns, "C"]).count()
        assert all(
            [row["count"] <= threshold for row in rows_per_group_per_id.collect()]
        )

    @parameterized.expand(
        [
            (
                3,
                1,
                3,
                IfGroupedBy(["B"], SumOf(IfGroupedBy(["A"], SymmetricDifference()))),
                SymmetricDifference(),
            ),
            (
                2,
                1,
                2,
                IfGroupedBy(
                    ["B"], RootSumOfSquared(IfGroupedBy(["A"], SymmetricDifference()))
                ),
                IfGroupedBy(["B"], RootSumOfSquared(SymmetricDifference())),
            ),
            (
                2,
                2,
                2,
                IfGroupedBy(["A"], SymmetricDifference()),
                IfGroupedBy(["A"], SymmetricDifference()),
            ),
        ]
    )
    def test_stability_function(
        self,
        threshold: int,
        d_in: int,
        expected_d_out: int,
        input_metric: IfGroupedBy,
        expected_output_metric: Union[SymmetricDifference, IfGroupedBy],
    ):
        """Tests that supported metrics have the correct stability functions."""
        transformation = LimitRowsPerGroupPerID(
            input_domain=SparkDataFrameDomain(self.schema),
            input_metric=input_metric,
            id_columns=["A"],
            grouping_column="B",
            threshold=threshold,
        )
        self.assertEqual(transformation.stability_function(d_in), expected_d_out)
        self.assertTrue(transformation.stability_relation(d_in, expected_d_out))
        self.assertEqual(transformation.output_metric, expected_output_metric)

    @parameterized.expand(
        [
            ({"threshold": -1}, ValueError, "Threshold must be nonnegative"),
            (
                {"input_metric": IfGroupedBy(["B"], SymmetricDifference())},
                ValueError,
                r"Input metric must be one of `IfGroupedBy\(\['B'\], "
                r"SumOf\(IfGroupedBy\(\['A'\], SymmetricDifference\(\)\)\)\)` "
                r"or `IfGroupedBy\(\['B'\], RootSumOfSquared\(IfGroupedBy\(\['A'\],"
                r" SymmetricDifference\(\)\)\)\)` "
                r"or `IfGroupedBy\(\['A'\], SymmetricDifference\(\)\)",
            ),
            (
                {"id_columns": ["A", "B"], "grouping_column": "B"},
                ValueError,
                "ID column cannot be a grouping column",
            ),
        ]
    )
    def test_invalid_parameters(
        self, updated_args: Dict, error_type: Type[Exception], error_msg: str
    ):
        """Tests that appropriate errors are raised for invalid params."""
        args = {
            "input_domain": SparkDataFrameDomain(self.schema),
            "id_columns": ["A"],
            "grouping_column": "B",
            "threshold": 1,
            "input_metric": IfGroupedBy(
                ["B"], SumOf(IfGroupedBy(["A"], SymmetricDifference()))
            ),
        }
        args.update(updated_args)
        with self.assertRaisesRegex(error_type, error_msg):
            LimitRowsPerGroupPerID(**args)  # type: ignore

    def test_format(self):
        """Tests that format returns the expected string."""
        transformation = LimitRowsPerGroupPerID(
            input_domain=SparkDataFrameDomain(
                {
                    "A": SparkStringColumnDescriptor(),
                    "B": SparkStringColumnDescriptor(),
                    "C": SparkStringColumnDescriptor(),
                }
            ),
            input_metric=IfGroupedBy(
                ["B"], SumOf(IfGroupedBy(["A"], SymmetricDifference()))
            ),
            id_columns=["A"],
            grouping_column="B",
            threshold=2,
        )
        assert transformation.format() == (
            "LimitRowsPerGroupPerID id_columns={'A'} grouping_column='B' threshold=2"
        )
