"""Unit tests for :mod:`~tmlt.core.transformations.spark_transformations.truncation`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022
from typing import Dict, Type, Union

from parameterized import parameterized

from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkStringColumnDescriptor,
)
from tmlt.core.metrics import IfGroupedBy, RootSumOfSquared, SumOf, SymmetricDifference
from tmlt.core.transformations.spark_transformations.truncation import (
    LimitKeysPerGroup,
    LimitRowsPerGroup,
    LimitRowsPerKeyPerGroup,
)
from tmlt.core.utils.testing import (
    PySparkTest,
    assert_property_immutability,
    get_all_props,
)
from tmlt.core.utils.truncation import limit_keys_per_group, truncate_large_groups


class TestLimitRowsPerGroup(PySparkTest):
    """Tests for class LimitRowsPerGroup."""

    def setUp(self):
        """Setup."""
        self.schema = {
            "A": SparkStringColumnDescriptor(),
            "B": SparkStringColumnDescriptor(),
        }
        self.df = self.spark.createDataFrame(
            [("x1", "y1"), ("x2", "y2")], schema=["A", "B"]
        )

    @parameterized.expand(get_all_props(LimitRowsPerGroup))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        truncate = LimitRowsPerGroup(
            input_domain=SparkDataFrameDomain(self.schema),
            output_metric=SymmetricDifference(),
            grouping_column="A",
            threshold=2,
        )
        assert_property_immutability(truncate, prop_name)

    def test_properties(self):
        """LimitRowsPerGroup's properties have the expected values."""
        transformation = LimitRowsPerGroup(
            input_domain=SparkDataFrameDomain(self.schema),
            output_metric=SymmetricDifference(),
            grouping_column="A",
            threshold=2,
        )
        self.assertEqual(transformation.input_domain, SparkDataFrameDomain(self.schema))
        self.assertEqual(
            transformation.input_metric, IfGroupedBy("A", SymmetricDifference())
        )
        self.assertEqual(
            transformation.output_domain, SparkDataFrameDomain(self.schema)
        )
        self.assertEqual(transformation.output_metric, SymmetricDifference())
        self.assertEqual(transformation.grouping_column, "A")
        self.assertEqual(transformation.threshold, 2)

    @parameterized.expand(
        [
            (grouping_column, threshold)
            for grouping_column in ["A", "B"]
            for threshold in [0, 1, 2]
        ]
    )
    def test_correctness(self, grouping_column: str, threshold: int):
        """Tests that LimitRowsPerGroup works correctly."""
        transformation = LimitRowsPerGroup(
            input_domain=SparkDataFrameDomain(self.schema),
            output_metric=SymmetricDifference(),
            grouping_column=grouping_column,
            threshold=threshold,
        )
        actual_df = transformation(self.df).toPandas()
        expected_df = truncate_large_groups(
            self.df, [grouping_column], threshold
        ).toPandas()
        self.assert_frame_equal_with_sort(actual_df, expected_df)

    @parameterized.expand([(3, 1, 3), (2, 2, 4), (0, 1, 0)])
    def test_stability_function(self, threshold: int, d_in: int, expected_d_out: int):
        """Tests that supported metrics have the correct stability functions."""
        transformation = LimitRowsPerGroup(
            input_domain=SparkDataFrameDomain(self.schema),
            output_metric=SymmetricDifference(),
            grouping_column="A",
            threshold=threshold,
        )
        self.assertEqual(transformation.stability_function(d_in), expected_d_out)
        self.assertTrue(transformation.stability_relation(d_in, expected_d_out))

    @parameterized.expand(
        [
            ({"threshold": -1}, ValueError, "Threshold must be nonnegative"),
            (
                {"grouping_column": "invalid"},
                ValueError,
                "Input metric .* and input domain .* are not compatible.",
            ),
            (
                {"output_metric": IfGroupedBy("notA", SymmetricDifference())},
                ValueError,
                r"Output metric must be `SymmetricDifference\(\)` or `IfGroupedBy\(A,"
                r" SymmetricDifference\(\)\)`",
            ),
            (
                {"output_metric": IfGroupedBy("A", SumOf(SymmetricDifference()))},
                ValueError,
                r"Output metric must be `SymmetricDifference\(\)` or `IfGroupedBy\(A,"
                r" SymmetricDifference\(\)\)`",
            ),
        ]
    )
    def test_invalid_parameters(
        self, updated_args: Dict, error_type: Type[Exception], error_msg: str
    ):
        """Tests that appropriate errors are raised for invalid params."""
        args = {
            "input_domain": SparkDataFrameDomain(self.schema),
            "grouping_column": "A",
            "threshold": 1,
            "output_metric": SymmetricDifference(),
        }
        args.update(updated_args)
        with self.assertRaisesRegex(error_type, error_msg):
            LimitRowsPerGroup(**args)  # type: ignore


class TestLimitKeysPerGroup(PySparkTest):
    """Tests for class LimitKeysPerGroup."""

    def setUp(self):
        """Setup."""
        self.schema = {
            "A": SparkStringColumnDescriptor(),
            "B": SparkStringColumnDescriptor(),
        }
        self.df = self.spark.createDataFrame(
            [("x1", "y1"), ("x2", "y2")], schema=["A", "B"]
        )

    @parameterized.expand(get_all_props(LimitKeysPerGroup))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        truncate = LimitKeysPerGroup(
            input_domain=SparkDataFrameDomain(self.schema),
            output_metric=IfGroupedBy(
                "B", SumOf(IfGroupedBy("A", SymmetricDifference()))
            ),
            grouping_column="A",
            key_column="B",
            threshold=2,
        )
        assert_property_immutability(truncate, prop_name)

    def test_properties(self):
        """LimitKeysPerGroup's properties have the expected values."""
        transformation = LimitKeysPerGroup(
            input_domain=SparkDataFrameDomain(self.schema),
            output_metric=IfGroupedBy(
                "B", SumOf(IfGroupedBy("A", SymmetricDifference()))
            ),
            grouping_column="A",
            key_column="B",
            threshold=2,
        )
        self.assertEqual(transformation.input_domain, SparkDataFrameDomain(self.schema))
        self.assertEqual(
            transformation.input_metric, IfGroupedBy("A", SymmetricDifference())
        )
        self.assertEqual(
            transformation.output_domain, SparkDataFrameDomain(self.schema)
        )

        self.assertEqual(
            transformation.output_metric,
            IfGroupedBy("B", SumOf(IfGroupedBy("A", SymmetricDifference()))),
        )
        self.assertEqual(transformation.grouping_column, "A")
        self.assertEqual(transformation.key_column, "B")
        self.assertEqual(transformation.threshold, 2)

    @parameterized.expand(
        [
            (grouping_column, threshold)
            for grouping_column in ["A", "B"]
            for threshold in [0, 1, 2]
        ]
    )
    def test_correctness(self, grouping_column: str, threshold: int):
        """Tests that LimitKeysPerGroup works correctly."""
        key_column = "A" if grouping_column == "B" else "B"
        transformation = LimitKeysPerGroup(
            input_domain=SparkDataFrameDomain(self.schema),
            output_metric=IfGroupedBy(
                key_column, SumOf(IfGroupedBy(grouping_column, SymmetricDifference()))
            ),
            grouping_column=grouping_column,
            key_column=key_column,
            threshold=threshold,
        )
        actual_df = transformation(self.df).toPandas()
        expected_df = limit_keys_per_group(
            self.df, [grouping_column], [key_column], threshold
        ).toPandas()
        self.assert_frame_equal_with_sort(actual_df, expected_df)

    @parameterized.expand(
        [
            (3, 1, 3, IfGroupedBy("B", SumOf(IfGroupedBy("A", SymmetricDifference())))),
            (2, 2, 4, IfGroupedBy("B", SumOf(IfGroupedBy("A", SymmetricDifference())))),
            (0, 1, 0, IfGroupedBy("B", SumOf(IfGroupedBy("A", SymmetricDifference())))),
            (
                9,
                1,
                3,
                IfGroupedBy(
                    "B", RootSumOfSquared(IfGroupedBy("A", SymmetricDifference()))
                ),
            ),
            (
                4,
                2,
                4,
                IfGroupedBy(
                    "B", RootSumOfSquared(IfGroupedBy("A", SymmetricDifference()))
                ),
            ),
            (
                0,
                1,
                0,
                IfGroupedBy(
                    "B", RootSumOfSquared(IfGroupedBy("A", SymmetricDifference()))
                ),
            ),
            (5, 2, 2, IfGroupedBy("A", SymmetricDifference())),
            (0, 4, 4, IfGroupedBy("A", SymmetricDifference())),
        ]
    )
    def test_stability_function(
        self, threshold: int, d_in: int, expected_d_out: int, output_metric: IfGroupedBy
    ):
        """Tests that supported metrics have the correct stability functions."""
        transformation = LimitKeysPerGroup(
            input_domain=SparkDataFrameDomain(self.schema),
            output_metric=output_metric,
            grouping_column="A",
            key_column="B",
            threshold=threshold,
        )
        self.assertEqual(transformation.stability_function(d_in), expected_d_out)
        self.assertTrue(transformation.stability_relation(d_in, expected_d_out))

    @parameterized.expand(
        [
            ({"threshold": -1}, ValueError, "Threshold must be nonnegative"),
            (
                {
                    "grouping_column": "invalid",
                    "output_metric": IfGroupedBy(
                        "B", SumOf(IfGroupedBy("invalid", SymmetricDifference()))
                    ),
                },
                ValueError,
                "Input metric .* and input domain .* are not compatible.",
            ),
            (
                {
                    "key_column": "invalid",
                    "output_metric": IfGroupedBy(
                        "invalid", SumOf(IfGroupedBy("A", SymmetricDifference()))
                    ),
                },
                ValueError,
                "Output metric .* and output domain .* are not compatible.",
            ),
            (
                {"output_metric": IfGroupedBy("B", SymmetricDifference())},
                ValueError,
                r"Output metric must be one of `IfGroupedBy\(B, SumOf\(IfGroupedBy\(A,"
                r" SymmetricDifference\(\)\)\)\)` "
                r"or `IfGroupedBy\(B, RootSumOfSquared\(IfGroupedBy\(A,"
                r" SymmetricDifference\(\)\)\)\)` "
                r"or `IfGroupedBy\(A, SymmetricDifference\(\)\)",
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
                "B", SumOf(IfGroupedBy("A", SymmetricDifference()))
            ),
            "grouping_column": "A",
            "key_column": "B",
            "threshold": 1,
        }
        args.update(updated_args)
        with self.assertRaisesRegex(error_type, error_msg):
            LimitKeysPerGroup(**args)  # type: ignore


class TestLimitRowsPerKeyPerGroup(PySparkTest):
    """Tests for class LimitRowsPerKeyPerGroup."""

    def setUp(self):
        """Setup."""
        self.schema = {
            "A": SparkStringColumnDescriptor(),
            "B": SparkStringColumnDescriptor(),
        }
        self.df = self.spark.createDataFrame(
            [("x1", "y1"), ("x2", "y2")], schema=["A", "B"]
        )

    @parameterized.expand(get_all_props(LimitRowsPerKeyPerGroup))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        truncate = LimitRowsPerKeyPerGroup(
            input_domain=SparkDataFrameDomain(self.schema),
            input_metric=IfGroupedBy(
                "B", SumOf(IfGroupedBy("A", SymmetricDifference()))
            ),
            grouping_column="A",
            key_column="B",
            threshold=2,
        )
        assert_property_immutability(truncate, prop_name)

    def test_properties(self):
        """LimitRowsPerKeyPerGroup's properties have the expected values."""
        transformation = LimitRowsPerKeyPerGroup(
            input_domain=SparkDataFrameDomain(self.schema),
            input_metric=IfGroupedBy(
                "B", SumOf(IfGroupedBy("A", SymmetricDifference()))
            ),
            grouping_column="A",
            key_column="B",
            threshold=2,
        )
        self.assertEqual(transformation.input_domain, SparkDataFrameDomain(self.schema))
        self.assertEqual(
            transformation.input_metric,
            IfGroupedBy("B", SumOf(IfGroupedBy("A", SymmetricDifference()))),
        )
        self.assertEqual(
            transformation.output_domain, SparkDataFrameDomain(self.schema)
        )
        self.assertEqual(transformation.output_metric, SymmetricDifference())
        self.assertEqual(transformation.grouping_column, "A")
        self.assertEqual(transformation.key_column, "B")
        self.assertEqual(transformation.threshold, 2)

    @parameterized.expand(
        [
            (grouping_column, threshold)
            for grouping_column in ["A", "B"]
            for threshold in [0, 1, 2]
        ]
    )
    def test_correctness(self, grouping_column: str, threshold: int):
        """Tests that LimitRowsPerKeyPerGroup works correctly."""
        key_column = "A" if grouping_column == "B" else "B"
        transformation = LimitRowsPerKeyPerGroup(
            input_domain=SparkDataFrameDomain(self.schema),
            input_metric=IfGroupedBy(
                key_column, SumOf(IfGroupedBy(grouping_column, SymmetricDifference()))
            ),
            grouping_column=grouping_column,
            key_column=key_column,
            threshold=threshold,
        )
        actual_df = transformation(self.df).toPandas()
        expected_df = truncate_large_groups(
            self.df, [grouping_column, key_column], threshold
        ).toPandas()
        self.assert_frame_equal_with_sort(actual_df, expected_df)

    @parameterized.expand(
        [
            (
                3,
                1,
                3,
                IfGroupedBy("B", SumOf(IfGroupedBy("A", SymmetricDifference()))),
                SymmetricDifference(),
            ),
            (
                2,
                1,
                2,
                IfGroupedBy(
                    "B", RootSumOfSquared(IfGroupedBy("A", SymmetricDifference()))
                ),
                IfGroupedBy("B", RootSumOfSquared(SymmetricDifference())),
            ),
            (
                2,
                2,
                2,
                IfGroupedBy("A", SymmetricDifference()),
                IfGroupedBy("A", SymmetricDifference()),
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
        transformation = LimitRowsPerKeyPerGroup(
            input_domain=SparkDataFrameDomain(self.schema),
            input_metric=input_metric,
            grouping_column="A",
            key_column="B",
            threshold=threshold,
        )
        self.assertEqual(transformation.stability_function(d_in), expected_d_out)
        self.assertTrue(transformation.stability_relation(d_in, expected_d_out))
        self.assertEqual(transformation.output_metric, expected_output_metric)

    @parameterized.expand(
        [
            ({"threshold": -1}, ValueError, "Threshold must be nonnegative"),
            (
                {"input_metric": IfGroupedBy("B", SymmetricDifference())},
                ValueError,
                r"Input metric must be one of `IfGroupedBy\(B, SumOf\(IfGroupedBy\(A,"
                r" SymmetricDifference\(\)\)\)\)` "
                r"or `IfGroupedBy\(B, RootSumOfSquared\(IfGroupedBy\(A,"
                r" SymmetricDifference\(\)\)\)\)` "
                r"or `IfGroupedBy\(A, SymmetricDifference\(\)\)",
            ),
        ]
    )
    def test_invalid_parameters(
        self, updated_args: Dict, error_type: Type[Exception], error_msg: str
    ):
        """Tests that appropriate errors are raised for invalid params."""
        args = {
            "input_domain": SparkDataFrameDomain(self.schema),
            "grouping_column": "A",
            "key_column": "B",
            "threshold": 1,
            "input_metric": IfGroupedBy(
                "B", SumOf(IfGroupedBy("A", SymmetricDifference()))
            ),
        }
        args.update(updated_args)
        with self.assertRaisesRegex(error_type, error_msg):
            LimitRowsPerKeyPerGroup(**args)  # type: ignore
