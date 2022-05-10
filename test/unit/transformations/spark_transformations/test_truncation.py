"""Unit tests for :mod:`~tmlt.core.transformations.spark_transformations.truncation`."""

# <placeholder: boilerplate>
from typing import Dict, Type

from parameterized import parameterized

from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.metrics import IfGroupedBy, RootSumOfSquared, SumOf, SymmetricDifference
from tmlt.core.transformations.spark_transformations.truncation import (
    LimitKeysPerGroup,
    LimitRowsPerGroup,
)
from tmlt.core.utils.testing import (
    TestComponent,
    assert_property_immutability,
    get_all_props,
)
from tmlt.core.utils.truncation import limit_keys_per_group, truncate_large_groups


class TestLimitRowsPerGroup(TestComponent):
    """Tests for class LimitRowsPerGroup."""

    @parameterized.expand(get_all_props(LimitRowsPerGroup))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        truncate = LimitRowsPerGroup(
            input_domain=SparkDataFrameDomain(self.schema_a),
            grouping_column="A",
            threshold=2,
        )
        assert_property_immutability(truncate, prop_name)

    def test_properties(self):
        """LimitRowsPerGroup's properties have the expected values."""
        transformation = LimitRowsPerGroup(
            input_domain=SparkDataFrameDomain(self.schema_a),
            grouping_column="A",
            threshold=2,
        )
        self.assertEqual(
            transformation.input_domain, SparkDataFrameDomain(self.schema_a)
        )
        self.assertEqual(
            transformation.input_metric, IfGroupedBy("A", SymmetricDifference())
        )
        self.assertEqual(
            transformation.output_domain, SparkDataFrameDomain(self.schema_a)
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
            input_domain=SparkDataFrameDomain(self.schema_a),
            grouping_column=grouping_column,
            threshold=threshold,
        )
        actual_df = transformation(self.df_a).toPandas()
        expected_df = truncate_large_groups(
            self.df_a, [grouping_column], threshold
        ).toPandas()
        self.assert_frame_equal_with_sort(actual_df, expected_df)

    @parameterized.expand([(3, 1, 3), (2, 2, 4), (0, 1, 0)])
    def test_stability_function(self, threshold: int, d_in: int, expected_d_out: int):
        """Tests that supported metrics have the correct stability functions."""
        transformation = LimitRowsPerGroup(
            input_domain=SparkDataFrameDomain(self.schema_a),
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
        ]
    )
    def test_invalid_parameters(
        self, updated_args: Dict, error_type: Type[Exception], error_msg: str
    ):
        """Tests that appropriate errors are raised for invalid params."""
        args = {
            "input_domain": SparkDataFrameDomain(self.schema_a),
            "grouping_column": "A",
            "threshold": 1,
        }
        args.update(updated_args)
        with self.assertRaisesRegex(error_type, error_msg):
            LimitRowsPerGroup(**args)  # type: ignore


class TestLimitKeysPerGroup(TestComponent):
    """Tests for class LimitKeysPerGroup."""

    @parameterized.expand(get_all_props(LimitKeysPerGroup))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        truncate = LimitKeysPerGroup(
            input_domain=SparkDataFrameDomain(self.schema_a),
            grouping_column="A",
            key_column="B",
            threshold=2,
            use_l2=False,
        )
        assert_property_immutability(truncate, prop_name)

    @parameterized.expand([(True,), (False,)])
    def test_properties(self, use_l2: bool):
        """LimitKeysPerGroup's properties have the expected values."""
        transformation = LimitKeysPerGroup(
            input_domain=SparkDataFrameDomain(self.schema_a),
            grouping_column="A",
            key_column="B",
            threshold=2,
            use_l2=use_l2,
        )
        self.assertEqual(
            transformation.input_domain, SparkDataFrameDomain(self.schema_a)
        )
        self.assertEqual(
            transformation.input_metric, IfGroupedBy("A", SymmetricDifference())
        )
        self.assertEqual(
            transformation.output_domain, SparkDataFrameDomain(self.schema_a)
        )

        self.assertEqual(
            transformation.output_metric,
            IfGroupedBy("B", RootSumOfSquared(IfGroupedBy("A", SymmetricDifference())))
            if use_l2
            else IfGroupedBy("B", SumOf(IfGroupedBy("A", SymmetricDifference()))),
        )
        self.assertEqual(transformation.grouping_column, "A")
        self.assertEqual(transformation.key_column, "B")
        self.assertEqual(transformation.threshold, 2)
        self.assertEqual(transformation.use_l2, use_l2)

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
            input_domain=SparkDataFrameDomain(self.schema_a),
            grouping_column=grouping_column,
            key_column=key_column,
            threshold=threshold,
            use_l2=False,
        )
        actual_df = transformation(self.df_a).toPandas()
        expected_df = limit_keys_per_group(
            self.df_a, [grouping_column], [key_column], threshold
        ).toPandas()
        self.assert_frame_equal_with_sort(actual_df, expected_df)

    @parameterized.expand(
        [
            (3, 1, 3, False),
            (2, 2, 4, False),
            (0, 1, 0, False),
            (9, 1, 3, True),
            (4, 2, 4, True),
            (0, 1, 0, True),
        ]
    )
    def test_stability_function(
        self, threshold: int, d_in: int, expected_d_out: int, use_l2: bool
    ):
        """Tests that supported metrics have the correct stability functions."""
        transformation = LimitKeysPerGroup(
            input_domain=SparkDataFrameDomain(self.schema_a),
            grouping_column="A",
            key_column="B",
            threshold=threshold,
            use_l2=use_l2,
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
                {"key_column": "invalid"},
                ValueError,
                "Output metric .* and output domain .* are not compatible.",
            ),
        ]
    )
    def test_invalid_parameters(
        self, updated_args: Dict, error_type: Type[Exception], error_msg: str
    ):
        """Tests that appropriate errors are raised for invalid params."""
        args = {
            "input_domain": SparkDataFrameDomain(self.schema_a),
            "grouping_column": "A",
            "key_column": "B",
            "threshold": 1,
            "use_l2": False,
        }
        args.update(updated_args)
        with self.assertRaisesRegex(error_type, error_msg):
            LimitKeysPerGroup(**args)  # type: ignore
