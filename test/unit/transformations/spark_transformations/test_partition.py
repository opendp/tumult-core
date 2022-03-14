"""Unit tests for partition transformation.

Tests :mod:`~tmlt.core.transformations.spark_transformations.partition`.
"""

# <placeholder: boilerplate>

import itertools
from typing import List, Tuple, Union

import pandas as pd
from parameterized import parameterized

from tmlt.core.domains.collections import ListDomain
from tmlt.core.domains.spark_domains import (
    SparkColumnsDescriptor,
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
)
from tmlt.core.metrics import RootSumOfSquared, SumOf, SymmetricDifference
from tmlt.core.transformations.spark_transformations.partition import PartitionByKeys
from tmlt.core.utils.testing import (
    TestComponent,
    assert_property_immutability,
    get_all_props,
)


class TestPartitionByKeys(TestComponent):
    """Tests for class PartitionByKeys.

    Tests
    :class:`~tmlt.core.transformations.spark_transformations.partition.
    PartitionByKeys`.
    """

    @parameterized.expand(get_all_props(PartitionByKeys))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        transformation = PartitionByKeys(
            input_domain=SparkDataFrameDomain(self.schema_a),
            input_metric=SymmetricDifference(),
            output_metric=SumOf(SymmetricDifference()),
            keys=["A"],
            list_values=[(1.2,), (2.2,)],
        )
        assert_property_immutability(transformation, prop_name)

    def test_properties(self):
        """PartitionByKeys's properties have the expected values."""
        domain = SparkDataFrameDomain(self.schema_a)
        transformation = PartitionByKeys(
            input_domain=domain,
            input_metric=SymmetricDifference(),
            output_metric=SumOf(SymmetricDifference()),
            keys=["A", "B"],
            list_values=[(1.2, "X"), (2.2, "Y")],
        )
        self.assertEqual(transformation.input_domain, domain)
        self.assertEqual(transformation.input_metric, SymmetricDifference())
        self.assertEqual(transformation.output_domain, ListDomain(domain, length=2))
        self.assertEqual(transformation.output_metric, SumOf(SymmetricDifference()))
        self.assertEqual(transformation.num_partitions, 2)
        self.assertEqual(transformation.keys, ["A", "B"])
        self.assertEqual(transformation.list_values, [(1.2, "X"), (2.2, "Y")])

    @parameterized.expand(
        itertools.chain.from_iterable(
            [
                [
                    (  # Single partition key
                        pd.DataFrame([[1.2, "X"], [2.2, "Y"]], columns=["A", "B"]),
                        {
                            "A": SparkFloatColumnDescriptor(),
                            "B": SparkStringColumnDescriptor(),
                        },
                        ["A"],
                        [(1.2,), (2.2,)],
                        [
                            pd.DataFrame([[1.2, "X"]], columns=["A", "B"]),
                            pd.DataFrame([[2.2, "Y"]], columns=["A", "B"]),
                        ],
                        output_metric,
                    ),
                    (  # Multiple partition key
                        pd.DataFrame(
                            [[1.2, "X", 50], [2.2, "Y", 100]], columns=["A", "B", "C"]
                        ),
                        {
                            "A": SparkFloatColumnDescriptor(),
                            "B": SparkStringColumnDescriptor(),
                            "C": SparkIntegerColumnDescriptor(),
                        },
                        ["A", "C"],
                        [(1.2, 50), (2.2, 100)],
                        [
                            pd.DataFrame([[1.2, "X", 50]], columns=["A", "B", "C"]),
                            pd.DataFrame([[2.2, "Y", 100]], columns=["A", "B", "C"]),
                        ],
                        output_metric,
                    ),
                    (  # Empty partition
                        pd.DataFrame([[1.2, "X"], [2.2, "Y"]], columns=["A", "B"]),
                        {
                            "A": SparkFloatColumnDescriptor(),
                            "B": SparkStringColumnDescriptor(),
                        },
                        ["A"],
                        [(1.2,), (2.2,), (3.3,)],
                        [
                            pd.DataFrame([[1.2, "X"]], columns=["A", "B"]),
                            pd.DataFrame([[2.2, "Y"]], columns=["A", "B"]),
                            pd.DataFrame([], columns=["A", "B"]),
                        ],
                        output_metric,
                    ),
                ]
                for output_metric in [
                    SumOf(SymmetricDifference()),
                    RootSumOfSquared(SymmetricDifference()),
                ]
            ]
        )
    )
    def test_partition_by_keys_works_correctly(
        self,
        df: pd.DataFrame,
        columns_descriptor: SparkColumnsDescriptor,
        keys: List[str],
        list_values: List[Tuple],
        actual_partitions: List[pd.DataFrame],
        output_metric: Union[SumOf, RootSumOfSquared],
    ):
        """Tests that partition by keys works correctly."""
        sdf = self.spark.createDataFrame(df)
        partition_op = PartitionByKeys(
            input_domain=SparkDataFrameDomain(columns_descriptor),
            input_metric=SymmetricDifference(),
            output_metric=output_metric,
            keys=keys,
            list_values=list_values,
        )
        self.assertEqual(partition_op.stability_function(1), 1)
        self.assertTrue(partition_op.stability_relation(1, 1))
        expected_partitions = partition_op(sdf)
        for expected, actual in zip(actual_partitions, expected_partitions):
            self.assert_frame_equal_with_sort(expected, actual.toPandas())

    def test_partition_by_keys_invalid_value(self):
        """Tests that partition by keys raises error when value is invalid for key."""
        with self.assertRaises(ValueError):
            PartitionByKeys(
                input_domain=SparkDataFrameDomain(self.schema_a),
                input_metric=SymmetricDifference(),
                output_metric=SumOf(SymmetricDifference()),
                keys=["A"],
                list_values=[(1.0,), ("InvalidValue",)],
            )

    def test_partition_by_keys_rejects_duplicates(self):
        """Tests that PartitionByKeys raises error with duplicate key values."""
        with self.assertRaisesRegex(
            ValueError, "Partition key values list contains duplicate"
        ):
            PartitionByKeys(
                input_domain=SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(),
                        "B": SparkStringColumnDescriptor(),
                    }
                ),
                input_metric=SymmetricDifference(),
                output_metric=SumOf(SymmetricDifference()),
                keys=["A"],
                list_values=[(1,), (1,)],
            )
