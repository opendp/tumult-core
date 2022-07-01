"""Unit tests for partition transformation.

Tests :mod:`~tmlt.core.transformations.spark_transformations.partition`.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

import itertools
import math
import re
from typing import List, Tuple, Type, Union

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
from tmlt.core.metrics import IfGroupedBy, RootSumOfSquared, SumOf, SymmetricDifference
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

    def test_constructor_mutable_arguments(self):
        """Tests that mutable constructor arguments are copied."""
        partition_keys = ["A"]
        key_values = [(0.1,), (1.1,)]
        transformation = PartitionByKeys(
            input_domain=SparkDataFrameDomain(
                {"A": SparkFloatColumnDescriptor(), "B": SparkStringColumnDescriptor()}
            ),
            input_metric=SymmetricDifference(),
            use_l2=False,
            keys=partition_keys,
            list_values=key_values,
        )
        key_values.append((2.2,))
        partition_keys.append("D")
        self.assertListEqual(transformation.keys, ["A"])
        self.assertListEqual(transformation.list_values, [(0.1,), (1.1,)])

    @parameterized.expand(get_all_props(PartitionByKeys))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        transformation = PartitionByKeys(
            input_domain=SparkDataFrameDomain(self.schema_a),
            input_metric=SymmetricDifference(),
            use_l2=False,
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
            use_l2=False,
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
        use_l2 = isinstance(output_metric, RootSumOfSquared)
        partition_op = PartitionByKeys(
            input_domain=SparkDataFrameDomain(columns_descriptor),
            input_metric=SymmetricDifference(),
            use_l2=use_l2,
            keys=keys,
            list_values=list_values,
        )
        self.assertEqual(partition_op.stability_function(1), 1)
        self.assertTrue(partition_op.stability_relation(1, 1))
        expected_partitions = partition_op(sdf)
        for expected, actual in zip(actual_partitions, expected_partitions):
            self.assert_frame_equal_with_sort(expected, actual.toPandas())

    def test_partition_by_special_value_keys(self):
        """PartitionByKeys works when one or more key contains a special value.

        NaNs, nulls and Infs are considered special values.
        """
        key_values = [0.1, float("nan"), float("inf"), float("-inf"), None]
        transformation = PartitionByKeys(
            input_domain=SparkDataFrameDomain(
                {
                    "A": SparkFloatColumnDescriptor(
                        allow_null=True, allow_nan=True, allow_inf=True
                    ),
                    "B": SparkIntegerColumnDescriptor(),
                }
            ),
            input_metric=SymmetricDifference(),
            use_l2=False,
            keys=["A"],
            list_values=[(value,) for value in key_values],
        )
        sdf = self.spark.createDataFrame(
            [(value, 1) for value in key_values], schema=["A", "B"]
        )
        partitions = transformation(sdf)
        for key, partition in zip(key_values, partitions):
            actual_rows = partition.collect()
            self.assertEqual(len(actual_rows), 1)
            assert (
                actual_rows[0].A == key
                or math.isnan(actual_rows[0].A)
                and math.isnan(key)
            )
            assert actual_rows[0].B == 1

    @parameterized.expand(
        [
            (SymmetricDifference(), SumOf(SymmetricDifference()), 2, 2),
            (SymmetricDifference(), RootSumOfSquared(SymmetricDifference()), 2, 2),
            (
                IfGroupedBy("A", SumOf(SymmetricDifference())),
                SumOf(SymmetricDifference()),
                2,
                2,
            ),
            (
                IfGroupedBy("A", RootSumOfSquared(SymmetricDifference())),
                RootSumOfSquared(SymmetricDifference()),
                2,
                2,
            ),
            (
                IfGroupedBy("A", SumOf(IfGroupedBy("B", SymmetricDifference()))),
                SumOf(IfGroupedBy("B", SymmetricDifference())),
                2,
                2,
            ),
        ]
    )
    def test_stability_function(
        self,
        input_metric: Union[IfGroupedBy, SymmetricDifference],
        expected_output_metric: Union[SumOf, RootSumOfSquared],
        d_in: int,
        expected_d_out: int,
    ):
        """Tests that supported metrics have the correct stability functions."""
        use_l2 = isinstance(expected_output_metric, RootSumOfSquared)
        partition_op = PartitionByKeys(
            input_domain=SparkDataFrameDomain(
                {"A": SparkStringColumnDescriptor(), "B": SparkStringColumnDescriptor()}
            ),
            input_metric=input_metric,
            use_l2=use_l2,
            keys=["A"],
            list_values=[("a1",), ("a2",)],
        )
        self.assertEqual(partition_op.input_metric, input_metric)
        self.assertEqual(partition_op.output_metric, expected_output_metric)
        self.assertEqual(partition_op.stability_function(d_in), expected_d_out)
        self.assertTrue(partition_op.stability_relation(d_in, expected_d_out))

    @parameterized.expand(
        [
            (
                IfGroupedBy("A", SumOf(SymmetricDifference())),
                RootSumOfSquared(SymmetricDifference()),
                ValueError,
                "IfGroupedBy inner metric must match use_l2",
            ),
            (
                IfGroupedBy("A", RootSumOfSquared(SymmetricDifference())),
                SumOf(SymmetricDifference()),
                ValueError,
                "IfGroupedBy inner metric must match use_l2",
            ),
            (
                IfGroupedBy("A", SymmetricDifference()),
                SymmetricDifference(),
                ValueError,
                "IfGroupedBy inner metric must match use_l2",
            ),
        ]
    )
    def test_invalid_output_metric(
        self,
        input_metric: Union[IfGroupedBy, SymmetricDifference],
        output_metric: Union[SumOf, RootSumOfSquared],
        error_class: Type[Exception],
        message: str,
    ):
        """Tests that invalid output metrics are handled correctly."""
        use_l2 = isinstance(output_metric, RootSumOfSquared)
        with self.assertRaisesRegex(error_class, re.escape(message)):
            PartitionByKeys(
                input_domain=SparkDataFrameDomain(
                    {
                        "A": SparkStringColumnDescriptor(),
                        "B": SparkStringColumnDescriptor(),
                    }
                ),
                input_metric=input_metric,
                use_l2=use_l2,
                keys=["A"],
                list_values=[("a1",), ("a2",)],
            )

    def test_partition_by_keys_invalid_value(self):
        """Tests that partition by keys raises error when value is invalid for key."""
        with self.assertRaises(ValueError):
            PartitionByKeys(
                input_domain=SparkDataFrameDomain(self.schema_a),
                input_metric=SymmetricDifference(),
                use_l2=False,
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
                use_l2=False,
                keys=["A"],
                list_values=[(1,), (1,)],
            )
