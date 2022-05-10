"""Unit tests for :mod:`~tmlt.core.transformations.spark_transformations.groupby`."""

# <placeholder: boilerplate>
import re
from typing import List, Tuple, Union

import pandas as pd
from parameterized import parameterized
from pyspark.sql.types import LongType, StringType, StructField, StructType

from tmlt.core.domains.base import OutOfDomainError
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkGroupedDataFrameDomain,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
)
from tmlt.core.metrics import (
    HammingDistance,
    IfGroupedBy,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.spark_transformations.groupby import (
    GroupBy,
    create_groupby_from_column_domains,
    create_groupby_from_list_of_keys,
)
from tmlt.core.utils.grouped_dataframe import GroupedDataFrame
from tmlt.core.utils.testing import (
    PySparkTest,
    assert_property_immutability,
    get_all_props,
)


class TestGroupBy(PySparkTest):
    """Tests for GroupBy transformation on Spark DataFrames."""

    def setUp(self):
        """Setup."""
        self.domain = SparkDataFrameDomain(
            {"A": SparkIntegerColumnDescriptor(), "B": SparkStringColumnDescriptor()}
        )
        self.df = self.spark.createDataFrame(
            [(1, "X"), (1, "Y"), (2, "Z")], schema=["A", "B"]
        )
        self.group_keys = self.spark.createDataFrame([(1,), (2,), (3,)], schema=["A"])

    @parameterized.expand(get_all_props(GroupBy))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        transformation = GroupBy(
            input_domain=self.domain,
            input_metric=SymmetricDifference(),
            use_l2=False,
            group_keys=self.group_keys,
        )
        assert_property_immutability(transformation, prop_name)

    @parameterized.expand([(False,), (True,)])
    def test_properties(self, use_l2: bool):
        """Tests that GroupBy's properties have expected values."""
        groupby = GroupBy(
            input_domain=self.domain,
            input_metric=SymmetricDifference(),
            use_l2=use_l2,
            group_keys=self.group_keys,
        )
        self.assertEqual(groupby.input_domain, self.domain)
        output_domain = groupby.output_domain
        self.assertTrue(isinstance(output_domain, SparkGroupedDataFrameDomain))
        assert isinstance(output_domain, SparkGroupedDataFrameDomain)
        self.assertEqual(output_domain.schema, self.domain.schema)
        self.assert_frame_equal_with_sort(
            output_domain.group_keys.toPandas(), self.group_keys.toPandas()
        )
        self.assertEqual(groupby.input_metric, SymmetricDifference())
        self.assertEqual(
            groupby.output_metric,
            RootSumOfSquared(SymmetricDifference())
            if use_l2
            else SumOf(SymmetricDifference()),
        )
        self.assertEqual(groupby.use_l2, use_l2)
        self.assertEqual(groupby.groupby_columns, ["A"])

    @parameterized.expand(
        [
            (
                IfGroupedBy("A", SumOf(SymmetricDifference())),
                [("1",), ("2",)],
                StructType([StructField("A", StringType())]),
                "Column must be LongType, instead it is StringType",
                OutOfDomainError,
            ),
            (
                IfGroupedBy("A", RootSumOfSquared(SymmetricDifference())),
                [(1,), (2,), (3,)],
                StructType([StructField("A", LongType())]),
                "Input metric does not have the expected inner metric. Maybe "
                "IfGroupedBy(column='A', inner_metric=SumOf("
                "inner_metric=SymmetricDifference()))?",
            ),
            (
                SymmetricDifference(),
                [],
                StructType([StructField("A", LongType())]),
                "Group keys cannot have no rows, unless it also has no columns",
            ),
        ]
    )
    def test_invalid_constructor_arguments(
        self,
        input_metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
        group_keys_list: List[Tuple],
        group_keys_schema: StructType,
        error_msg: str,
        error_type: type = ValueError,
    ):
        """Tests that GroupBy constructor raises appropriate error."""
        with self.assertRaisesRegex(error_type, re.escape(error_msg)):
            GroupBy(
                input_domain=self.domain,
                input_metric=input_metric,
                use_l2=False,
                group_keys=self.spark.createDataFrame(
                    group_keys_list, schema=group_keys_schema
                ),
            )

    def test_stability_function(self):
        """Tests that stability function is correct."""
        groupby_transformation = GroupBy(
            input_domain=self.domain,
            input_metric=SymmetricDifference(),
            use_l2=True,
            group_keys=self.group_keys,
        )
        self.assertTrue(groupby_transformation.stability_function(1), 1)
        groupby_hamming_to_symmetric = GroupBy(
            input_domain=self.domain,
            input_metric=HammingDistance(),
            use_l2=False,
            group_keys=self.group_keys,
        )
        self.assertTrue(groupby_hamming_to_symmetric.stability_function(1) == 2)

    def test_correctness(self):
        """Tests that GroupBy transformation works correctly."""
        # pylint: disable=no-member
        groupby_transformation = GroupBy(
            input_domain=self.domain,
            input_metric=SymmetricDifference(),
            use_l2=True,
            group_keys=self.group_keys,
        )
        grouped_dataframe = groupby_transformation(self.df)
        self.assertTrue(isinstance(grouped_dataframe, GroupedDataFrame))
        self.assert_frame_equal_with_sort(
            grouped_dataframe._dataframe.toPandas(),  # pylint: disable=protected-access
            self.df.toPandas(),
        )
        self.assert_frame_equal_with_sort(
            grouped_dataframe.group_keys.toPandas(), self.group_keys.toPandas()
        )

    def test_total(self):
        """Tests that GroupBy transformation works correctly with no group keys."""
        # pylint: disable=no-member
        groupby_transformation = GroupBy(
            input_domain=self.domain,
            input_metric=SymmetricDifference(),
            use_l2=True,
            group_keys=self.spark.createDataFrame([], schema=StructType()),
        )
        grouped_dataframe = groupby_transformation(self.df)
        self.assertTrue(isinstance(grouped_dataframe, GroupedDataFrame))
        self.assert_frame_equal_with_sort(
            grouped_dataframe._dataframe.toPandas(),  # pylint: disable=protected-access
            self.df.toPandas(),
        )
        self.assert_frame_equal_with_sort(
            grouped_dataframe.group_keys.toPandas(), pd.DataFrame()
        )


class TestDerivedTransformations(PySparkTest):
    """Unit tests for derived groupby transformations."""

    def setUp(self):
        """Test setup."""
        self.input_domain = SparkDataFrameDomain(
            {
                "A": SparkStringColumnDescriptor(),
                "B": SparkStringColumnDescriptor(),
                "C": SparkIntegerColumnDescriptor(),
            }
        )

    @parameterized.expand(
        [
            (
                SymmetricDifference(),
                False,
                {"A": ["x1", "x2"], "B": ["y1", "y2"]},
                pd.DataFrame(
                    {"A": ["x1", "x2", "x1", "x2"], "B": ["y1", "y1", "y2", "y2"]}
                ),
            ),
            (
                HammingDistance(),
                True,
                {"A": ["x1", "x2"], "B": ["y1"]},
                pd.DataFrame({"A": ["x1", "x2"], "B": ["y1", "y1"]}),
            ),
            (
                HammingDistance(),
                False,
                {"A": ["x1", "x2"]},
                pd.DataFrame({"A": ["x1", "x2"]}),
            ),
            (HammingDistance(), True, {}, pd.DataFrame()),
        ]
    )
    def test_create_groupby_from_column_domains(
        self, input_metric, use_l2, column_domains, expected_group_keys
    ):
        """create_groupby_from_column_domains constructs expected transformation."""
        groupby_transformation = create_groupby_from_column_domains(
            input_domain=self.input_domain,
            input_metric=input_metric,
            use_l2=use_l2,
            column_domains=column_domains,
        )
        self.assertEqual(groupby_transformation.input_metric, input_metric)
        self.assertEqual(groupby_transformation.use_l2, use_l2)
        # If there are no columns, toPandas removes all rows, so this check is also
        # needed.
        self.assertEqual(
            groupby_transformation.group_keys.count(), len(expected_group_keys)
        )
        self.assert_frame_equal_with_sort(
            groupby_transformation.group_keys.toPandas(), expected_group_keys
        )

    @parameterized.expand(
        [
            (SymmetricDifference(), False, ["A", "B"], [("x1", "y2"), ("x2", "y1")]),
            (SymmetricDifference(), False, ["A"], [("x1",), ("x2",)]),
            (HammingDistance(), True, [], []),
        ]
    )
    def test_create_groupby_from_list_of_keys(
        self,
        input_metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
        use_l2: bool,
        groupby_columns: List[str],
        group_keys: List[Tuple[Union[str, int], ...]],
    ):
        """create_groupby_from_list_of_keys constructs expected transformation."""
        groupby = create_groupby_from_list_of_keys(
            input_domain=self.input_domain,
            input_metric=input_metric,
            use_l2=use_l2,
            groupby_columns=groupby_columns,
            keys=group_keys,
        )
        self.assertEqual(groupby.input_metric, input_metric)
        self.assertEqual(groupby.use_l2, use_l2)
        expected_group_keys = pd.DataFrame(data=group_keys, columns=groupby_columns)
        self.assert_frame_equal_with_sort(
            groupby.group_keys.toPandas(), expected_group_keys
        )
