"""Unit tests for :mod:`~tmlt.core.transformations.spark_transformations.groupby`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from parameterized import parameterized
from pyspark.sql.types import (
    DataType,
    DateType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from tmlt.core.domains.base import OutOfDomainError
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
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
    _spark_type,
    compute_full_domain_df,
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
                r"Column must be LongType(\(\))?, instead it is StringType(\(\))?.",
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
            (
                SymmetricDifference(),
                [(1.0,), (2.0,), (3.0,)],
                StructType([StructField("C", DoubleType())]),
                "Can not group by a floating point column: C",
                ValueError,
                SparkDataFrameDomain(
                    {
                        "A": SparkIntegerColumnDescriptor(),
                        "B": SparkStringColumnDescriptor(),
                        "C": SparkFloatColumnDescriptor(),
                    }
                ),
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
        input_domain: Optional[SparkDataFrameDomain] = None,
    ):
        """Tests that GroupBy constructor raises appropriate error."""
        if input_domain is None:
            input_domain = self.domain
        with self.assertRaisesRegex(error_type, error_msg):
            GroupBy(
                input_domain=input_domain,
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


class TestComputeFullDomainDF(PySparkTest):
    """Tests for compute_full_domain_df."""

    @parameterized.expand(
        [
            (
                {"A": [1, 2, 3], "B": ["b1", "b2"], "C": [date(2000, 1, 1)]},
                pd.DataFrame(
                    [
                        [1, "b1", date(2000, 1, 1)],
                        [1, "b2", date(2000, 1, 1)],
                        [2, "b1", date(2000, 1, 1)],
                        [2, "b2", date(2000, 1, 1)],
                        [3, "b1", date(2000, 1, 1)],
                        [3, "b2", date(2000, 1, 1)],
                    ],
                    columns=["A", "B", "C"],
                ),
            ),
            (
                {"Date": [date(2010, 1, 1), date(2020, 1, 1)], "B": ["b1", "b2"]},
                pd.DataFrame(
                    [
                        [date(2010, 1, 1), "b1"],
                        [date(2010, 1, 1), "b2"],
                        [date(2020, 1, 1), "b1"],
                        [date(2020, 1, 1), "b2"],
                    ],
                    columns=["Date", "B"],
                ),
            ),
        ]
    )
    def test_without_null(
        self,
        domains: Dict[
            str,
            Union[
                List[str],
                List[Optional[str]],
                List[int],
                List[Optional[int]],
                List[date],
                List[Optional[date]],
            ],
        ],
        expected: pd.DataFrame,
    ) -> None:
        """Test compute_full_domain_df without null/none values."""
        actual = compute_full_domain_df(domains)
        self.assert_frame_equal_with_sort(actual.toPandas(), expected)

    @parameterized.expand(
        [
            (
                {"A": [None, 2, 3], "B": ["b1", "b2"], "C": [date(2000, 1, 1)]},
                pd.DataFrame(
                    [
                        [None, "b1", date(2000, 1, 1)],
                        [None, "b2", date(2000, 1, 1)],
                        [2, "b1", date(2000, 1, 1)],
                        [2, "b2", date(2000, 1, 1)],
                        [3, "b1", date(2000, 1, 1)],
                        [3, "b2", date(2000, 1, 1)],
                    ],
                    columns=["A", "B", "C"],
                ),
            ),
            (
                {"Date": [date(2010, 1, 1), None], "B": [None, "b1", "b2"]},
                pd.DataFrame(
                    [
                        [date(2010, 1, 1), None],
                        [date(2010, 1, 1), "b1"],
                        [date(2010, 1, 1), "b2"],
                        [None, None],
                        [None, "b1"],
                        [None, "b2"],
                    ],
                    columns=["Date", "B"],
                ),
            ),
        ]
    )
    def test_with_null(
        self,
        domains: Dict[
            str,
            Union[
                List[str],
                List[Optional[str]],
                List[int],
                List[Optional[int]],
                List[date],
                List[Optional[date]],
            ],
        ],
        expected: pd.DataFrame,
    ) -> None:
        """Test compute_full_domain_df when some values *are* null/None."""
        actual = compute_full_domain_df(domains)
        self.assert_frame_equal_with_sort(actual.toPandas(), expected)


class TestSparkType(PySparkTest):
    """Tests for _spark_type."""

    @parameterized.expand(
        [
            ([1, 2], LongType()),
            ([None, None, None, None, 17], LongType()),
            ([-1, 2, None], LongType()),
            (["a", "b"], StringType()),
            ([None, "a"], StringType()),
            (["a", "b", None], StringType()),
            ([date(2020, 1, 1), date(2000, 1, 1)], DateType()),
            ([None, None, None, date(1970, 1, 1)], DateType()),
            ([date(2010, 1, 1), None], DateType()),
        ]
    )
    def test_spark_type(self, l: List[Any], expected: DataType) -> None:
        """Test _spark_type."""
        actual = _spark_type(l)  # pylint: disable=protected-access
        self.assertEqual(actual, expected)

    def test_all_nones(self) -> None:
        """Test _spark_type raises an error when every list entry is None."""
        l = [None, None, None, None, None]
        with self.assertRaisesRegex(ValueError, "every entry is None"):
            _spark_type(l)  # pylint: disable=protected-access

    @parameterized.expand(
        [
            ([1.7, 2.3, 17.0],),
            ([None, None, -3.1],),
            ([["a"], ["b"]],),
            ([None, ["b"]],),
            ([{"key": "val"}],),
            ([None, {}],),
            ([datetime.now()],),
            ([None, datetime(2012, 1, 1, 0, 0, 0)],),
        ]
    )
    def test_disallowed_types(self, l: List[Any]) -> None:
        """Test _spark_type raises an error for invalid types.

        (Invalid types are those that cannot be used in a GroupBy.)
        """
        with self.assertRaisesRegex(ValueError, "Type .* is not supported"):
            _spark_type(l)  # pylint: disable=protected-access
