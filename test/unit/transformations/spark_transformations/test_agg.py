"""Unit tests for :mod:`~tmlt.core.transformations.spark_transformations.agg`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from parameterized import parameterized

from tmlt.core.domains.numpy_domains import NumpyIntegerDomain
from tmlt.core.domains.spark_domains import (
    SparkColumnsDescriptor,
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkGroupedDataFrameDomain,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
)
from tmlt.core.metrics import (
    AbsoluteDifference,
    HammingDistance,
    OnColumn,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.spark_transformations.agg import (
    Count,
    CountDistinct,
    CountDistinctGrouped,
    CountGrouped,
    Sum,
    SumGrouped,
    create_count_aggregation,
    create_count_distinct_aggregation,
    create_sum_aggregation,
)
from tmlt.core.utils.exact_number import ExactNumberInput
from tmlt.core.utils.grouped_dataframe import GroupedDataFrame
from tmlt.core.utils.testing import (
    PySparkTest,
    assert_property_immutability,
    get_all_props,
)


class TestCount(PySparkTest):
    """Unit tests for Count."""

    def setUp(self):
        """Test Setup."""
        self.domain = SparkDataFrameDomain(
            {"A": SparkIntegerColumnDescriptor(), "B": SparkStringColumnDescriptor()}
        )

    @parameterized.expand(get_all_props(Count))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        transformation = Count(
            input_domain=self.domain, input_metric=SymmetricDifference()
        )
        assert_property_immutability(transformation, prop_name)

    def test_properties(self):
        """Tests that CountGrouped has expected properties."""
        transformation = Count(
            input_domain=self.domain, input_metric=SymmetricDifference()
        )
        self.assertEqual(transformation.input_domain, self.domain)
        self.assertEqual(transformation.input_metric, SymmetricDifference())
        self.assertEqual(transformation.output_domain, NumpyIntegerDomain())
        self.assertEqual(transformation.output_metric, AbsoluteDifference())

    def test_correctness(self):
        """Tests that count transformation returns expected DataFrame."""
        transformation = Count(
            input_domain=self.domain, input_metric=SymmetricDifference()
        )
        self.assertEqual(
            transformation(
                self.spark.createDataFrame([(1, "x1"), (2, "x2")], schema=["A", "B"])
            ),
            2,
        )
        self.assertEqual(
            transformation(
                self.spark.createDataFrame(
                    [(1, "x1"), (2, "x2")], schema=["A", "B"]
                ).filter("A > 3")
            ),
            0,
        )

    @parameterized.expand([(SymmetricDifference(), 1), (HammingDistance(), 2)])
    def test_stability_function(
        self,
        input_metric: Union[SymmetricDifference, HammingDistance],
        expected_d_out: ExactNumberInput,
    ):
        """Tests that the stability function is correct."""
        count_transformation = Count(
            input_domain=self.domain, input_metric=input_metric
        )
        self.assertEqual(count_transformation.stability_function(1), expected_d_out)


class TestCountDistinct(PySparkTest):
    """Unit tests for CountDistinct."""

    def setUp(self):
        """Test Setup."""
        self.domain = SparkDataFrameDomain(
            {"A": SparkIntegerColumnDescriptor(), "B": SparkStringColumnDescriptor()}
        )

    @parameterized.expand(get_all_props(CountDistinct))
    def test_property_immutability(self, prop_name: str):
        """Tests that a given property is immutable."""
        transformation = CountDistinct(
            input_domain=self.domain, input_metric=SymmetricDifference()
        )
        assert_property_immutability(transformation, prop_name)

    def test_properties(self):
        """Tests that CountDistinct has the expected properties."""
        transformation = CountDistinct(
            input_domain=self.domain, input_metric=SymmetricDifference()
        )
        self.assertEqual(transformation.input_domain, self.domain)
        self.assertEqual(transformation.input_metric, SymmetricDifference())
        self.assertEqual(transformation.output_domain, NumpyIntegerDomain())
        self.assertEqual(transformation.output_metric, AbsoluteDifference())

    @parameterized.expand(
        [
            ([(1, "x1", 1.2), (2, "x2", 3.2)], 2),
            ([(1, "x1", 1.4), (1, "x1", 1.4), (2, "x2", 4.5)], 2),
            ([(1, None, 0.0), (1, "x1", 0.0), (2, "x2", 0.0)], 3),
            ([(1, None, 0.0), (1, "x1", 0.0), (2, "", 0.0)], 3),
            ([(1, None, 1.0), (1, None, 1.0), (2, "x2", 1.0)], 2),
            ([(None, "", 1.0), (0, None, 1.0)], 2),
            (
                [
                    (None, "", float("inf")),
                    (0, " ", float("inf")),
                    (None, "", float("inf")),
                ],
                2,
            ),
            (
                [
                    (None, "", float("nan")),
                    (0, " ", float("nan")),
                    (None, "", float("nan")),
                ],
                2,
            ),
        ]
    )
    def test_correctness(self, rows: List[Tuple], expected: int):
        """Tests that the CountDistinct transformation returns the expected
        result.
        """
        transformation = CountDistinct(
            input_domain=SparkDataFrameDomain(
                {
                    "A": SparkIntegerColumnDescriptor(allow_null=True),
                    "B": SparkStringColumnDescriptor(allow_null=True),
                    "C": SparkFloatColumnDescriptor(
                        allow_null=True, allow_inf=True, allow_nan=True
                    ),
                }
            ),
            input_metric=SymmetricDifference(),
        )
        self.assertEqual(
            transformation(self.spark.createDataFrame(rows, schema=["A", "B", "C"])),
            expected,
        )

    @parameterized.expand([(SymmetricDifference(), 1), (HammingDistance(), 2)])
    def test_stability_function(
        self,
        input_metric: Union[SymmetricDifference, HammingDistance],
        expected_d_out: ExactNumberInput,
    ):
        """Tests that the stability function of CountDistinct is correct."""
        count_transformation = CountDistinct(
            input_domain=self.domain, input_metric=input_metric
        )
        self.assertEqual(count_transformation.stability_function(1), expected_d_out)


class TestCountGrouped(PySparkTest):
    """Unit tests for CountGrouped."""

    def setUp(self):
        """Test Setup."""
        self.domain = SparkGroupedDataFrameDomain(
            schema={
                "A": SparkIntegerColumnDescriptor(),
                "B": SparkStringColumnDescriptor(),
            },
            group_keys=self.spark.createDataFrame([(1,), (2,), (3,)], schema=["A"]),
        )

    @parameterized.expand(get_all_props(CountGrouped))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        transformation = CountGrouped(
            input_domain=self.domain,
            input_metric=SumOf(SymmetricDifference()),
            count_column="count",
        )
        assert_property_immutability(transformation, prop_name)

    def test_properties(self):
        """Tests that CountGrouped has expected properties."""
        count_grouped_dataframe = CountGrouped(
            input_domain=self.domain,
            input_metric=SumOf(SymmetricDifference()),
            count_column="count",
        )

        self.assertTrue(count_grouped_dataframe.input_domain == self.domain)
        self.assertEqual(
            count_grouped_dataframe.input_metric, SumOf(SymmetricDifference())
        )
        self.assertEqual(count_grouped_dataframe.count_column, "count")

        expected_output_domain = SparkDataFrameDomain(
            {
                "A": SparkIntegerColumnDescriptor(),
                "count": SparkIntegerColumnDescriptor(),
            }
        )
        self.assertEqual(count_grouped_dataframe.output_domain, expected_output_domain)
        self.assertEqual(
            count_grouped_dataframe.output_metric,
            OnColumn("count", metric=SumOf(AbsoluteDifference())),
        )

    @parameterized.expand(
        [
            (
                pd.DataFrame({"A": [1, 1, 4, 3], "B": ["x1", "x2", "x3", "x4"]}),
                pd.DataFrame({"A": [1, 2, 3], "C": [2, 0, 1]}),
            )
        ]
    )
    def test_correctness(self, input_df, expected_counts_df):
        """Tests that count transformation returns expected DataFrame."""
        group_keys = self.spark.createDataFrame([(1,), (2,), (3,)], schema=["A"])
        count_groups = CountGrouped(
            input_domain=SparkGroupedDataFrameDomain(
                schema={
                    "A": SparkIntegerColumnDescriptor(),
                    "B": SparkStringColumnDescriptor(),
                },
                group_keys=group_keys,
            ),
            input_metric=SumOf(SymmetricDifference()),
            count_column="C",
        )
        actual_counts_df = count_groups(
            GroupedDataFrame(
                dataframe=self.spark.createDataFrame(input_df), group_keys=group_keys
            )
        ).toPandas()
        self.assert_frame_equal_with_sort(actual_counts_df, expected_counts_df)

    @parameterized.expand(
        [
            (
                SumOf(HammingDistance()),
                "count",
                "Inner metric for the input metric must be SymmetricDifference",
            ),
            (SumOf(SymmetricDifference()), "A", "Invalid count column name"),
        ]
    )
    def test_invalid_inputs(
        self, input_metric: SumOf, output_column_name: str, expected_error_msg: str
    ):
        """Tests that error is raised appropriately for invalid arguments."""
        with self.assertRaisesRegex(ValueError, expected_error_msg):
            CountGrouped(
                input_domain=self.domain,
                input_metric=input_metric,
                count_column=output_column_name,
            )

    @parameterized.expand(
        [
            (SumOf(SymmetricDifference()), 1),
            (RootSumOfSquared(SymmetricDifference()), 1),
        ]
    )
    def test_stability_function(
        self,
        input_metric: Union[SumOf, RootSumOfSquared],
        expected_d_out: ExactNumberInput,
    ):
        """Tests that the stability function is correct."""
        count_transformation = CountGrouped(
            input_domain=self.domain, input_metric=input_metric
        )
        self.assertEqual(count_transformation.stability_function(1), expected_d_out)


class TestCountDistinctGrouped(PySparkTest):
    """Unit tests for CountDistinctGrouped."""

    def setUp(self):
        """Test Setup."""
        self.domain = SparkGroupedDataFrameDomain(
            schema={
                "A": SparkIntegerColumnDescriptor(),
                "B": SparkStringColumnDescriptor(),
            },
            group_keys=self.spark.createDataFrame([(1,), (2,), (3,)], schema=["A"]),
        )

    @parameterized.expand(get_all_props(CountDistinctGrouped))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property of CountDistinctGrouped is immutable."""
        transformation = CountDistinctGrouped(
            input_domain=self.domain,
            input_metric=SumOf(SymmetricDifference()),
            count_column="count",
        )
        assert_property_immutability(transformation, prop_name)

    def test_properties(self):
        """Tests that CountDistinctGrouped has expected properties."""
        count_distinct_grouped_dataframe = CountDistinctGrouped(
            input_domain=self.domain,
            input_metric=SumOf(SymmetricDifference()),
            count_column="count",
        )

        self.assertTrue(count_distinct_grouped_dataframe.input_domain == self.domain)
        self.assertEqual(
            count_distinct_grouped_dataframe.input_metric, SumOf(SymmetricDifference())
        )
        self.assertEqual(count_distinct_grouped_dataframe.count_column, "count")

        expected_output_domain = SparkDataFrameDomain(
            {
                "A": SparkIntegerColumnDescriptor(),
                "count": SparkIntegerColumnDescriptor(),
            }
        )
        self.assertEqual(
            count_distinct_grouped_dataframe.output_domain, expected_output_domain
        )
        self.assertEqual(
            count_distinct_grouped_dataframe.output_metric,
            OnColumn("count", metric=SumOf(AbsoluteDifference())),
        )

    @parameterized.expand(
        [
            (
                [(1, "a1", 1.0), (1, "a1", 1.0), (1, "a1", 1.0), (2, "b2", 1.0)],
                [(1, 1), (2, 1), (3, 0)],
            ),
            (
                [(1, "", 1.0), (1, None, 1.0), (1, "None", 1.0), (1, "null", 1.0)],
                [(1, 4), (2, 0), (3, 0)],
            ),
            (
                [
                    (1, "a", 1.0),
                    (1, "b", 1.0),
                    (1, "b", 1.1),
                    (2, "", float("nan")),
                    (2, "", float("nan")),
                ],
                [(1, 3), (2, 1), (3, 0)],
            ),
            (
                [
                    (1, "a", 1.0),
                    (1, "b", 1.0),
                    (1, "b", 1.1),
                    (2, "", float("nan")),
                    (2, "", float("inf")),
                    (2, "", -float("inf")),
                    (4, "OOD", 1.2),
                ],
                [(1, 3), (2, 3), (3, 0)],
            ),
        ]
    )
    def test_correctness(
        self, input_rows: List[Tuple], expected_output_rows: List[Tuple]
    ):
        """CountDistinctGrouped returns expected output."""
        pd.DataFrame()
        group_keys = self.spark.createDataFrame([(1,), (2,), (3,)], schema=["X"])
        count_distinct_groups = CountDistinctGrouped(
            input_domain=SparkGroupedDataFrameDomain(
                schema={
                    "X": SparkIntegerColumnDescriptor(),
                    "Y": SparkStringColumnDescriptor(allow_null=True),
                    "Z": SparkFloatColumnDescriptor(
                        allow_nan=True, allow_null=True, allow_inf=True
                    ),
                },
                group_keys=group_keys,
            ),
            input_metric=SumOf(SymmetricDifference()),
            count_column="C",
        )
        actual_count_distinct_df = count_distinct_groups(
            GroupedDataFrame(
                dataframe=self.spark.createDataFrame(
                    input_rows, schema=["X", "Y", "Z"]
                ),
                group_keys=group_keys,
            )
        ).toPandas()
        expected_counts_df = self.spark.createDataFrame(
            expected_output_rows, schema=["X", "C"]
        ).toPandas()
        self.assert_frame_equal_with_sort(actual_count_distinct_df, expected_counts_df)

    @parameterized.expand(
        [
            (
                SumOf(HammingDistance()),
                "count",
                "Inner metric for the input metric must be SymmetricDifference",
            ),
            (SumOf(SymmetricDifference()), "A", "Invalid count column name"),
        ]
    )
    def test_invalid_inputs(
        self, input_metric: SumOf, output_column_name: str, expected_error_msg: str
    ):
        """Tests that error is raised appropriately for invalid arguments."""
        with self.assertRaisesRegex(ValueError, expected_error_msg):
            CountDistinctGrouped(
                input_domain=self.domain,
                input_metric=input_metric,
                count_column=output_column_name,
            )

    @parameterized.expand(
        [
            (SumOf(SymmetricDifference()), 1),
            (RootSumOfSquared(SymmetricDifference()), 1),
        ]
    )
    def test_stability_function(
        self,
        input_metric: Union[SumOf, RootSumOfSquared],
        expected_d_out: ExactNumberInput,
    ):
        """Tests that the stability function for CountDistinctGrouped is correct."""
        count_distinct_transformation = CountDistinctGrouped(
            input_domain=self.domain, input_metric=input_metric
        )
        self.assertEqual(
            count_distinct_transformation.stability_function(1), expected_d_out
        )


class TestSum(PySparkTest):
    """Unit tests for Sum."""

    def setUp(self):
        """Test Setup."""
        self.domain = SparkDataFrameDomain(
            {"A": SparkStringColumnDescriptor(), "B": SparkIntegerColumnDescriptor()}
        )
        self.sum_B = Sum(
            input_domain=self.domain,
            input_metric=SymmetricDifference(),
            measure_column="B",
            upper=4,
            lower=2,
        )

    @parameterized.expand(get_all_props(Sum))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        assert_property_immutability(self.sum_B, prop_name)

    def test_properties(self):
        """Tests that Sum has expected properties."""
        self.assertEqual(self.sum_B.input_domain, self.domain)
        self.assertEqual(self.sum_B.input_metric, SymmetricDifference())
        self.assertEqual(self.sum_B.output_domain, NumpyIntegerDomain())
        self.assertEqual(self.sum_B.output_metric, AbsoluteDifference())
        self.assertEqual(self.sum_B.measure_column, "B")
        self.assertEqual(self.sum_B.lower, 2)
        self.assertEqual(self.sum_B.upper, 4)

    @parameterized.expand(
        [
            (pd.DataFrame({"A": ["x1", "x2"], "B": [2, 4]}), np.int64(6)),
            (pd.DataFrame({"A": ["x1", "x2"], "B": [1, 3]}), np.int64(5)),
            (pd.DataFrame({"A": ["x1", "x2"], "B": [6, 5]}), np.int64(8)),
            (pd.DataFrame({"A": [None, "x2"], "B": [2, 4]}), np.int64(6)),
        ]
    )
    def test_correctness(self, input_df: pd.DataFrame, expected_sum: pd.DataFrame):
        """Tests that sum transformation returns expected answer."""
        self.assertEqual(self.sum_B(self.spark.createDataFrame(input_df)), expected_sum)

    @parameterized.expand([(SymmetricDifference(), 4), (HammingDistance(), 2)])
    def test_stability_function(
        self,
        input_metric: Union[SymmetricDifference, HammingDistance],
        expected_d_out: ExactNumberInput,
    ):
        """Tests that the stability function is correct."""
        self.assertEqual(
            Sum(
                input_domain=self.domain,
                input_metric=input_metric,
                measure_column="B",
                lower=2,
                upper=4,
            ).stability_function(1),
            expected_d_out,
        )

    @parameterized.expand(
        [
            (  # Non-existent measure column
                "Z",
                1,
                40,
                r"Invalid measure column: \(Z\) does not exist",
            ),
            (  # Non-numeric measure column
                "A",
                1,
                40,
                r"Measure column \(A\) must be numeric",
            ),
            (  # Non-integral clipping bounds for integral measure column
                "B",
                "1.1",
                40,
                "Clipping bounds must be integral",
            ),
            (  # non-finite clipping bounds for integral measure column
                "E",
                -float("inf"),
                40,
                "Clipping bounds must be finite",
            ),
            (  # lower > upper
                "B",
                40,
                1,
                "Lower clipping bound is larger than upper clipping bound.",
            ),
            (  # measure column permits nans
                "C",
                1,
                40,
                r"Input domain must not allow nulls or NaNs on the measure column"
                r" \(C\)",
            ),
            (  # measure column permits nulls
                "D",
                1,
                40,
                r"Input domain must not allow nulls or NaNs on the measure column"
                r" \(D\)",
            ),
            ("E", 0, 2 ** 970 + 1, r"Upper clipping bound should be at most 2\^970."),
            (
                "E",
                -(2 ** 970) - 1,
                0,
                r"Lower clipping bound should be at least -2\^970.",
            ),
        ]
    )
    def test_invalid_inputs(
        self,
        sum_column: str,
        lower: ExactNumberInput,
        upper: ExactNumberInput,
        error_message: str,
    ):
        """Sum raises appropriate error when constructor arguments are invalid."""
        with self.assertRaisesRegex(ValueError, error_message):
            Sum(
                input_domain=SparkDataFrameDomain(
                    {
                        "A": SparkStringColumnDescriptor(),
                        "B": SparkIntegerColumnDescriptor(),
                        "C": SparkFloatColumnDescriptor(allow_nan=True),
                        "D": SparkIntegerColumnDescriptor(allow_null=True),
                        "E": SparkFloatColumnDescriptor(),
                    }
                ),
                input_metric=SymmetricDifference(),
                measure_column=sum_column,
                upper=upper,
                lower=lower,
            )


class TestSumGrouped(PySparkTest):
    """Unit tests for SumGrouped."""

    def setUp(self):
        """Test setup."""
        self.domain = SparkGroupedDataFrameDomain(
            schema={
                "A": SparkStringColumnDescriptor(allow_null=True),
                "B": SparkIntegerColumnDescriptor(),
            },
            group_keys=self.spark.createDataFrame(
                [(None,), ("x1",), ("x2",), ("x3",)], schema=["A"]
            ),
        )

        self.groupby_A_sum_B = SumGrouped(
            input_domain=self.domain,
            input_metric=SumOf(SymmetricDifference()),
            measure_column="B",
            upper=4,
            lower=2,
            sum_column="sum",
        )

    @parameterized.expand(get_all_props(SumGrouped))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        assert_property_immutability(self.groupby_A_sum_B, prop_name)

    def test_properties(self):
        """Tests that SumGrouped has expected properties.."""
        self.assertEqual(self.groupby_A_sum_B.input_domain, self.domain)
        self.assertEqual(
            self.groupby_A_sum_B.input_metric, SumOf(SymmetricDifference())
        )
        self.assertEqual(self.groupby_A_sum_B.measure_column, "B")
        self.assertEqual(self.groupby_A_sum_B.upper, 4)
        self.assertEqual(self.groupby_A_sum_B.lower, 2)
        self.assertEqual(
            self.groupby_A_sum_B.output_domain,
            SparkDataFrameDomain(
                {
                    "A": SparkStringColumnDescriptor(allow_null=True),
                    "sum": SparkIntegerColumnDescriptor(),
                }
            ),
        )
        self.assertEqual(
            self.groupby_A_sum_B.output_metric,
            OnColumn("sum", SumOf(AbsoluteDifference())),
        )
        self.assertEqual(self.groupby_A_sum_B.sum_column, "sum")

    @parameterized.expand(
        [
            (
                pd.DataFrame({"A": ["x1", "x2"], "B": [2, 4]}),
                pd.DataFrame({"A": [None, "x1", "x2", "x3"], "sum": [0, 2, 4, 0]}),
            ),
            (  # value exceeds upper clamping bound
                pd.DataFrame({"A": ["x1", "x2"], "B": [20, 4]}),
                pd.DataFrame({"A": [None, "x1", "x2", "x3"], "sum": [0, 4, 4, 0]}),
            ),
            (  # value below lower clamping bound
                pd.DataFrame({"A": ["x1", "x2"], "B": [2, 0]}),
                pd.DataFrame({"A": [None, "x1", "x2", "x3"], "sum": [0, 2, 2, 0]}),
            ),
            (  # extra key 'x4' shouldn't appear
                pd.DataFrame({"A": ["x1", "x4"], "B": [2, 3]}),
                pd.DataFrame({"A": [None, "x1", "x2", "x3"], "sum": [0, 2, 0, 0]}),
            ),
            (  # grouping on nulls should work
                pd.DataFrame({"A": [None, "x1"], "B": [2, 3]}),
                pd.DataFrame({"A": [None, "x1", "x2", "x3"], "sum": [2, 3, 0, 0]}),
            ),
        ]
    )
    def test_correctness(self, input_df: pd.DataFrame, expected_df: pd.DataFrame):
        """Tests that sum transformation returns expected DataFrame."""
        grouped_dataframe = GroupedDataFrame(
            dataframe=self.spark.createDataFrame(input_df),
            group_keys=self.domain.group_keys,
        )
        self.assert_frame_equal_with_sort(
            self.groupby_A_sum_B(grouped_dataframe).toPandas(), expected_df
        )

    @parameterized.expand(
        [
            (SumOf(SymmetricDifference()), 4),
            (RootSumOfSquared(SymmetricDifference()), 4),
        ]
    )
    def test_stability_function(
        self,
        input_metric: Union[SumOf, RootSumOfSquared],
        expected_d_out: ExactNumberInput,
    ):
        """SumGrouped's stability function is correct."""
        self.assertEqual(
            SumGrouped(
                input_domain=self.domain,
                input_metric=input_metric,
                measure_column="B",
                upper=4,
                lower=2,
                sum_column="sum",
            ).stability_function(1),
            expected_d_out,
        )

    @parameterized.expand(
        [
            (  # Non-existent measure column
                SumOf(SymmetricDifference()),
                "Z",
                "sum(Z)",
                1,
                40,
                "Invalid measure column: Z",
            ),
            (  # Non-numeric measure column
                SumOf(SymmetricDifference()),
                "B",
                "sum(B)",
                1,
                40,
                r"Measure column \(B\) must be numeric",
                {
                    "A": SparkStringColumnDescriptor(),
                    "B": SparkStringColumnDescriptor(),
                },
            ),
            (  # lower > upper
                SumOf(SymmetricDifference()),
                "B",
                "sum(B)",
                40,
                1,
                "Lower clipping bound is larger than upper clipping bound.",
            ),
            (  # Sum column already exists
                SumOf(SymmetricDifference()),
                "B",
                "A",
                1,
                40,
                "Invalid sum column name: 'A' already exists",
            ),
            (  # Invalid input metric
                SumOf(AbsoluteDifference()),
                "B",
                "sum(B)",
                1,
                40,
                r"Input metric must be SumOf\(SymmetricDifference\(\)\) or"
                r" RootSumOfSquared\(SymmetricDifference\(\)\)",
            ),
            (  # Domain permits nans
                SumOf(SymmetricDifference()),
                "C",
                "sum(C)",
                1,
                40,
                r"Input domain must not allow nulls or NaNs on the sum column \(C\)",
            ),
            (  # Domain permits nulls
                SumOf(SymmetricDifference()),
                "D",
                "sum(D)",
                1,
                40,
                r"Input domain must not allow nulls or NaNs on the sum column \(D\)",
            ),
        ]
    )
    def test_invalid_inputs(
        self,
        input_metric: Union[SumOf, RootSumOfSquared],
        measure_column: str,
        sum_column: str,
        lower: ExactNumberInput,
        upper: ExactNumberInput,
        error_msg: str,
        schema: Optional[SparkColumnsDescriptor] = None,
    ):
        """Tests that error is raised appropriately for invalid arguments."""
        if schema is None:
            schema = {
                "A": SparkStringColumnDescriptor(),
                "B": SparkIntegerColumnDescriptor(),
                "C": SparkFloatColumnDescriptor(allow_nan=True),
                "D": SparkIntegerColumnDescriptor(allow_null=True),
            }
        with self.assertRaisesRegex(ValueError, error_msg):
            SumGrouped(
                input_domain=SparkGroupedDataFrameDomain(
                    schema=schema,
                    group_keys=self.spark.createDataFrame(
                        pd.DataFrame({"A": ["x1", "x2"]})
                    ),
                ),
                input_metric=input_metric,
                measure_column=measure_column,
                sum_column=sum_column,
                upper=upper,
                lower=lower,
            )


class TestDerivedTransformations(PySparkTest):
    """Unit tests for derived aggregations."""

    def setUp(self):
        """Test Setup."""
        self.dataframe_domain = SparkDataFrameDomain(
            schema={
                "A": SparkStringColumnDescriptor(),
                "B": SparkIntegerColumnDescriptor(),
            }
        )
        self.grouped_dataframe_domain = SparkGroupedDataFrameDomain(
            schema={
                "A": SparkStringColumnDescriptor(),
                "B": SparkIntegerColumnDescriptor(),
            },
            group_keys=self.spark.createDataFrame([("x1",), ("x2",)], schema=["A"]),
        )

    @parameterized.expand(
        [
            (
                True,
                SumOf(SymmetricDifference()),
                OnColumn("sum(B)", SumOf(AbsoluteDifference())),
                SparkDataFrameDomain(
                    {
                        "A": SparkStringColumnDescriptor(),
                        "sum(B)": SparkIntegerColumnDescriptor(),
                    }
                ),
            ),
            (
                True,
                RootSumOfSquared(SymmetricDifference()),
                OnColumn("sum(B)", RootSumOfSquared(AbsoluteDifference())),
                SparkDataFrameDomain(
                    {
                        "A": SparkStringColumnDescriptor(),
                        "sum(B)": SparkIntegerColumnDescriptor(),
                    }
                ),
            ),
            (False, SymmetricDifference(), AbsoluteDifference(), NumpyIntegerDomain()),
            (False, HammingDistance(), AbsoluteDifference(), NumpyIntegerDomain()),
        ]
    )
    def test_create_sum_aggregation(
        self,
        on_grouped_dataframe: bool,
        input_metric: Union[
            SymmetricDifference, HammingDistance, SumOf, RootSumOfSquared
        ],
        expected_output_metric: Union[SumOf, AbsoluteDifference, RootSumOfSquared],
        expected_output_domain: Union[NumpyIntegerDomain, SparkDataFrameDomain],
    ):
        """create_sum_aggregation works correctly."""
        input_domain = (
            self.grouped_dataframe_domain
            if on_grouped_dataframe
            else self.dataframe_domain
        )
        sum_transformation = create_sum_aggregation(  # type: ignore
            input_domain=input_domain,
            input_metric=input_metric,
            measure_column="B",
            lower=0,
            upper=3,
        )
        self.assertTrue(
            isinstance(sum_transformation, SumGrouped if on_grouped_dataframe else Sum)
        )
        self.assertEqual(sum_transformation.measure_column, "B")
        self.assertEqual(sum_transformation.input_metric, input_metric)
        self.assertEqual(sum_transformation.input_domain, input_domain)
        self.assertEqual(sum_transformation.output_domain, expected_output_domain)
        self.assertEqual(sum_transformation.output_metric, expected_output_metric)
        self.assertEqual(sum_transformation.lower, 0)
        self.assertEqual(sum_transformation.upper, 3)

    @parameterized.expand(
        [
            (
                True,
                SumOf(SymmetricDifference()),
                OnColumn("count", SumOf(AbsoluteDifference())),
                SparkDataFrameDomain(
                    {
                        "A": SparkStringColumnDescriptor(),
                        "count": SparkIntegerColumnDescriptor(),
                    }
                ),
            ),
            (
                True,
                RootSumOfSquared(SymmetricDifference()),
                OnColumn("count", RootSumOfSquared(AbsoluteDifference())),
                SparkDataFrameDomain(
                    {
                        "A": SparkStringColumnDescriptor(),
                        "count": SparkIntegerColumnDescriptor(),
                    }
                ),
            ),
            (False, SymmetricDifference(), AbsoluteDifference(), NumpyIntegerDomain()),
            (False, HammingDistance(), AbsoluteDifference(), NumpyIntegerDomain()),
        ]
    )
    def test_create_count_aggregation(
        self,
        on_grouped_dataframe: bool,
        input_metric: Union[SymmetricDifference, SumOf, RootSumOfSquared],
        expected_output_metric: Union[SumOf, AbsoluteDifference, RootSumOfSquared],
        expected_output_domain: Union[NumpyIntegerDomain, SparkDataFrameDomain],
    ):
        """create_count_aggregation works correctly."""
        input_domain = (
            self.grouped_dataframe_domain
            if on_grouped_dataframe
            else self.dataframe_domain
        )
        count_transformation = create_count_aggregation(  # type: ignore
            input_domain=input_domain, input_metric=input_metric
        )
        self.assertTrue(
            isinstance(
                count_transformation, CountGrouped if on_grouped_dataframe else Count
            )
        )
        self.assertEqual(count_transformation.input_metric, input_metric)
        self.assertEqual(count_transformation.input_domain, input_domain)
        self.assertEqual(count_transformation.output_domain, expected_output_domain)
        self.assertEqual(count_transformation.output_metric, expected_output_metric)

    @parameterized.expand(
        [
            (
                True,
                SumOf(SymmetricDifference()),
                OnColumn("count_distinct", SumOf(AbsoluteDifference())),
                SparkDataFrameDomain(
                    {
                        "A": SparkStringColumnDescriptor(),
                        "count_distinct": SparkIntegerColumnDescriptor(),
                    }
                ),
            ),
            (
                True,
                RootSumOfSquared(SymmetricDifference()),
                OnColumn("count_distinct", RootSumOfSquared(AbsoluteDifference())),
                SparkDataFrameDomain(
                    {
                        "A": SparkStringColumnDescriptor(),
                        "count_distinct": SparkIntegerColumnDescriptor(),
                    }
                ),
            ),
            (False, SymmetricDifference(), AbsoluteDifference(), NumpyIntegerDomain()),
            (False, HammingDistance(), AbsoluteDifference(), NumpyIntegerDomain()),
        ]
    )
    def test_create_count_distinct_aggregation(
        self,
        on_grouped_dataframe: bool,
        input_metric: Union[SymmetricDifference, SumOf, RootSumOfSquared],
        expected_output_metric: Union[SumOf, AbsoluteDifference, RootSumOfSquared],
        expected_output_domain: Union[NumpyIntegerDomain, SparkDataFrameDomain],
    ):
        """create_count_distinct_aggregation works correctly."""
        input_domain = (
            self.grouped_dataframe_domain
            if on_grouped_dataframe
            else self.dataframe_domain
        )

        # the "type: ignore" comment makes the next line 2 characters too long.
        # pylint: disable=line-too-long
        count_distinct_transformation = create_count_distinct_aggregation(  # type: ignore
            input_domain=input_domain, input_metric=input_metric
        )
        # pylint: enable=line-too-long

        self.assertTrue(
            isinstance(
                count_distinct_transformation,
                CountDistinctGrouped if on_grouped_dataframe else CountDistinct,
            )
        )
        self.assertEqual(count_distinct_transformation.input_metric, input_metric)
        self.assertEqual(count_distinct_transformation.input_domain, input_domain)
        self.assertEqual(
            count_distinct_transformation.output_domain, expected_output_domain
        )
        self.assertEqual(
            count_distinct_transformation.output_metric, expected_output_metric
        )
