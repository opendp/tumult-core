"""Unit tests for :mod:`~tmlt.core.measurements.spark_measurements`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from fractions import Fraction
from typing import Dict, List
from unittest.mock import patch

import numpy as np
import pandas as pd
import sympy as sp
from parameterized import parameterized
from pyspark.sql import functions as sf
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from tmlt.core.domains.numpy_domains import NumpyIntegerDomain
from tmlt.core.domains.pandas_domains import PandasDataFrameDomain, PandasSeriesDomain
from tmlt.core.domains.spark_domains import (
    SparkColumnDescriptor,
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkGroupedDataFrameDomain,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
)
from tmlt.core.exceptions import DomainColumnError
from tmlt.core.measurements.noise_mechanisms import AddGeometricNoise, AddLaplaceNoise
from tmlt.core.measurements.pandas_measurements.dataframe import AggregateByColumn
from tmlt.core.measurements.pandas_measurements.series import (
    AddNoiseToSeries,
    NoisyQuantile,
)
from tmlt.core.measurements.spark_measurements import (
    AddNoiseToColumn,
    ApplyInPandas,
    GeometricPartitionSelection,
    SparseVectorPrefixSums,
)
from tmlt.core.measures import ApproxDP, PureDP
from tmlt.core.metrics import AbsoluteDifference, OnColumn, SumOf, SymmetricDifference
from tmlt.core.utils.distributions import double_sided_geometric_cmf_exact
from tmlt.core.utils.exact_number import ExactNumber
from tmlt.core.utils.grouped_dataframe import GroupedDataFrame
from tmlt.core.utils.misc import get_materialized_df
from tmlt.core.utils.testing import (
    FakeAggregate,
    PySparkTest,
    assert_property_immutability,
    get_all_props,
)


class TestApplyInPandas(PySparkTest):
    """Tests for ApplyInPandas."""

    def setUp(self):
        """Setup."""
        self.aggregation_function = AggregateByColumn(
            input_domain=PandasDataFrameDomain(
                {"B": PandasSeriesDomain(NumpyIntegerDomain())}
            ),
            column_to_aggregation={
                "B": NoisyQuantile(
                    PandasSeriesDomain(NumpyIntegerDomain()),
                    output_measure=PureDP(),
                    quantile=0.5,
                    lower=22,
                    upper=29,
                    epsilon=sp.Integer(1),
                )
            },
        )
        self.domain = SparkGroupedDataFrameDomain(
            schema={
                "A": SparkStringColumnDescriptor(),
                "B": SparkIntegerColumnDescriptor(),
            },
            groupby_columns=["A"],
        )
        self.measurement = ApplyInPandas(
            input_domain=self.domain,
            input_metric=SumOf(SymmetricDifference()),
            aggregation_function=self.aggregation_function,
        )

    @parameterized.expand(get_all_props(ApplyInPandas))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        assert_property_immutability(self.measurement, prop_name)

    def test_properties(self):
        """ApplyInPandas's properties have the expected values."""
        aggregation_function = FakeAggregate()
        input_domain = SparkGroupedDataFrameDomain(
            schema={
                "A": SparkStringColumnDescriptor(),
                "B": SparkFloatColumnDescriptor(allow_nan=True),
            },
            groupby_columns=["A"],
        )
        measurement = ApplyInPandas(
            input_domain=input_domain,
            input_metric=SumOf(SymmetricDifference()),
            aggregation_function=aggregation_function,
        )
        self.assertEqual(measurement.input_domain, input_domain)
        self.assertEqual(measurement.input_metric, SumOf(SymmetricDifference()))
        self.assertEqual(measurement.output_measure, PureDP())
        self.assertEqual(measurement.is_interactive, False)
        self.assertEqual(measurement.aggregation_function, aggregation_function)

    @parameterized.expand(
        [
            # test with one groupby column
            (
                {
                    "A": ["1", "2", "2", "3"],
                    "B": [1.0, 2.0, 1.0, np.nan],
                    "C": [np.nan] * 4,
                    "D": [np.nan] * 4,
                },
                {
                    "A": SparkStringColumnDescriptor(),
                    "B": SparkFloatColumnDescriptor(allow_nan=True),
                    "C": SparkFloatColumnDescriptor(allow_nan=True),
                    "D": SparkFloatColumnDescriptor(allow_nan=True),
                },
                {"A": ["1", "2", "3", "4"]},
                {
                    "A": ["1", "2", "3", "4"],
                    "C": [1.0, 3.0, None, -1.0],
                    "C_str": ["1.0", "3.0", "nan", "-1.0"],
                },
            ),
            # test with two groupby columns
            (
                {
                    "A_1": ["1", "2", "2", "3"],
                    "A_2": ["1", "2", "2", "1"],
                    "B": [1.0, 2.0, 1.0, np.nan],
                    "C": [np.nan] * 4,
                    "D": [np.nan] * 4,
                },
                {
                    "A_1": SparkStringColumnDescriptor(),
                    "A_2": SparkStringColumnDescriptor(),
                    "B": SparkFloatColumnDescriptor(allow_nan=True),
                    "C": SparkFloatColumnDescriptor(allow_nan=True),
                    "D": SparkFloatColumnDescriptor(allow_nan=True),
                },
                {"A_1": ["1", "1", "2", "2"], "A_2": ["1", "2", "1", "2"]},
                {
                    "A_1": ["1", "1", "2", "2"],
                    "A_2": ["1", "2", "1", "2"],
                    "C": [1.0, -1.0, -1.0, 3.0],
                    "C_str": ["1.0", "-1.0", "-1.0", "3.0"],
                },
            ),
        ]
    )
    def test_correctness_test_measure(
        self,
        df_dict: Dict[str, List],
        schema: Dict[str, SparkColumnDescriptor],
        groupby_domains: Dict[str, List],
        expected_dict: Dict[str, List],
    ):
        """Test correctness for a GroupByApplyInPandas aggregation."""
        group_keys = self.spark.createDataFrame(pd.DataFrame(groupby_domains))
        input_domain = SparkGroupedDataFrameDomain(
            schema=schema, groupby_columns=list(groupby_domains)
        )
        grouped_dataframe = GroupedDataFrame(
            dataframe=self.spark.createDataFrame(pd.DataFrame(df_dict)),
            group_keys=group_keys,
        )
        actual = ApplyInPandas(
            input_domain=input_domain,
            input_metric=SumOf(SymmetricDifference()),
            aggregation_function=FakeAggregate(),
        )(grouped_dataframe).toPandas()
        expected = pd.DataFrame(expected_dict)
        # It looks like python nans get converted to nulls when the return value
        # from a python udf gets converted back to spark land.
        self.assert_frame_equal_with_sort(actual, expected)

    def test_privacy_function_and_relation(self):
        """Test that the privacy function and relation are computed correctly."""

        quantile_measurement = NoisyQuantile(
            PandasSeriesDomain(NumpyIntegerDomain()),
            output_measure=PureDP(),
            quantile=0.5,
            lower=22,
            upper=29,
            epsilon=sp.Integer(2),
        )

        df_aggregation_function = AggregateByColumn(
            input_domain=PandasDataFrameDomain(
                {"Age": PandasSeriesDomain(NumpyIntegerDomain())}
            ),
            column_to_aggregation={"Age": quantile_measurement},
        )
        measurement = ApplyInPandas(
            input_domain=SparkGroupedDataFrameDomain(
                schema={
                    "Gender": SparkStringColumnDescriptor(),
                    "Age": SparkIntegerColumnDescriptor(),
                },
                groupby_columns=["Gender"],
            ),
            input_metric=SumOf(SymmetricDifference()),
            aggregation_function=df_aggregation_function,
        )

        self.assertTrue(measurement.privacy_function(sp.Integer(1)), sp.Integer(2))
        self.assertTrue(measurement.privacy_relation(sp.Integer(1), sp.Integer(2)))
        self.assertFalse(
            measurement.privacy_relation(sp.Integer(1), sp.Rational("1.99999"))
        )


class TestAddNoiseToColumn(PySparkTest):
    """Tests for AddNoiseToColumn.

    Tests :class:`~tmlt.core.measurements.spark_measurements.AddNoiseToColumn`.
    """

    def setUp(self):
        """Test Setup."""
        self.input_domain = SparkDataFrameDomain(
            {
                "A": SparkStringColumnDescriptor(),
                "count": SparkIntegerColumnDescriptor(),
            }
        )

    @parameterized.expand(get_all_props(AddNoiseToColumn))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        measurement = AddNoiseToColumn(
            input_domain=self.input_domain,
            measurement=AddNoiseToSeries(
                AddLaplaceNoise(input_domain=NumpyIntegerDomain(), scale=sp.Integer(1))
            ),
            measure_column="count",
        )
        assert_property_immutability(measurement, prop_name)

    def test_correctness(self):
        """Tests that AddNoiseToColumn works correctly."""
        expected = pd.DataFrame({"A": [0, 1, 2, 3], "count": [0, 1, 2, 3]})
        sdf = self.spark.createDataFrame(expected)
        measurement = AddNoiseToColumn(
            input_domain=self.input_domain,
            measurement=AddNoiseToSeries(AddGeometricNoise(alpha=0)),
            measure_column="count",
        )
        actual = measurement(sdf).toPandas()
        self.assert_frame_equal_with_sort(actual, expected)


class TestGeometricPartitionSelection(PySparkTest):
    """Tests for GeometricPartitionSelection.

    Tests
    :class:`~tmlt.core.measurements.spark_measurements.GeometricPartitionSelection`.
    """

    def setUp(self):
        """Test Setup."""
        self.input_domain = SparkDataFrameDomain(
            {"A": SparkStringColumnDescriptor(), "B": SparkIntegerColumnDescriptor()}
        )
        self.threshold = 5
        self.alpha = ExactNumber(3)
        self.count_column = "noisy counts"
        self.measurement = GeometricPartitionSelection(
            input_domain=self.input_domain,
            alpha=self.alpha,
            threshold=self.threshold,
            count_column=self.count_column,
        )

    @parameterized.expand(get_all_props(GeometricPartitionSelection))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        assert_property_immutability(self.measurement, prop_name)

    def test_properties(self):
        """GeometricPartitionSelection has the expected properties."""
        self.assertEqual(self.measurement.input_domain, self.input_domain)
        self.assertEqual(self.measurement.input_metric, SymmetricDifference())
        self.assertEqual(self.measurement.output_measure, ApproxDP())
        self.assertEqual(self.measurement.alpha, self.alpha)
        self.assertEqual(self.measurement.threshold, self.threshold)
        self.assertEqual(self.measurement.count_column, self.count_column)

    def test_empty(self):
        """Tests that empty inputs/outputs don't cause any issues."""
        sdf = self.spark.createDataFrame(
            [],
            schema=StructType(
                [StructField("A", StringType()), StructField("B", IntegerType())]
            ),
        )
        expected = pd.DataFrame(
            {
                "A": pd.Series(dtype=str),
                "B": pd.Series(dtype=int),
                self.count_column: pd.Series(dtype=int),
            }
        )
        actual = self.measurement(sdf).toPandas()
        self.assert_frame_equal_with_sort(actual, expected)

    def test_negative_threshold(self):
        """Tests that negative thresholds don't cause any issues."""
        sdf = self.spark.createDataFrame(
            pd.DataFrame({"A": ["a1"] * 100, "B": [1] * 100})
        )
        measurement = GeometricPartitionSelection(
            input_domain=self.input_domain,
            alpha=1,
            threshold=-1,
            count_column=self.count_column,
        )
        actual = measurement(sdf).toPandas()
        expected_without_count = pd.DataFrame({"A": ["a1"], "B": [1]})
        self.assertIsInstance(actual, pd.DataFrame)
        assert isinstance(actual, pd.DataFrame)
        self.assert_frame_equal_with_sort(actual[["A", "B"]], expected_without_count)
        # Threshold -1 should give worse guarantee than for threshold of 0 or 1
        measurement_threshold_0 = GeometricPartitionSelection(
            input_domain=self.input_domain,
            alpha=1,
            threshold=0,
            count_column=self.count_column,
        )
        measurement_threshold_1 = GeometricPartitionSelection(
            input_domain=self.input_domain,
            alpha=1,
            threshold=1,
            count_column=self.count_column,
        )
        # Guarantee isn't infinitely bad
        self.assertFalse(
            ApproxDP().compare((sp.oo, 1), measurement.privacy_function(1))
        )
        # But is worse than for 0, which is worse than for 1
        self.assertFalse(
            ApproxDP().compare(
                measurement.privacy_function(1),
                measurement_threshold_0.privacy_function(1),
            )
        )
        self.assertFalse(
            ApproxDP().compare(
                measurement.privacy_function(1),
                measurement_threshold_1.privacy_function(1),
            )
        )
        self.assertFalse(
            ApproxDP().compare(
                measurement_threshold_0.privacy_function(1),
                measurement_threshold_1.privacy_function(1),
            )
        )

    def test_no_noise(self):
        """Tests that the no noise works correctly."""
        sdf = self.spark.createDataFrame(
            pd.DataFrame(
                {"A": ["a1", "a2", "a2", "a3", "a3", "a3"], "B": [1, 2, 2, 3, 3, 3]}
            )
        )
        expected = pd.DataFrame({"A": ["a2", "a3"], "B": [2, 3], "count": [2, 3]})
        measurement = GeometricPartitionSelection(
            input_domain=self.input_domain, alpha=0, threshold=2
        )
        actual = measurement(sdf).toPandas()
        self.assert_frame_equal_with_sort(actual, expected)

    def test_privacy_function(self):
        """GeometricPartitionSelection's privacy function is correct."""
        alpha = ExactNumber(3)
        threshold = 100
        measurement = GeometricPartitionSelection(
            input_domain=self.input_domain, alpha=alpha, threshold=threshold
        )
        self.assertEqual(measurement.privacy_function(0), (0, 0))
        base_epsilon = 1 / alpha
        base_delta = 1 - double_sided_geometric_cmf_exact(threshold - 2, alpha)
        self.assertEqual(measurement.privacy_function(1), (base_epsilon, base_delta))

        self.assertEqual(
            measurement.privacy_function(3),
            (
                3 * base_epsilon,
                3 * ExactNumber(sp.E) ** (3 * base_epsilon) * base_delta,
            ),
        )


class TestSparseVectorPrefixSums(PySparkTest):
    """Tests for SparseVectorPrefixSums."""

    def setUp(self):
        """Test Setup."""
        self.input_domain = SparkDataFrameDomain(
            {
                "grouping": SparkStringColumnDescriptor(),
                "rank": SparkIntegerColumnDescriptor(),
                "count": SparkIntegerColumnDescriptor(),
            }
        )
        self.alpha = ExactNumber(1)
        self.measurement = SparseVectorPrefixSums(
            input_domain=self.input_domain,
            count_column="count",
            rank_column="rank",
            alpha=self.alpha,
            grouping_columns=["grouping"],
            threshold_fraction=0.9,
        )

    @parameterized.expand(get_all_props(SparseVectorPrefixSums))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        assert_property_immutability(self.measurement, prop_name)

    def test_properties(self):
        """SparseVectorPrefixSums has the expected properties."""
        self.assertEqual(self.measurement.input_domain, self.input_domain)
        self.assertEqual(
            self.measurement.input_metric,
            OnColumn("count", SumOf(AbsoluteDifference())),
        )
        self.assertEqual(self.measurement.output_measure, PureDP())
        self.assertEqual(self.measurement.alpha, self.alpha)
        self.assertEqual(self.measurement.count_column, "count")
        self.assertEqual(self.measurement.rank_column, "rank")
        self.assertEqual(self.measurement.grouping_columns, ["grouping"])
        self.assertEqual(self.measurement.threshold_fraction, 0.9)

    @parameterized.expand(
        [
            (
                "missing_count_column",
                {
                    "rank_column": "rank",
                    "alpha": 1,
                    "count_column": "missing_column",
                    "error": "Column 'missing_column' is not in the input schema.",
                    "error_type": DomainColumnError,
                },
            ),
            (
                "missing_rank_column",
                {
                    "count_column": "count",
                    "alpha": 1,
                    "rank_column": "missing_column",
                    "error": "Column 'missing_column' is not in the input schema.",
                    "error_type": DomainColumnError,
                },
            ),
            (
                "invalid_grouping_column",
                {
                    "count_column": "count",
                    "rank_column": "rank",
                    "alpha": 1,
                    "grouping_columns": ["missing_column"],
                    "error": "Column 'missing_column' is not in the input schema.",
                    "error_type": DomainColumnError,
                },
            ),
            (
                "grouping_equals_count",
                {
                    "count_column": "count",
                    "rank_column": "rank",
                    "alpha": 1,
                    "grouping_columns": ["count"],
                    "error": (
                        "Grouping columns cannot contain the count or rank columns."
                    ),
                    "error_type": ValueError,
                },
            ),
            (
                "grouping_equals_rank",
                {
                    "count_column": "count",
                    "rank_column": "rank",
                    "alpha": 1,
                    "grouping_columns": ["rank"],
                    "error": (
                        "Grouping columns cannot contain the count or rank columns."
                    ),
                    "error_type": ValueError,
                },
            ),
            (
                "invalid_alpha",
                {
                    "count_column": "count",
                    "rank_column": "rank",
                    "alpha": -1,
                    "error": (
                        "Invalid noise scale: -1 is not greater than or equal to 0"
                    ),
                    "error_type": ValueError,
                },
            ),
            (
                "invalid_threshold",
                {
                    "count_column": "count",
                    "rank_column": "rank",
                    "alpha": 1,
                    "threshold_fraction": 0,
                    "error": r"Invalid threshold fraction: 0. Must be in \(0, 1].",
                    "error_type": ValueError,
                },
            ),
        ]
    )
    def test_init(self, _, test_params):
        """Test init function error handling."""
        with self.assertRaisesRegex(test_params["error_type"], test_params["error"]):
            SparseVectorPrefixSums(
                input_domain=self.input_domain,
                count_column=test_params.get("count_column", "count"),
                rank_column=test_params.get("rank_column", "rank"),
                alpha=test_params.get("alpha", 1),
                grouping_columns=test_params.get("grouping_columns", []),
                threshold_fraction=test_params.get("threshold_fraction", 0.9),
            )

    @parameterized.expand(
        [
            # Basic case with single group
            (
                pd.DataFrame(
                    {
                        "grouping1": ["A"] * 5,
                        "grouping2": ["X"] * 5,
                        "rank": [1, 2, 3, 4, 5],
                        "count": [1, 2, 10, 3, 4],
                    }
                ),
                0.5,
                None,
                pd.DataFrame(
                    {
                        "rank": [3],
                    }
                ),
            ),
            # Multiple groups, multiple grouping columns
            (
                pd.DataFrame(
                    {
                        "grouping1": ["A", "A", "A", "A", "A", "A"],
                        "grouping2": ["X", "X", "X", "Y", "Y", "Y"],
                        "rank": [1, 2, 3, 1, 2, 3],
                        "count": [1, 2, 3, 5, 10, 5],
                    }
                ),
                0.7,
                ["grouping1", "grouping2"],
                pd.DataFrame(
                    {"grouping1": "A", "grouping2": ["X", "Y"], "rank": [3, 2]}
                ),
            ),
            # Empty input
            (
                pd.DataFrame(
                    {"grouping1": [], "grouping2": [], "rank": [], "count": []}
                ),
                0.9,
                ["grouping1"],
                pd.DataFrame(
                    {
                        "grouping1": pd.Series(dtype=str),
                        "rank": pd.Series(dtype=int),
                    }
                ),
            ),
            # Single row per group
            (
                pd.DataFrame(
                    {
                        "grouping1": ["A", "B"],
                        "grouping2": ["X", "Y"],
                        "rank": [1, 1],
                        "count": [100, 200],
                    }
                ),
                0.5,
                ["grouping1"],
                pd.DataFrame({"grouping1": ["A", "B"], "rank": [1, 1]}),
            ),
            # Negative ranks
            (
                pd.DataFrame(
                    {
                        "grouping1": ["A"] * 4,
                        "grouping2": ["X"] * 4,
                        "rank": [-2, -1, 0, 1],
                        "count": [1, 5, 2, 2],
                    }
                ),
                0.5,
                None,
                pd.DataFrame({"rank": [-1]}),
            ),
            # zero counts
            (
                pd.DataFrame(
                    {
                        "grouping1": ["A"] * 3,
                        "grouping2": ["X"] * 3,
                        "rank": [1, 2, 3],
                        "count": [0, 0, 0],
                    }
                ),
                0.9,
                ["grouping1"],
                pd.DataFrame(
                    {
                        "grouping1": ["A"],
                        "rank": [1],
                    }
                ),
            ),
            # test that Nulls work correctly
            (
                pd.DataFrame(
                    {
                        "grouping1": [None] * 5,
                        "grouping2": ["X"] * 5,
                        "rank": [1, 2, 3, 4, 5],
                        "count": [1, 2, 10, 3, 4],
                    }
                ),
                0.5,
                ["grouping1", "grouping2"],
                pd.DataFrame(
                    {
                        "grouping1": [None],
                        "grouping2": ["X"],
                        "rank": [3],
                    }
                ),
            ),
        ]
    )
    def test_correctness(
        self, input_df, threshold_fraction, grouping_columns, expected
    ):
        """Tests that SparseVectorPrefixSums works correctly for various inputs."""

        domain = SparkDataFrameDomain(
            {
                "grouping1": SparkStringColumnDescriptor(allow_null=True),
                "grouping2": SparkStringColumnDescriptor(),
                "rank": SparkIntegerColumnDescriptor(),
                "count": SparkIntegerColumnDescriptor(),
            }
        )
        measurement = SparseVectorPrefixSums(
            input_domain=domain,
            count_column="count",
            rank_column="rank",
            alpha=0,
            grouping_columns=grouping_columns,
            threshold_fraction=threshold_fraction,
        )

        sdf = self.spark.createDataFrame(input_df, schema=domain.spark_schema)
        result = measurement(sdf).toPandas()

        self.assert_frame_equal_with_sort(result, expected)

    @patch.object(SparseVectorPrefixSums, "threshold_fraction", new=1.5)
    def test_correctness_high_threshold(self):
        """Tests that SparseVectorPrefixSums works correctly for high threshold.

        Although the threshold fraction cannot be >1, we simulate the case where the
        noise added to the threshold causes it to be larger than any of the prefix
        counts.
        """
        domain = SparkDataFrameDomain(
            {
                "grouping1": SparkStringColumnDescriptor(),
                "rank": SparkIntegerColumnDescriptor(),
                "count": SparkIntegerColumnDescriptor(),
            }
        )
        input_df = pd.DataFrame(
            {
                "grouping1": ["A", "A", "A", "B", "B", "B"],
                "rank": [1, 2, 3, 1, 2, 3],
                "count": [1, 2, 3, 5, 10, 5],
            }
        )
        expected = pd.DataFrame({"grouping1": ["A", "B"], "rank": [3, 3]})

        measurement = SparseVectorPrefixSums(
            input_domain=domain,
            count_column="count",
            rank_column="rank",
            alpha=0,
            grouping_columns=["grouping1"],
            threshold_fraction=0.1,
        )

        sdf = self.spark.createDataFrame(input_df, schema=domain.spark_schema)
        result = measurement(sdf).toPandas()

        self.assert_frame_equal_with_sort(result, expected)

    def test_privacy_function(self):
        """SparseVectorPrefixSums's privacy function is correct."""
        self.assertEqual(self.measurement.privacy_function(0), ExactNumber(0))
        self.assertEqual(self.measurement.privacy_function(1), ExactNumber(4))
        self.assertEqual(self.measurement.privacy_function(2), ExactNumber(8))
        with self.assertRaises(NotImplementedError):
            self.measurement.privacy_function(Fraction(1, 2))


class TestSanitization(PySparkTest):
    """Output DataFrames from Spark measurements are correctly sanitized."""

    @parameterized.expand(
        [
            (
                pd.DataFrame({"col1": [1, 2, 3], "col2": ["abc", "def", "ghi"]}),
                "simple_table",
            ),
            (
                pd.DataFrame(
                    {
                        "bad;column;name": ["a", "b", "c"],
                        "big_numbers": [
                            100000000000000,
                            100000000000000000,
                            99999999999999999,
                        ],
                    }
                ),
                "table_123456",
            ),
        ]
    )
    def test_get_materialized_df(self, df, table_name):
        """Tests that get_materialized_df works correctly."""
        current_db = self.spark.catalog.currentDatabase()
        sdf = self.spark.createDataFrame(df)
        materialized_df = get_materialized_df(sdf, table_name)
        self.assertEqual(current_db, self.spark.catalog.currentDatabase())
        self.assert_frame_equal_with_sort(materialized_df.toPandas(), df)

    def test_repartition_works_as_expected(self):
        """Tests that repartitioning randomly works as expected.

        Note: This is a sanity test that checks repartition by a random
        column works as expected regardless of the internal representation of
        the DataFrame being repartitioned. This does not test any unit
        in :mod:`~tmlt.core.measurements.spark_measurements`.
        """
        df = self.spark.createDataFrame(
            [(i, f"{j}") for i in range(10) for j in range(20)]
        )
        df = df.withColumn("partitioningColumn", sf.round(sf.rand() * 1000))
        # Random partitioning column
        partitions1 = df.repartition("partitioningColumn").rdd.glom().collect()
        df_shuffled = df.repartition(1000)
        partitions2 = df_shuffled.repartition("partitioningColumn").rdd.glom().collect()
        self.assertListEqual(partitions1, partitions2)
