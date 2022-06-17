"""Unit tests for :mod:`~tmlt.core.measurements.spark_measurements`."""

# SPDX-License-Identifier: Apache-2.0

# pylint: disable=no-self-use
from typing import Dict, List

import numpy as np
import pandas as pd
import sympy as sp
from parameterized import parameterized
from pyspark.sql import functions as sf

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
from tmlt.core.measurements.noise_mechanisms import AddGeometricNoise, AddLaplaceNoise
from tmlt.core.measurements.pandas_measurements.dataframe import AggregateByColumn
from tmlt.core.measurements.pandas_measurements.series import (
    AddNoiseToSeries,
    NoisyQuantile,
)
from tmlt.core.measurements.spark_measurements import (
    AddNoiseToColumn,
    ApplyInPandas,
    _get_materialized_df,
)
from tmlt.core.measures import PureDP
from tmlt.core.metrics import SumOf, SymmetricDifference
from tmlt.core.utils.grouped_dataframe import GroupedDataFrame
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
            group_keys=self.spark.createDataFrame([("x1",), ("x2",)], schema=["A"]),
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
            group_keys=self.spark.createDataFrame([("x1",), ("x2",)], schema=["A"]),
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
        input_domain = SparkGroupedDataFrameDomain(schema=schema, group_keys=group_keys)
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
                group_keys=self.spark.createDataFrame(
                    pd.DataFrame({"Gender": ["M", "F"]})
                ),
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
        """Tests that _get_materialized_df works correctly."""
        current_db = self.spark.catalog.currentDatabase()
        sdf = self.spark.createDataFrame(df)
        materialized_df = _get_materialized_df(sdf, table_name)
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
