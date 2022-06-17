"""Benchmarking script for adding noise to dataframes."""

# SPDX-License-Identifier: Apache-2.0

import time

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import IntegerType, StructField, StructType

from tmlt.core.domains.numpy_domains import NumpyIntegerDomain
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkIntegerColumnDescriptor,
)
from tmlt.core.measurements.noise_mechanisms import (
    AddDiscreteGaussianNoise,
    AddGeometricNoise,
    AddLaplaceNoise,
)
from tmlt.core.measurements.pandas_measurements.series import AddNoiseToSeries
from tmlt.core.measurements.spark_measurements import AddNoiseToColumn


def evaluate_runtime(
    input_domain: SparkDataFrameDomain,
    measure_column: str,
    measurement: AddNoiseToSeries,
    sdf: DataFrame,
) -> float:
    """Returns the running time for adding noise to dataframes."""
    start = time.time()
    measurement = AddNoiseToColumn(
        input_domain=input_domain,
        measure_column=measure_column,
        measurement=measurement,
    )
    _ = measurement(sdf).toPandas()
    running_time = time.time() - start
    return round(running_time, 3)


def main():
    """Evaluate running time for adding noise to dataframes."""
    spark = SparkSession.builder.getOrCreate()
    benchmark_result = pd.DataFrame(
        [], columns=["Row Number", "UDF", "Running Time (s)"]
    )
    input_domain = SparkDataFrameDomain({"count": SparkIntegerColumnDescriptor()})
    schema = StructType([StructField("count", IntegerType(), True)])
    empty_df = spark.createDataFrame([], schema=schema)
    _ = empty_df.collect()  # Help spark warm up.

    for size in [100, 400, 10000, 40000, 160000, 640000]:
        df = pd.DataFrame({"count": [0] * size})
        sdf = spark.createDataFrame(df)  # pylint: disable=no-member
        running_time = evaluate_runtime(
            input_domain=input_domain,
            measure_column="count",
            measurement=AddNoiseToSeries(AddGeometricNoise(alpha=1)),
            sdf=sdf,
        )
        row = {
            "Row Number": size,
            "UDF": "AddGeometricNoise",
            "Running Time (s)": running_time,
        }
        benchmark_result = benchmark_result.append(row, ignore_index=True)

    for size in [100, 400, 10000, 40000, 160000, 640000]:
        df = pd.DataFrame({"count": [0] * size})
        sdf = spark.createDataFrame(df)  # pylint: disable=no-member
        running_time = evaluate_runtime(
            input_domain=input_domain,
            measure_column="count",
            measurement=AddNoiseToSeries(
                AddLaplaceNoise(input_domain=NumpyIntegerDomain(), scale=1)
            ),
            sdf=sdf,
        )
        row = {
            "Row Number": size,
            "UDF": "AddLaplaceNoise",
            "Running Time (s)": running_time,
        }
        benchmark_result = benchmark_result.append(row, ignore_index=True)

    for size in [100, 400, 10000, 40000, 160000, 640000]:
        df = pd.DataFrame({"count": [0] * size})
        sdf = spark.createDataFrame(df)  # pylint: disable=no-member
        running_time = evaluate_runtime(
            input_domain=input_domain,
            measure_column="count",
            measurement=AddNoiseToSeries(AddDiscreteGaussianNoise(sigma_squared=1)),
            sdf=sdf,
        )
        row = {
            "Row Number": size,
            "UDF": "AddDiscreteGaussianNoise",
            "Running Time (s)": running_time,
        }
        benchmark_result = benchmark_result.append(row, ignore_index=True)

    spark.stop()
    benchmark_result_html = benchmark_result.to_html()
    print(benchmark_result_html)


if __name__ == "__main__":
    main()
