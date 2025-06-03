"""Benchmarking script for bounds aggregation."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2025

from math import log
from random import randint
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from benchmarking_utils import Timer, write_as_html
from pyspark.sql import SparkSession, functions as sf
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import LongType, StructField, StructType

from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkIntegerColumnDescriptor,
)
from tmlt.core.measurements.aggregations import create_bounds_measurement
from tmlt.core.measures import PureDP
from tmlt.core.metrics import SymmetricDifference
from tmlt.core.transformations.spark_transformations.groupby import (
    create_groupby_from_column_domains,
)
from tmlt.core.utils.testing import PySparkTest


def evaluate_runtime(
    groupby_domains: Dict[str, List[int]],
    dataframe: DataFrame,
    input_domain: SparkDataFrameDomain,
    measure_column: str,
) -> Tuple[float, float, float]:
    """Returns the runtimes for bounds aggregations."""
    groupby = create_groupby_from_column_domains(
        input_domain=input_domain,
        input_metric=SymmetricDifference(),
        use_l2=False,
        column_domains=groupby_domains,
    )
    groupby_bounds = create_bounds_measurement(
        input_domain=input_domain,
        input_metric=SymmetricDifference(),
        output_measure=PureDP(),
        # d_out=1,
        d_out=float("inf"),
        measure_column=measure_column,
        threshold=0.95,
        groupby_transformation=groupby,
    )
    with Timer() as timer:
        groupby_bounds(dataframe).toPandas()
    time = timer.elapsed

    return round(time, 3)


def main() -> None:
    """Evaluate bounds runtimes for different group counts and sizes."""
    spark = SparkSession.builder.getOrCreate()
    benchmark_result = pd.DataFrame(
        [],
        columns=[
            "domain_size",
            "group_size",
            "group_count",
            "num_records",
            "num_groupby_columns",
            "time (s)",
        ],
    )
    input_domain = SparkDataFrameDomain(
        {"A": SparkIntegerColumnDescriptor(), "X": SparkIntegerColumnDescriptor()}
    )

    # Runtimes on Empty DataFrame
    empty_df = spark.createDataFrame([], schema=input_domain.spark_schema)
    for domain_size in [100, 400, 10000]:
        time = evaluate_runtime(
            groupby_domains={"A": list(range(domain_size))},
            dataframe=empty_df,
            input_domain=input_domain,
            measure_column="X",
        )
        row = {
            "domain_size": domain_size,
            "group_size": 0,
            "group_count": domain_size,
            "num_records": 0,
            "num_groupby_columns": 1,
            "time (s)": time,
        }
        benchmark_result = pd.concat(
            [benchmark_result, pd.DataFrame([row])], ignore_index=True
        )


    # Single group. Crashes with OOM error for size 10M for a pure pandas
    # implementation, see #3342.
    for size in [100_000, 900_000, 10_000_000]:
        df = spark.createDataFrame(
            spark.sparkContext.parallelize(
                [
                    (0, randint(0, 1))
                    for i in range(size)
                ]
            ),
            schema=input_domain.spark_schema,
        )
        time = evaluate_runtime(
            {"A": [0]},
            df,
            input_domain,
            measure_column="X",
        )
        row = {
            "domain_size": 1,
            "group_size": size,
            "group_count": 1,
            "num_records": size,
            "num_groupby_columns": 1,
            "time (s)": time,
        }
        benchmark_result = pd.concat(
            [benchmark_result, pd.DataFrame([row])], ignore_index=True
        )

    # Two large groups.
    for size in [100_000, 900_000, 10_000_000]:
        df = spark.createDataFrame(
            spark.sparkContext.parallelize(
                [
                    (i, randint(0, 1))
                    for _ in range(int(size/2))
                    for i in range(2)
                ]
            ),
            schema=input_domain.spark_schema,
        )
        time = evaluate_runtime(
            {"A": [0, 1]},
            df,
            input_domain,
            measure_column="X",
        )
        row = {
            "domain_size": 2,
            "group_size": int(size/2),
            "group_count": 2,
            "num_records": size,
            "num_groupby_columns": 1,
            "time (s)": time,
        }
        benchmark_result = pd.concat(
            [benchmark_result, pd.DataFrame([row])], ignore_index=True
        )


    # Group size = 100
    for size in [10_000, 40_000, 160_000]:
        df = spark.createDataFrame(
            spark.sparkContext.parallelize(
                [(i, randint(0, 1)) for j in range(100) for i in range(int(size / 100))]
            ),
            schema=input_domain.spark_schema,
        )
        time = evaluate_runtime(
            groupby_domains={"A": list(range(int(size / 100)))},
            dataframe=df,
            input_domain=input_domain,
            measure_column="X",
        )
        row = {
            "domain_size": int(size / 100),
            "group_size": 100,
            "group_count": int(size / 100),
            "num_records": size,
            "num_groupby_columns": 1,
            "time (s)": time,
        }
        benchmark_result = pd.concat(
            [benchmark_result, pd.DataFrame([row])], ignore_index=True
        )


    write_as_html(benchmark_result, "bounds.html")


if __name__ == "__main__":
    PySparkTest.setUpClass()
    main()
    PySparkTest.tearDownClass()
