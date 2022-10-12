"""Benchmarking script for spark-based count and sum aggregations."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

from math import log
from random import randint
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as sf
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import IntegerType, StructField, StructType

from benchmarking_utils import Timer, write_as_html
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkIntegerColumnDescriptor,
)
from tmlt.core.metrics import SymmetricDifference
from tmlt.core.transformations.spark_transformations.agg import (
    create_count_aggregation,
    create_sum_aggregation,
)
from tmlt.core.transformations.spark_transformations.groupby import (
    create_groupby_from_column_domains,
)
from tmlt.core.utils.exact_number import ExactNumberInput
from tmlt.core.utils.testing import PySparkTest


def evaluate_runtime(
    groupby_domains: Dict[str, List[int]],
    dataframe: DataFrame,
    input_domain: SparkDataFrameDomain,
    measure_column: str,
    low: ExactNumberInput,
    high: ExactNumberInput,
) -> Tuple[float, float, float]:
    """Returns the runtimes for count, sum, and both aggregations."""
    groupby = create_groupby_from_column_domains(
        input_domain=input_domain,
        input_metric=SymmetricDifference(),
        use_l2=False,
        column_domains=groupby_domains,
    )
    count = create_count_aggregation(
        input_domain=groupby.output_domain,
        input_metric=groupby.output_metric,
    )
    groupby_count = groupby | count
    with Timer() as count_timer:
        groupby_count(dataframe).toPandas()
    count_time = count_timer.elapsed

    sum = create_sum_aggregation(
        input_domain=groupby.output_domain,
        input_metric=groupby.output_metric,
        measure_column=measure_column,
        lower=low,
        upper=high,
        sum_column=f"sum{measure_column}",
    )
    groupby_sum = groupby | sum
    with Timer() as sum_timer:
        groupby_sum(dataframe).toPandas()
    sum_time = sum_timer.elapsed
    return round(count_time, 3), round(sum_time, 3)


def main():
    """Evaluate count and sum runtimes for different group counts and sizes."""
    spark = SparkSession.builder.getOrCreate()
    benchmark_result = pd.DataFrame(
        [],
        columns=[
            "domain_size",
            "group_size",
            "group_count",
            "num_records",
            "num_groupby_columns",
            "count_time (s)",
            "sum_time (s)",
        ],
    )
    input_domain = SparkDataFrameDomain(
        {"A": SparkIntegerColumnDescriptor(), "X": SparkIntegerColumnDescriptor()}
    )

    # Runtimes on Empty DataFrame
    empty_df = spark.createDataFrame([], schema=input_domain.spark_schema)
    for domain_size in [100, 400, 10000, 40000, 160000, 640000]:
        count_time, sum_time = evaluate_runtime(
            groupby_domains={"A": list(range(domain_size))},
            dataframe=empty_df,
            input_domain=input_domain,
            measure_column="X",
            low=0,
            high=1,
        )
        row = {
            "domain_size": domain_size,
            "group_size": 0,
            "group_count": domain_size,
            "num_records": 0,
            "num_groupby_columns": 1,
            "count_time (s)": count_time,
            "sum_time (s)": sum_time,
        }
        benchmark_result = benchmark_result.append(row, ignore_index=True)

    # Single Groupby Column of varying domain sizes (1 row/group)
    for domain_size in [100, 400, 10000, 40000, 160000, 640000]:
        df = spark.createDataFrame(  # pylint: disable=no-member
            spark.sparkContext.parallelize(
                [(i, randint(0, 1)) for i in range(domain_size)]
            ),
            schema=input_domain.spark_schema,
        )
        count_time, sum_time = evaluate_runtime(
            groupby_domains={"A": list(range(domain_size))},
            dataframe=df,
            input_domain=input_domain,
            measure_column="X",
            low=0,
            high=1,
        )
        row = {
            "domain_size": domain_size,
            "group_size": 1,
            "group_count": domain_size,
            "num_records": domain_size,
            "num_groupby_columns": 1,
            "count_time (s)": count_time,
            "sum_time (s)": sum_time,
        }
        benchmark_result = benchmark_result.append(row, ignore_index=True)

    # Single groupby column, group size = 1M
    for size in [100000, 900000, 10000000]:
        df = spark.createDataFrame(  # pylint: disable=no-member
            spark.sparkContext.parallelize(
                [
                    (i, randint(0, 1))
                    for _ in range(100000)
                    for i in range(int(size / 100000))
                ]
            ),
            schema=input_domain.spark_schema,
        )
        count_time, sum_time = evaluate_runtime(
            {"A": list(range(int(size / 100000)))},
            df,
            input_domain,
            measure_column="X",
            low=0,
            high=1,
        )
        row = {
            "domain_size": int(size / 100000),
            "group_size": 100000,
            "group_count": int(size / 100000),
            "num_records": size,
            "num_groupby_columns": 1,
            "count_time (s)": count_time,
            "sum_time (s)": sum_time,
        }
        benchmark_result = benchmark_result.append(row, ignore_index=True)

    # Group size = 10K
    for size in [10000, 100000, 1000000, 10000000]:
        df = spark.createDataFrame(  # pylint: disable=no-member
            spark.sparkContext.parallelize(
                [
                    (i, randint(0, 1))
                    for j in range(10000)
                    for i in range(int(size / 10000))
                ]
            ),
            schema=input_domain.spark_schema,
        )
        count_time, sum_time = evaluate_runtime(
            groupby_domains={"A": list(range(int(size / 10000)))},
            dataframe=df,
            input_domain=input_domain,
            measure_column="X",
            low=0,
            high=1,
        )
        row = {
            "domain_size": int(size / 10000),
            "group_size": 10000,
            "group_count": int(size / 10000),
            "num_records": size,
            "num_groupby_columns": 1,
            "count_time (s)": count_time,
            "sum_time (s)": sum_time,
        }
        benchmark_result = benchmark_result.append(row, ignore_index=True)

    # Group size = 100
    for size in [10000, 40000, 160000, 640000, 2560000]:
        df = spark.createDataFrame(  # pylint: disable=no-member
            spark.sparkContext.parallelize(
                [(i, randint(0, 1)) for j in range(100) for i in range(int(size / 100))]
            ),
            schema=input_domain.spark_schema,
        )
        count_time, sum_time = evaluate_runtime(
            groupby_domains={"A": list(range(int(size / 100)))},
            dataframe=df,
            input_domain=input_domain,
            measure_column="X",
            low=0,
            high=1,
        )
        row = {
            "domain_size": int(size / 100),
            "group_size": 100,
            "group_count": int(size / 100),
            "num_records": size,
            "num_groupby_columns": 1,
            "count_time (s)": count_time,
            "sum_time (s)": sum_time,
        }
        benchmark_result = benchmark_result.append(row, ignore_index=True)

    # Multiple groupby columns
    domain_size = 2
    group_size = 1
    for num_cols in [1, 2, 4, 8, 12, 16, 18, 20]:
        group_count = domain_size ** num_cols
        num_records = group_count * group_size
        groupby_columns = ["Col_{}".format(i) for i in range(num_cols)]
        columns = groupby_columns + ["X"]
        input_domain = SparkDataFrameDomain(
            dict.fromkeys(columns, SparkIntegerColumnDescriptor())
        )
        schema = StructType(
            [
                StructField("Col_{}".format(i), IntegerType(), True)
                for i in range(num_cols)
            ]
        )
        sdf = spark.createDataFrame(  # pylint: disable=no-member
            spark.sparkContext.parallelize(
                np.repeat(
                    np.transpose(
                        np.unravel_index(
                            range(group_count), shape=(domain_size,) * num_cols
                        )
                    ),
                    group_size,
                    axis=0,
                ).tolist()
            ),
            schema=schema,
        )
        sdf = sdf.withColumn("X", sf.lit(100))
        groupby_domains = dict.fromkeys(groupby_columns, list(range(domain_size)))
        count_time, sum_time = evaluate_runtime(
            groupby_domains,
            sdf,
            input_domain,
            measure_column="X",
            low=0,
            high=1,
        )
        row = {
            "domain_size": 2,
            "group_size": group_size,
            "group_count": group_count,
            "num_records": num_records,
            "num_groupby_columns": num_cols,
            "count_time (s)": count_time,
            "sum_time (s)": sum_time,
        }
        benchmark_result = benchmark_result.append(row, ignore_index=True)

    # various domain sizes and columns
    group_size = 100
    group_count = 2 ** 8
    num_records = group_count * group_size
    for domain_size in [256, 16, 4, 2]:
        num_cols = int(log(group_count, domain_size))
        groupby_columns = ["Col_{}".format(i) for i in range(num_cols)]
        columns = groupby_columns + ["X"]
        input_domain = SparkDataFrameDomain(
            dict.fromkeys(columns, SparkIntegerColumnDescriptor())
        )
        schema = StructType(
            [
                StructField("Col_{}".format(i), IntegerType(), True)
                for i in range(num_cols)
            ]
        )
        sdf = spark.createDataFrame(  # pylint: disable=no-member
            spark.sparkContext.parallelize(
                np.repeat(
                    np.transpose(
                        np.unravel_index(
                            range(group_count), shape=(domain_size,) * num_cols
                        )
                    ),
                    group_size,
                    axis=0,
                ).tolist()
            ),
            schema=schema,
        )
        sdf = sdf.withColumn("X", sf.lit(100))
        groupby_domains = dict.fromkeys(groupby_columns, list(range(domain_size)))
        count_time, sum_time = evaluate_runtime(
            groupby_domains,
            sdf,
            input_domain,
            measure_column="X",
            low=0,
            high=1,
        )
        row = {
            "domain_size": domain_size,
            "group_size": group_size,
            "group_count": group_count,
            "num_records": num_records,
            "num_groupby_columns": num_cols,
            "count_time (s)": count_time,
            "sum_time (s)": sum_time,
        }
        benchmark_result = benchmark_result.append(row, ignore_index=True)

    write_as_html(benchmark_result, "count_sum.html")


if __name__ == "__main__":
    PySparkTest.setUpClass()
    main()
    PySparkTest.tearDownClass()
