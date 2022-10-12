"""Benchmarking script for Map."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

import time
from random import choice, randint
from typing import Callable

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from tmlt.core.domains.collections import ListDomain
from tmlt.core.domains.spark_domains import (
    SparkIntegerColumnDescriptor,
    SparkRowDomain,
    SparkStringColumnDescriptor,
)
from tmlt.core.metrics import SymmetricDifference
from tmlt.core.transformations.spark_transformations.map import (
    FlatMap,
    RowToRowsTransformation,
)

from benchmarking_utils import write_as_html


def evaluate_runtime(
    input_domain: SparkRowDomain,
    output_domain: ListDomain,
    trusted_f: Callable,
    augment: bool,
    max_num_rows: int,
    sdf: DataFrame,
) -> float:
    """Returns the runtimes."""
    start = time.time()
    duplicate_transformer = RowToRowsTransformation(
        input_domain=input_domain,
        output_domain=output_domain,
        trusted_f=trusted_f,
        augment=augment,
    )
    flat_map_transformation = FlatMap(
        row_transformer=duplicate_transformer,
        max_num_rows=max_num_rows,
        metric=SymmetricDifference(),
    )
    _ = flat_map_transformation(sdf).toPandas()
    running_time = time.time() - start
    return round(running_time, 3)


def main():
    """Evaluate running time for FlatMap with different settings.

    These settings include number of rows, number of columns, transform functions,
    augment flag (True vs False), and max_num_rows.
    """
    spark = SparkSession.builder.getOrCreate()
    benchmark_result = pd.DataFrame(
        [],
        columns=[
            "Row Number",
            "Column Number",
            "Transform Function",
            "Augment Flag",
            "Max Num Rows",
            "Running Time (s)",
        ],
    )
    input_domain = {
        "A": SparkIntegerColumnDescriptor(),
        "B": SparkStringColumnDescriptor(),
    }
    schema = StructType(
        [StructField("A", IntegerType(), True), StructField("B", StringType(), True)]
    )
    empty_df = spark.createDataFrame([], schema=schema)
    _ = empty_df.collect()  # Help spark warm up.

    # Various rows
    max_num_row = 10
    for size in [100, 10000, 100000]:
        sdf = spark.createDataFrame(  # pylint: disable=no-member
            spark.sparkContext.parallelize(
                [(i, choice(["X", "Y"])) for i in range(size)]
            ),
            schema=schema,
        )
        augment = False
        trans_func = lambda x: [x] * max_num_row
        running_time = evaluate_runtime(
            input_domain=SparkRowDomain(input_domain),
            output_domain=ListDomain(SparkRowDomain(input_domain)),
            trusted_f=trans_func,
            augment=augment,
            max_num_rows=max_num_row,
            sdf=sdf,
        )
        row = {
            "Row Number": size,
            "Column Number": 2,
            "Transform Function": "duplicates",
            "Augment Flag": augment,
            "Max Num Rows": max_num_row,
            "Running Time (s)": running_time,
        }
        benchmark_result = benchmark_result.append(row, ignore_index=True)

        augment = True
        trans_func = lambda row: [
            {f"{key}_copy": value for key, value in row.asDict().items()}
        ]
        output_domain = {
            "A": SparkIntegerColumnDescriptor(),
            "B": SparkStringColumnDescriptor(),
            "A_copy": SparkIntegerColumnDescriptor(),
            "B_copy": SparkStringColumnDescriptor(),
        }
        running_time = evaluate_runtime(
            input_domain=SparkRowDomain(input_domain),
            output_domain=ListDomain(SparkRowDomain(output_domain)),
            trusted_f=trans_func,
            augment=augment,
            max_num_rows=max_num_row,
            sdf=sdf,
        )
        row = {
            "Row Number": size,
            "Column Number": 2,
            "Transform Function": "duplicates",
            "Augment Flag": augment,
            "Max Num Rows": max_num_row,
            "Running Time (s)": running_time,
        }
        benchmark_result = benchmark_result.append(row, ignore_index=True)

    # various columns
    max_num_rows, rows = 10, 1000
    for cols in [100, 400, 800, 1600]:
        columns = ["Col_{}".format(i) for i in range(cols)]
        input_domain = dict.fromkeys(columns, SparkIntegerColumnDescriptor())
        schema = StructType(
            [StructField("Col_{}".format(i), IntegerType(), True) for i in range(cols)]
        )
        sdf = spark.createDataFrame(  # pylint: disable=no-member
            spark.sparkContext.parallelize([tuple(range(cols))] * rows), schema=schema
        )
        augment = False
        trans_func = lambda x: [x] * max_num_rows
        running_time = evaluate_runtime(
            input_domain=SparkRowDomain(input_domain),
            output_domain=ListDomain(SparkRowDomain(input_domain)),
            trusted_f=trans_func,
            augment=augment,
            max_num_rows=max_num_rows,
            sdf=sdf,
        )
        row = {
            "Row Number": rows,
            "Column Number": cols,
            "Transform Function": "duplicates",
            "Augment Flag": augment,
            "Max Num Rows": max_num_rows,
            "Running Time (s)": running_time,
        }
        benchmark_result = benchmark_result.append(row, ignore_index=True)

        augment = True
        trans_func = lambda row: [
            {f"{key}_copy": value for key, value in row.asDict().items()}
        ]
        output_domain = input_domain.copy()
        for i in range(cols):
            output_domain[f"Col_{i}_copy"] = SparkIntegerColumnDescriptor()
        running_time = evaluate_runtime(
            input_domain=SparkRowDomain(input_domain),
            output_domain=ListDomain(SparkRowDomain(output_domain)),
            trusted_f=trans_func,
            augment=augment,
            max_num_rows=max_num_rows,
            sdf=sdf,
        )
        row = {
            "Row Number": rows,
            "Column Number": cols,
            "Transform Function": "duplicates",
            "Augment Flag": augment,
            "Max Num Rows": max_num_rows,
            "Running Time (s)": running_time,
        }
        benchmark_result = benchmark_result.append(row, ignore_index=True)

    # various output rows
    cols, rows = 100, 10000
    columns = ["Col_{}".format(i) for i in range(cols)]
    input_domain = dict.fromkeys(columns, SparkIntegerColumnDescriptor())
    schema = StructType(
        [StructField("Col_{}".format(i), IntegerType(), True) for i in range(cols)]
    )
    sdf = spark.createDataFrame(  # pylint: disable=no-member
        spark.sparkContext.parallelize([tuple(range(cols))] * rows), schema=schema
    )
    for max_num_rows in [1, 10, 50]:
        trans_func = lambda x: [x] * max_num_rows
        augment = False
        running_time = evaluate_runtime(
            input_domain=SparkRowDomain(input_domain),
            output_domain=ListDomain(SparkRowDomain(input_domain)),
            trusted_f=trans_func,
            augment=augment,
            max_num_rows=max_num_rows,
            sdf=sdf,
        )
        row = {
            "Row Number": rows,
            "Column Number": cols,
            "Transform Function": "duplicates",
            "Augment Flag": augment,
            "Max Num Rows": max_num_rows,
            "Running Time (s)": running_time,
        }
        benchmark_result = benchmark_result.append(row, ignore_index=True)

        augment = True
        trans_func = lambda row: [
            {f"{key}_copy": value for key, value in row.asDict().items()}
        ]
        output_domain = input_domain.copy()
        for i in range(cols):
            output_domain[f"Col_{i}_copy"] = SparkIntegerColumnDescriptor()
        running_time = evaluate_runtime(
            input_domain=SparkRowDomain(input_domain),
            output_domain=ListDomain(SparkRowDomain(output_domain)),
            trusted_f=trans_func,
            augment=augment,
            max_num_rows=max_num_rows,
            sdf=sdf,
        )
        row = {
            "Row Number": rows,
            "Column Number": cols,
            "Transform Function": "duplicates",
            "Augment Flag": augment,
            "Max Num Rows": max_num_rows,
            "Running Time (s)": running_time,
        }
        benchmark_result = benchmark_result.append(row, ignore_index=True)

    # various transform functions
    for times in [10, 1000, 100000, 1000000]:

        def my_map(row):
            count = 0
            for i in range(times):
                count += 1
            return row.asDict()

        input_domain = {
            "A": SparkIntegerColumnDescriptor(),
            "B": SparkIntegerColumnDescriptor(),
        }
        schema = StructType(
            [
                StructField("A", IntegerType(), True),
                StructField("B", IntegerType(), True),
            ]
        )
        sdf = spark.createDataFrame(  # pylint: disable=no-member
            spark.sparkContext.parallelize([(i, randint(0, 1)) for i in range(10000)]),
            schema=schema,
        )
        augment = False
        trans_func = lambda row: [my_map(row)] * 5
        running_time = evaluate_runtime(
            input_domain=SparkRowDomain(input_domain),
            output_domain=ListDomain(SparkRowDomain(input_domain)),
            trusted_f=trans_func,
            augment=augment,
            max_num_rows=2,
            sdf=sdf,
        )
        row = {
            "Row Number": 10000,
            "Column Number": 2,
            "Transform Function": f"count_times_{times}",
            "Augment Flag": augment,
            "Max Num Rows": 5,
            "Running Time (s)": running_time,
        }
        benchmark_result = benchmark_result.append(row, ignore_index=True)

    spark.stop()
    write_as_html(benchmark_result, "sparkflatmap.html")


if __name__ == "__main__":
    main()
