"""Benchmarking script for Map."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

import time
from random import randint
from typing import Callable

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import IntegerType, StructField, StructType

from tmlt.core.domains.collections import ListDomain
from tmlt.core.domains.spark_domains import (
    SparkIntegerColumnDescriptor,
    SparkRowDomain,
)
from tmlt.core.metrics import SymmetricDifference
from tmlt.core.transformations.spark_transformations.map import (
    FlatMap,
    Map,
    RowToRowsTransformation,
    RowToRowTransformation,
)

from benchmarking_utils import write_as_html


def evaluate_runtime(
    input_domain: SparkRowDomain,
    output_domain: SparkRowDomain,
    trusted_f: Callable,
    augment: bool,
    flat_map: bool,
    sdf: DataFrame,
) -> float:
    """Returns the runtimes."""
    start = time.time()
    if flat_map:
        output_domain = ListDomain(output_domain)
        transformer = RowToRowsTransformation(
            input_domain=input_domain,
            output_domain=output_domain,
            trusted_f=trusted_f,
            augment=augment,
        )
        transformation = FlatMap(
            row_transformer=transformer, max_num_rows=1, metric=SymmetricDifference()
        )
    else:
        transformation = Map(
            row_transformer=RowToRowTransformation(
                input_domain=input_domain,
                output_domain=output_domain,
                trusted_f=trusted_f,
                augment=augment,
            ),
            metric=SymmetricDifference(),
        )
    _ = transformation(sdf).toPandas()
    running_time = time.time() - start
    return round(running_time, 3)


def main():
    """Evaluate running time for Map/FlatMap with different settings.

    These settings include number of rows, number of columns, transform functions, and
    augment flag (True vs False).
    """
    spark = SparkSession.builder.getOrCreate()
    benchmark_result = pd.DataFrame(
        [],
        columns=[
            "Map",
            "Row Number",
            "Column Number",
            "Transform Function",
            "Augment Flag",
            "Running Time (s)",
        ],
    )
    input_domain = {
        "A": SparkIntegerColumnDescriptor(),
        "B": SparkIntegerColumnDescriptor(),
    }
    output_domain = {
        "A": SparkIntegerColumnDescriptor(),
        "B": SparkIntegerColumnDescriptor(),
        "C": SparkIntegerColumnDescriptor(),
    }
    schema = StructType(
        [StructField("A", IntegerType(), True), StructField("B", IntegerType(), True)]
    )
    sdf = spark.createDataFrame(  # pylint: disable=no-member
        spark.sparkContext.parallelize([(i, randint(0, 1)) for i in range(1250000)]),
        schema=schema,
    )
    _ = sdf.collect()  # Help spark warm up.

    # various rows
    for size in [100, 400, 10000, 40000, 160000, 320000]:
        sdf = spark.createDataFrame(  # pylint: disable=no-member
            spark.sparkContext.parallelize([(i, randint(0, 1)) for i in range(size)]),
            schema=schema,
        )
        augment = False
        for flat_map in [True, False]:
            map_func = lambda row: {"A": row["A"] * 2, "B": row["B"]}
            if flat_map:
                map_func = lambda row: [{"A": row["A"] * 2, "B": row["B"]}]
            running_time = evaluate_runtime(
                input_domain=SparkRowDomain(input_domain),
                output_domain=SparkRowDomain(input_domain),
                trusted_f=map_func,
                augment=augment,
                flat_map=flat_map,
                sdf=sdf,
            )
            row = {
                "Map": "FlatMap" if flat_map else "Map",
                "Row Number": size,
                "Column Number": 2,
                "Transform Function": "times_two",
                "Augment Flag": augment,
                "Running Time (s)": running_time,
            }
            benchmark_result = benchmark_result.append(row, ignore_index=True)

        augment = True
        for flat_map in [True, False]:
            map_func = lambda row: {"C": row["A"] * 2}
            if flat_map:
                map_func = lambda row: [{"C": row["A"] * 2}]
            running_time = evaluate_runtime(
                input_domain=SparkRowDomain(input_domain),
                output_domain=SparkRowDomain(output_domain),
                trusted_f=map_func,
                augment=augment,
                flat_map=flat_map,
                sdf=sdf,
            )
            row = {
                "Map": "FlatMap" if flat_map else "Map",
                "Row Number": size,
                "Column Number": 2,
                "Transform Function": "times_two",
                "Augment Flag": augment,
                "Running Time (s)": running_time,
            }
            benchmark_result = benchmark_result.append(row, ignore_index=True)

    # various columns
    for size in [100, 400, 800, 1600]:
        columns = [f"Col_{i}" for i in range(size)]
        input_domains = dict.fromkeys(columns, SparkIntegerColumnDescriptor())
        output_domains = input_domains.copy()
        output_domains["Col_X"] = SparkIntegerColumnDescriptor()
        schema = StructType(
            [StructField(f"Col_{i}", IntegerType(), True) for i in range(size)]
        )
        sdf = spark.createDataFrame(  # pylint: disable=no-member
            spark.sparkContext.parallelize([tuple(range(size))] * 10000), schema=schema
        )
        augment = False
        for flat_map in [True, False]:
            map_func = lambda row: {
                f"Col_{i}": row["Col_1"] * 2 if i == 1 else row[f"Col_{i}"]
                for i in range(size)
            }
            if flat_map:
                map_func = lambda row: [
                    {
                        f"Col_{i}": row["Col_1"] * 2 if i == 1 else row[f"Col_{i}"]
                        for i in range(size)
                    }
                ]
            running_time = evaluate_runtime(
                input_domain=SparkRowDomain(input_domains),
                output_domain=SparkRowDomain(input_domains),
                trusted_f=map_func,
                augment=augment,
                flat_map=flat_map,
                sdf=sdf,
            )
            row = {
                "Map": "FlatMap" if flat_map else "Map",
                "Row Number": 10000,
                "Column Number": size,
                "Transform Function": "times_two",
                "Augment Flag": augment,
                "Running Time (s)": running_time,
            }
            benchmark_result = benchmark_result.append(row, ignore_index=True)

        augment = True
        for flat_map in [True, False]:
            map_func = lambda row: {"Col_X": row["Col_11"] * 2}
            if flat_map:
                map_func = lambda row: [{"Col_X": row["Col_11"] * 2}]
            running_time = evaluate_runtime(
                input_domain=SparkRowDomain(input_domains),
                output_domain=SparkRowDomain(output_domains),
                trusted_f=map_func,
                augment=augment,
                flat_map=flat_map,
                sdf=sdf,
            )
            row = {
                "Map": "FlatMap" if flat_map else "Map",
                "Row Number": 10000,
                "Column Number": size,
                "Transform Function": "times_two",
                "Augment Flag": augment,
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
        for flat_map in [True, False]:
            map_func = lambda row: my_map(row)
            if flat_map:
                map_func = lambda row: [my_map(row)]
            running_time = evaluate_runtime(
                input_domain=SparkRowDomain(input_domain),
                output_domain=SparkRowDomain(input_domain),
                trusted_f=map_func,
                augment=augment,
                flat_map=flat_map,
                sdf=sdf,
            )
            row = {
                "Map": "FlatMap" if flat_map else "Map",
                "Row Number": 10000,
                "Column Number": 2,
                "Transform Function": f"count_times_{times}",
                "Augment Flag": augment,
                "Running Time (s)": running_time,
            }
            benchmark_result = benchmark_result.append(row, ignore_index=True)

    spark.stop()
    write_as_html(benchmark_result, "sparkmap.html")


if __name__ == "__main__":
    main()
