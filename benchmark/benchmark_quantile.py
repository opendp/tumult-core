"""Benchmarking quantile script for the OpenDP-based privacy framework."""
# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

import time
from random import randint
from typing import Dict, List, Tuple

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame

from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkIntegerColumnDescriptor,
)
from tmlt.core.measurements.aggregations import create_quantile_measurement
from tmlt.core.measures import PureDP
from tmlt.core.metrics import SymmetricDifference
from tmlt.core.transformations.spark_transformations.groupby import (
    create_groupby_from_column_domains,
)
from tmlt.core.utils.exact_number import ExactNumberInput

from benchmarking_utils import write_as_html


def evaluate_runtime(
    df: DataFrame,
    groupby_domains: Dict,
    input_domain: SparkDataFrameDomain,
    measure_column: str,
    quantile: float,
    lower: ExactNumberInput,
    upper: ExactNumberInput,
    epsilon: ExactNumberInput,
) -> Tuple[float, pd.DataFrame]:
    """Evaluate quantile runtime with various params. Return a runtime in seconds as
    well as the resulting pandas dataframe.

    Args:
        df: The DF to run the test on.
        groupby_domains: Mapping from column names to possible values the column
            can take.
        input_domain: Domain of input DataFrames.
        measure_column: Which column to measure for quantile.
        quantile: Quantile to be computed. Must be in [0,1]. (e.g. 0.5 -> median)
        lower: Lower bound on measure values.
        upper: Upper bound on measure values.
        epsilon: The privacy budget.
    """
    start = time.time()  # start the benchmark here

    # perform the quantile calculation as per the privacy framework
    postprocessed_measurement = create_quantile_measurement(
        input_domain=input_domain,
        input_metric=SymmetricDifference(),
        output_measure=PureDP(),
        measure_column=measure_column,
        quantile=quantile,
        lower=lower,
        upper=upper,
        d_in=1,
        d_out=epsilon,
        groupby_transformation=create_groupby_from_column_domains(
            input_domain=input_domain,
            input_metric=SymmetricDifference(),
            use_l2=False,
            column_domains=groupby_domains,
        ),
        quantile_column="quantile",
    )
    ppm = postprocessed_measurement(df).toPandas()
    quantile_time = time.time() - start  # record the total time
    return quantile_time, ppm  # return the measurement time and the pandas DF


def wrap_evaluation_multiple_group_counts(
    spark: SparkSession,
    measure_column: str,
    quantile: float,
    lower: ExactNumberInput,
    upper: ExactNumberInput,
    epsilon: ExactNumberInput,
    group_size: int,
    group_counts: List[int],
    benchmark_result: pd.DataFrame,
) -> pd.DataFrame:
    # pylint: disable=unused-variable
    """Evaluate quantile runtime over multiple sizes = group_counts. Returns the
    resulting benchmarking information as a pandas dataframe.

    Args:
        spark: An existing spark session.
        measure_column: Which column to measure for quantile.
        quantile: Quantile to be computed. Must be in [0,1]. (e.g. 0.5 -> median)
        lower: Lower bound on measure values.
        upper: Upper bound on measure values.
        epsilon: The privacy budget.
        group_size: The size of a group.
        group_counts: List of the number of groups.
        benchmark_result: The benchmark results dataframe to be appended to.
    """

    # Input domain for all experiments
    input_domain = SparkDataFrameDomain(
        {"A": SparkIntegerColumnDescriptor(), "X": SparkIntegerColumnDescriptor()}
    )

    for size in group_counts:
        groupby_domains = {}
        divisor = group_size
        num_records = size
        if group_size == 0:
            groupby_domains = {"A": list(range(size))}
            num_records = 0
            divisor = 1
            df = spark.createDataFrame([], schema=input_domain.spark_schema)
            _ = df.collect()  # Help spark warm up.
        else:
            groupby_domains = {"A": list(range(int(size / group_size)))}
            df = spark.createDataFrame(  # pylint: disable=no-member
                spark.sparkContext.parallelize(
                    [
                        (i, randint(lower, upper))
                        for j in range(group_size)
                        for i in range(int(size / group_size))
                    ]
                ),
                schema=input_domain.spark_schema,
            )
        quantile_time, result_df = evaluate_runtime(
            df=df,
            groupby_domains=groupby_domains,
            input_domain=input_domain,
            measure_column=measure_column,
            quantile=quantile,
            lower=lower,
            upper=upper,
            epsilon=epsilon,
        )
        row = {
            "group_size": group_size,
            "group_count": int(size / divisor),
            "num_records": num_records,
            "quantile": quantile,
            "quantile_time (s)": quantile_time,
        }
        benchmark_result = benchmark_result.append(row, ignore_index=True)

    return benchmark_result


def benchmark_groupby_quantile(
    spark: SparkSession, quantile: float, epsilon: ExactNumberInput
) -> pd.DataFrame:
    # pylint: disable=unused-variable
    """Evaluate quantile runtime with various params. Return the resulting
    pandas dataframe.

    Args:
        spark: An existing spark session.
        quantile: Quantile to be computed. Must be in [0,1]. (e.g. 0.5 -> median)
        epsilon: The privacy budget.
    """

    # The initial DataFrame to be appended to
    benchmark_result = pd.DataFrame(
        [],
        columns=[
            "group_size",
            "group_count",
            "num_records",
            "quantile",
            "quantile_time (s)",
        ],
    )

    # Low / High Values
    low = 1
    high = 1000000

    # Empty dataframe to warm up spark
    # group size = 0, num records = 0
    # group_count = num_records / group_size (or 0 if group_size = 0)
    benchmark_result = wrap_evaluation_multiple_group_counts(
        spark=spark,
        measure_column="X",
        quantile=quantile,
        lower=low,
        upper=high,
        epsilon=epsilon,
        group_size=0,
        group_counts=[100, 400, 10000, 40000, 160000, 640000],
        benchmark_result=benchmark_result,
    )

    # Test various sizes
    # group size = 1, num records = [100, 400, 10000, 40000, 160000, 640000]
    # group_count = num_records / group_size (or 0 if group_size = 0)
    benchmark_result = wrap_evaluation_multiple_group_counts(
        spark=spark,
        measure_column="X",
        quantile=quantile,
        lower=low,
        upper=high,
        epsilon=epsilon,
        group_size=1,
        group_counts=[100, 400, 10000, 40000, 160000, 640000],
        benchmark_result=benchmark_result,
    )

    # Test various sizes
    # group size = 100000, num records = [100000, 900000, 10000000]
    # group_count = num_records / group_size (or 0 if group_size = 0)
    benchmark_result = wrap_evaluation_multiple_group_counts(
        spark=spark,
        measure_column="X",
        quantile=quantile,
        lower=low,
        upper=high,
        epsilon=epsilon,
        group_size=100000,
        group_counts=[100000, 900000, 10000000],
        benchmark_result=benchmark_result,
    )

    # Test various sizes
    # group size = 10000, num records = [10000, 90000, 1000000]
    # group_count = num_records / group_size (or 0 if group_size = 0)
    benchmark_result = wrap_evaluation_multiple_group_counts(
        spark=spark,
        measure_column="X",
        quantile=quantile,
        lower=low,
        upper=high,
        epsilon=epsilon,
        group_size=10000,
        group_counts=[10000, 90000, 1000000],
        benchmark_result=benchmark_result,
    )

    # Test various sizes
    # group size = 100, num records = [10000, 40000, 160000, 640000, 2560000]
    # group_count = num_records / group_size (or 0 if group_size = 0)
    benchmark_result = wrap_evaluation_multiple_group_counts(
        spark=spark,
        measure_column="X",
        quantile=quantile,
        lower=low,
        upper=high,
        epsilon=epsilon,
        group_size=100,
        group_counts=[10000, 40000, 160000, 640000, 2560000],
        benchmark_result=benchmark_result,
    )

    return benchmark_result


def main():
    """Evaluate quantile runtime for different group counts and sizes."""
    spark = SparkSession.builder.getOrCreate()
    benchmark_result = benchmark_groupby_quantile(spark, 0.5, 10)
    write_as_html(benchmark_result, "quantile.html")


if __name__ == "__main__":
    main()
