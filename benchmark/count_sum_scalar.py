"""Benchmarking script for scalar spark-based count and sum aggregations."""

# SPDX-License-Identifier: Apache-2.0

import argparse
import statistics
from random import randint
from typing import Tuple

import pandas as pd
from benchmarking_utils import Timer, write_as_csv, write_as_html
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame

from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkIntegerColumnDescriptor,
)
from tmlt.core.measurements.aggregations import (
    NoiseMechanism,
    create_count_measurement,
    create_sum_measurement,
)
from tmlt.core.measures import Measure, PureDP, RhoZCDP
from tmlt.core.metrics import Metric, SymmetricDifference
from tmlt.core.utils.testing import PySparkTest


def evaluate_runtime(
    dataframe: DataFrame,
    input_domain: SparkDataFrameDomain,
    input_metric: Metric,
    output_measure: Measure,
) -> Tuple[float, float]:
    """Returns the runtimes for a count with the given parameters."""
    count = create_count_measurement(
        input_domain=input_domain,
        input_metric=input_metric,
        output_measure=output_measure,
        d_out=1,
        noise_mechanism=(
            NoiseMechanism.GEOMETRIC
            if output_measure in [PureDP(), RhoZCDP()]
            else NoiseMechanism.DISCRETE_GAUSSIAN
        ),
    )
    with Timer() as count_timer:
        count(dataframe)
    count_time = count_timer.elapsed

    sum_meas = create_sum_measurement(
        input_domain=input_domain,
        input_metric=input_metric,
        output_measure=output_measure,
        d_out=1,
        noise_mechanism=(
            NoiseMechanism.GEOMETRIC
            if output_measure in [PureDP(), RhoZCDP()]
            else NoiseMechanism.DISCRETE_GAUSSIAN
        ),
        measure_column="X",
        lower=0,
        upper=1,
    )
    with Timer() as sum_timer:
        sum_meas(dataframe)
    sum_time = sum_timer.elapsed

    return count_time, sum_time


def main(trial_name: str = ""):
    """Evaluate count and sum runtimes for different group counts and sizes."""
    spark = SparkSession.builder.getOrCreate()
    count_column = f"count_time{f'_{trial_name}' if trial_name else ''} (s)"
    sum_column = f"sum_time{f'_{trial_name}' if trial_name else ''} (s)"
    benchmark_result = pd.DataFrame(
        [],
        columns=[
            "num_records",
            count_column,
            sum_column,
        ],
    )
    input_domain = SparkDataFrameDomain(
        {"A": SparkIntegerColumnDescriptor(), "X": SparkIntegerColumnDescriptor()}
    )
    input_metric = SymmetricDifference()
    output_measure = PureDP()

    for num_rows in [0, 1000, 1_000_000, 10_000_000]:
        df = spark.createDataFrame(
            spark.sparkContext.parallelize(
                [(i, randint(0, 1)) for i in range(num_rows)]
            ),
            schema=input_domain.spark_schema,
        ).cache()
        count_times = []
        sum_times = []
        for _ in range(11):
            count_time, sum_time = evaluate_runtime(
                df, input_domain, input_metric, output_measure
            )
            count_times.append(count_time)
            sum_times.append(sum_time)
        # Treat first trial as a warmup
        count_times.pop(0)
        sum_times.pop(0)
        row = {
            "num_records": num_rows,
            count_column: round(statistics.mean(count_times), 3),
            sum_column: round(statistics.mean(sum_times), 3),
        }
        benchmark_result = pd.concat(
            [benchmark_result, pd.DataFrame([row])], ignore_index=True
        )

    write_as_html(
        benchmark_result,
        f"count_sum_scalar{f'_{trial_name}' if trial_name else ''}.html",
    )
    write_as_csv(
        benchmark_result,
        f"count_sum_scalar{f'_{trial_name}' if trial_name else ''}.csv",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trial_name", default="", nargs="?")
    args = parser.parse_args()
    PySparkTest.setUpClass()
    main(args.trial_name)
    PySparkTest.tearDownClass()
