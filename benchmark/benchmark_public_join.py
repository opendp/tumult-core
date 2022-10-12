"""Benchmarking module for PublicJoin."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2022

import time
from random import randint
from typing import List, Optional, Tuple, Union

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import lit
from benchmarking_utils import write_as_html
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
)
from tmlt.core.metrics import IfGroupedBy, SumOf, SymmetricDifference
from tmlt.core.transformations.spark_transformations.join import PublicJoin


class BenchmarkSparkPublicJoin:
    """Benchmarking for PublicJoin"""

    def __init__(self):
        self.spark = SparkSession.builder.getOrCreate()
        self.benchmark_result = pd.DataFrame(
            [],
            columns=[
                "Rows in Private Table",
                "Rows in Public Table",
                "Columns in Private Table",
                "Columns in Public Table",
                "Join Columns",
                "Domain Size",
                "Transform Time (s)",
                "Running Time (s)",
            ],
        )
        input_domain = SparkDataFrameDomain(
            {"A": SparkIntegerColumnDescriptor(), "X": SparkIntegerColumnDescriptor()}
        )
        empty_df = self.spark.createDataFrame([], schema=input_domain.spark_schema)
        _ = empty_df.collect()  # Help spark warm up.
        self.schema = {
            "A": SparkFloatColumnDescriptor(),
            "B": SparkIntegerColumnDescriptor(),
        }

    def __call__(
        self,
        rows_public: Optional[List[int]] = None,
        rows_private: Optional[List[int]] = None,
        columns_public: Optional[List[int]] = None,
        columns_private: Optional[List[int]] = None,
        join_columns: Optional[List[int]] = None,
        domain_sizes: Optional[List[int]] = None,
    ) -> None:
        """Perform benchmarking.

        Args:
            rows_public: list of row numbers in public table, default is None.
            rows_private: list of row numbers in private table, default is None.
            columns_public: list of column numbers in public table, default is None.
            columns_private: list of column numbers in private table, default is None.
            join_columns: list of join column numbers, default is None.
            domain_sizes: numbers for unique values in join column, default is None.
        """
        # various rows in public table.
        if rows_public:
            domain_size = 2
            rows_in_private = 100
            input_domain = SparkDataFrameDomain(schema=self.schema)
            private_df = self.spark.createDataFrame(  # pylint: disable=no-member
                pd.DataFrame(
                    [[1.2, i] for i in range(domain_size)]
                    * int(rows_in_private / domain_size),
                    columns=["A", "B"],
                )
            )
            for rows in rows_public:
                public_df = pd.DataFrame(
                    [[i, 2.2] for i in range(domain_size)] * int(rows / domain_size),
                    columns=["B", "C"],
                )
                metric = IfGroupedBy("B", SumOf(SymmetricDifference()))
                join_cols = ["B"]
                transform_time, running_time = self.evaluate_runtime(
                    input_domain=input_domain,
                    public_df=self.spark.createDataFrame(public_df),
                    private_df=private_df,
                    metric=metric,
                    join_cols=join_cols,
                )
                row = {
                    "Rows in Private Table": rows_in_private,
                    "Rows in Public Table": rows,
                    "Columns in Private Table": 2,
                    "Columns in Public Table": 2,
                    "Join Columns": 1,
                    "Domain Size": domain_size,
                    "Transform Time (s)": transform_time,
                    "Running Time (s)": running_time,
                }
                self.benchmark_result = self.benchmark_result.append(
                    row, ignore_index=True
                )

        # various rows in private table.
        if rows_private:
            domain_size = 2
            rows_in_public = 100
            input_domain = SparkDataFrameDomain(schema=self.schema)
            public_df = pd.DataFrame(
                [[i, 2.2] for i in range(domain_size)]
                * int(rows_in_public / domain_size),
                columns=["B", "C"],
            )
            for rows in rows_private:
                private_df = self.spark.createDataFrame(  # pylint: disable=no-member
                    pd.DataFrame(
                        [[1.2, i] for i in range(domain_size)]
                        * int(rows / domain_size),
                        columns=["A", "B"],
                    )
                )
                metric = IfGroupedBy("B", SumOf(SymmetricDifference()))
                join_cols = ["B"]
                transform_time, running_time = self.evaluate_runtime(
                    input_domain=input_domain,
                    public_df=self.spark.createDataFrame(public_df),
                    private_df=private_df,
                    metric=metric,
                    join_cols=join_cols,
                )
                row = {
                    "Rows in Private Table": rows,
                    "Rows in Public Table": rows_in_public,
                    "Columns in Private Table": 2,
                    "Columns in Public Table": 2,
                    "Join Columns": 1,
                    "Domain Size": domain_size,
                    "Transform Time (s)": transform_time,
                    "Running Time (s)": running_time,
                }
                self.benchmark_result = self.benchmark_result.append(
                    row, ignore_index=True
                )

        # various columns in public table.
        if columns_public:
            domain_size = 2
            rows_in_public = 10000
            rows_in_private = 100
            input_domain = SparkDataFrameDomain(schema=self.schema)
            private_df = self.spark.createDataFrame(  # pylint: disable=no-member
                pd.DataFrame(
                    [[10.0, i] for i in range(domain_size)]
                    * int(rows_in_private / domain_size),
                    columns=["A", "B"],
                )
            )
            for cols in columns_public:
                columns = [f"Col_{i}" for i in range(cols)]
                public_df = pd.DataFrame(
                    [tuple(range(cols))] * rows_in_public, columns=columns
                )
                public_df["B"] = [randint(0, 1) for i in range(rows_in_public)]
                metric = IfGroupedBy("B", SumOf(SymmetricDifference()))
                join_cols = ["B"]
                transform_time, running_time = self.evaluate_runtime(
                    input_domain=input_domain,
                    public_df=self.spark.createDataFrame(public_df),
                    private_df=private_df,
                    metric=metric,
                    join_cols=join_cols,
                )
                row = {
                    "Rows in Private Table": rows_in_private,
                    "Rows in Public Table": rows_in_public,
                    "Columns in Private Table": 2,
                    "Columns in Public Table": cols,
                    "Join Columns": 1,
                    "Domain Size": domain_size,
                    "Transform Time (s)": transform_time,
                    "Running Time (s)": running_time,
                }
                self.benchmark_result = self.benchmark_result.append(
                    row, ignore_index=True
                )

        # various columns in private table.
        if columns_private:
            domain_size = 2
            rows_in_public = 100
            rows_in_private = 10000
            public_df = pd.DataFrame(
                [[i, 10.0] for i in range(domain_size)]
                * int(rows_in_public / domain_size),
                columns=["B", "C"],
            )
            for cols in columns_private:
                schema = {f"Col_{i}": SparkFloatColumnDescriptor() for i in range(cols)}
                private_df = self.spark.createDataFrame(  # pylint: disable=no-member
                    pd.DataFrame(
                        [tuple(range(cols))] * rows_in_private, columns=schema.keys()
                    )
                )
                private_df = private_df.withColumn("B", lit(randint(0, 1)))
                schema["B"] = SparkIntegerColumnDescriptor()
                input_domain = SparkDataFrameDomain(schema=schema)
                metric = IfGroupedBy("B", SumOf(SymmetricDifference()))
                join_cols = ["B"]
                transform_time, running_time = self.evaluate_runtime(
                    input_domain=input_domain,
                    public_df=self.spark.createDataFrame(public_df),
                    private_df=private_df,
                    metric=metric,
                    join_cols=join_cols,
                )
                row = {
                    "Rows in Private Table": rows_in_private,
                    "Rows in Public Table": rows_in_public,
                    "Columns in Private Table": cols,
                    "Columns in Public Table": 2,
                    "Join Columns": 1,
                    "Domain Size": domain_size,
                    "Transform Time (s)": transform_time,
                    "Running Time (s)": running_time,
                }
                self.benchmark_result = self.benchmark_result.append(
                    row, ignore_index=True
                )

        # various join columns
        if join_columns:
            rows = 1000
            domain_size = 2
            for num_cols in join_columns:
                columns = [f"Col_{i}" for i in range(num_cols)]
                data = [[str(randint(0, 1)) for i in range(num_cols)]] * rows
                public_df = pd.DataFrame(data, columns=columns)
                public_df["A"] = "A"
                schema = {
                    f"Col_{i}": SparkStringColumnDescriptor() for i in range(num_cols)
                }
                private_df = self.spark.createDataFrame(  # pylint: disable=no-member
                    pd.DataFrame(data, columns=columns)
                )
                private_df = private_df.withColumn("B", lit("B"))
                schema["B"] = SparkStringColumnDescriptor()
                input_domain = SparkDataFrameDomain(schema=schema)
                metric = SymmetricDifference()
                transform_time, running_time = self.evaluate_runtime(
                    input_domain=input_domain,
                    public_df=self.spark.createDataFrame(public_df),
                    private_df=private_df,
                    metric=metric,
                    join_cols=columns,
                )
                row = {
                    "Rows in Private Table": rows,
                    "Rows in Public Table": rows,
                    "Columns in Private Table": num_cols,
                    "Columns in Public Table": num_cols,
                    "Join Columns": num_cols,
                    "Domain Size": domain_size,
                    "Transform Time (s)": transform_time,
                    "Running Time (s)": running_time,
                }
                self.benchmark_result = self.benchmark_result.append(
                    row, ignore_index=True
                )

        # various domain sizes
        if domain_sizes:
            rows, join_columns = 4000, 1
            input_domain = SparkDataFrameDomain(schema=self.schema)
            for size in domain_sizes:
                private_df = self.spark.createDataFrame(  # pylint: disable=no-member
                    pd.DataFrame(
                        [[10.0, i] for i in range(size)] * int(rows / size),
                        columns=["A", "B"],
                    )
                )
                public_df = pd.DataFrame(
                    [[i, 100.0] for i in range(size)] * int(rows / size),
                    columns=["B", "C"],
                )
                metric = SymmetricDifference()
                join_cols = ["B"]
                transform_time, running_time = self.evaluate_runtime(
                    input_domain=input_domain,
                    public_df=self.spark.createDataFrame(public_df),
                    private_df=private_df,
                    metric=metric,
                    join_cols=join_cols,
                )
                row = {
                    "Rows in Private Table": rows,
                    "Rows in Public Table": rows,
                    "Columns in Private Table": 2,
                    "Columns in Public Table": 2,
                    "Join Columns": join_columns,
                    "Domain Size": size,
                    "Transform Time (s)": transform_time,
                    "Running Time (s)": running_time,
                }
                self.benchmark_result = self.benchmark_result.append(
                    row, ignore_index=True
                )

    @staticmethod
    def evaluate_runtime(
        input_domain: SparkDataFrameDomain,
        public_df: DataFrame,
        private_df: DataFrame,
        metric: Union[SymmetricDifference, IfGroupedBy] = SymmetricDifference(),
        join_cols: Optional[List[str]] = None,
    ) -> Tuple[float, float]:
        """Evaluate public join runtime with various params. Return a runtime in seconds.

        Args:
            input_domain: Domain of input DataFrames.
            public_df: Public DataFrame to join with.
            private_df: Private DataFrame to join public DataFrame with.
            metric: Metric for input and output DataFrames.
            join_cols: Names of columns to join on. If None, natural join is performed.
        """
        start = time.time()
        public_join_transformation = PublicJoin(
            input_domain=input_domain,
            public_df=public_df,
            metric=metric,
            join_cols=join_cols,
        )
        transform_time = time.time() - start
        _ = public_join_transformation(private_df).toPandas()
        running_time = time.time() - start
        return round(transform_time, 3), round(running_time, 3)

    def write_result(self):
        self.spark.stop()
        write_as_html(self.benchmark_result, "public_join.html")


def main():
    """Evaluate running time for PublicJoin with different settings.

    These settings include various numbers of join columns, rows in private table and
    public table, columns in private table and public table.
    """
    benchmark = BenchmarkSparkPublicJoin()
    rows_public = [1000, 4000, 16000, 64000, 128000]
    benchmark(rows_public=rows_public)
    rows_private = [1000, 4000, 16000, 64000, 128000]
    benchmark(rows_private=rows_private)
    columns_public = [2, 10, 20, 40, 80, 120]
    benchmark(columns_public=columns_public)
    columns_private = [2, 10, 20, 40, 80, 120]
    benchmark(columns_private=columns_private)
    join_columns = [2, 4, 8, 12, 14]
    benchmark(join_columns=join_columns)
    domain_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 384, 512]
    benchmark(domain_sizes=domain_sizes)
    benchmark.write_result()


if __name__ == "__main__":
    main()
