"""Benchmarking module for PrivateJoin and Truncation transformations."""

# <placeholder: boilerplate>

import itertools
from random import randint
from typing import List, Tuple

import pandas as pd
from pyspark.sql import DataFrame, SparkSession

from benchmarking_utils import Timer
from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
)
from tmlt.core.transformations.spark_transformations.join import (
    DropAllTruncation,
    HashTopKTruncation,
    PrivateJoin,
)

TRUNCATIONS = {"DROPALL": DropAllTruncation, "HASHTOPK": HashTopKTruncation}


def get_phsafe_runtimes_df(persons_df: DataFrame, units_df: DataFrame) -> pd.DataFrame:
    """Runs truncated join using three truncation functions and returns result."""

    persons_dom = SparkDataFrameDomain(
        {
            "RTYPE": SparkStringColumnDescriptor(),
            "MAFID": SparkIntegerColumnDescriptor(),
            "QAGE": SparkIntegerColumnDescriptor(),
            "CENHISP": SparkIntegerColumnDescriptor(),
            "CENRACE": SparkStringColumnDescriptor(),
            "RELSHIP": SparkStringColumnDescriptor(),
        }
    )

    units_dom = SparkDataFrameDomain(
        {
            "RTYPE": SparkStringColumnDescriptor(),
            "MAFID": SparkIntegerColumnDescriptor(),
            "FINAL_POP": SparkIntegerColumnDescriptor(),
            "NPF": SparkIntegerColumnDescriptor(),
            "HHSPAN": SparkIntegerColumnDescriptor(),
            "HHRACE": SparkStringColumnDescriptor(),
            "TEN": SparkStringColumnDescriptor(),
            "HHT": SparkStringColumnDescriptor(),
            "HHT2": SparkStringColumnDescriptor(),
            "CPLT": SparkStringColumnDescriptor(),
        }
    )

    runtimes = dict()
    for truncation_mech, TruncationType in TRUNCATIONS.items():
        with Timer() as t:
            PrivateJoin(
                input_domain=DictDomain({"persons": persons_dom, "units": units_dom}),
                left="persons",
                right="units",
                left_truncator=TruncationType(
                    domain=persons_dom, keys=["MAFID"], threshold=10
                ),
                right_truncator=TruncationType(
                    domain=units_dom, keys=["MAFID"], threshold=1
                ),
                join_cols=["MAFID"],
            )({"persons": persons_df, "units": units_df}).write.saveAsTable(
                "tbl", mode="overwrite"
            )
        runtimes[truncation_mech] = f"{t.elapsed:.2f}"
    return pd.DataFrame(
        list(runtimes.items()), columns=["Truncation Type", "PHSafe Runtime(sec)"]
    )


def benchmark_phsafe():
    """Truncate at 10/1 and join persons & units tables on MAFID."""
    spark = SparkSession.builder.getOrCreate()
    persons_df = spark.createDataFrame(
        pd.read_csv(
            "s3://tumult.data.census/pih/phsafe-full-size-input/persons.csv"
        )
    )
    units_df = spark.createDataFrame(
        pd.read_csv(
            "s3://tumult.data.census/pih/phsafe-full-size-input/units.csv"
        )
    )
    return get_phsafe_runtimes_df(persons_df, units_df)


def benchmark_join(hash_truncation: bool):
    """Benchmark PrivateJoin on different cell sizes and cell counts.

    In particular, this function runs PrivateJoin (with truncation) on the following
    configurations of left and right DataFrames having 10**i rows (i=2,4 or 5):
        Configuration 1:
            - Left has groups of sizes 1, 5 and 10 (approx.)
            - Right has groups of sizes 5, 10 (approx)
            - Truncation thresholds : left=2, right=6
        Configuration 2:
            - Left has groups of sizes 10, 20 and 30 (approx.)
            - Right has groups of sizes 20 and 40 (approx).
            - Truncation thresholds : left=20, right=40
        Configuration 3:
            - Left has groups of sizes 5, 25 and 100 (approx.)
            - Right has groups of sizes 15 and 20 (approx).
            - Truncation thresholds : left=25, right=18

    * Each configuration is run under both truncation strategies (hash & dropall)
    * There are approximately the same number of groups of each size.
    * Groups are identified by values in column 'K' which contains the "group index"
        for each row, identified by integers 0,...,[GROUPSIZE].

    """
    TruncationType = HashTopKTruncation if hash_truncation else DropAllTruncation
    group_counts = [10 ** i for i in (2, 4, 5)]
    runtimes = []

    for left_group_count, right_group_count in itertools.combinations_with_replacement(
        group_counts, r=2
    ):
        for left_group_sizes, right_group_sizes, (tau1, tau2) in [
            [[1, 5, 10], [5, 10], (2, 6)],
            [[10, 20, 30], [20, 40], (20, 40)],
            [[5, 25, 100], [15, 20], (25, 18)],
        ]:
            left_df, left_dom, left_table_size = generate_dataframe(
                group_sizes=left_group_sizes,
                group_count=left_group_count,
                num_cols=3,
                column_prefix="B",
            )
            right_df, right_dom, right_table_size = generate_dataframe(
                group_sizes=right_group_sizes,
                group_count=right_group_count,
                num_cols=5,
                column_prefix="C",
            )
            left_table_size, right_table_size = left_df.count(), right_df.count()
            join_transformation = PrivateJoin(
                input_domain=DictDomain({"left": left_dom, "right": right_dom}),
                left="left",
                right="right",
                left_truncator=TruncationType(
                    domain=left_dom, keys=["K"], threshold=tau1
                ),
                right_truncator=TruncationType(
                    domain=right_dom, keys=["K"], threshold=tau2
                ),
                join_cols=["K"],
            )
            with Timer() as t:
                joined_df = join_transformation({"left": left_df, "right": right_df})
                joined_df.write.saveAsTable("tbl", mode="overwrite")  # Materialize

            runtimes.append(
                {
                    "Left Table Size": left_table_size,
                    "Right Table Size": right_table_size,
                    "Left Group Sizes (approx.)": left_group_sizes,
                    "Right Group Sizes (approx.)": right_group_sizes,
                    "Left Group Count": left_group_count,
                    "Right Group Count": right_group_count,
                    "Left Threshold": tau1,
                    "Right Threshold": tau2,
                    "Join time(sec)": f"{t.elapsed:.2f}",
                    "# Output Rows": joined_df.count(),
                }
            )

    return pd.DataFrame.from_records(runtimes)


def benchmark_trunc():
    """Benchmark truncation functions.

    In particular, this runs truncation on DataFrames with 10**i groups (i=2, 4 or 5)
    for the following configurations
        - Configuration 1
            - Groups are of sizes 1, 5 and 10 (approximately)
            - For this configuration, truncation is benchmarked with thresholds of 1
             and 7 using both truncation strategies.
        - Configuration 2
            - Groups are of sizes 10, 20 and 50 (approximately)
            - For this configuration, truncation is benchmarked with thresholds of 10
             and 48 using both truncation strategies.
        - Configuration 3
            - Groups are of sizes 25, 30, 45, 50, 200 (approximately)
            - For this configuration, truncation is benchmarked with threshold of 40
             using both truncation strategies.

    * There are approximately the same number of groups of each size.
    """
    trunc_runtimes = []
    group_counts = [10 ** i for i in (2, 4, 5)]
    group_sizes_tau = [
        ([1, 5, 10], [1, 7]),
        ([10, 20, 50], [10, 48]),
        ([25, 30, 45, 50, 200], [40]),
    ]
    for group_count in group_counts:
        for group_sizes, taus in group_sizes_tau:
            for tau in taus:
                df, df_dom, df_size = generate_dataframe(
                    group_sizes=group_sizes, group_count=group_count
                )
                runtimes_record = {
                    "Group Sizes (approx.)": group_sizes,
                    "Group Count": group_count,
                    "Table Size (approx.)": df_size,
                    "Truncation Threshold": tau,
                }
                for truncation_mech, TruncationType in TRUNCATIONS.items():
                    truncator = TruncationType(domain=df_dom, keys=["K"], threshold=tau)
                    with Timer() as t:
                        truncated_df = truncator(df)
                        truncated_df.write.saveAsTable("tbl", mode="overwrite")
                    runtimes_record[f"{truncation_mech} time(sec)"] = f"{t.elapsed:.2f}"
                    runtimes_record[
                        f"{truncation_mech} #rows output"
                    ] = truncated_df.count()
                trunc_runtimes.append(runtimes_record)
    return pd.DataFrame.from_records(trunc_runtimes)


def generate_dataframe(
    group_sizes: List[int],
    group_count: int,
    num_cols: int = 2,
    fuzz: int = 5,
    column_prefix: str = "B",
) -> Tuple[DataFrame, SparkDataFrameDomain, int]:
    """Generates spark dataframe with specified number of groups.

    Returns tuple containing generated DataFrame, its domain and size.

    There are (almost) the same number of groups of each size specified in group_sizes.
    """
    spark = SparkSession.builder.getOrCreate()
    dom = SparkDataFrameDomain(
        {
            **{"K": SparkStringColumnDescriptor()},
            **{
                f"{column_prefix}_{i}": SparkIntegerColumnDescriptor()
                for i in range(num_cols - 1)
            },
        }
    )
    group_size_factory = itertools.cycle(group_sizes)
    data = [
        (str(i), *[randint(0, 1e6) for _ in range(num_cols - 1)])
        for i in range(group_count)
        for _ in range(next(group_size_factory) + randint(-fuzz, fuzz))
    ]
    df = spark.createDataFrame(  # pylint: disable=no-member
        spark.sparkContext.parallelize(data), schema=list(dom.schema)
    )
    return df, dom, len(data)


def main():
    """Evaluate runtimes for join and truncate components."""

    _ = (
        SparkSession.builder.master("local[*]")
        .config("spark.sql.warehouse.dir", "/tmp/hive_tables")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.executor.memory", "3g")
        .config("spark.driver.memory", "3g")
        .getOrCreate()
    )
    truncation_runtimes = benchmark_trunc().to_html()
    join_runtimes_hash_trunc = benchmark_join(hash_truncation=True).to_html()
    join_runtimes_dropall_trunc = benchmark_join(hash_truncation=False).to_html()
    phsafe_runtimes = benchmark_phsafe().to_html()

    all_tables_html = "\n".join(
        [
            "Truncation Runtimes",
            truncation_runtimes,
            "PrivateJoin with HashTopKTruncation",
            join_runtimes_hash_trunc,
            "PrivateJoin with DropAllTruncation",
            join_runtimes_dropall_trunc,
            "PHSafe Person-Household Join (~1% data)",
            phsafe_runtimes,
        ]
    )
    print(all_tables_html)


if __name__ == "__main__":
    main()
