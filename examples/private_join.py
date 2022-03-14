"""Example illustrating private joins."""

# <placeholder: boilerplate>

from functools import partial
from typing import Type

from pyspark.sql import DataFrame, SparkSession

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
from tmlt.core.utils.testing import PySparkTest


def perform_join(
    TruncationStrategy: Type,
    tau_1: int,
    tau_2: int,
    left_domain: SparkDataFrameDomain,
    right_domain: SparkDataFrameDomain,
    left_df: DataFrame,
    right_df: DataFrame,
):
    """Perform join after truncation as specified."""
    left_truncator = TruncationStrategy(domain=left_domain, keys=["Y"], threshold=tau_1)
    right_truncator = TruncationStrategy(
        domain=right_domain, keys=["Y"], threshold=tau_2
    )

    private_join = PrivateJoin(
        input_domain=DictDomain({"left": left_domain, "right": right_domain}),
        left="left",
        right="right",
        left_truncator=left_truncator,
        right_truncator=right_truncator,
        join_cols=["Y"],
    )
    private_join({"left": left_df, "right": right_df}).show()


def example():
    """Compute private joins using different thresholds."""
    spark = SparkSession.builder.getOrCreate()

    # Declare domains for left and right tables.
    left_domain = SparkDataFrameDomain(
        {"X": SparkIntegerColumnDescriptor(), "Y": SparkStringColumnDescriptor()}
    )
    right_domain = SparkDataFrameDomain(
        {"Y": SparkStringColumnDescriptor(), "Z": SparkIntegerColumnDescriptor()}
    )

    # Set up data for joining.
    left_df = spark.createDataFrame(
        [(1, "A"), (1, "B"), (2, "B"), (1, "C"), (2, "C"), (3, "C")], schema=["X", "Y"]
    )
    right_df = spark.createDataFrame(
        [("A", 4), ("A", 5), ("B", 4), ("C", 4), ("C", 5)], schema=["Y", "Z"]
    )

    print("Left DataFrame:")
    left_df.show()
    print("Right DataFrame:")
    right_df.show()

    # Helper function to avoid repetition.
    run_example_join = partial(
        perform_join,
        left_df=left_df,
        right_df=right_df,
        left_domain=left_domain,
        right_domain=right_domain,
    )

    # Example 1: Truncate both tables with threshold = 1 with DropAllTruncation.
    # Since keys B and C have multiplicity > 1 in the left table, all records
    # with Y=B or Y=C are dropped from left table. Similarly, all records with
    # Y=A or Y=C are dropped from the right table. Consequently, the output is empty.
    print("Private Join with threshold = 1 for both tables with DropAll truncation.")
    run_example_join(TruncationStrategy=DropAllTruncation, tau_1=1, tau_2=1)

    # Example 2: Truncate both tables with threshold = 1 with HashTopKTruncation.
    # Since keys B and C have multiplicity > 1 in the left table, one row with Y=B
    # and two rows with Y=C are discarded from the left table for the join. Similarly,
    # one record with Y=A and one record with Y=C are discarded from the right table.
    # Join output contains exactly one row for each key.
    print("Private Join with threshold = 1 for both tables with Random truncation.")
    run_example_join(TruncationStrategy=HashTopKTruncation, tau_1=1, tau_2=1)

    # Example 3: Truncate both tables with left threshold = 2 and right threshold = 1
    # with DropAllTruncation. Records with Y=C are dropped from the left table and
    # records with Y=A or Y=C are dropped from the right table. For all records in the
    # output, Y=B.
    print(
        "Private Join with left threshold = 2 and right threshold = 1 with DropAll"
        " truncation."
    )
    run_example_join(TruncationStrategy=DropAllTruncation, tau_1=2, tau_2=1)

    # Example 4: Truncate both tables with left threshold = 2 and right threshold = 1
    # with Random truncation. 1 random record with Y=C is discarded from the left table
    # Keys A and C have multiplicity > 1 in the right table so, one record with Y=A and
    # one record with Y=C are randomly discarded from the right table.
    print(
        "Private Join with left threshold = 2 and right threshold = 1 with Random"
        " truncation."
    )
    run_example_join(TruncationStrategy=HashTopKTruncation, tau_1=2, tau_2=1)

    # Example 5: Truncate both tables with left threshold = 3 and right threshold = 2
    # with Random truncation. Neither tables require truncation, so are joined as
    # provided.
    print(
        "Private Join with left threshold = 3 and right threshold = 2 with Random"
        " truncation."
    )
    run_example_join(TruncationStrategy=HashTopKTruncation, tau_1=3, tau_2=2)


def main():
    """Compute private joins using different thresholds."""
    PySparkTest.setUpClass()
    example()
    PySparkTest.tearDownClass()


if __name__ == "__main__":
    main()
