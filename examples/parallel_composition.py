"""Example illustrating parallel composition."""

# <placeholder: boilerplate>

import pandas as pd
import sympy as sp
from pyspark.sql import SparkSession

from tmlt.core.domains.collections import ListDomain
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
)
from tmlt.core.measurements.aggregations import (
    NoiseMechanism,
    create_count_measurement,
    create_sum_measurement,
)
from tmlt.core.measurements.composition import (
    ParallelComposition,
    unpack_parallel_composition_queryable,
)
from tmlt.core.measures import PureDP
from tmlt.core.metrics import SumOf, SymmetricDifference
from tmlt.core.transformations.spark_transformations.partition import PartitionByKeys
from tmlt.core.utils.testing import PySparkTest

### Example Description ###

# Input Data
################
# | A | STATE |#
#  ----------- #
# | 2 |   NY  |#
# | 4 |   NY  |#
# | 5 |   NC  |#
# | 3 |   NC  |#
################

# Partition by State
#   - Compute a noisy count for NY.
#   - Compute a noisy sum of column A for NC.


def main():
    """Perform parallel composition."""
    PySparkTest.setUpClass()
    # Using PySparkTest so spark configs are set appropriately
    # and temporary tables are cleared at the end.
    spark = SparkSession.builder.getOrCreate()

    # Input DataFrame domain
    input_domain = SparkDataFrameDomain(
        {"A": SparkIntegerColumnDescriptor(), "STATE": SparkStringColumnDescriptor()}
    )

    # Set privacy budget per partition
    epsilon = sp.Integer(10)

    noisy_count_measurement = create_count_measurement(
        input_domain=input_domain,
        input_metric=SymmetricDifference(),
        output_measure=PureDP(),
        noise_mechanism=NoiseMechanism.LAPLACE,
        d_in=1,
        d_out=epsilon,
        count_column="count",
    )
    noisy_sum_measurement = create_sum_measurement(
        input_domain=input_domain,
        input_metric=SymmetricDifference(),
        output_measure=PureDP(),
        measure_column="A",
        lower=2,
        upper=5,
        noise_mechanism=NoiseMechanism.LAPLACE,
        d_in=1,
        d_out=epsilon,
        sum_column="sumA",
    )

    # Define partition transformation
    partition_transformation = PartitionByKeys(
        input_domain=input_domain,
        input_metric=SymmetricDifference(),
        output_metric=SumOf(SymmetricDifference()),
        keys=["STATE"],
        list_values=[("NY",), ("NC",)],
    )
    # Define a ParallelComposition to be chained with partition
    parallel_composition = ParallelComposition(
        input_domain=ListDomain(input_domain, 2),
        input_metric=SumOf(SymmetricDifference()),
        output_measure=PureDP(),
        measurements=[noisy_count_measurement, noisy_sum_measurement],
    )
    # Chain Partition and ParallelMeasure
    parallel_composition = partition_transformation | parallel_composition

    assert parallel_composition.privacy_function(1) == epsilon

    # Construct DataFrame
    input_dataframe = spark.createDataFrame(
        pd.DataFrame({"A": [2, 4, 5, 3], "STATE": ["NY", "NY", "NC", "NC"]})
    )
    # Compute answers
    count_NC, sum_NY = unpack_parallel_composition_queryable(
        parallel_composition(input_dataframe)
    )
    print(count_NC)
    print(sum_NY)

    PySparkTest.tearDownClass()


if __name__ == "__main__":
    main()
