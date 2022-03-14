"""Example illustrating noisy sum and count aggregations."""

# <placeholder: boilerplate>
from typing import cast

import pandas as pd
import sympy as sp
from pyspark.sql import SparkSession

from tmlt.core.domains.collections import ListDomain
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkRowDomain,
    SparkStringColumnDescriptor,
)
from tmlt.core.measurements.aggregations import (
    NoiseMechanism,
    create_count_measurement,
    create_sum_measurement,
)
from tmlt.core.measurements.composition import Composition
from tmlt.core.measures import PureDP
from tmlt.core.metrics import SumOf, SymmetricDifference
from tmlt.core.transformations.spark_transformations.groupby import (
    create_groupby_from_column_domains,
)
from tmlt.core.transformations.spark_transformations.map import (
    FlatMap,
    RowToRowsTransformation,
)


def main():
    """Illustrate the privacy framework."""

    # Parameters used in the pipeline.
    duplication_factor = 2  # How many times we duplicate each row
    lower_bound = 1  # Lower clamping bound for the sum
    upper_bound = 3  # Upper clamping bound for the sum

    spark = SparkSession.builder.getOrCreate()
    # Create some fake data. In this initial dataset, we assume that each
    # individual contributes at most one record.
    df = spark.createDataFrame(
        pd.DataFrame(
            data=[
                [2.1, "X"],
                [0, "X"],
                [0, "X"],
                [0, "X"],
                [0, "X"],
                [47.3, "Y"],
                [1.5, "Y"],
                [-4, "Y"],
                [1, "W"],
            ],
            columns=["A", "B"],
        )
    )

    schema = {"A": SparkFloatColumnDescriptor(), "B": SparkStringColumnDescriptor()}

    row_domain = SparkRowDomain(schema)

    # Duplicate each record. Each record is transformed into a list of 4
    # identical records, but then, the FlatMap truncates this list to
    # only *2* records. The analyst can rely on the FlatMap parameter for
    # sensitivity calculation, ignoring pre-processing.
    row_duplicator = RowToRowsTransformation(
        input_domain=row_domain,
        output_domain=ListDomain(row_domain),
        trusted_f=lambda x: [x, x, x, x],
        augment=False,
    )
    duplicate_flat_map = FlatMap(
        metric=SymmetricDifference(),
        row_transformer=row_duplicator,
        max_num_rows=duplication_factor,
    )
    # duplicate_flat_map has a stability of 2, because adding or removing a row in the
    # input can add/remove 2 rows in the output
    assert duplicate_flat_map.stability_function(1) == 2

    # Group records by the value of column B, and in each group, counts the
    # number of records and sums the values of column A. The full list of groups is
    # indicated by the column_domains parameters: all these groups, and only
    # these groups, will appear in the output, regardless of the input data.
    groupby = create_groupby_from_column_domains(
        input_domain=cast(SparkDataFrameDomain, duplicate_flat_map.output_domain),
        input_metric=cast(SymmetricDifference, duplicate_flat_map.output_metric),
        output_metric=SumOf(
            cast(SymmetricDifference, duplicate_flat_map.output_metric)
        ),
        column_domains={"B": ["X", "Y", "Z"]},
    )

    # Create a private sum and count. So that create_x_measurement knows how much
    # noise to add, we need to tell it how different inputs to it can be, as well
    # as what privacy guarantee we want on the measurements.
    # For this, we will use d_in = the d_out from duplicate_flat_map's stability
    # function, and a d_out = 1 (this is the value of epsilon)
    noisy_groupby_sum = create_sum_measurement(
        input_domain=cast(SparkDataFrameDomain, duplicate_flat_map.output_domain),
        input_metric=cast(SymmetricDifference, duplicate_flat_map.output_metric),
        output_measure=PureDP(),
        measure_column="A",
        lower=lower_bound,
        upper=upper_bound,
        noise_mechanism=NoiseMechanism.LAPLACE,
        d_in=duplicate_flat_map.stability_function(1),
        d_out=1,
        groupby_transformation=groupby,
        sum_column="sum(A)",
    )
    noisy_groupby_count = create_count_measurement(
        input_domain=cast(SparkDataFrameDomain, duplicate_flat_map.output_domain),
        input_metric=cast(SymmetricDifference, duplicate_flat_map.output_metric),
        output_measure=PureDP(),
        noise_mechanism=NoiseMechanism.LAPLACE,
        d_in=duplicate_flat_map.stability_function(1),
        d_out=1,
        groupby_transformation=groupby,
        count_column="count",
    )

    # Now we can combine the sum and count into a single measurement, which returns
    # the sum and count as two elements in a tuple.
    noisy_groupby_sum_and_noisy_groupby_count = Composition(
        [noisy_groupby_sum, noisy_groupby_count]
    )

    # And we can combine this measurement with the duplicate flat map to create our
    # full algorithm.
    full_algorithm = duplicate_flat_map | noisy_groupby_sum_and_noisy_groupby_count

    # Now, what will be the total privacy guarantee? We wanted the noisy_groupby_sum
    # to have an epsilon of 1 if it was combined with duplicate_flat_map, and we wanted
    # the same for noisy_groupby_count. When we compose measurements, the resulting
    # epsilon is the sum of the individual epsilons, in this case 1 + 1 = 2
    assert full_algorithm.privacy_function(1) == 2

    noisy_sum, noisy_count = full_algorithm(df)
    print(f"Noisy sum :\n{noisy_sum.toPandas()}")
    print(f"Noisy count :\n{noisy_count.toPandas()}")


if __name__ == "__main__":
    main()
