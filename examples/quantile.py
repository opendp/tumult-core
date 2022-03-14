"""Example illustrating quantiles."""

# <placeholder: boilerplate>

import pandas as pd
from pyspark.sql import SparkSession

from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
)
from tmlt.core.measurements.aggregations import create_quantile_measurement
from tmlt.core.measures import PureDP
from tmlt.core.metrics import SumOf, SymmetricDifference
from tmlt.core.transformations.spark_transformations.groupby import (
    create_groupby_from_column_domains,
)


def main():
    """Main function."""
    spark = SparkSession.builder.config(
        "spark.ui.showConsoleProgress", "false"
    ).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    sdf = spark.createDataFrame(  # pylint: disable=no-member
        pd.DataFrame(
            [
                ["F", 28],
                ["F", 26],
                ["F", 27],
                ["M", 23],
                ["F", 29],
                ["M", 22],
                ["M", 24],
                ["M", 25],
            ],
            columns=["Sex", "Age"],
        )
    )

    print("Dataframe:")
    sdf.show()

    groupby = create_groupby_from_column_domains(
        input_domain=SparkDataFrameDomain(
            {
                "Sex": SparkStringColumnDescriptor(),
                "Age": SparkIntegerColumnDescriptor(),
            }
        ),
        input_metric=SymmetricDifference(),
        output_metric=SumOf(SymmetricDifference()),
        column_domains={"Sex": ["M", "F"]},
    )
    groupby_quantile = create_quantile_measurement(
        input_domain=groupby.input_domain,
        input_metric=groupby.input_metric,
        output_measure=PureDP(),
        measure_column="Age",
        quantile=0.5,
        lower=22,
        upper=29,
        d_out=1,
        groupby_transformation=groupby,
        quantile_column="Noisy Median Age",
    )

    print("Output:")
    groupby_quantile(sdf).show()

    # We can also aggregate on the entire dataframe.
    total_quantile = create_quantile_measurement(
        input_domain=groupby.input_domain,
        input_metric=groupby.input_metric,
        output_measure=PureDP(),
        measure_column="Age",
        quantile=0.5,
        lower=22,
        upper=29,
        d_out=1,
        groupby_transformation=None,
    )
    print()
    print(f"Output: {total_quantile(sdf)}")


if __name__ == "__main__":
    main()
