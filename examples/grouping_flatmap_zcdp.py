"""Example illustrating GroupingFlatMap under zCDP."""

# <placeholder: boilerplate>

import pandas as pd
from pyspark.sql import SparkSession

from tmlt.core.domains.collections import ListDomain
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkRowDomain,
    SparkStringColumnDescriptor,
)
from tmlt.core.measurements.aggregations import (
    NoiseMechanism,
    create_count_measurement,
)
from tmlt.core.measurements.composition import (
    ParallelComposition,
    unpack_parallel_composition_queryable,
)
from tmlt.core.measures import RhoZCDP
from tmlt.core.metrics import IfGroupedBy, RootSumOfSquared, SymmetricDifference
from tmlt.core.transformations.spark_transformations.groupby import (
    create_groupby_from_column_domains,
)
from tmlt.core.transformations.spark_transformations.map import (
    GroupingFlatMap,
    Map,
    RowToRowsTransformation,
    RowToRowTransformation,
)
from tmlt.core.transformations.spark_transformations.partition import PartitionByKeys
from tmlt.core.utils.testing import PySparkTest


def main():
    """Illustrate private analysis under zCDP.

    An overview of this example:
        - Private Table:
            JOB_TITLE         | SEX
            ------------------|-----
            CTO               | M
            Project Manager   | F
            Software Engineer | F
            Software Engineer | M
            Office Manager    | M

        - First, we map each row to one of more roles based on the JOB_TITLE.
          In particular we use the following map:
            * CTO -> executive, engineer
            * Project Manager -> manager, engineer
            * Software Engineer -> engineer
            * Office Manager -> manager

        - Analysis:
            - We compute the following DP counts:
                - Total counts for managerial or executive roles
                - Sex marginal for engineering roles
        * Note that a record in the private table may contribute to up to 2 counts
          since a job title may be associated with up to 2 roles.
    """
    PySparkTest.setUpClass()
    # Using PySparkTest so spark configs are set appropriately
    # and temporary tables are cleared at the end.
    spark = SparkSession.builder.getOrCreate()

    # Set up private df.
    sdf = spark.createDataFrame(
        pd.DataFrame(
            [
                ["CTO", "M"],
                ["Project Manager", "M"],
                ["Software Engineer", "F"],
                ["Office Manager", "F"],
            ],
            columns=["JOB_TITLE", "SEX"],
        )
    )
    schema = {
        "JOB_TITLE": SparkStringColumnDescriptor(),
        "SEX": SparkStringColumnDescriptor(),
    }
    print("Private data:")
    sdf.show()

    # Set up and run a grouping flat map.
    schema_after_flatmap = {**schema, "ROLE": SparkStringColumnDescriptor()}
    job_title_to_roles = {
        "CTO": ["engineer", "executive"],
        "Project Manager": ["engineer", "manager"],
        "Software Engineer": ["engineer"],
        "Office Manager": ["manager"],
    }
    row_transformer = RowToRowsTransformation(
        input_domain=SparkRowDomain(schema),
        output_domain=ListDomain(SparkRowDomain(schema_after_flatmap)),
        trusted_f=lambda x: [
            {"ROLE": role} for role in job_title_to_roles[x["JOB_TITLE"]]
        ],
        augment=True,
    )
    # By constructing a GroupingFlatMap (instead of a regular FlatMap),
    # we specify our intention to group by the column produced by this flat map.
    grouping_flatmap = GroupingFlatMap(
        output_metric=RootSumOfSquared(SymmetricDifference()),
        row_transformer=row_transformer,
        max_num_rows=2,
    )
    # The stability of GroupingFlatMap is the sqrt(2) - with FlatMap it
    # would be 2 instead.
    assert grouping_flatmap.stability_function(1) == "sqrt(2)"
    print("Mapped df:")
    grouping_flatmap(sdf).show()

    # Since we want to compute the total counts for managerial and executive roles
    # and sex marginals for engineering roles, we map each record to a COUNT_TYPE.

    role_to_count_type = {
        "engineer": "sex marginal",
        "manager": "total",
        "executive": "total",
    }
    schema_after_map = {
        "ROLE": SparkStringColumnDescriptor(),
        "JOB_TITLE": SparkStringColumnDescriptor(),
        "SEX": SparkStringColumnDescriptor(),
        "COUNT_TYPE": SparkStringColumnDescriptor(),
    }
    map_role_to_count_type = Map(
        row_transformer=RowToRowTransformation(
            input_domain=SparkRowDomain(grouping_flatmap.output_domain.schema),
            output_domain=SparkRowDomain(schema_after_map),
            trusted_f=lambda row: {"COUNT_TYPE": role_to_count_type[row["ROLE"]]},
            augment=True,
        ),
        metric=IfGroupedBy("ROLE", RootSumOfSquared(SymmetricDifference())),
    )

    plan_upto_map = grouping_flatmap | map_role_to_count_type
    assert plan_upto_map.stability_function(1) == "sqrt(2)"
    print("df after map:")
    plan_upto_map(sdf).show()

    # Partition.
    partition = PartitionByKeys(
        input_domain=SparkDataFrameDomain(schema_after_map),
        input_metric=(IfGroupedBy("ROLE", RootSumOfSquared(SymmetricDifference()))),
        output_metric=RootSumOfSquared(
            IfGroupedBy("ROLE", RootSumOfSquared(SymmetricDifference()))
        ),
        keys=["COUNT_TYPE"],
        list_values=[("sex marginal",), ("total",)],
    )
    plan_upto_partition = plan_upto_map | partition
    assert plan_upto_partition.stability_function(1) == "sqrt(2)"
    (total_count_sdf, sex_marginal_sdf) = plan_upto_partition(sdf)
    print("Partitioned dfs:")
    total_count_sdf.show()
    sex_marginal_sdf.show()

    # Create total count measurement for managerial and executive roles.
    rho = 1
    groupby_role = create_groupby_from_column_domains(
        input_domain=SparkDataFrameDomain(schema_after_map),
        input_metric=IfGroupedBy("ROLE", RootSumOfSquared(SymmetricDifference())),
        output_metric=RootSumOfSquared(SymmetricDifference()),
        column_domains={"ROLE": ["manager", "executive"]},
    )
    groupby_role_noisy_count = create_count_measurement(
        input_domain=SparkDataFrameDomain(schema_after_map),
        input_metric=IfGroupedBy("ROLE", RootSumOfSquared(SymmetricDifference())),
        output_measure=RhoZCDP(),
        noise_mechanism=NoiseMechanism.DISCRETE_GAUSSIAN,
        d_in="sqrt(2)",
        d_out=rho,
        groupby_transformation=groupby_role,
    )
    assert groupby_role_noisy_count.privacy_function("sqrt(2)") == rho

    # Create sex marginal measurement.
    groupby_role_sex_noisy_count = create_count_measurement(
        input_domain=SparkDataFrameDomain(schema_after_map),
        input_metric=IfGroupedBy("ROLE", RootSumOfSquared(SymmetricDifference())),
        output_measure=RhoZCDP(),
        noise_mechanism=NoiseMechanism.DISCRETE_GAUSSIAN,
        d_in="sqrt(2)",
        d_out=rho,
        groupby_transformation=groupby_role,
    )
    assert groupby_role_sex_noisy_count.privacy_function("sqrt(2)") == rho

    # Create parallel measure.
    parallel_measure = ParallelComposition(
        ListDomain(SparkDataFrameDomain(schema_after_map), 2),
        input_metric=RootSumOfSquared(
            IfGroupedBy("ROLE", RootSumOfSquared(SymmetricDifference()))
        ),
        output_measure=RhoZCDP(),
        measurements=[groupby_role_noisy_count, groupby_role_sex_noisy_count],
    )
    full_plan = plan_upto_partition | parallel_measure
    assert full_plan.privacy_function(1) == rho
    print("Final df:")
    (total_only_output, sex_marginal_output) = unpack_parallel_composition_queryable(
        full_plan(sdf)
    )
    total_only_output.show()
    sex_marginal_output.show()

    PySparkTest.tearDownClass()


if __name__ == "__main__":
    main()
