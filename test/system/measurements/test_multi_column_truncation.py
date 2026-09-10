"""Tests full queries that truncate by multiple columns."""

from pyspark.sql import SparkSession

from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
)
from tmlt.core.measurements.aggregations import NoiseMechanism, create_count_measurement
from tmlt.core.measures import PureDP
from tmlt.core.metrics import IfGroupedBy, SumOf, SymmetricDifference
from tmlt.core.transformations.base import Transformation
from tmlt.core.transformations.spark_transformations.groupby import GroupBy
from tmlt.core.transformations.spark_transformations.truncation import (
    LimitGroupsPerID,
    LimitRowsPerGroupPerID,
)
from tmlt.core.utils.testing import assert_dataframe_equal


def test_multi_column_truncation(spark: SparkSession):
    """Tests a query with multi-column truncation on a dataset with ids."""
    input_data = spark.createDataFrame(
        [
            ("id1", "a", "a", 1),
            ("id1", "a", "a", 1),
            ("id1", "a", "b", 1),
            ("id1", "a", "b", 1),
            ("id1", "b", "a", 1),
            ("id1", "b", "b", 1),
        ],
        ["id", "group1", "group2", "value"],
    )
    group_keys = spark.createDataFrame(
        [
            ("a", "a"),
            ("a", "b"),
            ("b", "a"),
            ("b", "b"),
        ],
        ["group1", "group2"],
    )
    input_domain = SparkDataFrameDomain(
        {
            "id": SparkStringColumnDescriptor(),
            "group1": SparkStringColumnDescriptor(),
            "group2": SparkStringColumnDescriptor(),
            "value": SparkIntegerColumnDescriptor(),
        }
    )

    truncation: Transformation = LimitGroupsPerID(
        input_domain=input_domain,
        output_metric=IfGroupedBy(
            ["id"], SumOf(IfGroupedBy(["group1", "group2"], SymmetricDifference()))
        ),
        id_columns=["group1", "group2"],
        grouping_column="id",
        threshold=1,
    )
    assert isinstance(truncation.output_domain, SparkDataFrameDomain)
    assert isinstance(truncation.output_metric, IfGroupedBy)
    truncation = truncation | LimitRowsPerGroupPerID(
        input_domain=truncation.output_domain,
        input_metric=truncation.output_metric,
        id_columns=["group1", "group2"],
        grouping_column="id",
        threshold=1,
    )
    assert isinstance(truncation.output_domain, SparkDataFrameDomain)
    assert isinstance(truncation.output_metric, SymmetricDifference)
    groupby = GroupBy(
        input_domain=truncation.output_domain,
        input_metric=truncation.output_metric,
        use_l2=False,
        group_keys=group_keys,
    )

    assert isinstance(groupby.input_domain, SparkDataFrameDomain)
    assert isinstance(groupby.input_metric, SymmetricDifference)
    measurement = truncation | create_count_measurement(
        input_domain=groupby.input_domain,
        input_metric=groupby.input_metric,
        output_measure=PureDP(),
        d_out=float("inf"),
        noise_mechanism=NoiseMechanism.GEOMETRIC,
        d_in=1,
        groupby_transformation=groupby,
        count_column="count",
    )

    expected_data = spark.createDataFrame(
        [
            ("a", "a", 1),
            ("a", "b", 1),
            ("b", "a", 1),
            ("b", "b", 1),
        ],
        ["group1", "group2", "count"],
    )

    got_data = measurement(input_data)

    assert_dataframe_equal(got_data, expected_data)
