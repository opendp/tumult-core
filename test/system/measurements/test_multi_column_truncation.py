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


def test_multi_column_id_truncation(spark: SparkSession):
    """Tests a query that truncates by IDs defined by multiple columns."""
    # IDs are (household, person) pairs, so (h1, p1) and (h1, p2) are different IDs
    # even though they share a household.
    input_data = spark.createDataFrame(
        [
            ("h1", "p1", "a", 1),
            ("h1", "p1", "a", 1),
            ("h1", "p1", "b", 1),
            ("h1", "p2", "a", 1),
            ("h1", "p2", "a", 1),
            ("h1", "p2", "a", 1),
            ("h2", "p1", "b", 1),
            ("h2", "p1", "b", 1),
        ],
        ["household", "person", "group", "value"],
    )
    group_keys = spark.createDataFrame([("a",), ("b",), ("c",)], ["group"])
    input_domain = SparkDataFrameDomain(
        {
            "household": SparkStringColumnDescriptor(),
            "person": SparkStringColumnDescriptor(),
            "group": SparkStringColumnDescriptor(),
            "value": SparkIntegerColumnDescriptor(),
        }
    )

    truncation: Transformation = LimitGroupsPerID(
        input_domain=input_domain,
        output_metric=IfGroupedBy(
            ["group"],
            SumOf(IfGroupedBy(["household", "person"], SymmetricDifference())),
        ),
        id_columns=["household", "person"],
        grouping_columns=["group"],
        threshold=1,
    )
    assert isinstance(truncation.output_domain, SparkDataFrameDomain)
    assert isinstance(truncation.output_metric, IfGroupedBy)
    truncation = truncation | LimitRowsPerGroupPerID(
        input_domain=truncation.output_domain,
        input_metric=truncation.output_metric,
        id_columns=["household", "person"],
        grouping_columns=["group"],
        threshold=1,
    )
    assert isinstance(truncation.output_domain, SparkDataFrameDomain)
    assert isinstance(truncation.output_metric, SymmetricDifference)
    # Each ID contributes to at most 1 group, with at most 1 row in it.
    assert truncation.stability_function(1) == 1
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

    got_rows = measurement(input_data).collect()

    # (h1, p1) is in groups a and b, so one of them is dropped (which one depends
    # on a hash), and its remaining rows are truncated to one. (h1, p2) and
    # (h2, p1) are each in a single group, and their duplicate rows are truncated
    # to one. If only household were used as the ID, h1's two people would be
    # truncated together and the total would be 2 instead of 3.
    counts = {row["group"]: row["count"] for row in got_rows}
    assert set(counts) == {"a", "b", "c"}
    assert sum(counts.values()) == 3
    assert counts["a"] in (1, 2)
    assert counts["b"] in (1, 2)
    assert counts["c"] == 0


def test_multi_column_grouping_truncation(spark: SparkSession):
    """Tests a query that truncates groups defined by multiple columns."""
    input_data = spark.createDataFrame(
        [
            ("id1", "a", "a", 1),
            ("id1", "a", "a", 1),
            ("id1", "a", "b", 1),
            ("id1", "a", "b", 1),
            ("id1", "b", "a", 1),
            ("id1", "b", "b", 1),
            ("id2", "a", "a", 1),
            ("id2", "a", "a", 1),
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
            ["group1", "group2"], SumOf(IfGroupedBy(["id"], SymmetricDifference()))
        ),
        id_columns=["id"],
        grouping_columns=["group1", "group2"],
        threshold=2,
    )
    assert isinstance(truncation.output_domain, SparkDataFrameDomain)
    assert isinstance(truncation.output_metric, IfGroupedBy)
    truncation = truncation | LimitRowsPerGroupPerID(
        input_domain=truncation.output_domain,
        input_metric=truncation.output_metric,
        id_columns=["id"],
        grouping_columns=["group1", "group2"],
        threshold=1,
    )
    assert isinstance(truncation.output_domain, SparkDataFrameDomain)
    assert isinstance(truncation.output_metric, SymmetricDifference)
    # Each ID contributes to at most 2 groups, with at most 1 row in each.
    assert truncation.stability_function(1) == 2
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

    got_rows = measurement(input_data).collect()

    # id1 has 4 distinct (group1, group2) groups, of which 2 are kept (which 2
    # depends on a hash), with one row each. id2 has a single group with
    # duplicate rows, which is truncated to one row.
    counts = {(row["group1"], row["group2"]): row["count"] for row in got_rows}
    assert set(counts) == {("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")}
    assert sum(counts.values()) == 3
    assert counts[("a", "a")] >= 1
    assert all(count <= 2 for count in counts.values())


def test_multi_column_id_and_grouping_truncation(spark: SparkSession):
    """Tests a query that truncates with multi-column IDs and multi-column groups."""
    # IDs are (household, person) pairs and groups are (region, category) pairs.
    input_data = spark.createDataFrame(
        [
            ("h1", "p1", "r1", "x", 1),
            ("h1", "p1", "r1", "x", 1),
            ("h1", "p1", "r1", "y", 1),
            ("h1", "p1", "r2", "x", 1),
            ("h1", "p2", "r1", "x", 1),
            ("h1", "p2", "r1", "x", 1),
            ("h2", "p1", "r2", "x", 1),
            ("h2", "p1", "r2", "y", 1),
        ],
        ["household", "person", "region", "category", "value"],
    )
    group_keys = spark.createDataFrame(
        [
            ("r1", "x"),
            ("r1", "y"),
            ("r2", "x"),
            ("r2", "y"),
        ],
        ["region", "category"],
    )
    input_domain = SparkDataFrameDomain(
        {
            "household": SparkStringColumnDescriptor(),
            "person": SparkStringColumnDescriptor(),
            "region": SparkStringColumnDescriptor(),
            "category": SparkStringColumnDescriptor(),
            "value": SparkIntegerColumnDescriptor(),
        }
    )
    id_columns = ["household", "person"]
    grouping_columns = ["region", "category"]

    truncation: Transformation = LimitGroupsPerID(
        input_domain=input_domain,
        output_metric=IfGroupedBy(
            grouping_columns, SumOf(IfGroupedBy(id_columns, SymmetricDifference()))
        ),
        id_columns=id_columns,
        grouping_columns=grouping_columns,
        threshold=2,
    )
    assert isinstance(truncation.output_domain, SparkDataFrameDomain)
    assert isinstance(truncation.output_metric, IfGroupedBy)
    truncation = truncation | LimitRowsPerGroupPerID(
        input_domain=truncation.output_domain,
        input_metric=truncation.output_metric,
        id_columns=id_columns,
        grouping_columns=grouping_columns,
        threshold=1,
    )
    assert isinstance(truncation.output_domain, SparkDataFrameDomain)
    assert isinstance(truncation.output_metric, SymmetricDifference)
    # Each ID contributes to at most 2 groups, with at most 1 row in each.
    assert truncation.stability_function(1) == 2
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

    got_rows = measurement(input_data).collect()

    # (h1, p1) is in 3 groups, so one of them is dropped (which one depends on a
    # hash), and its rows in the 2 remaining groups are truncated to one each.
    # (h1, p2) is in a single group, and its duplicate rows are truncated to one.
    # (h2, p1) is in 2 groups with one row each, so it is unaffected.
    #
    # If only household were used as the ID, h1's two people would be truncated
    # together to 2 groups, and the total would be 4. If only region were used
    # to define groups, (h1, p1) would have 1 row in each of 2 regions and
    # (h2, p1) would have 1 row in r2, and the total would also be 4.
    counts = {(row["region"], row["category"]): row["count"] for row in got_rows}
    assert set(counts) == {("r1", "x"), ("r1", "y"), ("r2", "x"), ("r2", "y")}
    assert sum(counts.values()) == 5
    assert counts[("r1", "x")] in (1, 2)
    assert counts[("r1", "y")] in (0, 1)
    assert counts[("r2", "x")] in (1, 2)
    assert counts[("r2", "y")] == 1
