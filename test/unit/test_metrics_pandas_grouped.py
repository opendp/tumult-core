"""Tests for the grouped pandas branches of :mod:`~tmlt.core.metrics`.

:class:`~tmlt.core.metrics.IfGroupedBy` over a
:class:`~tmlt.core.domains.pandas_domains.PandasTableDomain`, and
:class:`~tmlt.core.metrics.SumOf`/:class:`~tmlt.core.metrics.RootSumOfSquared`
over a :class:`~tmlt.core.domains.pandas_domains.PandasGroupedTableDomain`, are
copies of their Spark branches, so most of what is worth asserting is that the
two agree. The differential tests here put the corpus in
:mod:`test.unit.backend_testing` through both backends and require the same
distance; the pandas-only tests cover the branches with no Spark counterpart to
compare against.

The non-grouped metrics' pandas branches, which the inner metrics here bottom
out in, are real: they have their own suites in
:mod:`test.unit.test_pandas_metrics`, and this suite exercises them through the
grouped stack.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

from test.unit.backend_testing import (
    EdgeCase,
    grouped_symdiff_distance,
    utc_session_timezone,
)
from test.unit.pandas_grouped_testing import (
    GROUPABLE_CASES,
    key_schema,
    pandas_domain,
    spark_domain,
    spark_frame,
)
from typing import List, Tuple, Union

import pandas as pd
import pytest
import sympy as sp
from pyspark.sql import SparkSession

from tmlt.core.domains.pandas_domains import (
    PandasFloatColumnDescriptor,
    PandasGroupedTableDomain,
    PandasIntegerColumnDescriptor,
    PandasStringColumnDescriptor,
    PandasTableDomain,
)
from tmlt.core.domains.spark_domains import SparkGroupedDataFrameDomain
from tmlt.core.metrics import IfGroupedBy, RootSumOfSquared, SumOf, SymmetricDifference
from tmlt.core.utils.exact_number import ExactNumber
from tmlt.core.utils.grouped_dataframe import GroupedDataFrame
from tmlt.core.utils.pandas_grouped_table import PandasGroupedTable
from tmlt.core.utils.pandas_grouping import distinct_rows, group_ids
from tmlt.core.utils.testing import Case, parametrize

_SCHEMA = {
    "A": PandasStringColumnDescriptor(),
    "B": PandasIntegerColumnDescriptor(),
    "C": PandasIntegerColumnDescriptor(),
}
_DOMAIN = PandasTableDomain(_SCHEMA)

#: The cases the Spark comparisons are run over: nulls in the grouping and key
#: columns, and several grouping columns at once. Each Spark distance over a
#: grouped domain takes seconds, so the corpus-wide coverage is the oracle
#: comparison below instead.
_REPRESENTATIVE_CASE_IDS = frozenset(
    {"nulls-in-grouping-and-key-columns", "multi-column-grouping-and-keys"}
)

#: The inner metrics an IfGroupedBy over a table domain can carry.
_INNER_METRICS: Tuple[Union[SumOf, RootSumOfSquared, SymmetricDifference], ...] = (
    SumOf(SymmetricDifference()),
    RootSumOfSquared(SymmetricDifference()),
    SymmetricDifference(),
)


_REPRESENTATIVE_CASES = tuple(
    case for case in GROUPABLE_CASES if case.id in _REPRESENTATIVE_CASE_IDS
)


################################################################################
# supports_domain
################################################################################


@parametrize(
    Case("sum-of")(inner_metric=SumOf(SymmetricDifference())),
    Case("root-sum-of-squared")(inner_metric=RootSumOfSquared(SymmetricDifference())),
    Case("symmetric-difference")(inner_metric=SymmetricDifference()),
)
def test_if_grouped_by_supports_pandas_table_domain(inner_metric: object) -> None:
    """IfGroupedBy supports a pandas table domain with the grouping columns."""
    assert IfGroupedBy(["A"], inner_metric).supports_domain(_DOMAIN)  # type: ignore
    assert not IfGroupedBy(["Z"], inner_metric).supports_domain(_DOMAIN)  # type: ignore


def test_if_grouped_by_rejects_float_grouping_column() -> None:
    """Grouping by a floating point column raises, as it does for Spark."""
    domain = PandasTableDomain({"A": PandasFloatColumnDescriptor()})
    with pytest.raises(ValueError, match="Can not group by a floating point column"):
        IfGroupedBy(["A"], SumOf(SymmetricDifference())).supports_domain(domain)


@parametrize(
    Case("sum-of")(metric=SumOf(SymmetricDifference())),
    Case("root-sum-of-squared")(metric=RootSumOfSquared(SymmetricDifference())),
)
def test_aggregation_metric_supports_pandas_grouped_domain(
    metric: Union[SumOf, RootSumOfSquared],
) -> None:
    """SumOf and RootSumOfSquared support a pandas grouped table domain."""
    assert metric.supports_domain(PandasGroupedTableDomain(_SCHEMA, ["A"]))


################################################################################
# Pandas-only distances
################################################################################


def test_if_grouped_by_distance_with_no_groups_at_all() -> None:
    """Two empty tables are at distance zero, by the branch's hardcode."""
    empty = pd.DataFrame({"A": pd.Series([], dtype=object), "B": [], "C": []}).astype(
        {"B": "int64", "C": "int64"}
    )
    for inner_metric in _INNER_METRICS:
        metric = IfGroupedBy(["A"], inner_metric)
        assert metric.distance(empty, empty, _DOMAIN) == ExactNumber(0)


def test_aggregation_metric_distance_is_infinite_for_different_keys() -> None:
    """Grouped tables with different group keys are infinitely far apart."""
    frame = pd.DataFrame({"A": ["a1"], "B": [1], "C": [2]})
    domain = PandasGroupedTableDomain(_SCHEMA, ["A"])
    value1 = PandasGroupedTable(frame, pd.DataFrame({"A": ["a1"]}))
    value2 = PandasGroupedTable(frame, pd.DataFrame({"A": ["a2"]}))
    distance = SumOf(SymmetricDifference()).distance(value1, value2, domain)
    assert distance == ExactNumber(sp.oo)


def test_if_grouped_by_distance_keeps_null_and_nan_groups_apart() -> None:
    """A null group and a NaN group are two groups, so the distance sees both."""
    domain = PandasTableDomain({"A": PandasStringColumnDescriptor(allow_null=True)})
    value1 = pd.DataFrame({"A": pd.Series([None, float("nan")], dtype=object)})
    value2 = pd.DataFrame({"A": pd.Series([None], dtype=object)})
    metric = IfGroupedBy(["A"], SumOf(SymmetricDifference()))
    assert metric.distance(value1, value2, domain) == ExactNumber(1)
    assert metric.distance(value1, value1, domain) == ExactNumber(0)


################################################################################
# Differential tests
################################################################################

# Two backends' distances are compared in two ways here, because a Spark
# distance over a grouped domain costs several Spark jobs per group and takes
# seconds. Every case is checked against the harness's own oracle, which is
# pure pandas and independent of the implementation; a couple of cases are then
# checked against Spark itself, which is what ties the oracle to the backend it
# is standing in for.


def _variant_pairs(case: EdgeCase) -> List[Tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Returns pairs of frames derived from a case, to measure distances between.

    Every pair separates something a distance has to see: two identical frames,
    two differing by one row, two differing by a whole group -- which is what
    tells a group present on one side only from a group differing on both -- and
    a frame against nothing at all.

    Args:
        case: The corpus case to derive from.
    """
    full = case.to_pandas()
    empty = full.iloc[:0].reset_index(drop=True)
    pairs = [("full-vs-itself", full, full), ("full-vs-empty", full, empty)]
    if len(full) == 0:
        return pairs
    pairs.append(("without-first-row", full, full.iloc[1:].reset_index(drop=True)))
    ids = group_ids(full, list(case.grouping))
    pairs.append(
        ("without-first-group", full, full[ids != ids[0]].reset_index(drop=True))
    )
    return pairs


def _discriminating_pairs(
    case: EdgeCase,
) -> List[Tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Returns the variant pairs the Spark comparisons are run over.

    These are the two that tell the grouped distances apart: dropping one row
    changes a group both frames have, and dropping a whole group leaves a
    declared group empty on one side only.

    Args:
        case: The corpus case to derive from.
    """
    return [
        pair
        for pair in _variant_pairs(case)
        if pair[0] in ("without-first-row", "without-first-group")
    ]


def _grouping_columns(case: EdgeCase) -> List[str]:
    """Returns a case's grouping columns, in its schema's order.

    Args:
        case: The corpus case whose grouping columns are wanted.
    """
    return [name for name in case.columns if name in case.grouping]


@parametrize(Case(case.id)(case=case) for case in GROUPABLE_CASES)
def test_if_grouped_by_distance_matches_the_oracle(case: EdgeCase) -> None:
    """IfGroupedBy(SymmetricDifference) is the harness's grouped distance.

    :func:`~test.unit.backend_testing.comparison.grouped_symdiff_distance` is
    the harness's own definition of this metric, written independently of the
    implementation and in pure pandas, so this covers every case in the corpus
    without a Spark session.

    Args:
        case: The corpus case to measure distances within.
    """
    domain = pandas_domain(case)
    assert domain is not None
    grouping = _grouping_columns(case)
    metric = IfGroupedBy(grouping, SymmetricDifference())
    for name, frame1, frame2 in _variant_pairs(case):
        expected = grouped_symdiff_distance(frame1, frame2, grouping)
        assert metric.distance(frame1, frame2, domain) == ExactNumber(expected), (
            f"{metric} disagreed with the oracle on {name} of case {case.id}"
        )


@parametrize(Case(case.id)(case=case) for case in _REPRESENTATIVE_CASES)
def test_if_grouped_by_distance_matches_spark(spark: SparkSession, case: EdgeCase):
    """IfGroupedBy gives the same distance on both backends, for every inner metric.

    Args:
        spark: The Spark session.
        case: The corpus case to measure distances within.
    """
    pandas_table_domain = pandas_domain(case)
    assert pandas_table_domain is not None
    spark_table_domain = spark_domain(case)
    with utc_session_timezone(spark):
        for inner_metric in _INNER_METRICS:
            metric = IfGroupedBy(_grouping_columns(case), inner_metric)
            for name, frame1, frame2 in _discriminating_pairs(case):
                pandas_distance = metric.distance(frame1, frame2, pandas_table_domain)
                spark_distance = metric.distance(
                    spark_frame(spark, case, frame1),
                    spark_frame(spark, case, frame2),
                    spark_table_domain,
                )
                assert pandas_distance == spark_distance, (
                    f"{metric} disagreed between backends on {name} of case {case.id}"
                )


@parametrize(Case(case.id)(case=case) for case in _REPRESENTATIVE_CASES)
def test_aggregation_metric_distance_matches_spark(spark: SparkSession, case: EdgeCase):
    """SumOf and RootSumOfSquared agree across backends over declared keys.

    The keys are those of the case's own frame, so the variant with a whole
    group dropped leaves a declared group that is empty on one side only -- the
    case that separates a distance of 1 from one of 2.

    Args:
        spark: The Spark session.
        case: The corpus case to measure distances within.
    """
    pandas_table_domain = pandas_domain(case)
    assert pandas_table_domain is not None
    grouping = _grouping_columns(case)
    pandas_grouped_domain = PandasGroupedTableDomain(
        pandas_table_domain.schema, grouping
    )
    spark_table_domain = spark_domain(case)
    spark_grouped_domain = SparkGroupedDataFrameDomain(
        spark_table_domain.schema, grouping
    )
    keys = distinct_rows(case.to_pandas()[grouping])
    with utc_session_timezone(spark):
        spark_keys = spark.createDataFrame(
            [tuple(row) for row in keys.itertuples(index=False)],
            schema=key_schema(case),
        )
        for metric in (
            SumOf(SymmetricDifference()),
            RootSumOfSquared(SymmetricDifference()),
        ):
            for name, frame1, frame2 in _discriminating_pairs(case):
                pandas_distance = metric.distance(
                    PandasGroupedTable(frame1, keys),
                    PandasGroupedTable(frame2, keys),
                    pandas_grouped_domain,
                )
                spark_distance = metric.distance(
                    GroupedDataFrame(spark_frame(spark, case, frame1), spark_keys),
                    GroupedDataFrame(spark_frame(spark, case, frame2), spark_keys),
                    spark_grouped_domain,
                )
                assert pandas_distance == spark_distance, (
                    f"{metric} disagreed between backends on {name} of case {case.id}"
                )
