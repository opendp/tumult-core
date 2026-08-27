"""Unit tests for :mod:`~tmlt.core.transformations.pandas_transformations.agg`.

The two count transformations mirror their Spark twins, so the load-bearing
tests are differential: the stability functions are pinned against the Spark
ones over a grid of ``d_in`` values, and a whole ``GroupBy`` into ``Count``
chain is run on both backends over the corpus in
:mod:`test.unit.backend_testing` and required to produce the same counts.

The Spark twins' *stability* needs no Spark session -- a grouped domain is a
Python object -- so those tests run in the no-JVM lane alongside the pandas-only
ones. Only the end-to-end chains take a session.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

from test.unit.backend_testing import (
    EDGE_CASES_BY_ID,
    Backend,
    EdgeCase,
    assert_frames_equal_as_multisets,
    normalized_rows,
    spark_df_from_pandas,
    to_pandas,
    utc_session_timezone,
)
from test.unit.pandas_grouped_testing import (
    GROUPABLE_CASES,
    key_schema,
    keys_survive_spark_round_trip,
    pandas_domain,
    spark_domain,
    spark_frame,
)
from test.unit.transformations.pandas_transformations.structural_testing import (
    assert_stability_parity,
)
from typing import Any, List, Tuple, cast

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import DataFrame, SparkSession

from tmlt.core.domains.pandas_domains import (
    PandasGroupedTableDomain,
    PandasIntegerColumnDescriptor,
    PandasStringColumnDescriptor,
    PandasTableDomain,
)
from tmlt.core.domains.spark_domains import SparkGroupedDataFrameDomain
from tmlt.core.exceptions import UnsupportedMetricError
from tmlt.core.metrics import (
    AbsoluteDifference,
    OnColumn,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.pandas_transformations.agg import (
    CountDistinctGrouped,
    CountGrouped,
)
from tmlt.core.transformations.pandas_transformations.groupby import GroupBy
from tmlt.core.transformations.spark_transformations import agg as spark_agg
from tmlt.core.transformations.spark_transformations import groupby as spark_groupby
from tmlt.core.utils.exact_number import ExactNumber
from tmlt.core.utils.pandas_grouped_table import PandasGroupedTable
from tmlt.core.utils.pandas_grouping import distinct_rows, row_keys
from tmlt.core.utils.testing import (
    Case,
    assert_property_immutability,
    get_all_props,
    parametrize,
)

_SCHEMA = {
    "A": PandasStringColumnDescriptor(),
    "X": PandasIntegerColumnDescriptor(),
}
_DOMAIN = PandasGroupedTableDomain(_SCHEMA, ["A"])
_SPARK_SCHEMA = {
    column: descriptor.to_spark_descriptor() for column, descriptor in _SCHEMA.items()
}
_SPARK_DOMAIN = SparkGroupedDataFrameDomain(_SPARK_SCHEMA, ["A"])

_FRAME = pd.DataFrame(
    {
        "A": pd.Series(["a1", "a1", "a1", "a2", "a2"], dtype=object),
        "X": [2, 2, 3, 5, -1],
    }
)
#: One key with no rows in the frame, one with rows, and a group in the frame
#: that is not declared at all.
_KEYS = pd.DataFrame({"A": pd.Series(["a0", "a1"], dtype=object)})

#: The two transformations under test, with the Spark twin and default column
#: name of each.
_COUNTS: Tuple[Tuple[str, Any, Any, str], ...] = (
    ("count", CountGrouped, spark_agg.CountGrouped, "count"),
    (
        "count-distinct",
        CountDistinctGrouped,
        spark_agg.CountDistinctGrouped,
        "count_distinct",
    ),
)

#: The two aggregation metrics an input metric can be, by name.
_INPUT_METRICS: Tuple[Tuple[str, Any], ...] = (("l1", SumOf), ("l2", RootSumOfSquared))


def _count_cases(**extra: Any) -> List[Case]:
    """Returns one case per count transformation.

    Args:
        extra: Further arguments to pass to every case.
    """
    return [
        Case(name)(
            transformation_type=pandas_type,
            spark_type=spark_type,
            default_column=default_column,
            **extra,
        )
        for name, pandas_type, spark_type, default_column in _COUNTS
    ]


def _count_metric_cases() -> List[Case]:
    """Returns one case per count transformation and input metric."""
    return [
        Case(f"{name}-{metric_name}")(
            transformation_type=pandas_type,
            spark_type=spark_type,
            default_column=default_column,
            input_metric=metric,
        )
        for name, pandas_type, spark_type, default_column in _COUNTS
        for metric_name, metric in _INPUT_METRICS
    ]


################################################################################
# Properties
################################################################################


@parametrize(_count_metric_cases())
def test_properties(
    transformation_type: Any,
    spark_type: Any,
    default_column: str,
    input_metric: Any,
) -> None:
    """The transformation's properties have the expected values."""
    transformation = transformation_type(
        input_domain=_DOMAIN, input_metric=input_metric(SymmetricDifference())
    )
    assert transformation.input_domain == _DOMAIN
    assert transformation.output_domain == PandasTableDomain(
        {"A": _SCHEMA["A"], default_column: PandasIntegerColumnDescriptor()}
    )
    assert transformation.input_metric == input_metric(SymmetricDifference())
    assert transformation.output_metric == OnColumn(
        default_column, input_metric(AbsoluteDifference())
    )
    assert transformation.count_column == default_column


@parametrize(_count_cases())
def test_custom_count_column(
    transformation_type: Any,
    spark_type: Any,
    default_column: str,
) -> None:
    """The count column can be named."""
    transformation = transformation_type(
        input_domain=_DOMAIN,
        input_metric=SumOf(SymmetricDifference()),
        count_column="total",
    )
    assert transformation.count_column == "total"
    assert transformation.output_domain == PandasTableDomain(
        {"A": _SCHEMA["A"], "total": PandasIntegerColumnDescriptor()}
    )


@parametrize(
    [
        case
        for (prop,) in get_all_props(CountGrouped)
        for case in _count_cases(prop_name=prop)
    ]
)
def test_property_immutability(
    transformation_type: Any,
    spark_type: Any,
    default_column: str,
    prop_name: str,
) -> None:
    """The properties cannot be mutated through the values they return."""
    transformation = transformation_type(
        input_domain=_DOMAIN, input_metric=SumOf(SymmetricDifference())
    )
    assert_property_immutability(transformation, prop_name)


@parametrize(_count_cases())
def test_invalid_inner_metric(
    transformation_type: Any,
    spark_type: Any,
    default_column: str,
) -> None:
    """An input metric with the wrong inner metric is rejected."""
    with pytest.raises(UnsupportedMetricError, match="must be SymmetricDifference"):
        transformation_type(
            input_domain=_DOMAIN, input_metric=SumOf(AbsoluteDifference())
        )


@parametrize(_count_cases())
def test_count_column_already_exists(
    transformation_type: Any,
    spark_type: Any,
    default_column: str,
) -> None:
    """A count column that is already a groupby column is rejected."""
    with pytest.raises(ValueError, match="column already exists"):
        transformation_type(
            input_domain=_DOMAIN,
            input_metric=SumOf(SymmetricDifference()),
            count_column="A",
        )


################################################################################
# Stability, against the Spark twins
################################################################################


@parametrize(_count_metric_cases())
def test_stability_function_matches_spark(
    transformation_type: Any,
    spark_type: Any,
    default_column: str,
    input_metric: Any,
) -> None:
    """The stability function is its Spark twin's, over a grid of d_in values."""
    transformation = transformation_type(
        input_domain=_DOMAIN, input_metric=input_metric(SymmetricDifference())
    )
    spark_transformation = spark_type(
        input_domain=_SPARK_DOMAIN, input_metric=input_metric(SymmetricDifference())
    )
    assert_stability_parity(transformation, spark_transformation)
    assert transformation.stability_function(2) == ExactNumber(2)


################################################################################
# Counting
################################################################################


@parametrize(
    Case("count")(transformation_type=CountGrouped, expected=[0, 3]),
    Case("count-distinct")(transformation_type=CountDistinctGrouped, expected=[0, 2]),
)
def test_call(transformation_type: Any, expected: List[int]) -> None:
    """Each count fills absent keys, drops undeclared groups, and counts its own way."""
    transformation = transformation_type(
        input_domain=_DOMAIN, input_metric=SumOf(SymmetricDifference())
    )
    actual = transformation(PandasGroupedTable(_FRAME, _KEYS))
    assert actual in transformation.output_domain
    pd.testing.assert_frame_equal(
        actual,
        pd.DataFrame(
            {
                "A": pd.Series(["a0", "a1"], dtype=object),
                transformation.count_column: expected,
            }
        ),
    )


@parametrize(
    Case("count")(transformation_type=CountGrouped),
    Case("count-distinct")(transformation_type=CountDistinctGrouped),
)
def test_call_with_no_groups(transformation_type: Any) -> None:
    """An empty frame of group keys gives an empty output of the right dtype."""
    transformation = transformation_type(
        input_domain=_DOMAIN, input_metric=SumOf(SymmetricDifference())
    )
    keys = pd.DataFrame({"A": pd.Series([], dtype=object)})
    actual = transformation(PandasGroupedTable(_FRAME, keys))
    assert len(actual) == 0
    assert actual in transformation.output_domain


@parametrize(
    Case("count")(transformation_type=CountGrouped, expected=5),
    Case("count-distinct")(transformation_type=CountDistinctGrouped, expected=4),
)
def test_call_total_aggregation(transformation_type: Any, expected: int) -> None:
    """A total aggregation counts the whole frame, into a single row."""
    transformation = transformation_type(
        input_domain=PandasGroupedTableDomain(_SCHEMA, []),
        input_metric=SumOf(SymmetricDifference()),
    )
    actual = transformation(PandasGroupedTable(_FRAME, None))
    assert actual in transformation.output_domain
    pd.testing.assert_frame_equal(
        actual, pd.DataFrame({transformation.count_column: [expected]})
    )


@parametrize(
    Case("count")(transformation_type=CountGrouped),
    Case("count-distinct")(transformation_type=CountDistinctGrouped),
)
def test_call_does_not_modify_its_input(transformation_type: Any) -> None:
    """Counting leaves the frame and the group keys it was given unchanged."""
    frame = _FRAME.copy()
    keys = _KEYS.copy()
    frame_before = frame.copy()
    keys_before = keys.copy()
    transformation = transformation_type(
        input_domain=_DOMAIN, input_metric=SumOf(SymmetricDifference())
    )
    transformation(PandasGroupedTable(frame, keys))
    pd.testing.assert_frame_equal(frame, frame_before)
    pd.testing.assert_frame_equal(keys, keys_before)


def test_count_distinct_counts_rows_with_nulls() -> None:
    """count_distinct keeps rows holding nulls, and a null is not a NaN.

    A Spark ``count_distinct`` drops rows with nulls, which is why the Spark
    implementation counts a ``collect_set`` of structs instead; the pandas
    implementation counts distinct rows under
    :mod:`~tmlt.core.utils.pandas_grouping`' identity, where a null and a NaN
    are also two different rows.
    """
    schema = {
        "A": PandasStringColumnDescriptor(),
        "X": PandasStringColumnDescriptor(allow_null=True),
    }
    frame = pd.DataFrame(
        {
            "A": pd.Series(["a1"] * 5, dtype=object),
            "X": pd.Series([None, None, float("nan"), "x", "x"], dtype=object),
        }
    )
    transformation = CountDistinctGrouped(
        input_domain=PandasGroupedTableDomain(schema, ["A"]),
        input_metric=SumOf(SymmetricDifference()),
    )
    actual = transformation(
        PandasGroupedTable(frame, pd.DataFrame({"A": pd.Series(["a1"], dtype=object)}))
    )
    # {null}, {nan} and {"x"} are three distinct rows.
    assert list(actual["count_distinct"]) == [3]


@parametrize(
    Case("schema-order")(grouping=["A", "B", "C"]),
    Case("reversed")(grouping=["C", "B", "A"]),
    Case("shuffled")(grouping=["B", "A", "C"]),
)
def test_output_domain_orders_groupby_columns_as_the_schema_does(
    grouping: List[str],
) -> None:
    """The output's columns are the schema's order, whatever order the keys had.

    An aggregation emits the groupby columns in the order the *group keys* have
    them, and declares them in the order the *schema* has them. Those are one
    order because :class:`~.GroupBy` puts the group keys in the schema's order;
    without that, group keys built from a reversed keyset produced an output
    frame that the transformation's own output domain rejected.

    Args:
        grouping: The order the group keys' columns are given in.
    """
    schema = {
        "A": PandasStringColumnDescriptor(),
        "B": PandasStringColumnDescriptor(),
        "C": PandasStringColumnDescriptor(),
        "X": PandasIntegerColumnDescriptor(),
    }
    frame = pd.DataFrame(
        {
            "A": pd.Series(["a1"], dtype=object),
            "B": pd.Series(["b1"], dtype=object),
            "C": pd.Series(["c1"], dtype=object),
            "X": [1],
        }
    )
    groupby = GroupBy(
        input_domain=PandasTableDomain(schema),
        input_metric=SymmetricDifference(),
        use_l2=False,
        group_keys=frame[grouping],
    )
    assert groupby.groupby_columns == ["A", "B", "C"]
    for transformation_type in (CountGrouped, CountDistinctGrouped):
        transformation = transformation_type(
            input_domain=cast(PandasGroupedTableDomain, groupby.output_domain),
            input_metric=SumOf(SymmetricDifference()),
        )
        output = transformation(groupby(frame))
        assert list(output.columns)[:3] == ["A", "B", "C"]
        assert output in transformation.output_domain


################################################################################
# End-to-end parity with the Spark chain
################################################################################


def _pandas_counts(
    case: EdgeCase,
    frame: pd.DataFrame,
    keys: pd.DataFrame,
    transformation_type: Any,
) -> pd.DataFrame:
    """Returns the counts a pandas GroupBy into Count chain produces.

    Args:
        case: The corpus case being counted.
        frame: The frame to count.
        keys: The group keys to declare.
        transformation_type: The count transformation to use.
    """
    table_domain = pandas_domain(case)
    assert table_domain is not None
    groupby = GroupBy(
        input_domain=table_domain,
        input_metric=SymmetricDifference(),
        use_l2=False,
        group_keys=keys,
    )
    count = transformation_type(
        input_domain=cast(PandasGroupedTableDomain, groupby.output_domain),
        input_metric=SumOf(SymmetricDifference()),
    )
    output = (groupby | count)(frame)
    assert output in count.output_domain
    return output


def _spark_counts(
    spark: SparkSession,
    case: EdgeCase,
    frame: pd.DataFrame,
    keys: pd.DataFrame,
    spark_transformation_type: Any,
) -> DataFrame:
    """Returns the counts the equivalent Spark chain produces.

    Args:
        spark: The Spark session.
        case: The corpus case being counted.
        frame: The frame to count, as pandas.
        keys: The group keys to declare, as pandas.
        spark_transformation_type: The Spark count transformation to use.
    """
    groupby = spark_groupby.GroupBy(
        input_domain=spark_domain(case),
        input_metric=SymmetricDifference(),
        use_l2=False,
        group_keys=spark_df_from_pandas(spark, keys, schema=key_schema(case)),
    )
    count = spark_transformation_type(
        input_domain=groupby.output_domain, input_metric=SumOf(SymmetricDifference())
    )
    return (groupby | count)(spark_frame(spark, case, frame))


@parametrize(
    [
        Case(f"{case.id}-{name}")(
            case=case, transformation_type=pandas_type, spark_type=spark_type
        )
        for case in GROUPABLE_CASES
        for name, pandas_type, spark_type, _ in _COUNTS
    ]
)
def test_counts_match_spark(
    spark: SparkSession,
    case: EdgeCase,
    transformation_type: Any,
    spark_type: Any,
):
    """The two backends' chains produce the same counts, in the keys' order.

    The counts themselves are compared as multisets of rows, since Spark returns
    them in no particular order. That the *pandas* output is in the group keys'
    order -- the property a measurement downstream of it depends on -- is
    asserted directly against the public keys.

    Args:
        spark: The Spark session.
        case: The corpus case to count.
        transformation_type: The pandas count transformation to use.
        spark_type: Its Spark twin.
    """
    frame = case.to_pandas()
    grouping = [name for name in case.columns if name in case.grouping]
    keys = distinct_rows(frame[grouping])
    with utc_session_timezone(spark):
        pandas_output = _pandas_counts(case, frame, keys, transformation_type)
        spark_output = to_pandas(
            _spark_counts(spark, case, frame, keys, spark_type), Backend(name="spark")
        )
    if keys_survive_spark_round_trip(keys, grouping):
        assert_frames_equal_as_multisets(pandas_output, spark_output)
    else:
        # ``toPandas()`` widens a nullable integer column holding a null to
        # float64 and renders the null as a NaN, so the two frames' *keys* are
        # no longer comparable even though the groups they name are the same
        # (see test.unit.backend_testing.conversion.to_pandas). The counts
        # still are, and the keys are covered by every other case.
        count_column = pandas_output.columns[-1]
        assert sorted(pandas_output[count_column]) == sorted(spark_output[count_column])
    # The output's rows are the declared keys, in the declared order.
    assert list(row_keys(pandas_output[grouping], grouping)) == list(
        row_keys(keys, grouping)
    )


@parametrize(
    Case(name)(transformation_type=pandas_type, spark_type=spark_type)
    for name, pandas_type, spark_type, _ in _COUNTS
)
def test_counts_match_spark_row_for_row(
    spark: SparkSession,
    transformation_type: Any,
    spark_type: Any,
):
    """The two chains' outputs are equal frames once Spark's is put in key order.

    This is the strongest form of the comparison, and needs a case whose group
    keys survive ``toPandas()`` unchanged -- a string column with no nulls --
    since a Spark round trip is otherwise free to rewrite them (see
    :func:`~test.unit.backend_testing.conversion.to_pandas`).

    Args:
        spark: The Spark session.
        transformation_type: The pandas count transformation to use.
        spark_type: Its Spark twin.
    """
    case = _string_case()
    frame = case.to_pandas()
    keys = pd.DataFrame({"g": pd.Series(["g2", "g0", "g1"], dtype=object)})
    with utc_session_timezone(spark):
        pandas_output = _pandas_counts(case, frame, keys, transformation_type)
        spark_output = to_pandas(
            _spark_counts(spark, case, frame, keys, spark_type), Backend(name="spark")
        )
    order = {row: position for position, row in enumerate(normalized_rows(keys, ["g"]))}
    positions = [order[row] for row in normalized_rows(spark_output, ["g"])]
    reordered = spark_output.iloc[np.argsort(positions)].reset_index(drop=True)
    pd.testing.assert_frame_equal(reordered, pandas_output, check_dtype=False)


def _string_case() -> EdgeCase:
    """Returns a corpus case whose group keys survive a Spark round trip."""
    return EDGE_CASES_BY_ID["unicode-and-separator-strings"]
