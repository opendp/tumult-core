"""Unit tests for :mod:`~tmlt.core.transformations.pandas_transformations.truncation`.

The three transformations here are thin wrappers: the truncation itself is
:mod:`tmlt.core.utils.pandas_truncation`, whose agreement with
:mod:`tmlt.core.utils.truncation` is established row-for-row by
:mod:`test.unit.utils.test_truncation_differential`. What is left for this
module is the *wrapper*, and it is tested against the Spark wrapper rather than
against hard-coded expectations:

* The stability functions are copies of their Spark twins', and
  :func:`~test.unit.transformations.pandas_transformations.structural_testing.assert_stability_parity`
  is what keeps them copies -- over the whole
  :data:`~test.unit.transformations.pandas_transformations.structural_testing.D_IN_GRID`,
  including the distances the metrics reject, and over several thresholds.
* The constructors accept and reject the same arguments with the same errors,
  which is asserted by building both transformations and comparing the
  exceptions, not by matching a written-down message. The two differ in exactly
  one place: an error raised by
  :class:`~tmlt.core.transformations.base.Transformation` itself names the
  domain, and the two domains have different reprs.
* Each transformation hands the util the columns its Spark twin hands its own
  util. Passing the grouping and key columns the other way round is the
  plausible wiring mistake, and it is a *different truncation* rather than a
  differently ordered one, so the differential tests below catch it: they run
  the pandas transformation and its Spark twin over the curated
  :data:`~test.unit.backend_testing.EDGE_CASES` corpus and over seeded
  :func:`~test.unit.backend_testing.random_frame` sweeps, and compare the rows
  that survived.

Everything except the differential tests is pandas-only and runs in the
``test-nojvm`` lane: building a
:class:`~tmlt.core.domains.spark_domains.SparkDataFrameDomain` and a Spark
transformation over it needs no session. The differential tests take the
``spark`` fixture, so ``test/conftest.py`` marks them ``spark`` and
``-m "not spark"`` deselects them.

The sweeps draw no floating point columns. Spark renders a float with the JVM's
``Double.toString``/``Float.toString``, which on a JVM older than 19 sometimes
emits more digits than the shortest that round-trips and so hashes differently;
:mod:`test.unit.utils.test_truncation_differential` is where that is
characterized, and repeating it here would be testing the formatter rather than
the wrapper. The curated corpus's float cases are kept, since those are the
values that suite already runs against this JVM.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import random
import re
from collections import Counter
from test.unit.backend_testing import (
    EDGE_CASES,
    ROW_ID_COLUMN,
    Backend,
    EdgeCase,
    assert_frames_equal_as_multisets,
    df_for,
    frame_row_ids,
    random_frame,
    spark_df_from_case,
    to_pandas,
)
from test.unit.transformations.pandas_transformations.structural_testing import (
    assert_same_rejection,
    assert_stability_parity,
    describable_cases,
    pandas_domain_for_case,
    spark_domain_for_case,
)
from typing import Any, Callable, Collection, Dict, List, Tuple, Union

import pandas as pd
import pytest
from pyspark.sql import DataFrame, SparkSession

from tmlt.core.domains.pandas_domains import (
    PandasFloatColumnDescriptor,
    PandasStringColumnDescriptor,
    PandasTableDomain,
)
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkStringColumnDescriptor,
)
from tmlt.core.metrics import IfGroupedBy, RootSumOfSquared, SumOf, SymmetricDifference
from tmlt.core.transformations.base import Transformation
from tmlt.core.transformations.pandas_transformations.truncation import (
    LimitKeysPerGroup,
    LimitRowsPerGroup,
    LimitRowsPerKeyPerGroup,
)
from tmlt.core.transformations.spark_transformations.truncation import (
    LimitKeysPerGroup as SparkLimitKeysPerGroup,
)
from tmlt.core.transformations.spark_transformations.truncation import (
    LimitRowsPerGroup as SparkLimitRowsPerGroup,
)
from tmlt.core.transformations.spark_transformations.truncation import (
    LimitRowsPerKeyPerGroup as SparkLimitRowsPerKeyPerGroup,
)
from tmlt.core.utils.pandas_truncation import (
    limit_keys_per_group,
    truncate_large_groups,
)
from tmlt.core.utils.testing import (
    Case,
    assert_property_immutability,
    get_all_props,
    parametrize,
)

SCHEMA = {
    "A": PandasStringColumnDescriptor(),
    "B": PandasStringColumnDescriptor(),
    "C": PandasStringColumnDescriptor(),
}
"""The schema of :data:`DF`."""

SPARK_SCHEMA = {
    "A": SparkStringColumnDescriptor(),
    "B": SparkStringColumnDescriptor(),
    "C": SparkStringColumnDescriptor(),
}
"""The same schema, for the Spark twins the parity assertions build."""

DF = pd.DataFrame(
    {
        "A": ["a1", "a2", "a3", "a3", "a3", "a4", "a4", "a4", "a4"],
        "B": ["b1", "b1", "b2", "b2", "b2", "b1", "b2", "b3", "b4"],
        "C": ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"],
    },
    index=range(10, 19),
)
"""A frame in :data:`SCHEMA`, with groups of several sizes.

Its index deliberately does not start at zero, so that a transformation that
forgot to reindex its result is visible.
"""

#: The thresholds the behavioural and parity tests sweep over: nothing kept,
#: the two interesting small groups, and a threshold no group reaches.
THRESHOLDS: Tuple[int, ...] = (0, 1, 2, 7)

#: Seed for the randomized differential sweep.
SWEEP_SEED = 20260812

#: Number of random frames the differential sweep draws.
SWEEP_FRAMES = 8

#: The column kinds the sweep draws from: the kinds the pandas descriptors can
#: describe, minus the floating point ones. See the module docstring.
SWEEP_DTYPE_MENU: Tuple[str, ...] = ("int64", "Int64", "string", "date", "timestamp")

#: The thresholds the differential sweep uses for its generated frames.
SWEEP_THRESHOLDS: Tuple[int, ...] = (0, 1, 2, 5)

_SPARK_BACKEND = Backend(name="spark")

################################################################################
# Helpers
################################################################################


def assert_leaves_input_alone(
    transformation: Transformation, df: pd.DataFrame
) -> pd.DataFrame:
    """Asserts a transformation does not modify the frame it is given.

    Args:
        transformation: The transformation to apply.
        df: The frame to apply it to.

    Returns:
        The transformation's result.
    """
    before = df.copy(deep=True)
    result = transformation(df)
    pd.testing.assert_frame_equal(df, before)
    # A result sharing memory with the input would let a later write reach it.
    # The write goes to a second, discarded result, so that the one returned to
    # the caller is what the transformation produced.
    scratch = transformation(df)
    if len(scratch) and "A" in scratch.columns:
        scratch.loc[scratch.index[0], "A"] = "mutated"
        pd.testing.assert_frame_equal(df, before)
    return result


def _group_sizes(df: pd.DataFrame, columns: Collection[str]) -> Counter:
    """Returns the number of rows of each group of a frame.

    Args:
        df: The frame to count.
        columns: The columns defining the groups.
    """
    ordered = [column for column in df.columns if column in set(columns)]
    return Counter(df[ordered].itertuples(index=False, name=None))


################################################################################
# LimitRowsPerGroup
################################################################################


def _limit_rows_per_group(
    grouping_columns: Collection[str] = ("A",),
    threshold: int = 2,
    output_metric: Union[SymmetricDifference, IfGroupedBy, None] = None,
) -> LimitRowsPerGroup:
    """Returns a LimitRowsPerGroup over :data:`SCHEMA`.

    Args:
        grouping_columns: The columns defining the groups.
        threshold: The maximum number of rows per group.
        output_metric: The output metric, defaulting to SymmetricDifference().
    """
    return LimitRowsPerGroup(
        input_domain=PandasTableDomain(SCHEMA),
        output_metric=(
            SymmetricDifference() if output_metric is None else output_metric
        ),
        grouping_columns=grouping_columns,
        threshold=threshold,
    )


def _spark_limit_rows_per_group(
    grouping_columns: Collection[str] = ("A",),
    threshold: int = 2,
    output_metric: Union[SymmetricDifference, IfGroupedBy, None] = None,
) -> SparkLimitRowsPerGroup:
    """Returns the Spark twin of :func:`_limit_rows_per_group`.

    Args:
        grouping_columns: The columns defining the groups.
        threshold: The maximum number of rows per group.
        output_metric: The output metric, defaulting to SymmetricDifference().
    """
    return SparkLimitRowsPerGroup(
        input_domain=SparkDataFrameDomain(SPARK_SCHEMA),
        output_metric=(
            SymmetricDifference() if output_metric is None else output_metric
        ),
        grouping_columns=grouping_columns,
        threshold=threshold,
    )


@pytest.mark.parametrize(
    "prop_name", [prop[0] for prop in get_all_props(LimitRowsPerGroup)]
)
def test_limit_rows_per_group_property_immutability(prop_name: str):
    """LimitRowsPerGroup's properties are immutable."""
    assert_property_immutability(_limit_rows_per_group(), prop_name)


def test_limit_rows_per_group_properties():
    """LimitRowsPerGroup's properties have the expected values."""
    transformation = _limit_rows_per_group(grouping_columns=["A", "B"])
    assert transformation.input_domain == PandasTableDomain(SCHEMA)
    assert transformation.input_metric == IfGroupedBy(["A", "B"], SymmetricDifference())
    assert transformation.output_domain == PandasTableDomain(SCHEMA)
    assert transformation.output_metric == SymmetricDifference()
    assert transformation.grouping_columns == frozenset({"A", "B"})
    assert transformation.threshold == 2


def test_limit_rows_per_group_format():
    """LimitRowsPerGroup formats the way its Spark twin does."""
    assert (
        _limit_rows_per_group().format()
        == _spark_limit_rows_per_group().format()
        == "LimitRowsPerGroup grouping_columns={'A'} threshold=2"
    )


@parametrize(
    Case(f"{'-'.join(grouping)}-threshold-{threshold}-{type(metric).__name__}")(
        grouping_columns=grouping, threshold=threshold, output_metric=metric
    )
    for grouping in (("A",), ("B",), ("A", "B"))
    for threshold in THRESHOLDS
    for metric in (SymmetricDifference(), IfGroupedBy(grouping, SymmetricDifference()))
)
def test_limit_rows_per_group_delegates_to_the_util(
    grouping_columns: Tuple[str, ...],
    threshold: int,
    output_metric: Union[SymmetricDifference, IfGroupedBy],
):
    """LimitRowsPerGroup keeps the rows the pandas util keeps, and no more."""
    transformation = _limit_rows_per_group(
        grouping_columns=grouping_columns,
        threshold=threshold,
        output_metric=output_metric,
    )
    actual = assert_leaves_input_alone(transformation, DF.copy())
    expected = truncate_large_groups(DF, grouping_columns, threshold)
    pd.testing.assert_frame_equal(actual, expected)
    assert list(actual.index) == list(range(len(actual)))
    assert all(
        size <= threshold for size in _group_sizes(actual, grouping_columns).values()
    )


@parametrize(
    Case(f"threshold-{threshold}-{type(metric).__name__}")(
        threshold=threshold, output_metric=metric
    )
    for threshold in THRESHOLDS
    for metric in (
        SymmetricDifference(),
        IfGroupedBy(["A"], SymmetricDifference()),
    )
)
def test_limit_rows_per_group_stability_matches_spark(
    threshold: int, output_metric: Union[SymmetricDifference, IfGroupedBy]
):
    """LimitRowsPerGroup's stability function is its Spark twin's."""
    assert_stability_parity(
        _limit_rows_per_group(threshold=threshold, output_metric=output_metric),
        _spark_limit_rows_per_group(threshold=threshold, output_metric=output_metric),
    )


@parametrize(
    Case("negative-threshold")(
        kwargs={"threshold": -1},
        match="Threshold must be nonnegative",
        same_message=True,
    ),
    Case("nonexistent-grouping-column")(
        kwargs={"grouping_columns": ["invalid"]},
        match="Input metric .* and input domain .* are not compatible.",
        same_message=False,
    ),
    Case("output-metric-on-another-column")(
        kwargs={"output_metric": IfGroupedBy(["notA"], SymmetricDifference())},
        match=re.escape(
            "Output metric must be `SymmetricDifference()` or "
            "`IfGroupedBy(['A'], SymmetricDifference())`"
        ),
        same_message=True,
    ),
    Case("output-metric-with-inner-sum")(
        kwargs={"output_metric": IfGroupedBy(["A"], SumOf(SymmetricDifference()))},
        match=re.escape(
            "Output metric must be `SymmetricDifference()` or "
            "`IfGroupedBy(['A'], SymmetricDifference())`"
        ),
        same_message=True,
    ),
    Case("duplicate-grouping-columns")(
        kwargs={"grouping_columns": ["A", "A"]},
        match="IfGroupedBy cannot have duplicate grouping columns",
        same_message=True,
    ),
    Case("no-grouping-columns")(
        kwargs={"grouping_columns": []},
        match="Cannot instantiate an IfGroupedBy with empty columns",
        same_message=True,
    ),
)
def test_limit_rows_per_group_invalid_parameters(
    kwargs: Dict[str, Any], match: str, same_message: bool
):
    """LimitRowsPerGroup rejects what its Spark twin rejects, the same way."""
    args: Dict[str, Any] = {
        "grouping_columns": ["A"],
        "threshold": 1,
        "output_metric": SymmetricDifference(),
    }
    args.update(kwargs)
    assert_same_rejection(
        lambda: _limit_rows_per_group(**args),
        lambda: _spark_limit_rows_per_group(**args),
        match=match,
        same_message=same_message,
    )


################################################################################
# LimitKeysPerGroup
################################################################################


def _sum_of_metric(
    grouping_columns: Collection[str] = ("A",), key_column: str = "C"
) -> IfGroupedBy:
    """Returns the SumOf output metric of a LimitKeysPerGroup.

    Args:
        grouping_columns: The columns defining the groups.
        key_column: The column defining the keys.
    """
    return IfGroupedBy(
        [key_column], SumOf(IfGroupedBy(grouping_columns, SymmetricDifference()))
    )


def _root_sum_of_squared_metric(
    grouping_columns: Collection[str] = ("A",), key_column: str = "C"
) -> IfGroupedBy:
    """Returns the RootSumOfSquared output metric of a LimitKeysPerGroup.

    Args:
        grouping_columns: The columns defining the groups.
        key_column: The column defining the keys.
    """
    return IfGroupedBy(
        [key_column],
        RootSumOfSquared(IfGroupedBy(grouping_columns, SymmetricDifference())),
    )


def _limit_keys_per_group(
    grouping_columns: Collection[str] = ("A",),
    key_column: str = "C",
    threshold: int = 2,
    output_metric: Union[IfGroupedBy, None] = None,
) -> LimitKeysPerGroup:
    """Returns a LimitKeysPerGroup over :data:`SCHEMA`.

    Args:
        grouping_columns: The columns defining the groups.
        key_column: The column defining the keys.
        threshold: The maximum number of keys per group.
        output_metric: The output metric, defaulting to the SumOf one.
    """
    return LimitKeysPerGroup(
        input_domain=PandasTableDomain(SCHEMA),
        output_metric=(
            _sum_of_metric(grouping_columns, key_column)
            if output_metric is None
            else output_metric
        ),
        grouping_columns=grouping_columns,
        key_column=key_column,
        threshold=threshold,
    )


def _spark_limit_keys_per_group(
    grouping_columns: Collection[str] = ("A",),
    key_column: str = "C",
    threshold: int = 2,
    output_metric: Union[IfGroupedBy, None] = None,
) -> SparkLimitKeysPerGroup:
    """Returns the Spark twin of :func:`_limit_keys_per_group`.

    Args:
        grouping_columns: The columns defining the groups.
        key_column: The column defining the keys.
        threshold: The maximum number of keys per group.
        output_metric: The output metric, defaulting to the SumOf one.
    """
    return SparkLimitKeysPerGroup(
        input_domain=SparkDataFrameDomain(SPARK_SCHEMA),
        output_metric=(
            _sum_of_metric(grouping_columns, key_column)
            if output_metric is None
            else output_metric
        ),
        grouping_columns=grouping_columns,
        key_column=key_column,
        threshold=threshold,
    )


@pytest.mark.parametrize(
    "prop_name", [prop[0] for prop in get_all_props(LimitKeysPerGroup)]
)
def test_limit_keys_per_group_property_immutability(prop_name: str):
    """LimitKeysPerGroup's properties are immutable."""
    assert_property_immutability(_limit_keys_per_group(), prop_name)


def test_limit_keys_per_group_properties():
    """LimitKeysPerGroup's properties have the expected values."""
    transformation = _limit_keys_per_group(grouping_columns=["A", "B"], key_column="C")
    assert transformation.input_domain == PandasTableDomain(SCHEMA)
    assert transformation.input_metric == IfGroupedBy(["A", "B"], SymmetricDifference())
    assert transformation.output_domain == PandasTableDomain(SCHEMA)
    assert transformation.output_metric == _sum_of_metric(["A", "B"], "C")
    assert transformation.grouping_columns == frozenset({"A", "B"})
    assert transformation.key_column == "C"
    assert transformation.threshold == 2


def test_limit_keys_per_group_format():
    """LimitKeysPerGroup formats the way its Spark twin does."""
    assert (
        _limit_keys_per_group().format()
        == _spark_limit_keys_per_group().format()
        == "LimitKeysPerGroup grouping_columns={'A'} key_column='C' threshold=2"
    )


@parametrize(
    Case(f"{'-'.join(grouping)}-key-{key}-threshold-{threshold}")(
        grouping_columns=grouping, key_column=key, threshold=threshold
    )
    for grouping, key in ((("A",), "B"), (("B",), "C"), (("A", "B"), "C"))
    for threshold in THRESHOLDS
)
def test_limit_keys_per_group_delegates_to_the_util(
    grouping_columns: Tuple[str, ...], key_column: str, threshold: int
):
    """LimitKeysPerGroup keeps the rows the pandas util keeps, and no more.

    The util takes the grouping columns and the key columns as two separate
    arguments, so this also pins which of them the transformation passes where.

    The expected call is made with the transformation's own
    :attr:`~tmlt.core.transformations.pandas_transformations.truncation.LimitKeysPerGroup.grouping_columns`,
    not with the collection it was constructed from, because those are not
    interchangeable. ``limit_keys_per_group`` hashes the *concatenation* of the
    grouping and key columns, so its result depends on the order the grouping
    columns arrive in -- and both this transformation and its Spark twin hold
    them as a :class:`~tmlt.core.utils.misc.ConciseFrozenSet` and hand the util
    that. With two or more grouping columns the iteration order of the frozenset
    is what Python's string hashing makes it, so which rows survive can differ
    between *processes*, though never between the two backends within one. That
    is the Spark transformation's behaviour, faithfully mirrored, not something
    introduced here.
    """
    transformation = _limit_keys_per_group(
        grouping_columns=grouping_columns, key_column=key_column, threshold=threshold
    )
    actual = assert_leaves_input_alone(transformation, DF.copy())
    expected = limit_keys_per_group(
        DF, transformation.grouping_columns, [key_column], threshold
    )
    pd.testing.assert_frame_equal(actual, expected)
    assert list(actual.index) == list(range(len(actual)))
    keys_per_group: Dict[Any, set] = {}
    for row in actual.itertuples(index=False):
        group = tuple(getattr(row, column) for column in grouping_columns)
        keys_per_group.setdefault(group, set()).add(getattr(row, key_column))
    assert all(len(keys) <= threshold for keys in keys_per_group.values())


@parametrize(
    Case(f"threshold-{threshold}-{name}")(threshold=threshold, metric_name=name)
    for threshold in THRESHOLDS
    for name in ("sum-of", "root-sum-of-squared", "if-grouped-by")
)
def test_limit_keys_per_group_stability_matches_spark(threshold: int, metric_name: str):
    """LimitKeysPerGroup's stability function is its Spark twin's.

    All three valid output metrics are covered, since each takes a different
    branch of the stability function.
    """
    output_metric = {
        "sum-of": _sum_of_metric(),
        "root-sum-of-squared": _root_sum_of_squared_metric(),
        "if-grouped-by": IfGroupedBy(["A"], SymmetricDifference()),
    }[metric_name]
    assert_stability_parity(
        _limit_keys_per_group(threshold=threshold, output_metric=output_metric),
        _spark_limit_keys_per_group(threshold=threshold, output_metric=output_metric),
    )


@parametrize(
    Case("negative-threshold")(
        kwargs={"threshold": -1},
        match="Threshold must be nonnegative",
        same_message=True,
    ),
    Case("key-column-is-a-grouping-column")(
        kwargs={"grouping_columns": ["A", "C"], "key_column": "C"},
        match="Key column cannot be a grouping column",
        same_message=True,
    ),
    Case("nonexistent-grouping-column")(
        kwargs={
            "grouping_columns": ["invalid"],
            "output_metric": _sum_of_metric(["invalid"], "C"),
        },
        match="Input metric .* and input domain .* are not compatible.",
        same_message=False,
    ),
    Case("nonexistent-key-column")(
        kwargs={
            "key_column": "invalid",
            "output_metric": _sum_of_metric(["A"], "invalid"),
        },
        match="Output metric .* and output domain .* are not compatible.",
        same_message=False,
    ),
    Case("output-metric-without-inner-aggregation")(
        kwargs={"output_metric": IfGroupedBy(["C"], SymmetricDifference())},
        match=re.escape("Output metric must be one of `IfGroupedBy(['C'],"),
        same_message=True,
    ),
    Case("output-metric-on-the-grouping-column")(
        kwargs={"output_metric": _sum_of_metric(["C"], "A")},
        match=re.escape("Output metric must be one of `IfGroupedBy(['C'],"),
        same_message=True,
    ),
)
def test_limit_keys_per_group_invalid_parameters(
    kwargs: Dict[str, Any], match: str, same_message: bool
):
    """LimitKeysPerGroup rejects what its Spark twin rejects, the same way."""
    args: Dict[str, Any] = {
        "grouping_columns": ["A"],
        "key_column": "C",
        "threshold": 1,
        "output_metric": _sum_of_metric(),
    }
    args.update(kwargs)
    assert_same_rejection(
        lambda: _limit_keys_per_group(**args),
        lambda: _spark_limit_keys_per_group(**args),
        match=match,
        same_message=same_message,
    )


################################################################################
# LimitRowsPerKeyPerGroup
################################################################################


def _limit_rows_per_key_per_group(
    grouping_columns: Collection[str] = ("A",),
    key_column: str = "C",
    threshold: int = 2,
    input_metric: Union[IfGroupedBy, None] = None,
) -> LimitRowsPerKeyPerGroup:
    """Returns a LimitRowsPerKeyPerGroup over :data:`SCHEMA`.

    Args:
        grouping_columns: The columns defining the groups.
        key_column: The column defining the keys.
        threshold: The maximum number of rows per (key, group) pair.
        input_metric: The input metric, defaulting to the SumOf one.
    """
    return LimitRowsPerKeyPerGroup(
        input_domain=PandasTableDomain(SCHEMA),
        input_metric=(
            _sum_of_metric(grouping_columns, key_column)
            if input_metric is None
            else input_metric
        ),
        grouping_columns=grouping_columns,
        key_column=key_column,
        threshold=threshold,
    )


def _spark_limit_rows_per_key_per_group(
    grouping_columns: Collection[str] = ("A",),
    key_column: str = "C",
    threshold: int = 2,
    input_metric: Union[IfGroupedBy, None] = None,
) -> SparkLimitRowsPerKeyPerGroup:
    """Returns the Spark twin of :func:`_limit_rows_per_key_per_group`.

    Args:
        grouping_columns: The columns defining the groups.
        key_column: The column defining the keys.
        threshold: The maximum number of rows per (key, group) pair.
        input_metric: The input metric, defaulting to the SumOf one.
    """
    return SparkLimitRowsPerKeyPerGroup(
        input_domain=SparkDataFrameDomain(SPARK_SCHEMA),
        input_metric=(
            _sum_of_metric(grouping_columns, key_column)
            if input_metric is None
            else input_metric
        ),
        grouping_columns=grouping_columns,
        key_column=key_column,
        threshold=threshold,
    )


@pytest.mark.parametrize(
    "prop_name", [prop[0] for prop in get_all_props(LimitRowsPerKeyPerGroup)]
)
def test_limit_rows_per_key_per_group_property_immutability(prop_name: str):
    """LimitRowsPerKeyPerGroup's properties are immutable."""
    assert_property_immutability(_limit_rows_per_key_per_group(), prop_name)


def test_limit_rows_per_key_per_group_properties():
    """LimitRowsPerKeyPerGroup's properties have the expected values."""
    transformation = _limit_rows_per_key_per_group(
        grouping_columns=["A", "B"], key_column="C"
    )
    assert transformation.input_domain == PandasTableDomain(SCHEMA)
    assert transformation.input_metric == _sum_of_metric(["A", "B"], "C")
    assert transformation.output_domain == PandasTableDomain(SCHEMA)
    assert transformation.output_metric == SymmetricDifference()
    assert transformation.grouping_columns == frozenset({"A", "B"})
    assert transformation.key_column == "C"
    assert transformation.threshold == 2


def test_limit_rows_per_key_per_group_format():
    """LimitRowsPerKeyPerGroup formats the way its Spark twin does."""
    assert (
        _limit_rows_per_key_per_group().format()
        == _spark_limit_rows_per_key_per_group().format()
        == "LimitRowsPerKeyPerGroup grouping_columns={'A'} key_column='C' threshold=2"
    )


@parametrize(
    Case(f"{'-'.join(grouping)}-key-{key}-threshold-{threshold}")(
        grouping_columns=grouping, key_column=key, threshold=threshold
    )
    for grouping, key in ((("A",), "B"), (("B",), "C"), (("A", "B"), "C"))
    for threshold in THRESHOLDS
)
def test_limit_rows_per_key_per_group_delegates_to_the_util(
    grouping_columns: Tuple[str, ...], key_column: str, threshold: int
):
    """LimitRowsPerKeyPerGroup truncates over the grouping *and* key columns."""
    transformation = _limit_rows_per_key_per_group(
        grouping_columns=grouping_columns, key_column=key_column, threshold=threshold
    )
    actual = assert_leaves_input_alone(transformation, DF.copy())
    expected = truncate_large_groups(DF, [*grouping_columns, key_column], threshold)
    pd.testing.assert_frame_equal(actual, expected)
    assert list(actual.index) == list(range(len(actual)))
    assert all(
        size <= threshold
        for size in _group_sizes(actual, [*grouping_columns, key_column]).values()
    )


@parametrize(
    Case(f"threshold-{threshold}-{name}")(threshold=threshold, metric_name=name)
    for threshold in THRESHOLDS
    for name in ("sum-of", "root-sum-of-squared", "if-grouped-by")
)
def test_limit_rows_per_key_per_group_stability_matches_spark(
    threshold: int, metric_name: str
):
    """LimitRowsPerKeyPerGroup's stability function is its Spark twin's.

    The output metric each input metric induces is compared too, since the
    stability function branches on the input metric and the two must agree on
    both.
    """
    input_metric = {
        "sum-of": _sum_of_metric(),
        "root-sum-of-squared": _root_sum_of_squared_metric(),
        "if-grouped-by": IfGroupedBy(["A"], SymmetricDifference()),
    }[metric_name]
    pandas_transformation = _limit_rows_per_key_per_group(
        threshold=threshold, input_metric=input_metric
    )
    spark_transformation = _spark_limit_rows_per_key_per_group(
        threshold=threshold, input_metric=input_metric
    )
    assert pandas_transformation.output_metric == spark_transformation.output_metric
    assert_stability_parity(pandas_transformation, spark_transformation)


@parametrize(
    Case("negative-threshold")(
        kwargs={"threshold": -1},
        match="Threshold must be nonnegative",
        same_message=True,
    ),
    Case("key-column-is-a-grouping-column")(
        kwargs={"grouping_columns": ["A", "C"], "key_column": "C"},
        match="Key column cannot be a grouping column",
        same_message=True,
    ),
    Case("input-metric-without-inner-aggregation")(
        kwargs={"input_metric": IfGroupedBy(["C"], SymmetricDifference())},
        match=re.escape("Input metric must be one of `IfGroupedBy(['C'],"),
        same_message=True,
    ),
    Case("input-metric-on-the-grouping-column")(
        kwargs={"input_metric": _sum_of_metric(["C"], "A")},
        match=re.escape("Input metric must be one of `IfGroupedBy(['C'],"),
        same_message=True,
    ),
    Case("nonexistent-grouping-column")(
        kwargs={
            "grouping_columns": ["invalid"],
            "input_metric": _sum_of_metric(["invalid"], "C"),
        },
        match="Input metric .* and input domain .* are not compatible.",
        same_message=False,
    ),
)
def test_limit_rows_per_key_per_group_invalid_parameters(
    kwargs: Dict[str, Any], match: str, same_message: bool
):
    """LimitRowsPerKeyPerGroup rejects what its Spark twin rejects, the same way."""
    args: Dict[str, Any] = {
        "grouping_columns": ["A"],
        "key_column": "C",
        "threshold": 1,
        "input_metric": _sum_of_metric(),
    }
    args.update(kwargs)
    assert_same_rejection(
        lambda: _limit_rows_per_key_per_group(**args),
        lambda: _spark_limit_rows_per_key_per_group(**args),
        match=match,
        same_message=same_message,
    )


################################################################################
# Empty frames and a threshold of zero
################################################################################


@parametrize(
    Case("LimitRowsPerGroup")(build=_limit_rows_per_group),
    Case("LimitKeysPerGroup")(build=_limit_keys_per_group),
    Case("LimitRowsPerKeyPerGroup")(build=_limit_rows_per_key_per_group),
)
def test_empty_frame_gives_an_empty_frame(build: Callable[..., Transformation]):
    """Truncating an empty frame gives an empty frame with the same columns."""
    empty = DF.iloc[:0]
    actual = build()(empty)
    assert len(actual) == 0
    assert list(actual.columns) == list(DF.columns)
    assert actual.dtypes.to_dict() == DF.dtypes.to_dict()


@parametrize(
    Case("LimitRowsPerGroup")(build=_limit_rows_per_group),
    Case("LimitKeysPerGroup")(build=_limit_keys_per_group),
    Case("LimitRowsPerKeyPerGroup")(build=_limit_rows_per_key_per_group),
)
def test_threshold_zero_keeps_nothing(build: Callable[..., Transformation]):
    """A threshold of zero keeps no rows at all, and is not an error."""
    actual = build(threshold=0)(DF)
    assert len(actual) == 0
    assert list(actual.columns) == list(DF.columns)


################################################################################
# Differential tests against the Spark transformations
################################################################################


def _sweep_cases() -> List[EdgeCase]:
    """Returns the randomly generated cases of the differential sweep.

    Half the frames have two grouping columns, so that the multi-column wiring
    is swept as well as the single-column one.
    """
    rng = random.Random(SWEEP_SEED)
    return [
        random_frame(
            rng,
            dtype_menu=SWEEP_DTYPE_MENU,
            n_rows=16,
            n_groups=3,
            n_grouping_columns=1 + index % 2,
            n_key_columns=1,
            n_payload_columns=1,
            case_id=f"random-{index}",
        )
        for index in range(SWEEP_FRAMES)
    ]


def _differential_cases() -> Dict[str, EdgeCase]:
    """Returns the cases the differential tests run, by id.

    These are the corpus cases the pandas descriptors can describe -- a
    transformation over a column with no descriptor cannot be constructed at
    all -- together with the generated sweep.
    """
    cases = {case.id: case for case in describable_cases()}
    cases.update({case.id: case for case in _sweep_cases()})
    return cases


DIFFERENTIAL_CASES: Dict[str, EdgeCase] = _differential_cases()

CASE_PARAMS = [Case(case_id)(case_id=case_id) for case_id in DIFFERENTIAL_CASES]


def _thresholds_for(case: EdgeCase) -> List[int]:
    """Returns the nonnegative thresholds to run a case at.

    The corpus's ``threshold-extremes`` case includes -1, which every
    transformation here rejects in its constructor; that rejection is covered
    by the invalid-parameter tests instead.

    Args:
        case: The case to run.
    """
    thresholds = (
        case.thresholds if case.id in {c.id for c in EDGE_CASES} else SWEEP_THRESHOLDS
    )
    return [threshold for threshold in thresholds if threshold >= 0]


def _key_column_of(case: EdgeCase) -> str:
    """Returns the column a case's key-taking transformations should use.

    Args:
        case: The case to run.
    """
    return case.keys[0]


def _rejects_float_key(
    case: EdgeCase,
    key_column: str,
    pandas_build: Callable[[], Any],
    spark_build: Callable[[], Any],
) -> bool:
    """Asserts both backends refuse a floating point key column, if it is one.

    :class:`~tmlt.core.metrics.IfGroupedBy` on the key column is part of these
    two transformations' contract, and neither backend can group by a floating
    point column: a row's group would depend on which of ``NaN``, ``0.0`` and
    ``-0.0`` it holds. Both
    :class:`~tmlt.core.domains.pandas_domains.PandasGroupedTableDomain` and
    :class:`~tmlt.core.domains.spark_domains.SparkGroupedDataFrameDomain` say
    so, with the same error, so the corpus's float-keyed cases are turned into
    an assertion that they still do rather than skipped.

    Args:
        case: The case being run.
        key_column: The key column it would be run with.
        pandas_build: A callable constructing the pandas transformation.
        spark_build: A callable constructing its Spark twin.

    Returns:
        Whether the key column was a floating point one, in which case the
        caller has nothing left to compare.
    """
    pandas_domain = pandas_domain_for_case(case)
    assert pandas_domain is not None
    if not isinstance(pandas_domain[key_column], PandasFloatColumnDescriptor):
        return False
    assert_same_rejection(
        pandas_build,
        spark_build,
        match=f"Can not group by a floating point column: {key_column}",
    )
    return True


def _assert_same_survivors(
    context: str, spark_result: DataFrame, pandas_result: pd.DataFrame
) -> None:
    """Asserts a Spark transformation and a pandas one kept the same rows.

    Cases carrying a unique :data:`~test.unit.backend_testing.ROW_ID_COLUMN` are
    compared by the surviving ids. That comparison is exact, and it has to be
    the one used wherever it is available: ``toPandas()`` widens a nullable
    integer column to ``float64`` and renders a null in a floating point column
    as ``NaN``, and the harness's normalized comparison deliberately keeps
    ``NaN`` distinct from ``NULL``, so comparing such frames cell by cell would
    fail on the round trip rather than on the truncation. The rest -- the cases
    with duplicate rows, which are the ones exercising the per-duplicate salt,
    and which have no such columns -- are compared as multisets of rows, which
    is the only thing observable about them.

    Args:
        context: A description of what is being compared, for failure messages.
        spark_result: The Spark transformation's output.
        pandas_result: The pandas transformation's output.
    """
    spark_pandas = to_pandas(spark_result, _SPARK_BACKEND)
    assert set(spark_pandas.columns) == set(pandas_result.columns), (
        f"{context}: different columns."
    )
    if ROW_ID_COLUMN not in pandas_result.columns:
        try:
            assert_frames_equal_as_multisets(spark_pandas, pandas_result)
        except AssertionError as error:
            raise AssertionError(f"{context}: {error}") from error
        return
    pandas_ids = Counter(frame_row_ids(pandas_result))
    spark_ids = Counter(frame_row_ids(spark_pandas))
    # Row ids are unique in the input and truncation only selects rows, so a
    # result with a repeated id has duplicated rows -- which comparing sets of
    # ids would hide.
    for name, result, ids in (
        ("pandas", pandas_result, pandas_ids),
        ("Spark", spark_pandas, spark_ids),
    ):
        assert len(result) == len(ids), (
            f"{context}: the {name} result has {len(result)} rows but only "
            f"{len(ids)} distinct row ids, so it duplicated rows."
        )
    assert pandas_ids == spark_ids, (
        f"{context}: kept different rows. Only pandas kept row ids "
        f"{sorted((pandas_ids - spark_ids).elements())}; only Spark kept "
        f"{sorted((spark_ids - pandas_ids).elements())}."
    )


@parametrize(*CASE_PARAMS)
def test_limit_rows_per_group_matches_spark(utc_spark: SparkSession, case_id: str):
    """LimitRowsPerGroup keeps the rows its Spark twin keeps."""
    case = DIFFERENTIAL_CASES[case_id]
    pandas_domain = pandas_domain_for_case(case)
    assert pandas_domain is not None
    spark_domain = spark_domain_for_case(case)
    sdf = spark_df_from_case(utc_spark, case)
    for threshold in _thresholds_for(case):
        pandas_result = LimitRowsPerGroup(
            input_domain=pandas_domain,
            output_metric=SymmetricDifference(),
            grouping_columns=case.grouping,
            threshold=threshold,
        )(case.to_pandas())
        spark_result = SparkLimitRowsPerGroup(
            input_domain=spark_domain,
            output_metric=SymmetricDifference(),
            grouping_columns=case.grouping,
            threshold=threshold,
        )(sdf)
        _assert_same_survivors(
            f"case {case.id}, LimitRowsPerGroup, threshold {threshold}",
            spark_result,
            pandas_result,
        )


@parametrize(*CASE_PARAMS)
def test_limit_keys_per_group_matches_spark(utc_spark: SparkSession, case_id: str):
    """LimitKeysPerGroup keeps the rows its Spark twin keeps.

    Its util takes the grouping columns and the key column as separate
    arguments and truncates on their concatenation, so swapping them at the
    call site is a different truncation, which this catches.
    """
    case = DIFFERENTIAL_CASES[case_id]
    pandas_domain = pandas_domain_for_case(case)
    assert pandas_domain is not None
    spark_domain = spark_domain_for_case(case)
    key_column = _key_column_of(case)
    output_metric = _sum_of_metric(case.grouping, key_column)

    def build_pandas(threshold: int) -> LimitKeysPerGroup:
        return LimitKeysPerGroup(
            input_domain=pandas_domain,
            output_metric=output_metric,
            grouping_columns=case.grouping,
            key_column=key_column,
            threshold=threshold,
        )

    def build_spark(threshold: int) -> SparkLimitKeysPerGroup:
        return SparkLimitKeysPerGroup(
            input_domain=spark_domain,
            output_metric=output_metric,
            grouping_columns=case.grouping,
            key_column=key_column,
            threshold=threshold,
        )

    if _rejects_float_key(
        case, key_column, lambda: build_pandas(1), lambda: build_spark(1)
    ):
        return
    sdf = spark_df_from_case(utc_spark, case)
    for threshold in _thresholds_for(case):
        pandas_result = build_pandas(threshold)(case.to_pandas())
        spark_result = build_spark(threshold)(sdf)
        _assert_same_survivors(
            f"case {case.id}, LimitKeysPerGroup, threshold {threshold}",
            spark_result,
            pandas_result,
        )


@parametrize(*CASE_PARAMS)
def test_limit_rows_per_key_per_group_matches_spark(
    utc_spark: SparkSession, case_id: str
):
    """LimitRowsPerKeyPerGroup keeps the rows its Spark twin keeps."""
    case = DIFFERENTIAL_CASES[case_id]
    pandas_domain = pandas_domain_for_case(case)
    assert pandas_domain is not None
    spark_domain = spark_domain_for_case(case)
    key_column = _key_column_of(case)
    input_metric = _sum_of_metric(case.grouping, key_column)

    def build_pandas(threshold: int) -> LimitRowsPerKeyPerGroup:
        return LimitRowsPerKeyPerGroup(
            input_domain=pandas_domain,
            input_metric=input_metric,
            grouping_columns=case.grouping,
            key_column=key_column,
            threshold=threshold,
        )

    def build_spark(threshold: int) -> SparkLimitRowsPerKeyPerGroup:
        return SparkLimitRowsPerKeyPerGroup(
            input_domain=spark_domain,
            input_metric=input_metric,
            grouping_columns=case.grouping,
            key_column=key_column,
            threshold=threshold,
        )

    if _rejects_float_key(
        case, key_column, lambda: build_pandas(1), lambda: build_spark(1)
    ):
        return
    sdf = spark_df_from_case(utc_spark, case)
    for threshold in _thresholds_for(case):
        pandas_result = build_pandas(threshold)(case.to_pandas())
        spark_result = build_spark(threshold)(sdf)
        _assert_same_survivors(
            f"case {case.id}, LimitRowsPerKeyPerGroup, threshold {threshold}",
            spark_result,
            pandas_result,
        )


def test_differential_frames_are_built_the_same_way(spark: SparkSession):
    """The harness renders the shared fixture frame identically for both backends.

    The differential tests above build their Spark frames from the corpus's own
    explicit schemas rather than through :func:`df_for`; this pins the two
    routes together on the frame the rest of this module uses, so that a
    difference in the *inputs* cannot be mistaken for one in the outputs.
    """
    spark_frame = df_for(DF.reset_index(drop=True), _SPARK_BACKEND, spark=spark)
    assert_frames_equal_as_multisets(
        to_pandas(spark_frame, _SPARK_BACKEND),
        df_for(DF.reset_index(drop=True), Backend(name="pandas")),
    )
