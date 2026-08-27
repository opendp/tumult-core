"""Tests for :mod:`~tmlt.core.utils.pandas_grouping`.

The module under test answers one question -- when do two rows belong to the
same group? -- so the tests come in two flavors:

* Differential tests, which put the same frame through Spark and assert that
  :func:`~tmlt.core.utils.pandas_grouping.distinct_rows` keeps as many rows as
  ``distinct()``/``dropDuplicates()`` do, and that every group
  :func:`~tmlt.core.utils.pandas_grouping.group_indices` forms holds exactly
  the rows a null-safe (``eqNullSafe``) Spark join selects. The inputs are
  the curated :data:`~test.unit.utils.truncation_testing.EDGE_CASES` corpus,
  which already covers every corner where pandas and Spark could plausibly
  disagree about identity.
* Pandas-only tests, which pin the identity properties that have no Spark
  counterpart to compare against -- an ``object`` column holding both a null
  and a NaN, a nullable float column holding both ``pd.NA`` and ``np.nan``, a
  signed zero, a date beside a datetime, and timestamps carrying precision
  Spark cannot see -- together with the invariants that tie the public
  functions to each other.

Only the differential tests request a Spark session, so a run restricted to
the rest of this module never starts a JVM.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import datetime
from test.unit.backend_testing import floating_array
from test.unit.utils.truncation_testing import (
    EDGE_CASES,
    EDGE_CASES_BY_ID,
    ROW_ID_COLUMN,
    EdgeCase,
    spark_df_from_case,
    spark_df_from_pandas,
)
from typing import Any, Dict, List, Sequence, Set

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as sf
from pyspark.sql.types import LongType, StructField, StructType

from tmlt.core.utils.pandas_grouping import (
    distinct_rows,
    group_codes,
    group_ids,
    group_indices,
    row_keys,
)
from tmlt.core.utils.testing import Case, parametrize

#: Whether pandas is version 2 or later, where a ``datetime64`` column may
#: carry a unit other than nanoseconds.
_PANDAS_2 = int(pd.__version__.split(".")[0]) >= 2

#: The column the differential tests add to hold each row's pandas group
#: index. It is deliberately not a name any edge case uses.
_GROUP_INDEX_COLUMN = "__group_index"

#: Prefix distinguishing a group representative's columns from the columns of
#: the frame it is joined to, which carry the same names.
_REPRESENTATIVE_PREFIX = "__representative_"

################################################################################
# Helpers
################################################################################


def _content_columns(case: EdgeCase) -> List[str]:
    """Returns a case's columns without its unique row id.

    The row id is unique by construction, so a whole-row distinct over a case
    that carries one would trivially keep every row. Dropping it is what makes
    the case's duplicate rows visible.

    Args:
        case: The case to read.

    Returns:
        The case's columns, in order, minus :data:`ROW_ID_COLUMN`.
    """
    return [name for name in case.columns if name != ROW_ID_COLUMN]


def _identity_columns(case: EdgeCase) -> List[str]:
    """Returns the grouping and key columns of a case, deduplicated.

    These are the columns whose values the corpus was written to stress, so
    they are what the grouping tests group by.

    Args:
        case: The case to read.

    Returns:
        The case's grouping columns followed by its key columns, without
        repetitions.
    """
    return list(dict.fromkeys([*case.grouping, *case.keys]))


def _group_representatives(
    spark: SparkSession,
    sdf: DataFrame,
    df: pd.DataFrame,
    columns: Sequence[str],
    groups: Dict[Any, np.ndarray],
) -> DataFrame:
    """Returns the first row of each pandas group, as a Spark frame.

    The representatives are ingested exactly as the frame under test was --
    Python row tuples plus the very schema ``sdf`` carries -- rather than
    turned into ``lit`` literals. The two paths are not interchangeable: a
    ``lit`` built from a Python datetime reaches the JVM as a legacy
    ``java.sql.Timestamp``, whose pre-1900 wall clocks land minutes away from
    where ``createDataFrame`` puts the same value. The grouping columns are
    renamed so that the join condition can name both sides unambiguously.

    Args:
        spark: The Spark session to build the frame with.
        sdf: The Spark frame the representatives are matched against, which
            supplies the column types.
        df: The pandas frame the groups were computed on.
        columns: The columns defining the groups.
        groups: The groups, as
            :func:`~tmlt.core.utils.pandas_grouping.group_indices` returns
            them.

    Returns:
        One row per group, with the grouping columns renamed and a
        :data:`_GROUP_INDEX_COLUMN` holding the group's position.
    """
    first_rows = [positions[0] for positions in groups.values()]
    frame = df.iloc[first_rows][list(columns)].reset_index(drop=True).copy()
    frame[_GROUP_INDEX_COLUMN] = np.arange(len(frame), dtype=np.int64)
    schema = StructType(
        [
            StructField(f"{_REPRESENTATIVE_PREFIX}{name}", sdf.schema[name].dataType)
            for name in columns
        ]
        + [StructField(_GROUP_INDEX_COLUMN, LongType(), False)]
    )
    return spark_df_from_pandas(spark, frame, schema)


def _null_safe_match(
    sdf: DataFrame, representatives: DataFrame, columns: Sequence[str]
) -> Column:
    """Returns the join condition matching each row to its group, null-safely.

    ``eqNullSafe`` is Spark's ``<=>``, under which two nulls match and (with
    Spark's NaN semantics) two NaNs do too, which is exactly the identity
    :mod:`~tmlt.core.utils.pandas_grouping` implements.

    Args:
        sdf: The frame whose rows are being matched.
        representatives: The group representatives, as
            :func:`_group_representatives` builds them.
        columns: The columns to match on.

    Returns:
        The Spark join condition.
    """
    condition = sf.lit(True)
    for name in columns:
        condition = condition & sdf[name].eqNullSafe(
            representatives[f"{_REPRESENTATIVE_PREFIX}{name}"]
        )
    return condition


def _first_occurrence_positions(df: pd.DataFrame, columns: Sequence[str]) -> List[int]:
    """Returns the position of the first row of each group, in group order.

    This is an independent oracle for what
    :func:`~tmlt.core.utils.pandas_grouping.distinct_rows` keeps: it walks the
    frame once and remembers the keys it has seen, where the implementation
    works from vectorized codes.

    Args:
        df: The frame to walk.
        columns: The columns defining the groups.

    Returns:
        One position per group, ascending.
    """
    seen: Set[Any] = set()
    positions: List[int] = []
    for position, key in enumerate(row_keys(df, columns)):
        if key not in seen:
            seen.add(key)
            positions.append(position)
    return positions


################################################################################
# Differential tests against Spark
################################################################################


@parametrize(Case(case.id)(case_id=case.id) for case in EDGE_CASES)
def test_distinct_rows_matches_spark(utc_spark: SparkSession, case_id: str) -> None:
    """``distinct_rows`` keeps as many rows as Spark's distinct does.

    Which row of a group Spark keeps is not defined -- ``dropDuplicates`` picks
    whatever its shuffle produced -- so the two are compared by how many groups
    they find, over the whole row and over the case's identity columns.
    """
    case = EDGE_CASES_BY_ID[case_id]
    df = case.to_pandas()
    sdf = spark_df_from_case(utc_spark, case)
    content = _content_columns(case)
    assert len(distinct_rows(df[content])) == sdf.select(*content).distinct().count(), (
        f"case {case_id}: distinct over {content} disagrees with Spark."
    )
    identity = _identity_columns(case)
    assert len(distinct_rows(df, identity)) == sdf.dropDuplicates(identity).count(), (
        f"case {case_id}: dropDuplicates over {identity} disagrees with Spark."
    )


@parametrize(Case(case.id)(case_id=case.id) for case in EDGE_CASES)
def test_group_indices_matches_spark_null_safe_grouping(
    utc_spark: SparkSession, case_id: str
) -> None:
    """Each group holds the rows a null-safe Spark match selects.

    The first row of each pandas group is sent back to Spark and joined to the
    frame under ``eqNullSafe``, so every Spark row is tagged with the group it
    belongs to. A row matching no group leaves the tag null, and a row matching
    two is counted twice; both show up in the comparison below. Cases carrying
    a unique row id are compared by the actual set of rows in each group; the
    others, by the group's size.
    """
    case = EDGE_CASES_BY_ID[case_id]
    df = case.to_pandas()
    sdf = spark_df_from_case(utc_spark, case)
    columns = _identity_columns(case)
    groups = group_indices(df, columns)
    representatives = _group_representatives(utc_spark, sdf, df, columns, groups)
    aggregations = [sf.count(sf.lit(1)).alias("size")]
    if case.has_row_id:
        aggregations.append(sf.collect_set(ROW_ID_COLUMN).alias("ids"))
    spark_groups = (
        sdf.join(
            representatives, _null_safe_match(sdf, representatives, columns), "left"
        )
        .groupBy(_GROUP_INDEX_COLUMN)
        .agg(*aggregations)
        .collect()
    )
    unmatched = [row for row in spark_groups if row[_GROUP_INDEX_COLUMN] is None]
    assert not unmatched, (
        f"case {case_id}: {unmatched[0]['size']} Spark rows matched none of the "
        f"{len(groups)} groups found in pandas."
    )
    spark_sizes = {row[_GROUP_INDEX_COLUMN]: row["size"] for row in spark_groups}
    pandas_sizes = {
        position: len(positions) for position, positions in enumerate(groups.values())
    }
    assert spark_sizes == pandas_sizes, (
        f"case {case_id}: group sizes disagree. pandas {pandas_sizes}, "
        f"Spark {spark_sizes}. Groups keyed by {columns}."
    )
    if not case.has_row_id:
        return
    ids = df[ROW_ID_COLUMN].to_numpy()
    spark_ids = {row[_GROUP_INDEX_COLUMN]: set(row["ids"]) for row in spark_groups}
    for position, positions in enumerate(groups.values()):
        expected = {int(value) for value in ids[positions]}
        assert spark_ids[position] == expected, (
            f"case {case_id}: group {position} holds row ids "
            f"{sorted(spark_ids[position])} in Spark but {sorted(expected)} in pandas."
        )


################################################################################
# Identity properties
################################################################################


def test_row_keys_separates_nulls_from_nans_in_object_columns() -> None:
    """A null and a NaN in an object column are different groups.

    A pandas ``groupby`` puts them in the same group -- and then drops it --
    where Spark keeps ``NULL`` and ``NaN`` apart.
    """
    df = pd.DataFrame(
        {"v": pd.Series([None, float("nan"), pd.NA, float("nan"), 1.0], dtype=object)}
    )
    keys = row_keys(df)
    assert keys[0] == keys[2], "None and pd.NA are both nulls."
    assert keys[1] == keys[3], "Two NaNs are one group."
    assert keys[0] != keys[1], "A null and a NaN are two groups."
    assert keys[4] not in (keys[0], keys[1]), "An ordinary value is its own group."
    assert list(group_ids(df, ["v"])) == [0, 1, 0, 1, 2]


def test_row_keys_separates_na_from_nan_in_nullable_float_columns() -> None:
    """``pd.NA`` and ``np.nan`` in a ``Float64`` column are different groups."""
    df = pd.DataFrame(
        {"v": pd.Series(floating_array([1.0, np.nan, 0.0], [False, False, True]))}
    )
    keys = row_keys(df)
    assert df["v"].dtype == pd.Float64Dtype()
    assert keys[1] != keys[2], "A NaN value and pd.NA are two groups."
    assert keys[0] not in (keys[1], keys[2])
    assert list(group_ids(df, ["v"])) == [0, 1, 2]


def test_row_keys_merges_signed_zeros() -> None:
    """``-0.0`` and ``0.0`` are one group, as they are in Spark."""
    for dtype in ("float64", "float32", "object"):
        df = pd.DataFrame({"v": pd.Series([0.0, -0.0], dtype=dtype)})
        keys = row_keys(df)
        assert keys[0] == keys[1], f"Signed zeros split in a {dtype} column."
        assert list(group_ids(df, ["v"])) == [0, 0]
    nullable = pd.DataFrame(
        {"v": pd.Series(floating_array([0.0, -0.0], [False, False]))}
    )
    assert row_keys(nullable)[0] == row_keys(nullable)[1]


def test_row_keys_separates_dates_from_datetimes() -> None:
    """A date and the datetime at its midnight are different groups.

    Spark holds the two in different types, so no single Spark column can mix
    them; an ``object`` column can, and keying them alike would merge values
    that hash differently.
    """
    df = pd.DataFrame(
        {
            "v": pd.Series(
                [
                    datetime.date(2020, 1, 1),
                    datetime.datetime(2020, 1, 1),
                    datetime.date(2020, 1, 1),
                ],
                dtype=object,
            )
        }
    )
    keys = row_keys(df)
    assert keys[0] == keys[2], "Two equal dates are one group."
    assert keys[0] != keys[1], "A date and a datetime are two groups."
    assert list(group_ids(df, ["v"])) == [0, 1, 0]


def test_row_keys_discards_sub_microsecond_timestamp_precision() -> None:
    """Timestamps agreeing at microsecond resolution are one group.

    Spark's ``TimestampType`` stores microseconds, so a ``datetime64[ns]``
    column must not split a Spark group over nanoseconds Spark cannot see.
    """
    df = pd.DataFrame(
        {
            "t": pd.to_datetime(
                [
                    "2020-01-01 00:00:00.000001001",
                    "2020-01-01 00:00:00.000001999",
                    "2020-01-01 00:00:00.000002000",
                ]
            )
        }
    )
    keys = row_keys(df)
    assert keys[0] == keys[1], "Nanoseconds within one microsecond split a group."
    assert keys[0] != keys[2], "Different microseconds are different groups."
    assert list(group_ids(df, ["t"])) == [0, 0, 1]


@pytest.mark.skipif(
    not _PANDAS_2, reason="pandas 1 has no datetime64 unit other than nanoseconds"
)
def test_row_keys_agree_across_timestamp_units() -> None:
    """The same wall clock is one group whatever unit its column carries.

    On pandas 2 a ``datetime64`` column may be in seconds, milliseconds or
    microseconds as well as nanoseconds, and the coarser units reach wall
    clocks the nanosecond range cannot hold. The unit is a storage detail, so
    it must not decide identity.
    """
    stamp = "2020-01-01 00:00:01"
    keys = [
        row_keys(
            pd.DataFrame({"t": pd.to_datetime([stamp]).astype(f"datetime64[{unit}]")})
        )[0]
        for unit in ("s", "ms", "us", "ns")
    ]
    assert len(set(keys)) == 1, f"The same wall clock gave {len(set(keys))} keys."
    far_future = pd.DataFrame(
        {"t": pd.to_datetime(["9999-12-31 00:00:00"]).astype("datetime64[us]")}
    )
    assert len(group_indices(far_future, ["t"])) == 1


def test_row_keys_keys_bytearrays_by_content() -> None:
    """Binary values group by content, whether or not they are hashable."""
    df = pd.DataFrame({"b": pd.Series([b"ab", bytearray(b"ab"), b"ba"], dtype=object)})
    keys = row_keys(df)
    assert keys[0] == keys[1], "bytes and an equal bytearray are one group."
    assert keys[0] != keys[2]
    assert list(group_ids(df, ["b"])) == [0, 0, 1]


################################################################################
# Invariants tying the public functions together
################################################################################


@parametrize(Case(case.id)(case_id=case.id) for case in EDGE_CASES)
def test_row_keys_induce_the_same_partition_as_group_ids(case_id: str) -> None:
    """Two rows share a key exactly when they share a group id.

    ``row_keys`` builds one key per value the slow way and ``group_ids`` codes
    whole columns at once, so this pins the vectorized branches against the
    per-value specification they implement, over every column of the corpus.
    """
    case = EDGE_CASES_BY_ID[case_id]
    df = case.to_pandas()
    column_sets: List[List[str]] = [
        list(case.columns),
        _identity_columns(case),
        list(case.grouping),
        list(case.keys),
        [],
    ]
    for columns in column_sets:
        keys = list(row_keys(df, columns))
        ids = list(group_ids(df, columns))
        by_key: Dict[Any, int] = {}
        by_id: Dict[int, Any] = {}
        for key, identifier in zip(keys, ids):
            assert by_key.setdefault(key, identifier) == identifier, (
                f"case {case_id}: key {key} spans several group ids over {columns}."
            )
            assert by_id.setdefault(identifier, key) == key, (
                f"case {case_id}: group id {identifier} spans several keys "
                f"over {columns}."
            )


@parametrize(Case(case.id)(case_id=case.id) for case in EDGE_CASES)
def test_group_codes_matches_group_ids_of_one_column(case_id: str) -> None:
    """Coding one column is grouping a frame by that one column."""
    case = EDGE_CASES_BY_ID[case_id]
    df = case.to_pandas()
    for name in case.columns:
        codes = group_codes(df[name])
        assert list(codes) == list(group_ids(df, [name])), (
            f"case {case_id}: column {name} codes differently on its own."
        )
        assert codes.dtype == np.int64
        assert list(codes) == list(pd.factorize(codes)[0]), (
            f"case {case_id}: column {name} codes are not first-occurrence dense."
        )


@parametrize(Case(case.id)(case_id=case.id) for case in EDGE_CASES)
def test_distinct_rows_keeps_the_first_row_of_each_group(case_id: str) -> None:
    """The survivors are the first row of each group, with every column intact.

    The expected frame is built by walking the rows and remembering the keys
    seen so far, which is the definition ``distinct_rows`` implements with
    codes. Comparing with ``check_dtype`` is what pins dtype preservation: an
    ``Int64`` or ``Float64`` column must not come back as a float64 one.
    """
    case = EDGE_CASES_BY_ID[case_id]
    df = case.to_pandas()
    for columns in (list(case.columns), _identity_columns(case)):
        result = distinct_rows(df, columns)
        expected = df.iloc[_first_occurrence_positions(df, columns)].reset_index(
            drop=True
        )
        pd.testing.assert_frame_equal(result, expected)
        assert result.dtypes.equals(df.dtypes), (
            f"case {case_id}: dtypes changed over {columns}."
        )
        assert list(result.index) == list(range(len(result)))


@parametrize(Case(case.id)(case_id=case.id) for case in EDGE_CASES)
def test_group_indices_partitions_the_frame(case_id: str) -> None:
    """Every row is in exactly one group, and each group is in input order."""
    case = EDGE_CASES_BY_ID[case_id]
    df = case.to_pandas()
    columns = _identity_columns(case)
    groups = group_indices(df, columns)
    covered = np.concatenate(list(groups.values())) if groups else np.zeros(0)
    assert sorted(covered.tolist()) == list(range(len(df))), (
        f"case {case_id}: the groups do not partition the {len(df)} rows."
    )
    for key, positions in groups.items():
        assert list(positions) == sorted(positions), (
            f"case {case_id}: group {key} is not in input order."
        )
    keys = row_keys(df, columns)
    for key, positions in groups.items():
        assert {keys[position] for position in positions} == {key}, (
            f"case {case_id}: group {key} holds rows with other keys."
        )
    assert list(groups) == list(dict.fromkeys(keys)), (
        f"case {case_id}: the groups are not in order of first appearance."
    )
    assert list(distinct_rows(df, columns).index) == list(range(len(groups)))


def test_no_columns_makes_the_whole_frame_one_group() -> None:
    """Grouping by no columns at all gives a single group."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert list(group_ids(df, [])) == [0, 0, 0]
    assert list(row_keys(df, [])) == [(), (), ()]
    groups = group_indices(df, [])
    assert list(groups) == [()]
    assert list(groups[()]) == [0, 1, 2]
    assert len(distinct_rows(df, [])) == 1


################################################################################
# Empty frames
################################################################################


@parametrize(
    Case("no-rows-with-columns")(df=EDGE_CASES_BY_ID["empty-frame"].to_pandas()),
    Case("no-rows-no-columns")(df=pd.DataFrame()),
)
def test_empty_frames_have_no_groups(df: pd.DataFrame) -> None:
    """An empty frame has no groups, no keys, and no distinct rows."""
    columns = list(df.columns)
    assert group_indices(df, columns) == {}
    assert list(group_ids(df, columns)) == []
    assert list(row_keys(df)) == []
    result = distinct_rows(df)
    assert len(result) == 0
    assert list(result.columns) == columns
    assert result.dtypes.equals(df.dtypes)


def test_empty_column_has_no_codes() -> None:
    """Coding a column with no values gives no codes."""
    codes = group_codes(pd.Series([], dtype="Int64"))
    assert list(codes) == []
    assert codes.dtype == np.int64


################################################################################
# Error contracts
################################################################################


@parametrize(
    Case("group_ids")(call=lambda df: group_ids(df, ["missing"])),
    Case("row_keys")(call=lambda df: row_keys(df, ["missing"])),
    Case("distinct_rows")(call=lambda df: distinct_rows(df, ["missing"])),
    Case("group_indices")(call=lambda df: group_indices(df, ["missing"])),
)
def test_unknown_column_raises_key_error(call: Any) -> None:
    """Naming a column the frame does not have raises ``KeyError``."""
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(KeyError, match="missing"):
        call(df)


@parametrize(
    Case("dict")(value={"a": 1}),
    Case("list")(value=[1, 2]),
    Case("set")(value={1, 2}),
)
def test_unhashable_value_raises_not_implemented(value: Any) -> None:
    """A value with no Python hash has no group key.

    Spark cannot hold such a value either, so it is reported as the unsupported
    type it is rather than as a bare ``TypeError`` from inside pandas.
    """
    df = pd.DataFrame({"v": pd.Series([value], dtype=object)})
    message = f"Unsupported data type {type(value).__name__}"
    for call in (
        lambda: group_codes(df["v"]),
        lambda: group_ids(df, ["v"]),
        lambda: row_keys(df),
        lambda: distinct_rows(df),
        lambda: group_indices(df, ["v"]),
    ):
        with pytest.raises(NotImplementedError, match=message):
            call()


def test_boolean_and_categorical_columns_group() -> None:
    """Dtypes the hashing functions reject still group.

    :mod:`~tmlt.core.utils.pandas_grouping` rejects no dtype; only a value with
    no Python hash has no group key.
    """
    df = pd.DataFrame(
        {
            "flag": pd.Series([True, False, True], dtype=bool),
            "kind": pd.Series(["a", "b", "a"], dtype="category"),
        }
    )
    assert list(group_ids(df, ["flag", "kind"])) == [0, 1, 0]
    assert len(distinct_rows(df)) == 2
    assert distinct_rows(df).dtypes.equals(df.dtypes)


def test_a_categorical_missing_entry_is_a_null() -> None:
    """A categorical's missing entry is a null, not a NaN.

    A categorical stores one as the code ``-1`` and hands it back as
    ``np.nan``, which everywhere else here is a *value* -- pandas does not
    allow a NaN to be a category, so there is nothing else it could be. It used
    to be grouped as a NaN, which put a filled-in join key or payload in a
    group of its own rather than in the null group, and made a join's
    ``nulls_are_equal`` inert for a categorical key.
    """
    categorical = pd.DataFrame(
        {"v": pd.Series(["a", None, "b", None], dtype="category")}
    )
    objects = pd.DataFrame({"v": pd.Series(["a", None, "b", None], dtype=object)})

    assert list(group_ids(categorical, ["v"])) == [0, 1, 2, 1]
    assert list(row_keys(categorical)) == list(row_keys(objects))


def test_repeated_column_names_group_once() -> None:
    """Naming a column twice groups by it once, as Spark allows."""
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 2, 1]})
    assert list(group_ids(df, ["a", "a"])) == list(group_ids(df, ["a"]))
    assert len(group_indices(df, ["a", "a"])) == 2


def test_distinct_rows_keeps_every_column() -> None:
    """A subset decides which rows survive, never which columns are returned."""
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "z"]})
    result = distinct_rows(df, ["a"])
    assert list(result.columns) == ["a", "b"]
    assert list(result["b"]) == ["x", "z"]


def test_group_indices_keys_are_row_keys() -> None:
    """A group is reachable by the key ``row_keys`` gives any of its rows."""
    df = pd.DataFrame({"g": pd.Series(["a", None, "a", float("nan")], dtype=object)})
    groups = group_indices(df, ["g"])
    keys = row_keys(df, ["g"])
    for position, key in enumerate(keys):
        assert position in set(groups[key].tolist())


def test_row_keys_keeps_the_frames_index() -> None:
    """The keys are aligned with the frame's own index, not with its positions."""
    df = pd.DataFrame({"a": [1, 2]}, index=["x", "y"])
    assert list(row_keys(df).index) == ["x", "y"]


def test_distinct_rows_and_group_indices_ignore_the_frames_index() -> None:
    """A non-default index changes neither the groups nor their positions."""
    df = pd.DataFrame({"a": [1, 2, 1]}, index=[10, 20, 30])
    assert list(distinct_rows(df).index) == [0, 1]
    assert list(distinct_rows(df)["a"]) == [1, 2]
    assert [positions.tolist() for positions in group_indices(df, ["a"]).values()] == [
        [0, 2],
        [1],
    ]
