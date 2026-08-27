"""Tests that hold for both truncation implementations.

Every test in this module is parametrized over the ``backend`` fixture, and so
runs twice: once against the Spark implementations in
:mod:`~tmlt.core.utils.truncation`, and once against their pandas counterparts
in :mod:`~tmlt.core.utils.pandas_truncation`. The Spark session is only
requested by the Spark parameter, so the pandas runs never start a JVM.

The behavioral cases that used to live in
:mod:`test.unit.utils.test_truncation` are re-expressed here (that module now
keeps only its Spark-specific tests), together with the cases that only exist
because there are now two implementations to agree: an empty collection of
grouping columns, several grouping or key columns, nulls in grouping and key
columns, and thresholds outside the interesting range.

Which rows survive truncation depends on SHA-256 digests, so the assertions
here are the ones that do not: exact results where the hashes cannot matter
(``drop_large_groups``, thresholds that keep everything or nothing), and
otherwise the surviving row and key counts per group, whether the output is a
sub-multiset of the input, and whether every row of a surviving key is kept.
The digest-level agreement of the two implementations is checked by
:mod:`test.unit.utils.test_truncation_differential`.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import itertools
from collections import Counter
from test.unit.utils.truncation_testing import (
    TRUNCATION_FUNCTIONS,
    TruncationBackend,
    apply_truncation,
    normalized_rows,
)
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from tmlt.core.utils.testing import Case, assert_dataframe_equal, parametrize

################################################################################
# Helpers
################################################################################


def _frame(columns: Sequence[str], rows: Sequence[Tuple[Any, ...]]) -> pd.DataFrame:
    """Returns a dataframe with the given columns and rows.

    A column holding only integers gets an ``int64`` dtype, and every other
    column an ``object`` dtype. In particular a null never widens an integer
    column to a float one, which matters because a float NaN and a SQL null are
    different values to both implementations.

    Args:
        columns: The column names, in order.
        rows: The rows, as tuples in the order given by ``columns``.

    Returns:
        The assembled dataframe.
    """
    data: Dict[str, pd.Series] = {}
    for index, name in enumerate(columns):
        values = [row[index] for row in rows]
        series = pd.Series(values, dtype=object)
        if values and all(
            isinstance(value, int) and not isinstance(value, bool) for value in values
        ):
            series = series.astype("int64")
        data[name] = series
    return pd.DataFrame(data, columns=list(columns))


def _rows(
    df: pd.DataFrame, columns: Optional[Sequence[str]] = None
) -> List[Tuple[Any, ...]]:
    """Returns the given columns of a dataframe as normalized row tuples.

    The values are normalized by
    :func:`~test.unit.utils.truncation_testing.normalize_value`, so that nulls
    of every flavor collapse onto one sentinel, NaN onto another, and numbers
    are compared by value rather than by type -- a Spark round trip can widen
    an integer column.

    Args:
        df: The dataframe to read.
        columns: The columns to take, or None for all of them. An empty
            collection gives one empty tuple per row, which is what makes the
            whole frame a single group.

    Returns:
        One tuple per row of ``df``, in the frame's own row order.
    """
    names = [str(name) for name in (df.columns if columns is None else columns)]
    if not names:
        return [()] * len(df)
    return normalized_rows(df, names)


def _group_sizes(df: pd.DataFrame, grouping: Sequence[str]) -> Dict[Any, int]:
    """Returns the number of rows of each group of a dataframe."""
    return dict(Counter(_rows(df, grouping)))


def _assert_sub_multiset(actual: pd.DataFrame, original: pd.DataFrame) -> None:
    """Asserts that every row of ``actual`` is a row of ``original``.

    Multiplicities are respected: a row kept twice must appear at least twice
    in the input.
    """
    assert list(actual.columns) == list(original.columns), (
        f"Columns changed: {list(actual.columns)} vs {list(original.columns)}."
    )
    original_counts = Counter(_rows(original))
    for row, count in Counter(_rows(actual, list(original.columns))).items():
        assert count <= original_counts[row], (
            f"Row {row} was kept {count} times but appears "
            f"{original_counts[row]} times in the input."
        )


def _assert_truncated(
    actual: pd.DataFrame,
    original: pd.DataFrame,
    grouping: Sequence[str],
    threshold: int,
) -> None:
    """Asserts that each group kept as many of its rows as it could.

    Args:
        actual: The output of ``truncate_large_groups``.
        original: The input it was called on.
        grouping: The grouping columns it was called with.
        threshold: The threshold it was called with.
    """
    _assert_sub_multiset(actual, original)
    expected = {
        group: min(size, max(threshold, 0))
        for group, size in _group_sizes(original, grouping).items()
    }
    expected = {group: size for group, size in expected.items() if size}
    assert _group_sizes(actual, grouping) == expected


def _expected_dropped(
    df: pd.DataFrame, grouping: Sequence[str], threshold: int
) -> pd.DataFrame:
    """Returns the rows ``drop_large_groups`` must keep.

    Unlike the other two functions, this one hashes nothing, so its result is
    fully determined by the group sizes.

    Args:
        df: The input dataframe.
        grouping: The grouping columns.
        threshold: The threshold above which a group is dropped.

    Returns:
        The rows of ``df`` whose group has at most ``threshold`` rows.
    """
    sizes = _group_sizes(df, grouping)
    mask = np.array(
        [sizes[group] <= threshold for group in _rows(df, grouping)], dtype=bool
    )
    return df[mask].reset_index(drop=True)


def _assert_key_limited(
    actual: pd.DataFrame,
    original: pd.DataFrame,
    grouping: Sequence[str],
    keys: Sequence[str],
    threshold: int,
) -> None:
    """Asserts that each group kept as many whole keys as it could.

    A key is kept in full or not at all: every row of a surviving (group, key)
    pair must be in the output, with its original multiplicity.

    Args:
        actual: The output of ``limit_keys_per_group``.
        original: The input it was called on.
        grouping: The grouping columns it was called with.
        keys: The key columns it was called with.
        threshold: The threshold it was called with.
    """
    _assert_sub_multiset(actual, original)
    pair_columns = [*grouping, *keys]
    original_pairs = _rows(original, pair_columns)
    distinct: Dict[Any, Set[Any]] = {}
    for group, pair in zip(_rows(original, grouping), original_pairs):
        distinct.setdefault(group, set()).add(pair)
    surviving: Dict[Any, Set[Any]] = {}
    for group, pair in zip(_rows(actual, grouping), _rows(actual, pair_columns)):
        surviving.setdefault(group, set()).add(pair)

    assert set(surviving) <= set(distinct), "Truncation invented a group."
    for group, pairs in distinct.items():
        kept = surviving.get(group, set())
        assert not kept - pairs, f"Group {group} invented the keys {kept - pairs}."
        expected_count = min(len(pairs), max(threshold, 0))
        assert len(kept) == expected_count, (
            f"Group {group} kept {len(kept)} of its {len(pairs)} keys, "
            f"expected {expected_count}."
        )

    kept_pairs = {pair for pairs in surviving.values() for pair in pairs}
    expected_rows = Counter(
        row for row, pair in zip(_rows(original), original_pairs) if pair in kept_pairs
    )
    assert Counter(_rows(actual, list(original.columns))) == expected_rows, (
        "The rows of a surviving key were not all kept."
    )


_ALL_FUNCTIONS = tuple(
    Case(function)(function=function) for function in TRUNCATION_FUNCTIONS
)


################################################################################
# truncate_large_groups
################################################################################


@parametrize(
    Case("group-over-threshold")(
        threshold=2,
        rows=[(1, "x"), (1, "y"), (1, "z"), (1, "w")],
        expected_count=2,
    ),
    Case("group-under-threshold")(threshold=2, rows=[(1, "x")], expected_count=1),
    Case("zero-threshold")(
        threshold=0,
        rows=[(1, "x"), (1, "y"), (1, "z"), (1, "w")],
        expected_count=0,
    ),
)
def test_truncate_correctness(
    backend: TruncationBackend,
    threshold: int,
    rows: List[Tuple[Any, ...]],
    expected_count: int,
) -> None:
    """Tests that truncate_large_groups keeps the right number of rows."""
    df = _frame(["A", "B"], rows)
    actual = backend.truncate_large_groups(df, ["A"], threshold)
    assert len(actual) == expected_count
    _assert_truncated(actual, df, ["A"], threshold)


@parametrize(
    Case("one-row-per-group")(
        columns=["A"], rows=[(i,) for i in range(1000)], threshold=5
    ),
    Case("hundred-rows-per-group")(
        columns=["A", "B"], rows=[(i % 10, i) for i in range(1000)], threshold=5
    ),
)
def test_truncate_consistency(
    backend: TruncationBackend,
    columns: List[str],
    rows: List[Tuple[Any, ...]],
    threshold: int,
) -> None:
    """Tests that truncate_large_groups does not truncate randomly across calls."""
    df = _frame(columns, rows)
    expected = backend.truncate_large_groups(df, ["A"], threshold)
    for _ in range(5):
        assert_dataframe_equal(
            backend.truncate_large_groups(df, ["A"], threshold), expected
        )


def test_truncate_rows_dropped_consistently(backend: TruncationBackend) -> None:
    """Tests that truncate_large_groups drops the same rows for unchanged keys."""
    df1 = _frame(["W", "X"], [("A", 1), ("B", 2), ("B", 3)])
    df2 = _frame(["W", "X"], [("A", 0), ("A", 1), ("B", 2), ("B", 3)])

    df1_truncated = backend.truncate_large_groups(df1, ["W"], 1)
    df2_truncated = backend.truncate_large_groups(df2, ["W"], 1)
    assert_dataframe_equal(
        df1_truncated[df1_truncated["W"] == "B"],
        df2_truncated[df2_truncated["W"] == "B"],
    )


def test_truncate_order_agnostic(backend: TruncationBackend) -> None:
    """Tests that truncate_large_groups doesn't depend on row order."""
    rows = [(1, 2, "A"), (3, 4, "A"), (5, 6, "A"), (7, 8, "B")]
    truncated = [
        backend.truncate_large_groups(
            _frame(["W", "X", "Y"], list(permutation)), ["Y"], 1
        )
        for permutation in itertools.permutations(rows, 4)
    ]
    for other in truncated[1:]:
        assert_dataframe_equal(truncated[0], other)


def test_truncate_duplicates_not_clumped(backend: TruncationBackend) -> None:
    """Tests that truncate_large_groups doesn't clump duplicate rows together."""
    df = _frame(
        ["X", "Y", "Z"],
        [(1, 2, "A")] * 5 + [(2, 4, "A")] * 5,
    )
    actual = backend.truncate_large_groups(df, ["Z"], 5)
    assert len(actual) == 5
    # Both distinct rows must survive: five copies of one of them would mean the
    # duplicates were ordered as a block.
    assert len(actual.drop_duplicates()) == 2


################################################################################
# drop_large_groups
################################################################################


@parametrize(
    Case("one-group-too-large")(
        threshold=1,
        rows=[(1, "A"), (1, "B"), (2, "C")],
        expected=[(2, "C")],
    ),
    Case("no-group-too-large")(
        threshold=1,
        rows=[(1, "A"), (2, "C")],
        expected=[(1, "A"), (2, "C")],
    ),
    Case("group-one-over-threshold")(
        threshold=2,
        rows=[(1, "A"), (2, "C"), (2, "D"), (2, "E")],
        expected=[(1, "A")],
    ),
    Case("every-group-too-large")(
        threshold=1,
        rows=[(1, "A"), (1, "B"), (2, "C"), (2, "D"), (2, "E")],
        expected=[],
    ),
    Case("zero-threshold")(
        threshold=0,
        rows=[(1, "x"), (2, "y"), (3, "z"), (3, "w")],
        expected=[],
    ),
)
def test_drop_correctness(
    backend: TruncationBackend,
    threshold: int,
    rows: List[Tuple[Any, ...]],
    expected: List[Tuple[Any, ...]],
) -> None:
    """Tests that drop_large_groups keeps exactly the small enough groups."""
    df = _frame(["A", "B"], rows)
    actual = backend.drop_large_groups(df, ["A"], threshold)
    assert_dataframe_equal(
        actual, pd.DataFrame.from_records(expected, columns=["A", "B"])
    )


################################################################################
# Thresholds outside the interesting range
################################################################################


@parametrize(*_ALL_FUNCTIONS)
@parametrize(
    Case("zero")(threshold=0),
    Case("negative")(threshold=-1),
    Case("very-negative")(threshold=-(10**9)),
)
def test_nonpositive_threshold_keeps_nothing(
    backend: TruncationBackend, function: str, threshold: int
) -> None:
    """Tests that a threshold of zero or less keeps nothing, and does not raise."""
    df = _frame(
        ["A", "B"],
        [("g1", "k1"), ("g1", "k2"), ("g2", "k1"), ("g2", "k1")],
    )
    actual = apply_truncation(backend, function, df, ["A"], ["B"], threshold)
    assert len(actual) == 0
    assert list(actual.columns) == ["A", "B"]


@parametrize(*_ALL_FUNCTIONS)
def test_huge_threshold_keeps_everything(
    backend: TruncationBackend, function: str
) -> None:
    """Tests that a threshold larger than the frame keeps every row."""
    df = _frame(
        ["A", "B"],
        [("g1", "k1"), ("g1", "k2"), ("g2", "k1"), ("g2", "k1")],
    )
    actual = apply_truncation(backend, function, df, ["A"], ["B"], 10**9)
    assert_dataframe_equal(actual, df)


################################################################################
# Empty grouping columns
################################################################################

_UNGROUPED_ROWS = [
    ("g1", "b1"),
    ("g1", "b2"),
    ("g2", "b1"),
    ("g2", "b2"),
    ("g2", "b3"),
]


@parametrize(
    Case("keeps-nothing")(threshold=0),
    Case("keeps-some")(threshold=2),
    Case("keeps-all-exactly")(threshold=5),
    Case("keeps-all-with-room")(threshold=10),
)
def test_truncate_empty_grouping_columns(
    backend: TruncationBackend, threshold: int
) -> None:
    """Tests that no grouping columns makes the whole frame one group."""
    df = _frame(["A", "B"], _UNGROUPED_ROWS)
    actual = backend.truncate_large_groups(df, [], threshold)
    assert len(actual) == min(threshold, len(df))
    _assert_truncated(actual, df, [], threshold)


@parametrize(
    Case("frame-too-large")(threshold=4),
    Case("frame-exactly-at-threshold")(threshold=5),
    Case("frame-under-threshold")(threshold=6),
)
def test_drop_empty_grouping_columns(
    backend: TruncationBackend, threshold: int
) -> None:
    """Tests that with no grouping columns the whole frame is kept or dropped."""
    df = _frame(["A", "B"], _UNGROUPED_ROWS)
    actual = backend.drop_large_groups(df, [], threshold)
    assert_dataframe_equal(actual, _expected_dropped(df, [], threshold))


@parametrize(
    Case("one-key")(threshold=1),
    Case("two-keys")(threshold=2),
    Case("every-key")(threshold=3),
)
def test_limit_keys_empty_grouping_columns(
    backend: TruncationBackend, threshold: int
) -> None:
    """Tests that with no grouping columns the keys of the whole frame are limited."""
    df = _frame(["A", "B"], _UNGROUPED_ROWS)
    actual = backend.limit_keys_per_group(df, [], ["B"], threshold)
    _assert_key_limited(actual, df, [], ["B"], threshold)


@parametrize(
    Case("keeps-nothing")(threshold=0),
    Case("keeps-everything")(threshold=1),
    Case("keeps-everything-with-room")(threshold=10**9),
)
def test_limit_keys_empty_grouping_and_key_columns(
    backend: TruncationBackend, threshold: int
) -> None:
    """Tests that with no grouping or key columns the frame is one group, one key."""
    df = _frame(["A", "B"], _UNGROUPED_ROWS)
    actual = backend.limit_keys_per_group(df, [], [], threshold)
    if threshold >= 1:
        assert_dataframe_equal(actual, df)
    _assert_key_limited(actual, df, [], [], threshold)


################################################################################
# Several grouping and key columns
################################################################################

# Group sizes are 3, 1, 2, and 4.
_MULTI_GROUPING_ROWS = [
    ("a", 1, "x"),
    ("a", 1, "y"),
    ("a", 1, "z"),
    ("a", 2, "x"),
    ("b", 1, "x"),
    ("b", 1, "y"),
    ("b", 2, "x"),
    ("b", 2, "y"),
    ("b", 2, "z"),
    ("b", 2, "w"),
]

# Group g1 has three distinct keys and group g2 four, two of which have a null
# in one of the key columns.
_MULTI_KEY_ROWS = [
    ("g1", "x", "1", "p"),
    ("g1", "x", "2", "q"),
    ("g1", "y", "1", "r"),
    ("g1", "x", "1", "s"),
    ("g2", "x", "1", "t"),
    ("g2", None, "1", "u"),
    ("g2", "x", None, "v"),
    ("g2", None, None, "w"),
]


@parametrize(
    Case("threshold-1")(threshold=1),
    Case("threshold-2")(threshold=2),
    Case("threshold-3")(threshold=3),
    Case("threshold-4")(threshold=4),
)
def test_truncate_multi_column_grouping(
    backend: TruncationBackend, threshold: int
) -> None:
    """Tests truncate_large_groups with a group defined by two columns."""
    df = _frame(["G1", "G2", "V"], _MULTI_GROUPING_ROWS)
    actual = backend.truncate_large_groups(df, ["G1", "G2"], threshold)
    _assert_truncated(actual, df, ["G1", "G2"], threshold)


@parametrize(
    Case("threshold-1")(threshold=1),
    Case("threshold-2")(threshold=2),
    Case("threshold-3")(threshold=3),
    Case("threshold-4")(threshold=4),
)
def test_drop_multi_column_grouping(backend: TruncationBackend, threshold: int) -> None:
    """Tests drop_large_groups with a group defined by two columns."""
    df = _frame(["G1", "G2", "V"], _MULTI_GROUPING_ROWS)
    actual = backend.drop_large_groups(df, ["G1", "G2"], threshold)
    assert_dataframe_equal(actual, _expected_dropped(df, ["G1", "G2"], threshold))


@parametrize(
    Case("threshold-1")(threshold=1),
    Case("threshold-2")(threshold=2),
    Case("threshold-3")(threshold=3),
    Case("threshold-4")(threshold=4),
    Case("threshold-5")(threshold=5),
)
def test_limit_keys_multi_column_keys(
    backend: TruncationBackend, threshold: int
) -> None:
    """Tests limit_keys_per_group with a key defined by two columns."""
    df = _frame(["G", "K1", "K2", "V"], _MULTI_KEY_ROWS)
    actual = backend.limit_keys_per_group(df, ["G"], ["K1", "K2"], threshold)
    _assert_key_limited(actual, df, ["G"], ["K1", "K2"], threshold)


@parametrize(
    Case("threshold-1")(threshold=1),
    Case("threshold-2")(threshold=2),
    Case("threshold-3")(threshold=3),
)
def test_limit_keys_multi_column_grouping_and_keys(
    backend: TruncationBackend, threshold: int
) -> None:
    """Tests limit_keys_per_group with two grouping and two key columns."""
    df = _frame(
        ["G1", "G2", "K1", "K2", "V"],
        [
            ("a", 1, "x", "1", "p"),
            ("a", 1, "x", "2", "q"),
            ("a", 1, "y", "1", "r"),
            ("a", 2, "x", "1", "s"),
            ("b", 1, "x", "1", "t"),
            ("b", 1, "y", "2", "u"),
            ("b", 1, "y", "2", "v"),
            ("b", 1, "z", "3", "w"),
        ],
    )
    actual = backend.limit_keys_per_group(df, ["G1", "G2"], ["K1", "K2"], threshold)
    _assert_key_limited(actual, df, ["G1", "G2"], ["K1", "K2"], threshold)


################################################################################
# Nulls in grouping and key columns
################################################################################

# The null group has three rows and two distinct keys, group g1 has two rows and
# two distinct keys (one of them null), and group g2 has two rows sharing a
# single null key.
_NULL_KEY_ROWS = [
    (None, "k1", "a"),
    (None, "k2", "b"),
    (None, "k1", "c"),
    ("g1", "k1", "d"),
    ("g1", None, "e"),
    ("g2", None, "f"),
    ("g2", None, "g"),
]


@parametrize(
    Case("threshold-1")(threshold=1),
    Case("threshold-2")(threshold=2),
    Case("threshold-3")(threshold=3),
)
def test_truncate_nulls_in_grouping_column(
    backend: TruncationBackend, threshold: int
) -> None:
    """Tests that a null grouping value is a group of its own, not a dropped row."""
    df = _frame(["G", "K", "V"], _NULL_KEY_ROWS)
    actual = backend.truncate_large_groups(df, ["G"], threshold)
    _assert_truncated(actual, df, ["G"], threshold)


@parametrize(
    Case("threshold-1")(threshold=1),
    Case("threshold-2")(threshold=2),
    Case("threshold-3")(threshold=3),
)
def test_drop_nulls_in_grouping_column(
    backend: TruncationBackend, threshold: int
) -> None:
    """Tests that drop_large_groups counts a null group like any other."""
    df = _frame(["G", "K", "V"], _NULL_KEY_ROWS)
    actual = backend.drop_large_groups(df, ["G"], threshold)
    assert_dataframe_equal(actual, _expected_dropped(df, ["G"], threshold))


@parametrize(
    Case("threshold-1")(threshold=1),
    Case("threshold-2")(threshold=2),
    Case("threshold-3")(threshold=3),
)
def test_limit_keys_nulls_in_grouping_and_key_columns(
    backend: TruncationBackend, threshold: int
) -> None:
    """Tests that a null key is a key of its own, kept or dropped as a whole."""
    df = _frame(["G", "K", "V"], _NULL_KEY_ROWS)
    actual = backend.limit_keys_per_group(df, ["G"], ["K"], threshold)
    _assert_key_limited(actual, df, ["G"], ["K"], threshold)


################################################################################
# Thresholds at or above the number of distinct keys
################################################################################


@parametrize(
    Case("exactly-the-largest-key-count")(threshold=3),
    Case("one-more-than-the-largest-key-count")(threshold=4),
    Case("far-more-than-the-largest-key-count")(threshold=10**9),
)
def test_limit_keys_threshold_at_least_distinct_keys_is_identity(
    backend: TruncationBackend, threshold: int
) -> None:
    """Tests that limit_keys_per_group keeps everything once no group is over."""
    # The largest number of distinct keys in a group is three, in group a1.
    df = _frame(
        ["A", "B", "C"],
        [
            ("a1", "b1", "x"),
            ("a1", "b2", "y"),
            ("a1", "b3", "z"),
            ("a2", "b1", "x"),
            ("a2", "b1", "y"),
            ("a3", "b9", "z"),
        ],
    )
    actual = backend.limit_keys_per_group(df, ["A"], ["B"], threshold)
    assert_dataframe_equal(actual, df)
    _assert_key_limited(actual, df, ["A"], ["B"], threshold)
