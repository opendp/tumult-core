"""Tests for :mod:`~tmlt.core.utils.pandas_grouped_table`.

These mirror ``test_grouped_dataframe.py``, which covers the Spark twin, and add
the properties that only the pandas implementation can get wrong or that only it
promises:

* the guards the constructor copies from the Spark one, including the one case
  -- group keys with rows but no columns -- where an empty pandas frame and an
  empty Spark one part company,
* that a group key holding a null is a group of its own, distinct from one
  holding a NaN, which is :mod:`~tmlt.core.utils.pandas_grouping`' notion of
  identity rather than ``pandas.DataFrame.groupby``'s,
* that an aggregation's output is a function of the *public* group keys alone,
  pinned by permuting the input's rows and requiring byte-identical output, and
* that nothing here modifies the frames it is given.

Nothing in this module needs a Spark session: the differential tests against the
Spark implementation live with the transformations that use this one.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

from typing import Any, List, Optional

import numpy as np
import pandas as pd
import pytest

from tmlt.core.utils.pandas_grouped_table import PandasGroupedTable
from tmlt.core.utils.pandas_grouping import row_keys
from tmlt.core.utils.testing import (
    Case,
    assert_property_immutability,
    get_all_props,
    parametrize,
)

# The frame the aggregation examples group, and the keys they group it by: one
# key with no rows in the frame ("a0"), one with rows ("a1"), and one group in
# the frame that is not a key at all ("a2").
_FRAME = pd.DataFrame({"A": ["a1", "a1", "a2", "a2"], "X": [2, 3, 5, -1]})
_KEYS = pd.DataFrame({"A": ["a0", "a1"]})


def _rows_without_columns(rows: int) -> pd.DataFrame:
    """Returns a DataFrame with the given number of rows and no columns.

    Args:
        rows: The number of rows.
    """
    return pd.DataFrame(index=pd.RangeIndex(rows))


################################################################################
# Constructor
################################################################################


@parametrize(
    Case("duplicate-dataframe-columns")(
        dataframe=pd.DataFrame([(1, 2)], columns=["A", "A"]),
        group_keys=pd.DataFrame([(1,)], columns=["A"]),
        error_msg="DataFrame contains duplicate column names",
    ),
    Case("duplicate-group-key-columns")(
        dataframe=pd.DataFrame([(1, 2)], columns=["A", "B"]),
        group_keys=pd.DataFrame([(1, 2)], columns=["A", "A"]),
        error_msg="Group keys contains duplicate column names",
    ),
    Case("invalid-groupby-columns")(
        dataframe=pd.DataFrame([(1, 2)], columns=["A", "Z"]),
        group_keys=pd.DataFrame([(1,)], columns=["B"]),
        error_msg="Invalid groupby columns",
    ),
    Case("rows-without-columns")(
        dataframe=pd.DataFrame([(1, 2)], columns=["A", "B"]),
        group_keys=_rows_without_columns(3),
        error_msg="Groupby keys cannot have records without columns.",
    ),
)
def test_constructor_invalid_inputs(
    dataframe: pd.DataFrame, group_keys: pd.DataFrame, error_msg: str
) -> None:
    """An invalid combination of frames is rejected."""
    with pytest.raises(ValueError, match=error_msg):
        PandasGroupedTable(dataframe=dataframe, group_keys=group_keys)


def test_constructor_drops_duplicate_group_keys() -> None:
    """Duplicate group keys are silently dropped."""
    table = PandasGroupedTable(
        dataframe=pd.DataFrame({"A": [1], "B": [2]}),
        group_keys=pd.DataFrame({"A": [1, 1]}),
    )
    assert table.group_keys is not None
    pd.testing.assert_frame_equal(table.group_keys, pd.DataFrame({"A": [1]}))


def test_constructor_keeps_null_and_nan_keys_apart() -> None:
    """A null group key and a NaN group key are two keys, as they are in Spark."""
    keys = pd.DataFrame({"A": pd.Series([None, float("nan"), None], dtype=object)})
    table = PandasGroupedTable(
        dataframe=pd.DataFrame({"A": pd.Series([None], dtype=object), "B": [1]}),
        group_keys=keys,
    )
    assert table.group_keys is not None
    assert len(table.group_keys) == 2
    assert table.group_keys["A"][0] is None
    assert np.isnan(table.group_keys["A"][1])


@parametrize(
    Case("none")(group_keys=None),
    Case("no-columns-no-rows")(group_keys=_rows_without_columns(0)),
    Case("no-columns-no-rows-plain")(group_keys=pd.DataFrame()),
)
def test_total_aggregation_group_keys(group_keys: Optional[pd.DataFrame]) -> None:
    """Group keys with no columns and no rows mean a total aggregation."""
    table = PandasGroupedTable(dataframe=_FRAME, group_keys=group_keys)
    assert table.group_keys is None
    assert table.groupby_columns == []
    assert table.get_groups() == {}


@parametrize(
    Case(prop)(prop_name=prop) for (prop,) in get_all_props(PandasGroupedTable)
)
def test_property_immutability(prop_name: str) -> None:
    """The properties cannot be mutated through the values they return."""
    table = PandasGroupedTable(dataframe=_FRAME, group_keys=_KEYS)
    assert_property_immutability(table, prop_name)


def test_properties() -> None:
    """The properties have the expected values."""
    table = PandasGroupedTable(dataframe=_FRAME, group_keys=_KEYS)
    pd.testing.assert_frame_equal(table.dataframe, _FRAME)
    assert table.group_keys is not None
    pd.testing.assert_frame_equal(table.group_keys, _KEYS)
    assert table.groupby_columns == ["A"]


################################################################################
# agg
################################################################################


def test_agg_fills_and_drops_group_keys() -> None:
    """Declared keys with no rows are filled and undeclared groups are dropped."""
    actual = PandasGroupedTable(dataframe=_FRAME, group_keys=_KEYS).agg(
        len, fill_value=0, output_column="count"
    )
    pd.testing.assert_frame_equal(
        actual, pd.DataFrame({"A": ["a0", "a1"], "count": [0, 2]})
    )


def test_agg_orders_output_by_group_keys() -> None:
    """The output is in the group keys' order, whatever order the data is in."""
    keys = pd.DataFrame({"A": ["a2", "a0", "a1"]})
    actual = PandasGroupedTable(dataframe=_FRAME, group_keys=keys).agg(
        len, fill_value=0, output_column="count"
    )
    pd.testing.assert_frame_equal(
        actual, pd.DataFrame({"A": ["a2", "a0", "a1"], "count": [2, 0, 2]})
    )
    assert list(actual.index) == [0, 1, 2]


def test_agg_is_permutation_invariant() -> None:
    """Permuting the input's rows leaves the output byte-identical.

    The output's row order is a side channel unless it is a function of the
    public group keys alone; this is what pins that it is.
    """
    keys = pd.DataFrame({"A": ["a0", "a1", "a2"]})
    aggregations: List[Any] = [len, lambda group: int(group["X"].sum())]
    for func in aggregations:
        expected = PandasGroupedTable(dataframe=_FRAME, group_keys=keys).agg(
            func, fill_value=0, output_column="v"
        )
        for permutation in ([3, 1, 0, 2], [2, 3, 1, 0], [1, 0, 3, 2]):
            permuted = _FRAME.iloc[permutation].reset_index(drop=True)
            actual = PandasGroupedTable(dataframe=permuted, group_keys=keys).agg(
                func, fill_value=0, output_column="v"
            )
            pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_agg_is_invariant_to_group_key_duplicates_and_index() -> None:
    """Duplicated group keys and an unusual index do not change the output."""
    expected = PandasGroupedTable(dataframe=_FRAME, group_keys=_KEYS).agg(
        len, fill_value=0, output_column="count"
    )
    keys = pd.DataFrame({"A": ["a0", "a1", "a0"]}, index=[7, 2, 9])
    actual = PandasGroupedTable(dataframe=_FRAME, group_keys=keys).agg(
        len, fill_value=0, output_column="count"
    )
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_agg_on_empty_frame_fills_every_key() -> None:
    """Every declared key is filled when the frame has no rows at all."""
    empty = _FRAME.iloc[:0]
    actual = PandasGroupedTable(dataframe=empty, group_keys=_KEYS).agg(
        len, fill_value=-1, output_column="count"
    )
    pd.testing.assert_frame_equal(
        actual, pd.DataFrame({"A": ["a0", "a1"], "count": [-1, -1]})
    )


def test_agg_with_no_group_keys_at_all() -> None:
    """An empty frame of group keys with columns produces no rows."""
    keys = pd.DataFrame({"A": pd.Series([], dtype=object)})
    actual = PandasGroupedTable(dataframe=_FRAME, group_keys=keys).agg(
        len, fill_value=0, output_column="count"
    )
    assert list(actual.columns) == ["A", "count"]
    assert len(actual) == 0


@parametrize(
    Case("none")(group_keys=None),
    Case("no-columns")(group_keys=_rows_without_columns(0)),
)
def test_total_agg(group_keys: Optional[pd.DataFrame]) -> None:
    """A total aggregation produces exactly one row, holding one column."""
    actual = PandasGroupedTable(dataframe=_FRAME, group_keys=group_keys).agg(
        len, fill_value=0, output_column="count"
    )
    pd.testing.assert_frame_equal(actual, pd.DataFrame({"count": [4]}))


def test_total_agg_fill_value() -> None:
    """A total aggregation of an empty frame is the fill value, not an aggregate."""
    actual = PandasGroupedTable(dataframe=_FRAME.iloc[:0], group_keys=None).agg(
        _unreachable, fill_value=10, output_column="count"
    )
    pd.testing.assert_frame_equal(actual, pd.DataFrame({"count": [10]}))


def _unreachable(group: pd.DataFrame) -> Any:
    """Fails if called; a total aggregation of an empty frame must not call func.

    Args:
        group: The group that should never have been passed.
    """
    raise AssertionError(f"func was called with {len(group)} rows")


def test_agg_does_not_override_valid_nulls() -> None:
    """A group that exists keeps its aggregate, even when that is null."""
    frame = pd.DataFrame({"X": ["A", "B"], "Y": pd.array([1, None], dtype="Int64")})
    actual = PandasGroupedTable(
        dataframe=frame, group_keys=pd.DataFrame({"X": ["A", "B", "C"]})
    ).agg(lambda group: group["Y"].sum(min_count=1), fill_value=0, output_column="sum")
    assert list(actual["X"]) == ["A", "B", "C"]
    assert actual["sum"][0] == 1
    assert pd.isna(actual["sum"][1])
    assert actual["sum"][2] == 0


def test_agg_passes_every_column_of_the_group() -> None:
    """The aggregation function sees the groupby columns too, indexed from zero."""
    seen = {}

    def record(group: pd.DataFrame) -> int:
        seen[group["A"][0]] = list(group.columns)
        assert list(group.index) == list(range(len(group)))
        return len(group)

    PandasGroupedTable(dataframe=_FRAME, group_keys=_KEYS).agg(
        record, fill_value=0, output_column="count"
    )
    assert seen == {"a1": ["A", "X"]}


def test_agg_groups_null_keys() -> None:
    """Null group keys are grouped rather than dropped, and NaN is a key of its own."""
    frame = pd.DataFrame(
        {
            "A": pd.Series([None, float("nan"), None, "a0"], dtype=object),
            "B": [1, 2, 3, 4],
        }
    )
    keys = pd.DataFrame(
        {"A": pd.Series([None, float("nan"), "a1"], dtype=object)},
    )
    actual = PandasGroupedTable(dataframe=frame, group_keys=keys).agg(
        len, fill_value=0, output_column="count"
    )
    assert list(actual["count"]) == [2, 1, 0]


################################################################################
# agg_by_position
################################################################################


def _size(positions: np.ndarray) -> int:
    """Returns how many positions there are.

    Args:
        positions: One group's row positions.
    """
    return int(positions.size)


@parametrize(
    Case("groups")(group_keys=_KEYS),
    Case("empty-group-keys")(group_keys=pd.DataFrame({"A": []})),
    Case("total")(group_keys=None),
)
def test_agg_by_position_agrees_with_agg(group_keys: Optional[pd.DataFrame]) -> None:
    """Counting positions is counting rows, output frame for output frame."""
    table = PandasGroupedTable(dataframe=_FRAME, group_keys=group_keys)
    pd.testing.assert_frame_equal(
        table.agg_by_position(_size, fill_value=0, output_column="count"),
        table.agg(len, fill_value=0, output_column="count"),
    )


def test_agg_by_position_of_an_empty_frame() -> None:
    """An empty frame fills every group key, and a total aggregation's one row."""
    empty = _FRAME.iloc[:0]
    grouped = PandasGroupedTable(dataframe=empty, group_keys=_KEYS)
    pd.testing.assert_frame_equal(
        grouped.agg_by_position(_size, fill_value=-1, output_column="count"),
        pd.DataFrame({"A": ["a0", "a1"], "count": [-1, -1]}),
    )
    total = PandasGroupedTable(dataframe=empty, group_keys=None)
    pd.testing.assert_frame_equal(
        total.agg_by_position(_size, fill_value=-1, output_column="count"),
        pd.DataFrame({"count": [-1]}),
    )


def test_agg_by_position_passes_the_groups_positions() -> None:
    """The positions are the group's rows in the table, in ascending order."""
    frame = pd.DataFrame({"A": ["a1", "a2", "a1", "a1"], "X": [1, 2, 3, 4]})
    seen = {}

    def record(positions: np.ndarray) -> int:
        seen[frame["A"][positions[0]]] = list(positions)
        return int(positions.size)

    PandasGroupedTable(
        dataframe=frame, group_keys=pd.DataFrame({"A": ["a1", "a2"]})
    ).agg_by_position(record, fill_value=0, output_column="count")
    assert seen == {"a1": [0, 2, 3], "a2": [1]}


################################################################################
# get_groups and select
################################################################################


def test_get_groups() -> None:
    """get_groups returns one frame per group key, empty ones included."""
    frame = pd.DataFrame(
        {
            "X": pd.Series(["A", "B", "B", None, None], dtype=object),
            "Y": [1, 2, 4, 2, 3],
        }
    )
    keys = pd.DataFrame({"X": pd.Series(["A", "B", "C", None], dtype=object)})
    groups = PandasGroupedTable(dataframe=frame, group_keys=keys).get_groups()
    expected = {
        "A": [1],
        "B": [2, 4],
        "C": [],
        None: [2, 3],
    }
    keys_by_value = dict(zip(keys["X"], row_keys(keys, ["X"])))
    assert list(groups) == [keys_by_value[value] for value in expected]
    for value, values in expected.items():
        group = groups[keys_by_value[value]]
        assert list(group.columns) == ["Y"]
        assert list(group["Y"]) == values
        assert list(group.index) == list(range(len(values)))


def test_select() -> None:
    """Select keeps the named columns and the group keys."""
    table = PandasGroupedTable(
        dataframe=pd.DataFrame({"A": [1], "B": [2], "C": [3]}),
        group_keys=pd.DataFrame({"A": [1]}),
    )
    selected = table.select(["A", "B"])
    pd.testing.assert_frame_equal(
        selected.dataframe, pd.DataFrame({"A": [1], "B": [2]})
    )
    assert selected.group_keys is not None
    pd.testing.assert_frame_equal(selected.group_keys, pd.DataFrame({"A": [1]}))
    assert selected.groupby_columns == ["A"]


@parametrize(
    Case("duplicates")(columns=["A", "A"], error_msg="List contains duplicate"),
    Case("missing-groupby-column")(
        columns=["B"], error_msg="Groupby columns must be selected."
    ),
    Case("unknown-column")(columns=["A", "Z"], error_msg=r"Invalid columns: \['Z'\]"),
)
def test_select_invalid_inputs(columns: List[str], error_msg: str) -> None:
    """Select rejects a column list it cannot honor."""
    table = PandasGroupedTable(
        dataframe=pd.DataFrame({"A": [1], "B": [2]}),
        group_keys=pd.DataFrame({"A": [1]}),
    )
    with pytest.raises(ValueError, match=error_msg):
        table.select(columns)


################################################################################
# Immutability of the inputs
################################################################################


def test_nothing_modifies_the_input_frames() -> None:
    """Constructing a table and using it leaves the caller's frames unchanged."""
    frame = _FRAME.copy()
    keys = pd.DataFrame({"A": ["a0", "a1", "a1"]})
    frame_before = frame.copy()
    keys_before = keys.copy()

    table = PandasGroupedTable(dataframe=frame, group_keys=keys)
    table.agg(len, fill_value=0, output_column="count")
    table.get_groups()
    table.select(["A", "X"])

    pd.testing.assert_frame_equal(frame, frame_before)
    pd.testing.assert_frame_equal(keys, keys_before)


def test_nothing_reindexes_the_frame_the_table_holds() -> None:
    """A frame with an index of its own comes back with it, and its groups do not.

    The per-group selections are reindexed in place, which costs nothing
    because a selection is already a copy. The frame the table holds is not one
    of those, and reindexing *it* in place would reindex the caller's frame
    under it -- the one way this optimization can go wrong.
    """
    frame = pd.DataFrame(
        {"A": ["a1", "a1", "a2", "a2"], "X": [2, 3, 5, -1]}, index=[10, 11, 12, 13]
    )
    frame_before = frame.copy()
    indices: List[List[int]] = []

    def record(group: pd.DataFrame) -> int:
        indices.append(list(group.index))
        return len(group)

    for group_keys in (_KEYS, None):
        table = PandasGroupedTable(dataframe=frame, group_keys=group_keys)
        table.agg(record, fill_value=0, output_column="count")
        table.agg_by_position(_size, fill_value=0, output_column="count")
        for group in table.get_groups().values():
            indices.append(list(group.index))

    # Every frame handed out is indexed from zero: the "a1" group's two rows,
    # then get_groups' empty "a0" and its "a1", then the whole frame for the
    # total aggregation (whose get_groups has no groups to hand out).
    assert indices == [[0, 1], [], [0, 1], [0, 1, 2, 3]]
    pd.testing.assert_frame_equal(frame, frame_before)


def test_agg_output_is_not_a_view_of_the_group_keys() -> None:
    """Modifying an aggregation's output does not modify the group keys."""
    table = PandasGroupedTable(dataframe=_FRAME, group_keys=_KEYS)
    output = table.agg(len, fill_value=0, output_column="count")
    output.loc[0, "A"] = "modified"
    assert table.group_keys is not None
    assert list(table.group_keys["A"]) == ["a0", "a1"]
    pd.testing.assert_frame_equal(_KEYS, pd.DataFrame({"A": ["a0", "a1"]}))
