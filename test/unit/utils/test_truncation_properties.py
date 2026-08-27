"""Stability properties of the truncation utilities.

Every test in this module runs against both truncation implementations -- the
Spark one in :mod:`~tmlt.core.utils.truncation` and the pandas one in
:mod:`~tmlt.core.utils.pandas_truncation` -- through the backend fixture, and
checks a property that must hold for whatever rows the implementation picks. The
parity and differential suites pin down *which* rows are kept; this module pins
down what the choice is allowed to be at all:

* **Bounded output.** ``truncate_large_groups`` keeps at most ``threshold`` rows
  per group, and ``limit_keys_per_group`` at most ``threshold`` distinct key
  tuples per group.
* **Subset.** The output is a sub-multiset of the input: no row is invented,
  duplicated, or altered.
* **Pass-through.** A group that is already within the threshold is kept whole.
* **All or nothing.** ``drop_large_groups`` keeps every row of a group or none.
* **Key completeness.** ``limit_keys_per_group`` keeps every row of every key it
  keeps.
* **Locality.** Editing one group never changes which rows of another group
  survive.
* **Stability.** For neighboring inputs, the distance between the outputs is at
  most ``threshold`` times the input distance under
  ``IfGroupedBy(grouping_columns, SymmetricDifference())``, whose distance
  counts an added or removed group once and a modified group twice (see
  :func:`~test.unit.utils.truncation_testing.grouped_symdiff_distance`). The
  output distance is measured with the multiset symmetric difference for
  ``truncate_large_groups`` and ``drop_large_groups``, and per (group, key) pair
  for ``limit_keys_per_group``, which bounds keys rather than rows.

Frames come from the seeded generator in
:mod:`~test.unit.utils.truncation_testing`, and neighbors are built from them by
removing a row, removing a group, duplicating a row, or editing one payload
value. Most frames carry a unique ``row_id`` column, which lets every property
be stated over sets of surviving row ids and evaluated against the *input*
rows -- so none of the assertions depend on how a Spark round trip renders
nulls, NaNs, or nullable dtypes. One shape deliberately has no ``row_id`` and
many duplicate rows, to exercise the per-duplicate salt; its properties are
stated over whole rows instead.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import random
from collections import Counter
from dataclasses import replace
from functools import lru_cache
from test.unit.utils.truncation_testing import (
    ROW_ID_COLUMN,
    SIMPLE_DTYPE_MENU,
    TRUNCATION_FUNCTIONS,
    EdgeCase,
    TruncationBackend,
    apply_truncation,
    assert_no_conflating_values,
    frame_row_ids,
    grouped_symdiff_distance,
    multiset_symdiff,
    normalize_value,
    random_frame,
)
from typing import Any, Dict, List, Sequence, Set, Tuple

import pandas as pd
import pytest

################################################################################
# Frames
################################################################################

#: Frame shapes, as keyword arguments for
#: :func:`~test.unit.utils.truncation_testing.random_frame`. Column kinds are
#: taken from the menu in order -- grouping columns, then key columns, then
#: payload columns -- so every menu here keeps strings and integers in the
#: grouping and key columns and puts the floating point and nullable kinds in
#: the payload. None of the menus include timestamps, which would need a UTC
#: Spark session; the differential suite covers those.
_SHAPES: Dict[str, Dict[str, Any]] = {
    "string-groups": {
        "dtype_menu": ("string", "string", "float64", "Float64"),
        "n_rows": 40,
        "n_groups": 4,
        "dup_rate": 0.25,
        "n_grouping_columns": 1,
        "n_key_columns": 1,
        "n_payload_columns": 2,
        "n_key_values": 5,
        "null_rate": 0.15,
    },
    "int-groups": {
        "dtype_menu": ("int64", "Int64", "string_dtype", "float32"),
        "n_rows": 36,
        "n_groups": 3,
        "dup_rate": 0.3,
        "n_grouping_columns": 1,
        "n_key_columns": 1,
        "n_payload_columns": 2,
        "n_key_values": 4,
        "null_rate": 0.2,
    },
    "multi-column": {
        "dtype_menu": ("string", "int64", "string", "Int64", "float64"),
        "n_rows": 45,
        "n_groups": 3,
        "dup_rate": 0.2,
        "n_grouping_columns": 2,
        "n_key_columns": 2,
        "n_payload_columns": 1,
        "n_key_values": 3,
        "null_rate": 0.1,
    },
}

#: The shape with duplicate rows and no ``row_id`` column, which is what makes
#: the per-duplicate salt observable. Its columns are nulls-free strings and
#: integers, so its rows survive a Spark round trip unchanged and can be
#: compared directly.
_DUPLICATES_SHAPE: Dict[str, Any] = {
    "dtype_menu": SIMPLE_DTYPE_MENU,
    "n_rows": 30,
    "n_groups": 3,
    "dup_rate": 0.5,
    "n_grouping_columns": 1,
    "n_key_columns": 1,
    "n_payload_columns": 1,
    "n_key_values": 4,
    "null_rate": 0.0,
    "with_row_id": False,
}

_SHAPE_IDS: Tuple[str, ...] = tuple(_SHAPES)

#: The shapes the stability tests that are not marked slow use: one whose groups
#: are all much larger than the thresholds, and one whose group sizes straddle
#: them, so that ``drop_large_groups`` keeps some groups and drops others.
_STABILITY_SHAPE_IDS: Tuple[str, ...] = ("string-groups", "multi-column")

_DUPLICATES_ID = "duplicates"

#: Thresholds for the tests that are not marked slow. Against the generated
#: group sizes, 2 truncates every group, while 5 leaves some groups untouched
#: and truncates others -- which is where the pass-through and all-or-nothing
#: properties have something to say.
_THRESHOLDS: Tuple[int, ...] = (2, 5)

#: Thresholds for the duplicate-row shape, whose groups all hold ten rows: 2
#: truncates them all, and 10 is exactly their size, so a neighbor with one
#: extra row has a group cross the boundary.
_DUPLICATE_THRESHOLDS: Tuple[int, ...] = (2, 10)

#: The ways a neighboring frame is derived from a frame. Removing a row from a
#: group with other rows in it, duplicating a row, and editing a payload value
#: all modify a single group (an input distance of 2); removing a whole group,
#: or the only row of one, removes it (a distance of 1).
_NEIGHBOR_OPERATIONS: Tuple[str, ...] = (
    "drop-row",
    "drop-group",
    "add-duplicate",
    "edit-payload",
)

_BASE_SEED = 20260809

_SWEEP_SEEDS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)

#: Thresholds for the slow sweeps. Zero and one are the degenerate ends, and
#: three and seven sit below and above the generated group sizes.
_SWEEP_THRESHOLDS: Tuple[int, ...] = (0, 1, 3, 7)


@lru_cache(maxsize=None)
def _frame(shape: str, seed: int) -> EdgeCase:
    """Returns the generated frame for a shape and seed.

    Args:
        shape: A key of :data:`_SHAPES`, or :data:`_DUPLICATES_ID`.
        seed: The seed for the generator.

    Returns:
        The generated case. Repeated calls return the same object.
    """
    arguments = _DUPLICATES_SHAPE if shape == _DUPLICATES_ID else _SHAPES[shape]
    case = random_frame(random.Random(seed), case_id=f"{shape}-{seed}", **arguments)
    # Every oracle in this module reads group and key identity through
    # normalize_value, which is coarser than the identity the implementations
    # use: limit_keys_per_group counts (group, digest, key) pairs, and the
    # digest splits int 1 from float 1.0 and 0.0 from -0.0 in a grouping
    # column exactly as it does in a key column (see
    # assert_no_conflating_values). Both column lists are therefore guarded,
    # deduplicated because a column may be both. Every frame the oracles read
    # comes from here -- the neighbor and perturbed frames only reuse values
    # already in the base frame -- so this one guard makes a generator change
    # that starts mixing conflating values fail loudly instead of silently
    # weakening the oracle.
    assert_no_conflating_values(
        case.to_pandas(), list(dict.fromkeys([*case.grouping, *case.keys]))
    )
    return case


#: Outputs already computed, keyed by backend, function, frame, and threshold.
#: The truncation functions are deterministic, so several properties can share a
#: single (comparatively expensive) Spark run.
_OUTPUTS: Dict[Tuple[Any, ...], pd.DataFrame] = {}


def _run(
    backend_: TruncationBackend, function: str, case: EdgeCase, threshold: int
) -> pd.DataFrame:
    """Returns the result of running one truncation function on a frame.

    Args:
        backend_: The backend to run.
        function: One of
            :data:`~test.unit.utils.truncation_testing.TRUNCATION_FUNCTIONS`.
        case: The frame to truncate.
        threshold: The threshold to truncate with.

    Returns:
        The truncated frame. The returned object is cached and shared between
        tests, so callers must not modify it.
    """
    key = (backend_.name, function, case.id, case.rows, threshold)
    if key not in _OUTPUTS:
        _OUTPUTS[key] = apply_truncation(
            backend_, function, case.to_pandas(), case.grouping, case.keys, threshold
        )
    return _OUTPUTS[key]


################################################################################
# Reading frames
################################################################################


def _tuples(case: EdgeCase, columns: Sequence[str]) -> List[Tuple[Any, ...]]:
    """Returns the given columns of a case's rows, as hashable tuples.

    The values are made hashable by
    :func:`~test.unit.utils.truncation_testing.normalize_value`, which stands
    NaN in with a sentinel (NaN is not equal to itself, so it cannot be used
    as a dictionary key directly). Note that 0.0 and -0.0 stay conflated, as
    Python conflates them, and so do int 1 and float 1.0. Telling them apart
    would matter in a grouping or key column, where Spark counts each pair as
    two keys because their hashes differ, but the generator never mixes them
    in one; :func:`_frame` checks that assumption for both column lists with
    :func:`~test.unit.utils.truncation_testing.assert_no_conflating_values`.

    Args:
        case: The case to read.
        columns: The columns to take, in order.

    Returns:
        One tuple per row of the case.
    """
    indices = [case.columns.index(name) for name in columns]
    return [
        tuple(normalize_value(row[index]) for index in indices) for row in case.rows
    ]


def _row_ids(case: EdgeCase) -> List[int]:
    """Returns the row ids of a case's rows, in order.

    Args:
        case: The case to read. It must carry a ``row_id`` column.

    Returns:
        One row id per row.
    """
    index = case.columns.index(ROW_ID_COLUMN)
    return [int(row[index]) for row in case.rows]


def _members(
    case: EdgeCase, columns: Sequence[str]
) -> Dict[Tuple[Any, ...], List[int]]:
    """Returns the row ids belonging to each distinct value of some columns.

    Args:
        case: The case to read. It must carry a ``row_id`` column.
        columns: The columns whose distinct values define the buckets.

    Returns:
        A mapping from column-value tuple to the row ids that have it, in the
        order the rows appear.
    """
    members: Dict[Tuple[Any, ...], List[int]] = {}
    for row_id, key in zip(_row_ids(case), _tuples(case, columns)):
        members.setdefault(key, []).append(row_id)
    return members


def _keys_by_group(case: EdgeCase) -> Dict[Tuple[Any, ...], Set[Tuple[Any, ...]]]:
    """Returns the distinct key tuples of each group of a case.

    Args:
        case: The case to read.

    Returns:
        A mapping from grouping-column tuple to the set of key tuples in it.
    """
    keys_by_group: Dict[Tuple[Any, ...], Set[Tuple[Any, ...]]] = {}
    groups = _tuples(case, case.grouping)
    keys = _tuples(case, case.keys)
    for group, key in zip(groups, keys):
        keys_by_group.setdefault(group, set()).add(key)
    return keys_by_group


def _group_of(case: EdgeCase) -> Dict[int, Tuple[Any, ...]]:
    """Returns the group of each row id of a case.

    Args:
        case: The case to read.

    Returns:
        A mapping from row id to grouping-column tuple.
    """
    return dict(zip(_row_ids(case), _tuples(case, case.grouping)))


def _pair_of(case: EdgeCase) -> Dict[int, Tuple[Any, ...]]:
    """Returns the (group, key) pair of each row id of a case.

    Args:
        case: The case to read.

    Returns:
        A mapping from row id to the tuple of its grouping and key columns.
    """
    return dict(zip(_row_ids(case), _tuples(case, (*case.grouping, *case.keys))))


def _payload_columns(case: EdgeCase) -> List[str]:
    """Returns the columns of a case that are neither grouping, key, nor row id.

    Args:
        case: The case to read.

    Returns:
        The payload column names, in the case's column order.
    """
    reserved = {ROW_ID_COLUMN, *case.grouping, *case.keys}
    return [name for name in case.columns if name not in reserved]


################################################################################
# Neighboring frames
################################################################################


def _largest_group(case: EdgeCase) -> Tuple[Any, ...]:
    """Returns the grouping-column tuple of the case's largest group.

    Args:
        case: The case to read. It must have at least one row.

    Returns:
        The group with the most rows; ties are broken by first appearance, so
        the result is deterministic.
    """
    groups = _tuples(case, case.grouping)
    counts = Counter(groups)
    return max(dict.fromkeys(groups), key=lambda group: counts[group])


def _with_fresh_row_id(case: EdgeCase, row: Tuple[Any, ...]) -> Tuple[Any, ...]:
    """Returns a copy of a row with a row id no other row of the case has.

    Args:
        case: The case the row comes from.
        row: The row to copy.

    Returns:
        The row unchanged if the case has no row id column, and otherwise a copy
        with a fresh one.
    """
    if not case.has_row_id:
        return row
    values = list(row)
    values[case.columns.index(ROW_ID_COLUMN)] = max(_row_ids(case)) + 1
    return tuple(values)


def _edit_one_payload(case: EdgeCase, rows: List[Tuple[Any, ...]]) -> None:
    """Replaces one payload value of one row with another row's value.

    Taking the replacement from another row of the same column keeps the value
    within the column's dtype, and keeps the frame free of the values the
    generator avoids on purpose. If every row already agrees on the column, the
    rows are left alone, which makes the neighbor identical to the original --
    a degenerate but still valid neighbor, at distance zero.

    Args:
        case: The case the rows come from.
        rows: The rows to edit, in place.
    """
    index = case.columns.index(_payload_columns(case)[0])
    target = len(rows) // 2
    current = normalize_value(rows[target][index])
    for other in rows:
        if normalize_value(other[index]) != current:
            values = list(rows[target])
            values[index] = other[index]
            rows[target] = tuple(values)
            return


def _neighbor(case: EdgeCase, operation: str) -> EdgeCase:
    """Returns a frame near ``case`` under the grouped symmetric difference.

    Args:
        case: The frame to derive a neighbor from. It must have at least one
            row.
        operation: One of :data:`_NEIGHBOR_OPERATIONS`.

    Returns:
        The neighboring frame, with the same columns and dtypes.
    """
    rows = list(case.rows)
    if operation == "drop-row":
        del rows[len(rows) // 3]
    elif operation == "drop-group":
        target = _largest_group(case)
        groups = _tuples(case, case.grouping)
        rows = [row for row, group in zip(rows, groups) if group != target]
    elif operation == "add-duplicate":
        rows.append(_with_fresh_row_id(case, rows[len(rows) // 2]))
    elif operation == "edit-payload":
        _edit_one_payload(case, rows)
    else:
        raise ValueError(f"Unknown neighbor operation {operation}")
    return replace(case, id=f"{case.id}/{operation}", rows=tuple(rows))


def _perturb_one_group(case: EdgeCase) -> Tuple[Tuple[Any, ...], EdgeCase]:
    """Returns a frame that differs from ``case`` inside a single group.

    The largest group loses its first row and gains a copy of its last one; no
    other row is touched, and the surviving rows keep their relative order, so
    the duplicate-row salt of the other groups cannot shift either.

    Args:
        case: The frame to perturb. It must carry a ``row_id`` column.

    Returns:
        The group that was edited, and the perturbed frame.
    """
    target = _largest_group(case)
    groups = _tuples(case, case.grouping)
    positions = [index for index, group in enumerate(groups) if group == target]
    rows = [row for index, row in enumerate(case.rows) if index != positions[0]]
    rows.append(_with_fresh_row_id(case, case.rows[positions[-1]]))
    return target, replace(case, id=f"{case.id}/perturbed", rows=tuple(rows))


################################################################################
# Property checks
################################################################################


def _assert_within_threshold(
    case: EdgeCase, output: pd.DataFrame, threshold: int
) -> None:
    """Asserts that no group of a truncated frame exceeds the threshold.

    Args:
        case: The input frame.
        output: The result of ``truncate_large_groups``.
        threshold: The threshold it was called with.
    """
    group_of = _group_of(case)
    counts = Counter(group_of[row_id] for row_id in frame_row_ids(output))
    too_large = {group: count for group, count in counts.items() if count > threshold}
    assert not too_large, f"groups over the threshold of {threshold}: {too_large}"


def _assert_keys_within_threshold(
    case: EdgeCase, output: pd.DataFrame, threshold: int
) -> None:
    """Asserts that no group of a key-limited frame has too many keys.

    Args:
        case: The input frame.
        output: The result of ``limit_keys_per_group``.
        threshold: The threshold it was called with.
    """
    pair_of = _pair_of(case)
    grouping_size = len(case.grouping)
    surviving: Dict[Tuple[Any, ...], Set[Tuple[Any, ...]]] = {}
    for row_id in frame_row_ids(output):
        pair = pair_of[row_id]
        surviving.setdefault(pair[:grouping_size], set()).add(pair[grouping_size:])
    too_many = {
        group: keys for group, keys in surviving.items() if len(keys) > threshold
    }
    assert not too_many, f"groups with more than {threshold} keys: {too_many}"


def _assert_submultiset_by_row_id(case: EdgeCase, output: pd.DataFrame) -> None:
    """Asserts that a truncated frame's rows are a sub-multiset of the input's.

    Args:
        case: The input frame.
        output: The truncated frame.
    """
    assert list(output.columns) == list(case.columns)
    available = Counter(_row_ids(case))
    kept = Counter(frame_row_ids(output))
    extra = {
        row_id: count for row_id, count in kept.items() if count > available[row_id]
    }
    assert not extra, f"row ids kept more often than they appear in the input: {extra}"


def _assert_submultiset_by_row(case: EdgeCase, output: pd.DataFrame) -> None:
    """Asserts the sub-multiset property by comparing whole rows.

    ``multiset_symdiff`` counts the rows that would have to be added to or
    removed from one frame to reach the other. It is exactly the number of rows
    that were dropped if and only if nothing else changed, so this also rules
    out altered or invented rows.

    Args:
        case: The input frame.
        output: The truncated frame.
    """
    df = case.to_pandas()
    assert list(output.columns) == list(case.columns)
    assert multiset_symdiff(df, output) == len(df) - len(output)


def _assert_untouched_groups_pass_through(
    case: EdgeCase, function: str, output: pd.DataFrame, threshold: int
) -> None:
    """Asserts that groups already within the threshold are kept whole.

    Args:
        case: The input frame.
        function: The truncation function that produced the output.
        output: The truncated frame.
        threshold: The threshold it was called with.
    """
    kept = set(frame_row_ids(output))
    keys_by_group = _keys_by_group(case)
    for group, row_ids in _members(case, case.grouping).items():
        if function == "limit_keys_per_group":
            load = len(keys_by_group[group])
        else:
            load = len(row_ids)
        if load <= threshold:
            missing = set(row_ids) - kept
            assert not missing, (
                f"group {group} is within {threshold} but lost {missing}"
            )


def _assert_drop_is_all_or_nothing(case: EdgeCase, output: pd.DataFrame) -> None:
    """Asserts that each group is either kept whole or dropped entirely.

    Args:
        case: The input frame.
        output: The result of ``drop_large_groups``.
    """
    kept = set(frame_row_ids(output))
    for group, row_ids in _members(case, case.grouping).items():
        survivors = kept & set(row_ids)
        assert survivors in (set(), set(row_ids)), (
            f"group {group} was partially dropped: kept {survivors} of {set(row_ids)}"
        )


def _assert_surviving_keys_are_complete(case: EdgeCase, output: pd.DataFrame) -> None:
    """Asserts that every row of every surviving key is kept.

    Args:
        case: The input frame.
        output: The result of ``limit_keys_per_group``.
    """
    kept = set(frame_row_ids(output))
    pair_of = _pair_of(case)
    surviving = {pair_of[row_id] for row_id in kept}
    for pair, row_ids in _members(case, (*case.grouping, *case.keys)).items():
        if pair in surviving:
            missing = set(row_ids) - kept
            assert not missing, f"key {pair} survived but lost rows {missing}"


def _assert_stability(
    backend_: TruncationBackend,
    function: str,
    case: EdgeCase,
    operation: str,
    threshold: int,
) -> None:
    """Asserts the stability bound for one function on one neighboring pair.

    The input distance is the one induced by
    ``IfGroupedBy(grouping_columns, SymmetricDifference())``. The output
    distance is the multiset symmetric difference for the two row-truncating
    functions, and the same grouped distance taken over (group, key) pairs for
    ``limit_keys_per_group``, which bounds the number of keys per group rather
    than the number of rows.

    Args:
        backend_: The backend to run.
        function: One of
            :data:`~test.unit.utils.truncation_testing.TRUNCATION_FUNCTIONS`.
        case: The first frame of the pair.
        operation: The neighbor operation producing the second frame.
        threshold: The threshold to truncate with.
    """
    neighbor = _neighbor(case, operation)
    distance_in = grouped_symdiff_distance(
        case.to_pandas(), neighbor.to_pandas(), case.grouping
    )
    output_a = _run(backend_, function, case, threshold)
    output_b = _run(backend_, function, neighbor, threshold)
    if function == "limit_keys_per_group":
        distance_out = grouped_symdiff_distance(
            output_a, output_b, (*case.grouping, *case.keys)
        )
    else:
        distance_out = multiset_symdiff(output_a, output_b)
    assert distance_out <= threshold * distance_in, (
        f"{function} on {case.id} with {operation} and threshold {threshold}: "
        f"output distance {distance_out} exceeds {threshold} * {distance_in}"
    )


def _assert_group_locality(
    backend_: TruncationBackend,
    function: str,
    case: EdgeCase,
    threshold: int,
) -> None:
    """Asserts that editing one group leaves the survivors of every other alone.

    Args:
        backend_: The backend to run.
        function: One of
            :data:`~test.unit.utils.truncation_testing.TRUNCATION_FUNCTIONS`.
        case: The frame to perturb. It must carry a ``row_id`` column.
        threshold: The threshold to truncate with.
    """
    edited_group, perturbed = _perturb_one_group(case)
    elsewhere = {
        row_id for row_id, group in _group_of(case).items() if group != edited_group
    }
    before = set(frame_row_ids(_run(backend_, function, case, threshold)))
    after = set(frame_row_ids(_run(backend_, function, perturbed, threshold)))
    assert before & elsewhere == after & elsewhere, (
        f"{function} on {case.id} at threshold {threshold}: editing group "
        f"{edited_group} changed the survivors of other groups: "
        f"{(before ^ after) & elsewhere}"
    )


################################################################################
# Bounded output
################################################################################


@pytest.mark.parametrize("threshold", _THRESHOLDS)
@pytest.mark.parametrize("shape", _SHAPE_IDS)
def test_truncate_keeps_at_most_threshold_rows_per_group(
    backend: TruncationBackend, shape: str, threshold: int
) -> None:
    """Tests that truncate_large_groups bounds the size of every group."""
    case = _frame(shape, _BASE_SEED)
    output = _run(backend, "truncate_large_groups", case, threshold)
    _assert_within_threshold(case, output, threshold)


@pytest.mark.parametrize("threshold", _DUPLICATE_THRESHOLDS)
def test_truncate_bounds_groups_of_duplicate_rows(
    backend: TruncationBackend, threshold: int
) -> None:
    """Tests the group size bound on a frame that is mostly duplicate rows."""
    case = _frame(_DUPLICATES_ID, _BASE_SEED)
    output = _run(backend, "truncate_large_groups", case, threshold)
    sizes = output.groupby(list(case.grouping), dropna=False).size()
    assert all(size <= threshold for size in sizes), f"group sizes {dict(sizes)}"


@pytest.mark.parametrize("threshold", _THRESHOLDS)
@pytest.mark.parametrize("shape", _SHAPE_IDS)
def test_limit_keys_keeps_at_most_threshold_keys_per_group(
    backend: TruncationBackend, shape: str, threshold: int
) -> None:
    """Tests that limit_keys_per_group bounds the key count of every group."""
    case = _frame(shape, _BASE_SEED)
    output = _run(backend, "limit_keys_per_group", case, threshold)
    _assert_keys_within_threshold(case, output, threshold)


################################################################################
# Subset of the input
################################################################################


@pytest.mark.parametrize("threshold", _THRESHOLDS)
@pytest.mark.parametrize("shape", _SHAPE_IDS)
@pytest.mark.parametrize("function", TRUNCATION_FUNCTIONS)
def test_output_is_submultiset_of_input(
    backend: TruncationBackend, function: str, shape: str, threshold: int
) -> None:
    """Tests that truncation only ever removes rows."""
    case = _frame(shape, _BASE_SEED)
    _assert_submultiset_by_row_id(case, _run(backend, function, case, threshold))


@pytest.mark.parametrize("threshold", _DUPLICATE_THRESHOLDS)
@pytest.mark.parametrize("function", TRUNCATION_FUNCTIONS)
def test_output_is_submultiset_of_duplicate_rows(
    backend: TruncationBackend, function: str, threshold: int
) -> None:
    """Tests the sub-multiset property where rows repeat and cannot be told apart."""
    case = _frame(_DUPLICATES_ID, _BASE_SEED)
    _assert_submultiset_by_row(case, _run(backend, function, case, threshold))


################################################################################
# Groups within the threshold
################################################################################


@pytest.mark.parametrize("threshold", _THRESHOLDS)
@pytest.mark.parametrize("shape", _SHAPE_IDS)
@pytest.mark.parametrize("function", TRUNCATION_FUNCTIONS)
def test_untouched_groups_pass_through(
    backend: TruncationBackend, function: str, shape: str, threshold: int
) -> None:
    """Tests that a group that is already small enough is kept whole."""
    case = _frame(shape, _BASE_SEED)
    output = _run(backend, function, case, threshold)
    _assert_untouched_groups_pass_through(case, function, output, threshold)


@pytest.mark.parametrize("threshold", _THRESHOLDS)
@pytest.mark.parametrize("shape", _SHAPE_IDS)
def test_drop_large_groups_is_all_or_nothing(
    backend: TruncationBackend, shape: str, threshold: int
) -> None:
    """Tests that drop_large_groups never keeps part of a group."""
    case = _frame(shape, _BASE_SEED)
    _assert_drop_is_all_or_nothing(
        case, _run(backend, "drop_large_groups", case, threshold)
    )


@pytest.mark.parametrize("threshold", _THRESHOLDS)
@pytest.mark.parametrize("shape", _SHAPE_IDS)
def test_limit_keys_keeps_every_row_of_surviving_keys(
    backend: TruncationBackend, shape: str, threshold: int
) -> None:
    """Tests that limit_keys_per_group truncates keys, not the rows under them."""
    case = _frame(shape, _BASE_SEED)
    output = _run(backend, "limit_keys_per_group", case, threshold)
    _assert_surviving_keys_are_complete(case, output)


################################################################################
# Locality
################################################################################


@pytest.mark.parametrize("threshold", _THRESHOLDS)
@pytest.mark.parametrize("shape", _SHAPE_IDS)
@pytest.mark.parametrize("function", TRUNCATION_FUNCTIONS)
def test_group_locality(
    backend: TruncationBackend, function: str, shape: str, threshold: int
) -> None:
    """Tests that editing one group leaves the survivors of every other alone."""
    case = _frame(shape, _BASE_SEED)
    _assert_group_locality(backend, function, case, threshold)


################################################################################
# Stability
################################################################################


@pytest.mark.parametrize("threshold", _THRESHOLDS)
@pytest.mark.parametrize("shape", _STABILITY_SHAPE_IDS)
@pytest.mark.parametrize("operation", _NEIGHBOR_OPERATIONS)
@pytest.mark.parametrize("function", TRUNCATION_FUNCTIONS)
def test_stability(
    backend: TruncationBackend,
    function: str,
    operation: str,
    shape: str,
    threshold: int,
) -> None:
    """Tests each function's stability bound on neighboring frames.

    For truncate_large_groups and drop_large_groups the bound is on rows;
    for limit_keys_per_group it is per (group, key) pair.
    """
    case = _frame(shape, _BASE_SEED)
    _assert_stability(backend, function, case, operation, threshold)


@pytest.mark.parametrize("threshold", _DUPLICATE_THRESHOLDS)
@pytest.mark.parametrize("operation", ["drop-row", "add-duplicate"])
@pytest.mark.parametrize("function", TRUNCATION_FUNCTIONS)
def test_stability_with_duplicate_rows(
    backend: TruncationBackend, function: str, operation: str, threshold: int
) -> None:
    """Tests the stability bound where the duplicate-row salt decides the order."""
    case = _frame(_DUPLICATES_ID, _BASE_SEED)
    _assert_stability(backend, function, case, operation, threshold)


################################################################################
# Sweeps
################################################################################


@pytest.mark.slow
@pytest.mark.parametrize("shape", (*_SHAPE_IDS, _DUPLICATES_ID))
@pytest.mark.parametrize("function", TRUNCATION_FUNCTIONS)
def test_structural_property_sweep(
    backend: TruncationBackend, function: str, shape: str
) -> None:
    """Tests every structural property over a sweep of seeded frames."""
    for seed in _SWEEP_SEEDS:
        for threshold in _SWEEP_THRESHOLDS:
            case = _frame(shape, seed)
            output = _run(backend, function, case, threshold)
            if case.has_row_id:
                _assert_submultiset_by_row_id(case, output)
                _assert_untouched_groups_pass_through(case, function, output, threshold)
                if function == "truncate_large_groups":
                    _assert_within_threshold(case, output, threshold)
                elif function == "drop_large_groups":
                    _assert_drop_is_all_or_nothing(case, output)
                else:
                    _assert_keys_within_threshold(case, output, threshold)
                    _assert_surviving_keys_are_complete(case, output)
            else:
                _assert_submultiset_by_row(case, output)
                if function == "truncate_large_groups":
                    sizes = output.groupby(list(case.grouping), dropna=False).size()
                    assert all(size <= threshold for size in sizes)


@pytest.mark.slow
@pytest.mark.parametrize("shape", (*_SHAPE_IDS, _DUPLICATES_ID))
@pytest.mark.parametrize("function", TRUNCATION_FUNCTIONS)
def test_stability_sweep(backend: TruncationBackend, function: str, shape: str) -> None:
    """Tests the stability bound over a sweep of seeded neighboring pairs."""
    for seed in _SWEEP_SEEDS[:4]:
        case = _frame(shape, seed)
        for operation in _NEIGHBOR_OPERATIONS:
            for threshold in (1, 7):
                _assert_stability(backend, function, case, operation, threshold)


@pytest.mark.slow
@pytest.mark.parametrize("shape", _SHAPE_IDS)
@pytest.mark.parametrize("function", TRUNCATION_FUNCTIONS)
def test_group_locality_sweep(
    backend: TruncationBackend, function: str, shape: str
) -> None:
    """Tests locality over a sweep of seeded frames and thresholds."""
    for seed in _SWEEP_SEEDS[:3]:
        case = _frame(shape, seed)
        for threshold in (1, 7):
            _assert_group_locality(backend, function, case, threshold)
