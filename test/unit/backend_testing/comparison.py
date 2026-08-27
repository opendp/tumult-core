"""Backend-neutral frame comparison for the parity harness.

This module is part of the frozen harness API; see
:mod:`test.unit.backend_testing` for the freeze contract.

Two backends never return *identical* frames, only equivalent ones: row order
is not observable, and a Spark round trip widens dtypes. Every comparison here
is therefore over multisets of rows whose cells have been mapped through a
*key function*, and the harness has exactly two:

* :func:`normalize_value` canonicalizes away the distinctions a Spark round
  trip destroys -- every null flavor becomes one value (still distinct from
  NaN), and numbers compare by value across types. It is the right key for
  comparing one backend's output against another's.
* :func:`exact_value` merges nothing beyond what pandas itself makes
  unobservable. It is the right key for asserting that a single backend
  preserved its input.

:func:`assert_frames_equal_as_multisets` selects between them with its
``normalize`` argument; everything else here uses :func:`normalize_value`.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import datetime
import math
from collections import Counter
from test.unit.backend_testing.conversion import is_null_value
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

################################################################################
# Distances
################################################################################

# Sentinels standing in for values that are not usable as dictionary keys, or
# that must not be conflated with each other.
_NULL = "\x00tmlt-null"
_NAN = "\x00tmlt-nan"

# How many distinct rows an assertion message renders per side before eliding.
_DIFF_ROW_LIMIT = 10


def normalize_value(value: Any) -> Any:
    """Returns a hashable, backend-independent stand-in for a cell value.

    Missing values of every flavor (``None``, ``pd.NA``, ``NaT``) collapse onto
    one sentinel, and NaN onto another, so that the two are never confused with
    each other. Numbers are compared by value rather than by type, because a
    Spark round trip widens nullable integer columns to floats; this does mean
    that 1 and 1.0 -- and, in an all-integer column, 0.0 and -0.0 -- are treated
    as one value.

    Args:
        value: The value to normalize.

    Returns:
        A hashable stand-in for the value.
    """
    if is_null_value(value):
        return _NULL
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        as_float = float(value)
        if math.isnan(as_float):
            return _NAN
        if as_float.is_integer() and abs(as_float) < 2.0**63:
            return int(as_float)
        return as_float
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, np.datetime64):
        value = pd.Timestamp(value)
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime(warn=False)
    return value


def assert_no_conflating_values(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Asserts that no column mixes values :func:`normalize_value` conflates.

    The oracle identity :func:`normalize_value` induces is deliberately
    coarser than the identity of ``limit_keys_per_group``, which counts
    (group, *digest*, key) pairs: the oracle compares numbers by value, so
    int ``1`` and float ``1.0`` -- which render, and therefore hash, as
    ``"1"`` and ``"1.0"``, two distinct pairs -- collapse onto one oracle
    key, as do ``0.0`` and ``-0.0``. The digest covers the grouping columns
    as well as the key columns, so mixing such a pair in either kind of
    column splits a pair the oracle keeps whole. It would not fail any test
    on its own; it would silently weaken every assertion built on the
    oracle. Calling this guard on the frames an oracle reads turns that
    generator assumption into a loud failure instead. Two exemptions are
    deliberate: the null flavors, which contribute nothing to a digest and
    so can never be two keys, and equal values whose *types* differ but
    whose renderings do not (int ``1`` and ``np.int64(1)``, bytes and
    bytearrays of the same content), which a stricter type-tagged identity
    would wrongly split.

    Args:
        df: The frame to check.
        columns: The columns whose values feed an oracle's group or key
            identity, deduplicated by the caller when the two lists overlap.
    """
    for name in columns:
        merged: Dict[Any, List[Any]] = {}
        for value in df[name]:
            if not is_null_value(value):
                merged.setdefault(normalize_value(value), []).append(value)
        for values in merged.values():
            int_typed = [
                value
                for value in values
                if isinstance(value, (int, np.integer))
                and not isinstance(value, (bool, np.bool_))
            ]
            float_or_bool_typed = [
                value
                for value in values
                if isinstance(value, (float, np.floating, bool, np.bool_))
            ]
            assert not (int_typed and float_or_bool_typed), (
                f"Column {name} mixes {int_typed[0]!r} with "
                f"{float_or_bool_typed[0]!r}: normalize_value merges them, but "
                "they render differently and so are distinct keys."
            )
            zero_signs = {
                math.copysign(1.0, float(value))
                for value in values
                if isinstance(value, (float, np.floating)) and float(value) == 0.0
            }
            assert len(zero_signs) <= 1, (
                f"Column {name} mixes 0.0 and -0.0: normalize_value merges "
                "them, but they hash differently and so are distinct keys."
            )


def normalized_rows(df: pd.DataFrame, columns: Sequence[str]) -> List[Tuple[Any, ...]]:
    """Returns the given columns of a dataframe as normalized row tuples."""
    if not len(df):
        return []
    series = [[normalize_value(value) for value in df[name]] for name in columns]
    return [tuple(values) for values in zip(*series)]


def _aligned_columns(a: pd.DataFrame, b: pd.DataFrame) -> List[str]:
    """Returns the shared column order of two dataframes, or raises."""
    columns = [str(name) for name in a.columns]
    if sorted(columns) != sorted(str(name) for name in b.columns):
        raise ValueError(
            "Dataframes must have matching columns, got "
            f"{sorted(columns)} and {sorted(str(n) for n in b.columns)}."
        )
    return columns


def multiset_symdiff(a: pd.DataFrame, b: pd.DataFrame) -> int:
    """Returns the size of the multiset symmetric difference of two frames.

    Rows are compared by value, ignoring order and dtypes (see
    :func:`normalize_value`); a row appearing twice in ``a`` and once in ``b``
    contributes 1.

    Args:
        a: The first dataframe.
        b: The second dataframe. It must have the same columns as ``a``, in any
            order.

    Returns:
        The number of rows that would have to be added to or removed from ``a``
        to turn it into ``b``.
    """
    columns = _aligned_columns(a, b)
    counts_a = Counter(normalized_rows(a, columns))
    counts_b = Counter(normalized_rows(b, columns))
    return sum(
        abs(counts_a[row] - counts_b[row]) for row in set(counts_a) | set(counts_b)
    )


def exact_value(value: Any) -> Any:
    """Returns a hashable stand-in for a cell value that merges nothing.

    This is the strict counterpart of :func:`normalize_value`, used by
    :func:`assert_frames_equal_as_multisets` with ``normalize=False``. Where
    :func:`normalize_value` deliberately collapses the distinctions a Spark
    round trip destroys, this one keeps every distinction a test could care
    about, by tagging each value with the kind it belongs to:

    * The three null flavors -- ``None``, ``pd.NA``, ``pd.NaT`` -- are three
      different values, and none of them equals a float NaN.
    * ``1`` and ``1.0`` are different values, and ``True`` is neither.
    * ``0.0`` and ``-0.0`` are different values.
    * All NaNs are one value, so that a frame compares equal to itself.
      (NaN payloads are not observable through pandas, and ``nan != nan``
      would otherwise make every NaN-bearing frame unequal to itself.)
    * ``bytes`` and ``bytearray`` of the same content are one value, since
      pandas hands back whichever the constructor happened to build.
    * A :class:`~pandas.Timestamp` and the equal :class:`~datetime.datetime`
      are one value, for the same reason.

    Use it directly when writing a bespoke comparison; the common cases are
    covered by :func:`assert_frames_equal_as_multisets`.

    Args:
        value: The value to key.

    Returns:
        A hashable tuple whose first element names the kind of value. Its
        shape past that is an implementation detail; only equality between
        two of these is meaningful.
    """
    if value is None:
        return ("none", None)
    if value is pd.NA:
        return ("na", None)
    if value is pd.NaT:
        return ("nat", None)
    if isinstance(value, (bool, np.bool_)):
        return ("bool", bool(value))
    if isinstance(value, (float, np.floating)):
        as_float = float(value)
        if math.isnan(as_float):
            return ("nan", None)
        # copysign, not a sign test: -0.0 == 0.0, so only the sign bit tells
        # the two apart, and they must not be one key here.
        return ("float", as_float, math.copysign(1.0, as_float))
    if isinstance(value, (int, np.integer)):
        return ("int", int(value))
    if isinstance(value, (bytes, bytearray)):
        return ("bytes", bytes(value))
    if isinstance(value, str):
        return ("str", str(value))
    if isinstance(value, np.datetime64):
        value = pd.Timestamp(value)
        if value is pd.NaT:
            return ("nat", None)
    if isinstance(value, pd.Timestamp):
        return ("datetime", value.to_pydatetime(warn=False))
    if isinstance(value, datetime.datetime):
        return ("datetime", value)
    return (type(value).__name__, value)


def _keyed_rows(
    df: pd.DataFrame, columns: Sequence[str], normalize: bool
) -> List[Tuple[Any, ...]]:
    """Returns a frame's rows as tuples of comparison keys.

    Args:
        df: The frame to read.
        columns: The columns to read, in the order they should appear in the
            tuples.
        normalize: Whether to key cells with :func:`normalize_value` rather
            than :func:`exact_value`.

    Returns:
        One tuple per row.
    """
    if normalize:
        return normalized_rows(df, columns)
    if not len(df):
        return []
    series = [[exact_value(value) for value in df[name]] for name in columns]
    return [tuple(values) for values in zip(*series)]


def _describe_rows(
    counts: Counter, columns: Sequence[str], limit: int = _DIFF_ROW_LIMIT
) -> str:
    """Returns a short, readable rendering of a multiset of comparison keys.

    Args:
        counts: The rows and their multiplicities.
        columns: The column names the tuples are aligned to.
        limit: The maximum number of distinct rows to render.

    Returns:
        One indented line per row, plus an elision line if there are more.
    """
    if not counts:
        return "    (none)"
    rendered = []
    for row, count in sorted(counts.items(), key=lambda item: repr(item[0]))[:limit]:
        cells = ", ".join(
            f"{name}={_render_key(key)}" for name, key in zip(columns, row)
        )
        suffix = f" (x{count})" if count > 1 else ""
        rendered.append(f"    {{{cells}}}{suffix}")
    if len(counts) > limit:
        rendered.append(f"    ... and {len(counts) - limit} more")
    return "\n".join(rendered)


def _render_key(key: Any) -> str:
    """Returns a readable rendering of one comparison key.

    Args:
        key: A key produced by :func:`normalize_value` or :func:`exact_value`.

    Returns:
        The rendering.
    """
    if key == _NULL:
        return "null"
    if key == _NAN:
        return "nan"
    # An exact_value key is a (kind, payload, ...) tuple. A normalize_value key
    # never is, unless the cell itself held a tuple, which the length checks
    # below fall through for.
    if isinstance(key, tuple) and len(key) >= 2 and isinstance(key[0], str):
        kind = key[0]
        if kind in ("none", "na", "nat"):
            return {"none": "None", "na": "pd.NA", "nat": "pd.NaT"}[kind]
        if kind == "nan":
            return "nan"
        if kind == "float" and len(key) == 3 and key[1] == 0.0:
            return "-0.0" if key[2] < 0 else "0.0"
        return repr(key[1])
    return repr(key)


def assert_frames_equal_as_multisets(
    left: pd.DataFrame, right: pd.DataFrame, *, normalize: bool = True
) -> None:
    """Asserts that two frames hold the same rows, ignoring order.

    This is the harness's default equality: two backends may return the same
    data in any order and, after a Spark round trip, in different dtypes, so
    neither row order nor the column dtypes are asserted on. Column *names*
    are: the frames must have the same set of columns, in any order.

    Multiset, not set: a row appearing twice on one side and once on the other
    is a failure. That matters -- truncation and grouping operations are
    defined on multisets of rows, and a comparison that deduplicated would not
    see a dropped duplicate.

    ``normalize`` selects which values count as the same value:

    * ``True`` (the default) keys cells with :func:`normalize_value`. Every
      null flavor collapses onto one value, distinct from NaN; numbers compare
      by value across types, so ``1`` equals ``1.0`` and ``0.0`` equals
      ``-0.0``. This is the mode for comparing *across* backends, because
      ``toPandas()`` destroys exactly those distinctions.
    * ``False`` keys cells with :func:`exact_value`, which merges nothing:
      the three null flavors differ from each other and from NaN, ``1``
      differs from ``1.0``, and ``0.0`` differs from ``-0.0``. This is the
      mode for asserting that a *single* backend preserved its input exactly.

    Args:
        left: The first frame.
        right: The second frame. It must have the same columns as ``left``, in
            any order.
        normalize: Whether to compare canonicalized values rather than exact
            ones.

    Raises:
        ValueError: If the two frames do not have the same columns.
    """
    columns = _aligned_columns(left, right)
    counts_left = Counter(_keyed_rows(left, columns, normalize))
    counts_right = Counter(_keyed_rows(right, columns, normalize))
    if counts_left == counts_right:
        return

    only_left: Counter = Counter()
    only_right: Counter = Counter()
    for row in set(counts_left) | set(counts_right):
        surplus = counts_left[row] - counts_right[row]
        if surplus > 0:
            only_left[row] = surplus
        elif surplus < 0:
            only_right[row] = -surplus

    mode = "normalized" if normalize else "exact"
    raise AssertionError(
        f"Frames differ as multisets of rows ({mode} values); "
        f"{sum(only_left.values())} row(s) only in the left frame and "
        f"{sum(only_right.values())} only in the right.\n"
        f"  columns: {columns}\n"
        f"  only in left:\n{_describe_rows(only_left, columns)}\n"
        f"  only in right:\n{_describe_rows(only_right, columns)}"
    )


def _group_pairs(
    df: pd.DataFrame, columns: Sequence[str], group_columns: Sequence[str]
) -> Set[Any]:
    """Returns the set of (group key, row multiset) pairs of a dataframe."""
    indices = [list(columns).index(name) for name in group_columns]
    groups: Dict[Tuple[Any, ...], Counter] = {}
    for row in normalized_rows(df, columns):
        key = tuple(row[index] for index in indices)
        groups.setdefault(key, Counter())[row] += 1
    return {(key, frozenset(rows.items())) for key, rows in groups.items()}


def grouped_symdiff_distance(
    a: pd.DataFrame, b: pd.DataFrame, group_cols: Sequence[str]
) -> int:
    """Returns the distance between two frames under a grouped symmetric metric.

    This is the distance of ``IfGroupedBy(group_cols, SymmetricDifference())``:
    the symmetric difference of the sets of (group key, group row multiset)
    pairs. A group present in only one of the two frames contributes 1, and a
    group present in both but with different rows contributes 2.

    Args:
        a: The first dataframe.
        b: The second dataframe. It must have the same columns as ``a``, in any
            order.
        group_cols: The columns defining the groups. An empty collection makes
            the whole frame one group.

    Returns:
        The distance between the two dataframes.
    """
    columns = _aligned_columns(a, b)
    for name in group_cols:
        if name not in columns:
            raise ValueError(f"Grouping column {name} is not in the dataframes.")
    pairs_a = _group_pairs(a, columns, list(group_cols))
    pairs_b = _group_pairs(b, columns, list(group_cols))
    return len(pairs_a ^ pairs_b)


################################################################################
# Value labels
################################################################################


def label_value(value: Any) -> str:
    """Returns a string label for a cell value, keeping NaN and null apart.

    ``None`` and ``pd.NA`` are labelled ``"null"``, and a float NaN ``"nan"``;
    any other value -- ``pd.NaT`` included, deliberately, unlike the null
    taxonomy of :func:`normalize_value` -- is labelled with its ``repr``.

    Args:
        value: The value to label.

    Returns:
        The value's label.
    """
    if value is None or value is pd.NA:
        return "null"
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return repr(value)
