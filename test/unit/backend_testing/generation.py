"""Seeded random frame generation for the parity harness.

This module is part of the frozen harness API; see
:mod:`test.unit.backend_testing` for the freeze contract.

:func:`random_frame` returns an :class:`~test.unit.backend_testing.corpus.EdgeCase`
rather than a bare dataframe, so that generated frames go through exactly the
same pandas and Spark construction paths as the curated ones, and so that a
failing seed can be reported, re-rendered, and pinned as a curated case.

The generator only draws values that are *comparable across backends at all*;
the constraints are spelled out on :func:`random_frame`.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import datetime
import math
from dataclasses import dataclass
from test.unit.backend_testing.corpus import (
    _ROW_ID_FIELD,
    CJK,
    E_ACUTE,
    E_COMBINING_ACUTE,
    EMOJI,
    EdgeCase,
    _make_case,
)
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import pandas as pd
from pyspark.sql.types import (
    BinaryType,
    DataType,
    DateType,
    DoubleType,
    FloatType,
    LongType,
    StringType,
    TimestampType,
)

################################################################################
# Random frame generation
################################################################################


class RandomLike(Protocol):
    """A seeded source of randomness.

    Only ``random()`` is used, so ``random.Random``, ``np.random.Generator``,
    and ``np.random.RandomState`` are all acceptable.
    """

    def random(self) -> float:
        """Returns a float uniformly distributed in [0, 1)."""
        ...  # pragma: no cover


@dataclass(frozen=True)
class ColumnKind:
    """A dtype a generated column can take.

    Attributes:
        name: The kind's name, as used in a dtype menu.
        spark_type: The Spark type of a column of this kind.
        pandas_dtype: The pandas dtype of a column of this kind.
        nullable: Whether values of this kind may be null. Plain float columns
            are not nullable: a null in a float64 column would have to be
            ``np.nan``, which the implementations read as a NaN value. The
            ``object_float`` kind is the one floating point kind that is
            nullable *and* can hold a NaN, since an object column holds both as
            themselves.
    """

    name: str
    spark_type: DataType
    pandas_dtype: str
    nullable: bool


COLUMN_KINDS: Dict[str, ColumnKind] = {
    kind.name: kind
    for kind in (
        ColumnKind("int64", LongType(), "int64", False),
        ColumnKind("Int64", LongType(), "Int64", True),
        ColumnKind("string", StringType(), "object", True),
        ColumnKind("string_dtype", StringType(), "string", True),
        ColumnKind("float64", DoubleType(), "float64", False),
        ColumnKind("Float64", DoubleType(), "Float64", True),
        ColumnKind("object_float", DoubleType(), "object", True),
        ColumnKind("float32", FloatType(), "float32", False),
        ColumnKind("date", DateType(), "object", True),
        ColumnKind("timestamp", TimestampType(), "datetime64[ns]", True),
        ColumnKind("binary", BinaryType(), "object", True),
    )
}

#: Every supported kind. Frames drawn from this menu need a UTC session
#: timezone, because it includes timestamps.
DEFAULT_DTYPE_MENU: Tuple[str, ...] = (
    "int64",
    "string",
    "float64",
    "Int64",
    "Float64",
    "object_float",
    "float32",
    "date",
    "timestamp",
    "binary",
    "string_dtype",
)

#: Strings and integers only: the menu for sweeps that focus on duplicate rows
#: and the row salt rather than on value rendering.
SIMPLE_DTYPE_MENU: Tuple[str, ...] = ("int64", "string")

_STRING_POOL: Tuple[str, ...] = (
    "",
    "a",
    "b",
    "c",
    "a,",
    ",b",
    "a,b",
    " ",
    "\t",
    E_ACUTE,
    E_COMBINING_ACUTE,
    CJK,
    EMOJI,
    "0",
    "00",
    "1e3",
)

_INT_POOL: Tuple[int, ...] = (
    -9223372036854775808,
    9223372036854775807,
    -4294967296,
    -1,
    0,
    1,
    2,
    7,
    42,
    1000000,
)

_FLOAT_SPECIALS: Tuple[float, ...] = (
    float("nan"),
    float("inf"),
    float("-inf"),
    0.0,
    5e-324,
    1.7976931348623157e308,
    1e7,
    0.001,
    0.0009,
)

_FLOAT32_SPECIALS: Tuple[float, ...] = (
    float("nan"),
    float("inf"),
    float("-inf"),
    0.0,
    1.401298464324817e-45,
    3.4028234663852886e38,
    1e7,
    0.001,
)

_BYTES_POOL: Tuple[bytes, ...] = (
    b"",
    b"\x00",
    b"a",
    b"ab",
    b"\xff",
    b"\xff\xfe",
    b"\x00\x01\x02",
)

# The first and last ordinals of datetime.date, i.e. 0001-01-01 and 9999-12-31.
_MIN_DATE_ORDINAL = datetime.date(1, 1, 1).toordinal()
_MAX_DATE_ORDINAL = datetime.date(9999, 12, 31).toordinal()

# Timestamps are drawn from a window comfortably inside the range of pandas'
# datetime64[ns] dtype, which only spans 1677-09-21 to 2262-04-11.
_MIN_TIMESTAMP = datetime.datetime(1700, 1, 1)
_TIMESTAMP_SPAN_SECONDS = int(
    (datetime.datetime(2260, 1, 1) - _MIN_TIMESTAMP).total_seconds()
)

_MICROSECOND_SHAPES: Tuple[int, ...] = (0, 1, 500000, 123456, 999999, 100000)


def _index(rng: RandomLike, size: int) -> int:
    """Returns a random index in ``range(size)``."""
    return min(int(rng.random() * size), size - 1)


def _pick(rng: RandomLike, values: Sequence[Any]) -> Any:
    """Returns a uniformly random element of ``values``."""
    return values[_index(rng, len(values))]


def _sample_short_decimal_float(rng: RandomLike) -> float:
    """Returns a float parsed from a decimal literal with at most 12 digits.

    Doubles needing 14 or more significant digits are where Java's pre-19
    ``Double.toString`` mostly emits extra (still round-tripping) digits, so
    drawing short literals keeps that divergence rare -- but not impossible,
    since a short literal of large magnitude, such as ``2.35206429e19``, can
    still be one of them.

    Args:
        rng: The source of randomness.

    Returns:
        A finite, nonzero float.
    """
    digit_count = 1 + _index(rng, 12)
    mantissa = 1 + _index(rng, 9)
    for _ in range(digit_count - 1):
        mantissa = mantissa * 10 + _index(rng, 10)
    exponent = -12 + _index(rng, 25)
    value = float(f"{mantissa}e{exponent}")
    return -value if rng.random() < 0.5 else value


def _sample_value(
    rng: RandomLike, kind: ColumnKind, null_rate: float, allow_negative_zero: bool
) -> Any:
    """Returns one random value of the given kind.

    Args:
        rng: The source of randomness.
        kind: The kind of value to draw.
        null_rate: The probability of drawing a null, for nullable kinds.
        allow_negative_zero: Whether -0.0 may be drawn. It is only allowed in
            columns that are neither grouping nor key columns (see the
            signed-zeros edge case).

    Returns:
        A Python-native value, or None.
    """
    if kind.nullable and rng.random() < null_rate:
        return None
    if kind.name in ("int64", "Int64"):
        if rng.random() < 0.3:
            return _pick(rng, _INT_POOL)
        return _index(rng, 2001) - 1000
    if kind.name in ("string", "string_dtype"):
        return _pick(rng, _STRING_POOL)
    if kind.name in ("float64", "Float64", "object_float"):
        if rng.random() < 0.25:
            specials = _FLOAT_SPECIALS + ((-0.0,) if allow_negative_zero else ())
            return _pick(rng, specials)
        return _sample_short_decimal_float(rng)
    if kind.name == "float32":
        if rng.random() < 0.25:
            specials = _FLOAT32_SPECIALS + ((-0.0,) if allow_negative_zero else ())
            return _pick(rng, specials)
        # float32 has at most 9 significant digits, so short literals are drawn
        # here too. Unlike for doubles that does not avoid Java's pre-19
        # Float.toString emitting extra digits, which it does for about a tenth
        # of all floats however short the literal they came from.
        mantissa = 1 + _index(rng, 999999)
        exponent = -6 + _index(rng, 13)
        value = float(np.float32(float(f"{mantissa}e{exponent}")))
        return -value if rng.random() < 0.5 else value
    if kind.name == "date":
        ordinal = _MIN_DATE_ORDINAL + _index(
            rng, _MAX_DATE_ORDINAL - _MIN_DATE_ORDINAL + 1
        )
        return datetime.date.fromordinal(ordinal)
    if kind.name == "timestamp":
        offset = datetime.timedelta(
            seconds=int(rng.random() * _TIMESTAMP_SPAN_SECONDS),
            microseconds=_pick(rng, _MICROSECOND_SHAPES),
        )
        return _MIN_TIMESTAMP + offset
    if kind.name == "binary":
        return _pick(rng, _BYTES_POOL)
    raise ValueError(f"Unknown column kind {kind.name}")


def _zero_sign_key(row: Sequence[Any]) -> Tuple[Any, ...]:
    """Returns a row key that ignores the sign of zeros (and NaN's identity)."""
    key: List[Any] = []
    for value in row:
        if isinstance(value, float):
            if math.isnan(value):
                key.append("nan")
            elif value == 0.0:
                key.append("zero")
            else:
                key.append(value)
        else:
            key.append(value)
    return tuple(key)


def _canonicalize_zero_signs(
    rows: List[Tuple[Any, ...]],
) -> List[Tuple[Any, ...]]:
    """Returns rows with no two differing only in the sign of a zero.

    Spark's duplicate-row salt partitions by every column, where -0.0 and 0.0
    compare equal, but hashes the stored value, where they differ. Two rows that
    are identical except for a zero's sign therefore get a nondeterministic salt
    in Spark itself. This collapses any such pair into a genuine duplicate,
    which both implementations then handle deterministically.

    Args:
        rows: The generated rows.

    Returns:
        The repaired rows, in the same order.
    """
    seen: Dict[Tuple[Any, ...], Tuple[Any, ...]] = {}
    repaired = []
    for row in rows:
        key = _zero_sign_key(row)
        canonical = seen.setdefault(key, row)
        repaired.append(canonical)
    return repaired


def random_frame(
    rng: RandomLike,
    dtype_menu: Sequence[str] = DEFAULT_DTYPE_MENU,
    n_rows: int = 20,
    n_groups: int = 3,
    dup_rate: float = 0.3,
    *,
    n_grouping_columns: int = 1,
    n_key_columns: int = 1,
    n_payload_columns: int = 1,
    n_key_values: int = 4,
    null_rate: float = 0.15,
    with_row_id: bool = True,
    case_id: Optional[str] = None,
) -> EdgeCase:
    """Returns a randomly generated frame, as an :class:`EdgeCase`.

    The result is an :class:`EdgeCase` so that generated frames go through the
    same pandas and Spark construction paths as the curated ones. Column kinds
    are taken from ``dtype_menu`` in order, cycling as needed: the grouping
    columns first, then the key columns, then the payload columns.

    The generator respects the constraints that make the two implementations
    comparable at all:

    * Nulls are ``None`` (becoming ``pd.NA`` or ``NaT``), never ``np.nan``, and
      only appear in columns whose dtype can hold SQL NULL.
    * Floats come from decimal literals with at most 12 significant digits,
      plus the special values, which keeps most of them out of the population
      that a pre-Java-19 ``Double.toString`` renders with extra digits. That is
      a bias, not a guarantee: short literals of large magnitude and float32
      values of any magnitude can still be rendered differently by such a JVM,
      so a caller comparing generated frames across the two backends has to
      handle those itself (the differential suite does).
    * -0.0 only appears in payload columns, and no two rows differ only in the
      sign of a zero.

    Args:
        rng: A seeded source of randomness.
        dtype_menu: The column kinds to draw from, by name. See
            :data:`COLUMN_KINDS`.
        n_rows: The number of rows to generate.
        n_groups: The number of distinct values to draw the grouping columns
            from.
        dup_rate: The probability that a row repeats an earlier one. When
            ``with_row_id`` is set the repeat still gets a fresh row id, so
            exercising the duplicate-row salt needs ``with_row_id=False``.
        n_grouping_columns: The number of grouping columns.
        n_key_columns: The number of key columns.
        n_payload_columns: The number of columns that are neither grouping nor
            key columns.
        n_key_values: The number of distinct values to draw each key column
            from.
        null_rate: The probability that a nullable column's value is null.
        with_row_id: Whether to add a unique integer ``row_id`` column.
        case_id: The id of the returned case, or None for a generated one.

    Returns:
        The generated case, with grouping columns ``g0..``, key columns
        ``k0..``, and payload columns ``c0..``.
    """
    if not dtype_menu:
        raise ValueError("dtype_menu must not be empty")
    kinds = [COLUMN_KINDS[name] for name in dtype_menu]

    fields: List[Tuple[str, DataType, str]] = []
    if with_row_id:
        fields.append(_ROW_ID_FIELD)
    grouping = tuple(f"g{i}" for i in range(n_grouping_columns))
    keys = tuple(f"k{i}" for i in range(n_key_columns))
    payload = tuple(f"c{i}" for i in range(n_payload_columns))
    generated = grouping + keys + payload
    kind_by_column = {
        name: kinds[index % len(kinds)] for index, name in enumerate(generated)
    }
    for name in generated:
        kind = kind_by_column[name]
        fields.append((name, kind.spark_type, kind.pandas_dtype))

    def pool(name: str, size: int) -> List[Any]:
        """Returns a pool of values for a grouping or key column."""
        kind = kind_by_column[name]
        values: List[Any] = []
        for _ in range(20 * size):
            if len(values) >= size:
                break
            value = _sample_value(rng, kind, null_rate, allow_negative_zero=False)
            if not any(value is other or value == other for other in values):
                values.append(value)
        return values or [None]

    pools = {name: pool(name, n_groups) for name in grouping}
    pools.update({name: pool(name, n_key_values) for name in keys})

    rows: List[Tuple[Any, ...]] = []
    values: List[Any]
    for row_id in range(n_rows):
        if rows and rng.random() < dup_rate:
            values = list(rows[_index(rng, len(rows))])
            if with_row_id:
                values[0] = row_id
        else:
            values = [row_id] if with_row_id else []
            for name in grouping + keys:
                values.append(_pick(rng, pools[name]))
            for name in payload:
                values.append(
                    _sample_value(
                        rng,
                        kind_by_column[name],
                        null_rate,
                        allow_negative_zero=True,
                    )
                )
        rows.append(tuple(values))

    if not with_row_id:
        rows = _canonicalize_zero_signs(rows)

    return _make_case(
        case_id or f"random-{n_rows}rows-{n_groups}groups",
        fields,
        rows,
        grouping,
        keys,
        (0, 1, 2, 3),
        notes="Randomly generated by backend_testing.random_frame.",
    )


################################################################################
# Building a column that holds both a NaN and a null
################################################################################


def floating_array(
    values: Sequence[float], mask: Sequence[bool], size: int = 64
) -> pd.arrays.FloatingArray:
    """Returns a nullable float array holding both NaNs and nulls.

    A nullable float column is the only pandas column other than an object one
    that can hold both, and this is the only way to build one: every Series
    constructor, ``pd.array`` and ``astype`` read a ``np.nan`` as a missing
    value and would turn the NaNs into ``pd.NA``. The distinction is one the
    two backends are compared over -- a Spark double column holds both, and
    :mod:`tmlt.core.utils.pandas_grouping` calls them different values -- so
    several suites need such a column.

    Args:
        values: The underlying float values. A masked position's value is never
            read, so anything will do there.
        mask: True at the positions that hold a null.
        size: The float size, 32 or 64.

    Returns:
        The array, of ``Float64`` or ``Float32`` dtype.
    """
    return pd.arrays.FloatingArray(
        np.array(values, dtype=np.dtype(f"float{size}")), np.array(mask, dtype=bool)
    )
