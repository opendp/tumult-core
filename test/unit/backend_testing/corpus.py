"""The curated edge-case corpus the parity harness compares backends over.

This module is part of the frozen harness API; see
:mod:`test.unit.backend_testing` for the freeze contract.

:data:`EDGE_CASES` is a corpus of small, hand-written frames, each aimed at one
corner where two backends could plausibly disagree: the null flavors, signed
zeros, float specials, unicode that normalizes or sorts unexpectedly, integer
extremes, dates and naive timestamps, binary values, and the degenerate shapes
(empty, single row). Each :class:`EdgeCase` carries both renderings of the same
rows -- a pandas dtype per column and an explicit Spark schema -- so that a test
can hand the *same* data to either backend without an inference step in between
changing it.

The corpus is deliberately declarative: a case is data, not code, so a new
backend inherits every case ever written without touching this file.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import datetime
from dataclasses import dataclass
from test.unit.backend_testing.conversion import (
    _require_utc_session_timezone,
    _schema_has_timestamps,
    _to_spark_value,
)
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    BinaryType,
    DataType,
    DateType,
    DoubleType,
    FloatType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# Name of the unique-integer column carried by edge cases whose dtypes cannot
# survive a Spark round trip unambiguously. Comparing the set of surviving
# row_ids sidesteps the NULL/NaN conflation that toPandas() introduces.
ROW_ID_COLUMN = "row_id"


def frame_row_ids(df: pd.DataFrame) -> List[int]:
    """Returns the row ids of a dataframe, in its row order.

    Args:
        df: A frame carrying a :data:`ROW_ID_COLUMN` column.

    Returns:
        One int per row.
    """
    return [int(value) for value in df[ROW_ID_COLUMN]]


################################################################################
# Edge case corpus
################################################################################


@dataclass(frozen=True)
class EdgeCase:
    """A hand-written frame exercising one corner where backends could disagree.

    A case is the *same data* in two renderings -- a pandas dtype per column and
    an explicit Spark schema, over one set of Python-native row tuples -- so
    that a parity test can hand both backends the same rows with no inference
    step in between to change them. :meth:`to_pandas` and
    :func:`spark_df_from_case` are the two renderings.

    The ``grouping``, ``keys``, and ``thresholds`` fields describe how an
    operation should be *applied* to the case. They are named after the
    truncation functions the corpus was first written for, but they are just as
    usable as the group-by columns, join keys, and sizes of any other grouped
    operation; an operation with no use for one of them ignores it.

    Attributes:
        id: A unique, human-readable identifier, used as a pytest test ID.
        columns: The column names, in order.
        rows: The rows, as Python-native tuples in the order given by
            ``columns``. Missing values are ``None`` (never ``np.nan``), naive
            datetimes denote UTC wall clocks, and the values are shared by both
            the pandas and the Spark rendering of the case.
        spark_schema: The explicit Spark schema for the case. All fields are
            nullable.
        pandas_dtypes: The pandas dtype of each column, by name.
        grouping: The columns to group the case by.
        keys: The columns that act as keys within a group.
        thresholds: The group sizes worth exercising for this case, chosen to
            straddle the case's actual group sizes.
        notes: Why this case exists, and any subtlety it encodes.
    """

    id: str
    columns: Tuple[str, ...]
    rows: Tuple[Tuple[Any, ...], ...]
    spark_schema: StructType
    pandas_dtypes: Mapping[str, str]
    grouping: Tuple[str, ...]
    keys: Tuple[str, ...]
    thresholds: Tuple[int, ...]
    notes: str = ""

    def to_pandas(self) -> pd.DataFrame:
        """Returns a fresh pandas dataframe holding this case's rows.

        Each column is built as an object-dtype Series and then cast, so that
        pandas never infers a dtype of its own (which would, for instance, turn
        ``None`` in an integer column into a float ``NaN``).

        Returns:
            The pandas rendering of this case.
        """
        data: Dict[str, pd.Series] = {}
        for index, name in enumerate(self.columns):
            values = [row[index] for row in self.rows]
            data[name] = pd.Series(values, dtype=object).astype(
                self.pandas_dtypes[name]
            )
        return pd.DataFrame(data, columns=list(self.columns))

    @property
    def has_row_id(self) -> bool:
        """Whether the case carries a unique :data:`ROW_ID_COLUMN` column."""
        return ROW_ID_COLUMN in self.columns

    @property
    def has_timestamps(self) -> bool:
        """Whether the case has a timestamp column (needing a UTC session)."""
        return _schema_has_timestamps(self.spark_schema)


def _make_case(
    case_id: str,
    fields: Sequence[Tuple[str, DataType, str]],
    rows: Sequence[Tuple[Any, ...]],
    grouping: Sequence[str],
    keys: Sequence[str],
    thresholds: Sequence[int],
    notes: str = "",
) -> EdgeCase:
    """Returns an :class:`EdgeCase` built from a compact field description.

    Args:
        case_id: The case's identifier.
        fields: One ``(name, spark type, pandas dtype)`` triple per column.
        rows: The case's rows.
        grouping: The grouping columns.
        keys: The key columns.
        thresholds: The thresholds worth exercising.
        notes: Why the case exists.

    Returns:
        The assembled edge case.
    """
    columns = tuple(name for name, _, _ in fields)
    for row in rows:
        if len(row) != len(columns):
            raise ValueError(f"Case {case_id} has a row with the wrong arity: {row}")
    return EdgeCase(
        id=case_id,
        columns=columns,
        rows=tuple(rows),
        spark_schema=StructType(
            [StructField(name, spark_type, True) for name, spark_type, _ in fields]
        ),
        pandas_dtypes={name: dtype for name, _, dtype in fields},
        grouping=tuple(grouping),
        keys=tuple(keys),
        thresholds=tuple(thresholds),
        notes=notes,
    )


_ROW_ID_FIELD = (ROW_ID_COLUMN, LongType(), "int64")

# Non-ASCII string values, written as escapes so that the source stays ASCII
# and no editor can normalize them away: a precomposed e-acute, an ASCII e
# followed by a combining acute accent (which renders identically but is a
# different string, and so must hash differently), three CJK characters, and an
# emoji from outside the basic multilingual plane.
E_ACUTE = "\u00e9"
E_COMBINING_ACUTE = "e\u0301"
CJK = "\u65e5\u672c\u8a9e"
EMOJI = "\U0001f642"

EDGE_CASES: Tuple[EdgeCase, ...] = (
    _make_case(
        "nulls-in-grouping-and-key-columns",
        [
            _ROW_ID_FIELD,
            ("g", StringType(), "object"),
            ("k", StringType(), "object"),
            ("payload", LongType(), "int64"),
        ],
        [
            (1, None, "k1", 10),
            (2, None, None, 11),
            (3, "g1", None, 12),
            (4, "g1", "k1", 13),
            (5, "g1", "k2", 14),
            (6, "g2", None, 15),
            (7, None, "k1", 16),
            (8, "g1", "k1", 17),
        ],
        ["g"],
        ["k"],
        [0, 1, 2, 3],
        notes=(
            "Null groups and null keys must be kept and grouped together, not "
            "dropped. Note that a null column contributes nothing to the "
            "combined hash, so (g=NULL, k='k1') and (g='k1', k=NULL) hash "
            "identically; they are in different groups, so that is harmless."
        ),
    ),
    _make_case(
        "empty-string-vs-null",
        [
            _ROW_ID_FIELD,
            ("g", StringType(), "object"),
            ("k", StringType(), "object"),
        ],
        [
            (1, "", "k1"),
            (2, None, "k1"),
            (3, "g1", "k1"),
            (4, "", ""),
            (5, None, None),
            (6, "g1", ""),
            (7, "", None),
        ],
        ["g"],
        ["k"],
        [1, 2],
        notes=(
            "The empty string is hashed (as the digest of no bytes) while a "
            "null is skipped by the combiner, so the two must never collide."
        ),
    ),
    _make_case(
        "unicode-and-separator-strings",
        [
            _ROW_ID_FIELD,
            ("g", StringType(), "object"),
            ("k", StringType(), "object"),
        ],
        [
            (1, "a,", "b"),
            (2, "a", ",b"),
            (3, "a,b", ""),
            (4, E_ACUTE, E_COMBINING_ACUTE),
            (5, E_COMBINING_ACUTE, E_ACUTE),
            (6, CJK, EMOJI),
            (7, "a", "b"),
            (8, "\t\n", " "),
        ],
        ["g"],
        ["k"],
        [1, 2, 3],
        notes=(
            "Rows 1 and 2 are the pair the per-column hashing exists to "
            "separate: naive concatenation would give both 'a,b'. The unicode "
            "values check that both implementations hash UTF-8 bytes, and that "
            "canonically equivalent strings stay distinct."
        ),
    ),
    _make_case(
        "int64-extremes",
        [
            _ROW_ID_FIELD,
            ("g", LongType(), "int64"),
            ("v", LongType(), "int64"),
        ],
        [
            (1, -9223372036854775808, 0),
            (2, 9223372036854775807, -1),
            (3, -1, 9223372036854775807),
            (4, 0, -9223372036854775808),
            (5, -1, 1),
            (6, 0, 0),
            (7, -1, -1),
        ],
        ["g"],
        ["v"],
        [1, 2],
        notes="Integers hash as their decimal rendering, including the sign.",
    ),
    _make_case(
        "nullable-int64-with-na",
        [
            _ROW_ID_FIELD,
            ("g", LongType(), "Int64"),
            ("k", LongType(), "Int64"),
            ("payload", StringType(), "object"),
        ],
        [
            (1, None, 5, "x"),
            (2, 7, None, "y"),
            (3, 7, 5, None),
            (4, 7, 6, "z"),
            (5, None, None, ""),
            (6, 7, 5, "w"),
            (7, None, 5, "x"),
        ],
        ["g"],
        ["k"],
        [1, 2],
        notes=(
            "pandas' nullable Int64 is the only integer dtype that can express "
            "SQL NULL, so it is what a null-bearing integer column must use."
        ),
    ),
    _make_case(
        "float-specials",
        [
            _ROW_ID_FIELD,
            ("g", StringType(), "object"),
            ("v", DoubleType(), "float64"),
        ],
        [
            (1, "g1", float("nan")),
            (2, "g1", float("inf")),
            (3, "g1", float("-inf")),
            (4, "g1", 0.0),
            (5, "g2", 1.5),
            (6, "g2", 5e-324),
            (7, "g2", 1.7976931348623157e308),
            (8, "g1", 1e7),
            (9, "g2", 0.0009),
            (10, "g1", 9999999.999),
            (11, "g2", float("nan")),
        ],
        ["g"],
        ["v"],
        [1, 2, 3],
        notes=(
            "NaN and the infinities take the special-cased hash strings, and "
            "the remaining values sit on the boundaries of Java's plain/"
            "scientific rendering window. There is deliberately no -0.0 here: "
            "see the signed-zeros case."
        ),
    ),
    _make_case(
        "signed-zeros-in-payload",
        [
            _ROW_ID_FIELD,
            ("g", StringType(), "object"),
            ("k", StringType(), "object"),
            ("v", DoubleType(), "float64"),
        ],
        [
            (1, "g1", "k1", 0.0),
            (2, "g1", "k1", -0.0),
            (3, "g1", "k2", 0.0),
            (4, "g2", "k1", -0.0),
            (5, "g1", "k1", 1.0),
            (6, "g2", "k2", -0.0),
        ],
        ["g"],
        ["k"],
        [1, 2],
        notes=(
            "-0.0 hashes differently from 0.0 but compares equal for grouping "
            "and ordering. It is kept out of the grouping and key columns "
            "because Spark's dense_rank would then see two zero signs as two "
            "distinct keys (their hashes differ) while a pandas groupby "
            "normalizes them into one. Every row here has a distinct row_id, "
            "so no two rows are identical except for a zero's sign -- which "
            "would make Spark's own duplicate-row salt nondeterministic."
        ),
    ),
    _make_case(
        "float32-column",
        [
            _ROW_ID_FIELD,
            ("g", StringType(), "object"),
            ("v", FloatType(), "float32"),
        ],
        [
            (1, "g1", 1.0),
            (2, "g1", 0.1),
            (3, "g1", float("nan")),
            (4, "g2", float("inf")),
            (5, "g2", float("-inf")),
            (6, "g2", 3.4028234663852886e38),
            (7, "g1", 1.401298464324817e-45),
            (8, "g2", 1e7),
            (9, "g1", 0.0009),
        ],
        ["g"],
        ["v"],
        [1, 2, 3],
        notes=(
            "float32 values are rendered from the shortest float32 repr, not "
            "the float64 one: 0.1 must hash as '0.1', not '0.10000000149...'. "
            "The values include the largest finite float32 and the smallest "
            "subnormal."
        ),
    ),
    _make_case(
        "dates-with-year-padding",
        [
            _ROW_ID_FIELD,
            ("g", StringType(), "object"),
            ("d", DateType(), "object"),
        ],
        [
            (1, "g1", datetime.date(1, 1, 1)),
            (2, "g1", datetime.date(999, 12, 31)),
            (3, "g1", datetime.date(1969, 12, 31)),
            (4, "g2", datetime.date(1970, 1, 1)),
            (5, "g2", datetime.date(2024, 2, 29)),
            (6, "g2", datetime.date(9999, 12, 31)),
            (7, "g1", None),
        ],
        ["g"],
        ["d"],
        [1, 2, 3],
        notes=(
            "Dates render as yyyy-MM-dd with the year zero-padded to four "
            "digits, which is what date.isoformat() produces. Dates live in "
            "object columns: datetime64[ns] would turn them into timestamps."
        ),
    ),
    _make_case(
        "timestamps-wall-clocks",
        [
            _ROW_ID_FIELD,
            ("g", StringType(), "object"),
            ("t", TimestampType(), "datetime64[ns]"),
        ],
        [
            (1, "g1", datetime.datetime(2026, 3, 8, 2, 30, 0)),
            (2, "g1", datetime.datetime(2026, 11, 1, 1, 30, 0)),
            (3, "g1", datetime.datetime(2020, 1, 1, 0, 0, 0, 500000)),
            (4, "g2", datetime.datetime(2020, 1, 1, 0, 0, 0, 123456)),
            (5, "g2", datetime.datetime(2020, 1, 1, 0, 0, 0, 1)),
            (6, "g2", datetime.datetime(1969, 12, 31, 23, 59, 59, 999999)),
            (7, "g1", None),
            (8, "g2", datetime.datetime(1700, 1, 1, 0, 0, 0)),
            (9, "g1", datetime.datetime(2020, 1, 1, 0, 0, 0)),
        ],
        ["g"],
        ["t"],
        [1, 2, 3],
        notes=(
            "Rows 1 and 2 are wall clocks that do not exist / occur twice in "
            "US Eastern, which must not matter: timestamps are hashed as their "
            "own wall clock. Rows 3-5 cover the fractional-second renderings "
            "(trailing zeros trimmed, six digits, one microsecond) and row 9 "
            "has no fraction at all. All timestamps stay inside the range of "
            "pandas' datetime64[ns]. Build with utc_session_timezone."
        ),
    ),
    _make_case(
        "binary-values",
        [
            _ROW_ID_FIELD,
            ("g", StringType(), "object"),
            ("b", BinaryType(), "object"),
        ],
        [
            (1, "g1", b""),
            (2, "g1", b"\x00"),
            (3, "g1", b"\xff\xfe"),
            (4, "g2", b"abc"),
            (5, "g2", None),
            (6, "g2", b"\x00\x01\x02"),
            (7, "g1", b"\xff\xff\xff\xff"),
        ],
        ["g"],
        ["b"],
        [1, 2, 3],
        notes=(
            "Binary values are hashed as raw bytes, so they are not "
            "interchangeable with the strings that would decode to them. Note "
            "that toPandas() returns bytearrays for binary columns."
        ),
    ),
    _make_case(
        "bytearray-binary-values",
        [
            _ROW_ID_FIELD,
            ("g", BinaryType(), "object"),
            ("b", BinaryType(), "object"),
        ],
        [
            (1, bytearray(b"g1"), bytearray(b"")),
            (2, bytearray(b"g1"), bytearray(b"\x00")),
            (3, bytearray(b"g1"), b"\x00"),
            (4, bytearray(b"g2"), bytearray(b"\xff\xfe")),
            (5, b"g2", None),
            (6, bytearray(b"g1"), bytearray(b"\x00\x01\x02")),
        ],
        ["g"],
        ["b"],
        [1, 2, 3],
        notes=(
            "The same binary values, but held as bytearrays, which is what "
            "toPandas() returns for a binary column when Arrow is disabled. A "
            "bytearray is not hashable, and a pandas groupby needs its keys to "
            "be. Spark compares binary values by content, so rows 2 and 3 hold "
            "one key and rows 4 and 5 one group."
        ),
    ),
    _make_case(
        "object-column-with-nan-and-null",
        [
            _ROW_ID_FIELD,
            ("g", DoubleType(), "object"),
            ("k", DoubleType(), "object"),
        ],
        [
            (1, float("nan"), 1.0),
            (2, None, 1.0),
            (3, float("nan"), None),
            (4, None, float("nan")),
            (5, float("nan"), 1.0),
            (6, None, 2.5),
            (7, 1.0, float("nan")),
            (8, float("nan"), 2.5),
            (9, None, None),
        ],
        ["g"],
        ["k"],
        [1, 2, 3],
        notes=(
            "An object column is the only pandas column that can hold both a "
            "NaN and a null, which is exactly what a Spark double column holds. "
            "The two are different values everywhere: they hash differently, "
            "they are different groups and different keys, and Spark's "
            "ascending order puts nulls first and NaNs last -- while a pandas "
            "groupby puts them in one group and no na_position separates them."
        ),
    ),
    _make_case(
        "duplicate-rows-past-threshold",
        [
            ("x", LongType(), "int64"),
            ("y", LongType(), "int64"),
            ("z", StringType(), "object"),
        ],
        [
            (1, 2, "A"),
            (1, 2, "A"),
            (1, 2, "A"),
            (1, 2, "A"),
            (1, 2, "A"),
            (2, 4, "A"),
            (2, 4, "A"),
            (2, 4, "A"),
            (2, 4, "A"),
            (2, 4, "A"),
            (3, 6, "B"),
        ],
        ["z"],
        ["x"],
        [1, 2, 5, 10],
        notes=(
            "No row_id: identical rows exercise the per-duplicate salt, which "
            "is what stops truncate_large_groups from keeping five copies of "
            "one row while dropping another row entirely."
        ),
    ),
    _make_case(
        "all-null-rows",
        [
            ("g", StringType(), "object"),
            ("k", StringType(), "object"),
        ],
        [
            (None, None),
            (None, None),
            ("a", None),
            (None, "b"),
            (None, None),
        ],
        ["g"],
        ["k"],
        [1, 2],
        notes=(
            "A row whose every hashed column is null hashes the empty "
            "concatenation, and the identical all-null rows also exercise the "
            "duplicate-row salt."
        ),
    ),
    _make_case(
        "groups-exactly-at-threshold",
        [
            ("g", StringType(), "object"),
            ("k", StringType(), "object"),
        ],
        [
            ("g1", "k1"),
            ("g1", "k2"),
            ("g2", "k1"),
            ("g2", "k2"),
            ("g2", "k3"),
            ("g3", "k1"),
        ],
        ["g"],
        ["k"],
        [1, 2, 3],
        notes=(
            "Group sizes 2, 3, and 1 against thresholds 1, 2, and 3 put every "
            "group just under, exactly at, and just over the threshold, where "
            "the <= versus < boundary shows up."
        ),
    ),
    _make_case(
        "multi-column-grouping-and-keys",
        [
            _ROW_ID_FIELD,
            ("g1", StringType(), "object"),
            ("g2", LongType(), "int64"),
            ("k1", StringType(), "object"),
            ("k2", LongType(), "Int64"),
            ("payload", StringType(), "object"),
        ],
        [
            (1, "a", 1, "x", 1, "p"),
            (2, "a", 1, "x", 2, "q"),
            (3, "a", 1, "y", 1, "r"),
            (4, "a", 2, "x", 1, "s"),
            (5, "b", 1, "x", 1, "t"),
            (6, "b", 1, "y", 2, "u"),
            (7, "b", 1, "y", 2, "v"),
            (8, "b", 1, "z", 3, "w"),
            (9, "a", 1, "x", 1, "p"),
            (10, None, 1, "x", None, "p"),
        ],
        ["g1", "g2"],
        ["k1", "k2"],
        [1, 2, 3],
        notes=(
            "Multi-column grouping and multi-column keys, including a row that "
            "repeats a (group, key) pair and one with nulls in both."
        ),
    ),
    _make_case(
        "pandas-string-dtype",
        [
            _ROW_ID_FIELD,
            ("g", StringType(), "string"),
            ("k", StringType(), "string"),
        ],
        [
            (1, "g1", "k1"),
            (2, "g1", None),
            (3, None, "k1"),
            (4, "g1", "k2"),
            (5, "g2", ""),
            (6, None, None),
        ],
        ["g"],
        ["k"],
        [1, 2],
        notes=(
            "The same string data as the object-dtype cases, but stored in "
            "pandas' nullable string dtype, where missing values are pd.NA."
        ),
    ),
    _make_case(
        "nullable-float-na-vs-nan",
        [
            _ROW_ID_FIELD,
            ("g", StringType(), "object"),
            ("v", DoubleType(), "Float64"),
            ("w", DoubleType(), "float64"),
        ],
        [
            (1, "g1", None, float("nan")),
            (2, "g1", None, 1.0),
            (3, "g2", 1.0, float("nan")),
            (4, "g2", None, 0.0),
            (5, "g1", 2.5, 2.5),
            (6, "g1", None, float("inf")),
        ],
        ["g"],
        ["v"],
        [1, 2],
        notes=(
            "SQL NULL and NaN are different values in a floating point column, "
            "and only the nullable Float64 dtype can hold the former. The "
            "nullable column v therefore carries the nulls and the plain "
            "float64 column w the NaNs: a NaN cannot be put in v at all, "
            "because pandas' masked float arrays read np.nan as missing on "
            "construction, so astype('Float64') would silently turn it into "
            "pd.NA and hand Spark and pandas different data."
        ),
    ),
    _make_case(
        "empty-frame",
        [
            _ROW_ID_FIELD,
            ("g", StringType(), "object"),
            ("k", StringType(), "object"),
        ],
        [],
        ["g"],
        ["k"],
        [0, 1],
        notes=(
            "An empty frame still has a schema, which is what column "
            "validation has to work from."
        ),
    ),
    _make_case(
        "single-row",
        [
            _ROW_ID_FIELD,
            ("g", StringType(), "object"),
            ("k", StringType(), "object"),
        ],
        [(1, "g1", "k1")],
        ["g"],
        ["k"],
        [0, 1, 2],
        notes="One row, one group, one key: nothing can be truncated at all.",
    ),
    _make_case(
        "threshold-extremes",
        [
            _ROW_ID_FIELD,
            ("g", StringType(), "object"),
            ("k", StringType(), "object"),
        ],
        [
            (1, "g1", "k1"),
            (2, "g1", "k2"),
            (3, "g1", "k3"),
            (4, "g2", "k1"),
            (5, "g2", "k1"),
        ],
        ["g"],
        ["k"],
        [-1, 0, 1, 10**9],
        notes=(
            "A negative threshold keeps nothing and must not raise, matching "
            "the Spark filter, and a huge threshold keeps everything."
        ),
    ),
)

EDGE_CASES_BY_ID: Dict[str, EdgeCase] = {case.id: case for case in EDGE_CASES}


def spark_df_from_case(spark: SparkSession, case: EdgeCase) -> DataFrame:
    """Returns the Spark rendering of an edge case.

    The frame is built from the case's row tuples and its explicit schema, never
    from a pandas dataframe, so that NaNs are not turned into nulls and dtypes
    are not widened. Naive datetimes are read as UTC wall clocks, so a case with
    timestamps may only be built inside :func:`utc_session_timezone`.

    Args:
        spark: The Spark session to build the dataframe with.
        case: The case to render.

    Returns:
        The Spark dataframe for the case.
    """
    if case.has_timestamps:
        _require_utc_session_timezone(spark)
    rows = [tuple(_to_spark_value(value) for value in row) for row in case.rows]
    return spark.createDataFrame(rows, case.spark_schema)
