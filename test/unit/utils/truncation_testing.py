"""Shared helpers for the truncation test suites.

This module has no ``test_`` prefix, so pytest never collects it. It is imported
by the parity, differential, and property test modules for
:mod:`~tmlt.core.utils.truncation` and its pandas counterpart, and provides:

* :class:`TruncationBackend`, plus :func:`make_spark_backend` and
  :func:`make_pandas_backend`, which put the Spark and pandas truncation
  utilities behind a single pandas-in/pandas-out API.
* :class:`EdgeCase` and the curated :data:`EDGE_CASES` corpus, which covers the
  corners where the two implementations could plausibly disagree.
* :func:`random_frame`, a seeded frame generator for randomized sweeps.
* :func:`multiset_symdiff` and :func:`grouped_symdiff_distance`, the distances
  used by the stability property tests.
* :func:`utc_session_timezone`, :func:`spark_df_from_case`, and
  :func:`spark_df_from_pandas`, which build Spark frames in a way that survives
  the pandas/Arrow ingestion hazards described below.

Ingestion hazards this module works around:

* ``spark.createDataFrame(pandas_df)`` goes through Arrow, which converts float
  ``NaN`` to SQL ``NULL`` and can silently change dtypes. Every Spark frame
  built here is instead created from Python row tuples with an explicit
  :class:`~pyspark.sql.types.StructType`.
* PySpark converts *naive* :class:`~datetime.datetime` objects to timestamps
  using the Python process's local timezone, while rendering them using
  ``spark.sql.session.timeZone``. Rows built here therefore attach UTC to naive
  datetimes, and frames containing timestamps may only be built inside
  :func:`utc_session_timezone`; together those make a naive pandas timestamp and
  its Spark counterpart denote the same wall clock, which is what the pandas
  implementation's rendering assumes.
* A pandas ``float64`` column cannot hold SQL ``NULL``. Following the module
  contract of the pandas implementation, ``NaN`` in a ``float64``/``float32``
  column is a genuine NaN value, and SQL ``NULL`` in a floating point column is
  expressed with the nullable ``Float64``/``Float32`` dtypes and ``pd.NA``.
  Neither the corpus nor the generator ever uses ``np.nan`` to mean ``NULL``.
  An ``object`` column -- the ``object_float`` column kind, and the
  ``object-column-with-nan-and-null`` case -- is the only pandas column that
  can hold both, which is what a Spark double column does.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import datetime
import math
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
)

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    BinaryType,
    BooleanType,
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

from tmlt.core.utils import pandas_truncation, truncation

__all__ = [
    "CJK",
    "COLUMN_KINDS",
    "DEFAULT_DTYPE_MENU",
    "EDGE_CASES",
    "EDGE_CASES_BY_ID",
    "EMOJI",
    "E_ACUTE",
    "E_COMBINING_ACUTE",
    "ROW_ID_COLUMN",
    "SIMPLE_DTYPE_MENU",
    "TRUNCATION_FUNCTIONS",
    "EdgeCase",
    "TruncationBackend",
    "apply_truncation",
    "assert_no_conflating_values",
    "frame_row_ids",
    "grouped_symdiff_distance",
    "is_null_value",
    "label_value",
    "make_pandas_backend",
    "make_spark_backend",
    "multiset_symdiff",
    "normalize_value",
    "normalized_rows",
    "python_rows_from_pandas",
    "random_frame",
    "spark_df_from_case",
    "spark_df_from_pandas",
    "spark_schema_from_pandas",
    "utc_session_timezone",
]

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


_UTC = datetime.timezone.utc

_SESSION_TIMEZONE_KEY = "spark.sql.session.timeZone"

# Session timezone settings that render timestamps as UTC wall clocks.
_UTC_TIMEZONES = frozenset(
    {"UTC", "Etc/UTC", "Etc/GMT", "GMT", "UCT", "Universal", "Z", "Zulu", "+00:00"}
)


################################################################################
# Null taxonomy
################################################################################


def is_null_value(value: Any) -> bool:
    """Returns whether a value is a null value, as opposed to a float NaN.

    This deliberately re-states the null taxonomy of
    :func:`tmlt.core.utils.pandas_truncation._is_null` rather than importing
    it: this module is the oracle the implementation is judged against, so a
    taxonomy regression in the code under test must surface as a test failure
    here instead of silently moving the oracle in lockstep with the bug.
    ``test_pandas_truncation.test_is_null_matches_the_harness_taxonomy`` pins
    the two functions against each other over a canonical corpus of values.

    Args:
        value: The value to classify.

    Returns:
        Whether the value is a null.
    """
    return value is None or value is pd.NA or value is pd.NaT


################################################################################
# Session timezone
################################################################################


@contextmanager
def utc_session_timezone(spark: SparkSession, timezone: str = "UTC") -> Iterator[None]:
    """Sets the Spark session timezone, restoring the previous value on exit.

    Timestamps are only comparable across the two truncation implementations
    when Spark renders them as the same wall clock the pandas implementation
    sees, which requires a UTC session timezone (see the module docstring).

    Args:
        spark: The Spark session to configure.
        timezone: The session timezone to set. Only UTC (or an alias of it)
            makes the two implementations comparable; the argument exists so
            that tests can deliberately set a different timezone.

    Yields:
        Nothing; the timezone is set for the duration of the ``with`` block.
    """
    previous = spark.conf.get(_SESSION_TIMEZONE_KEY, None)
    spark.conf.set(_SESSION_TIMEZONE_KEY, timezone)
    try:
        yield
    finally:
        if previous is None:
            spark.conf.unset(_SESSION_TIMEZONE_KEY)
        else:
            spark.conf.set(_SESSION_TIMEZONE_KEY, previous)


def _require_utc_session_timezone(spark: SparkSession) -> None:
    """Raises if the Spark session is not rendering timestamps as UTC."""
    timezone = spark.conf.get(_SESSION_TIMEZONE_KEY, None)
    if timezone not in _UTC_TIMEZONES:
        raise RuntimeError(
            "Building a Spark dataframe with timestamps requires a UTC session "
            f"timezone, but {_SESSION_TIMEZONE_KEY} is {timezone}. Wrap the test "
            "in truncation_testing.utc_session_timezone(spark)."
        )


################################################################################
# Pandas to Spark conversion
################################################################################


def _to_spark_value(value: Any) -> Any:
    """Returns ``value`` as a Python object PySpark accepts in a row tuple.

    Missing values (``None``, ``pd.NA``, ``NaT``) become ``None``; float ``NaN``
    is passed through as a value, not as a null. Naive datetimes get UTC
    attached so that PySpark does not reinterpret them in the process's local
    timezone.

    Args:
        value: A value taken from a pandas Series.

    Returns:
        The corresponding Python object.
    """
    if is_null_value(value):
        return None
    if isinstance(value, np.datetime64):
        value = pd.Timestamp(value)
        if value is pd.NaT:
            return None
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime(warn=False)
    if isinstance(value, datetime.datetime):
        return value.replace(tzinfo=_UTC) if value.tzinfo is None else value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.str_):
        return str(value)
    if isinstance(value, bytearray):
        return bytes(value)
    return value


def python_rows_from_pandas(df: pd.DataFrame) -> List[Tuple[Any, ...]]:
    """Returns the rows of a pandas dataframe as Python-native tuples.

    The tuples are suitable for ``spark.createDataFrame(rows, schema)``, which
    avoids the Arrow conversions that ``createDataFrame(pandas_df)`` performs.

    Args:
        df: The dataframe to convert.

    Returns:
        One tuple per row, in the dataframe's column order.
    """
    columns = [[_to_spark_value(value) for value in df[name]] for name in df.columns]
    return [tuple(values) for values in zip(*columns)] if columns else []


def _spark_type_for_object_column(series: pd.Series, name: str) -> DataType:
    """Returns the Spark type matching the values of an object-dtype column."""
    for value in series:
        if is_null_value(value):
            continue
        if isinstance(value, str):
            return StringType()
        if isinstance(value, (bytes, bytearray)):
            return BinaryType()
        if isinstance(value, datetime.datetime):
            return TimestampType()
        if isinstance(value, datetime.date):
            return DateType()
        if isinstance(value, bool):
            return BooleanType()
        if isinstance(value, (int, np.integer)):
            return LongType()
        if isinstance(value, float):
            return DoubleType()
        raise NotImplementedError(
            f"Cannot infer a Spark type for column {name} from value {value!r}; "
            "pass an explicit schema."
        )
    # An all-null object column carries no type information at all.
    return StringType()


def spark_schema_from_pandas(df: pd.DataFrame) -> StructType:
    """Returns a Spark schema matching a pandas dataframe's dtypes.

    Integer dtypes map to ``LongType``, ``float64``/``Float64`` to
    ``DoubleType``, ``float32``/``Float32`` to ``FloatType``, ``datetime64[ns]``
    to ``TimestampType``, and pandas string dtypes to ``StringType``. Object
    columns are typed from their first non-null value; an all-null object column
    is assumed to be a string column. Every field is nullable.

    Args:
        df: The dataframe whose schema should be inferred.

    Returns:
        The inferred schema.
    """
    fields: List[StructField] = []
    for name in df.columns:
        series = df[name]
        dtype = series.dtype
        spark_type: DataType
        if pd.api.types.is_bool_dtype(dtype):
            spark_type = BooleanType()
        elif pd.api.types.is_integer_dtype(dtype):
            spark_type = LongType()
        elif dtype == np.float32 or str(dtype) == "Float32":
            spark_type = FloatType()
        elif pd.api.types.is_float_dtype(dtype):
            spark_type = DoubleType()
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            spark_type = TimestampType()
        elif str(dtype) in ("string", "str"):
            spark_type = StringType()
        elif pd.api.types.is_object_dtype(dtype):
            spark_type = _spark_type_for_object_column(series, str(name))
        else:
            raise NotImplementedError(
                f"Cannot infer a Spark type for column {name} with dtype {dtype}; "
                "pass an explicit schema."
            )
        fields.append(StructField(str(name), spark_type, True))
    return StructType(fields)


def _schema_has_timestamps(schema: StructType) -> bool:
    """Returns whether any top-level field of the schema is a timestamp."""
    return any(isinstance(f.dataType, TimestampType) for f in schema.fields)


def spark_df_from_pandas(
    spark: SparkSession, df: pd.DataFrame, schema: Optional[StructType] = None
) -> DataFrame:
    """Returns a Spark dataframe with the contents of a pandas dataframe.

    The frame is built from Python row tuples and an explicit schema, never from
    the pandas dataframe directly, so that NaNs, nulls, and dtypes survive
    unchanged.

    Args:
        spark: The Spark session to build the dataframe with.
        df: The dataframe to convert.
        schema: The schema to build the dataframe with, or None to infer one
            with :func:`spark_schema_from_pandas`.

    Returns:
        The equivalent Spark dataframe.
    """
    schema = spark_schema_from_pandas(df) if schema is None else schema
    if _schema_has_timestamps(schema):
        _require_utc_session_timezone(spark)
    return spark.createDataFrame(python_rows_from_pandas(df), schema)


################################################################################
# Backends
################################################################################


@dataclass(frozen=True)
class TruncationBackend:
    """A truncation implementation behind a pandas-in/pandas-out API.

    The three callables take and return pandas dataframes and accept the same
    arguments as their counterparts in :mod:`~tmlt.core.utils.truncation`, so a
    single test body can exercise either implementation.

    Attributes:
        name: A short name for the backend, used in assertion messages.
        truncate_large_groups: Counterpart of
            :func:`~tmlt.core.utils.truncation.truncate_large_groups`.
        drop_large_groups: Counterpart of
            :func:`~tmlt.core.utils.truncation.drop_large_groups`.
        limit_keys_per_group: Counterpart of
            :func:`~tmlt.core.utils.truncation.limit_keys_per_group`.
    """

    name: str
    truncate_large_groups: Callable[..., pd.DataFrame]
    drop_large_groups: Callable[..., pd.DataFrame]
    limit_keys_per_group: Callable[..., pd.DataFrame]


#: The names of the three truncation functions, as dispatched by
#: :func:`apply_truncation`. Each name is defined in both
#: :mod:`~tmlt.core.utils.truncation` and
#: :mod:`~tmlt.core.utils.pandas_truncation`; the parity, differential, and
#: property suites all parametrize over this one tuple, so adding or renaming
#: a truncation function is one edit rather than one per module.
TRUNCATION_FUNCTIONS: Tuple[str, ...] = (
    "truncate_large_groups",
    "drop_large_groups",
    "limit_keys_per_group",
)


def apply_truncation(
    implementation: Any,
    function: str,
    df: Any,
    grouping_columns: Sequence[str],
    key_columns: Sequence[str],
    threshold: int,
) -> Any:
    """Calls one of the three truncation functions of an implementation by name.

    This is the single dispatch point for tests and helpers that are
    parametrized over the function name, so that adding or renaming a
    truncation function is one edit rather than one per call site.

    Args:
        implementation: Anything carrying the three truncation functions as
            attributes with their usual signatures: a
            :class:`TruncationBackend`, :mod:`tmlt.core.utils.truncation`, or
            :mod:`tmlt.core.utils.pandas_truncation`.
        function: The name of the function to call.
        df: The dataframe to truncate, of whatever type ``implementation``
            accepts.
        grouping_columns: The grouping columns.
        key_columns: The key columns. Only ``limit_keys_per_group`` reads
            them.
        threshold: The truncation threshold.

    Returns:
        Whatever the called function returns.
    """
    if function == "truncate_large_groups":
        return implementation.truncate_large_groups(
            df, list(grouping_columns), threshold
        )
    if function == "drop_large_groups":
        return implementation.drop_large_groups(df, list(grouping_columns), threshold)
    if function == "limit_keys_per_group":
        return implementation.limit_keys_per_group(
            df, list(grouping_columns), list(key_columns), threshold
        )
    raise ValueError(f"Unknown truncation function {function}")


def make_spark_backend(
    spark: SparkSession, schema: Optional[StructType] = None
) -> TruncationBackend:
    """Returns a backend running the Spark truncation utilities.

    Each call converts the pandas input to a Spark dataframe (via row tuples and
    an explicit schema), runs the Spark function, and converts the result back
    with ``toPandas()``. Note that ``toPandas()`` is lossy for some types -- a
    nullable integer column comes back as ``float64``, and nulls in a floating
    point column come back as ``NaN`` -- which is why the edge cases with tricky
    dtypes are compared by surviving :data:`ROW_ID_COLUMN` values.

    Args:
        spark: The Spark session used to build the intermediate dataframes.
        schema: The schema to build the intermediate dataframes with, or None to
            infer one from the pandas dtypes of each input.

    Returns:
        A backend wrapping :mod:`~tmlt.core.utils.truncation`.
    """

    def _to_spark(df: pd.DataFrame) -> DataFrame:
        return spark_df_from_pandas(spark, df, schema)

    def truncate_large_groups(
        df: pd.DataFrame, grouping_columns: Collection[str], threshold: int
    ) -> pd.DataFrame:
        """Calls :func:`~tmlt.core.utils.truncation.truncate_large_groups`."""
        result = truncation.truncate_large_groups(
            _to_spark(df), list(grouping_columns), threshold
        )
        return result.toPandas()

    def drop_large_groups(
        df: pd.DataFrame, grouping_columns: Collection[str], threshold: int
    ) -> pd.DataFrame:
        """Calls :func:`~tmlt.core.utils.truncation.drop_large_groups`."""
        result = truncation.drop_large_groups(
            _to_spark(df), list(grouping_columns), threshold
        )
        return result.toPandas()

    def limit_keys_per_group(
        df: pd.DataFrame,
        grouping_columns: Collection[str],
        key_columns: Collection[str],
        threshold: int,
    ) -> pd.DataFrame:
        """Calls :func:`~tmlt.core.utils.truncation.limit_keys_per_group`."""
        result = truncation.limit_keys_per_group(
            _to_spark(df), list(grouping_columns), list(key_columns), threshold
        )
        return result.toPandas()

    return TruncationBackend(
        name="spark",
        truncate_large_groups=truncate_large_groups,
        drop_large_groups=drop_large_groups,
        limit_keys_per_group=limit_keys_per_group,
    )


def make_pandas_backend() -> TruncationBackend:
    """Returns a backend running the pandas truncation utilities.

    Returns:
        A backend wrapping :mod:`~tmlt.core.utils.pandas_truncation`.
    """

    def truncate_large_groups(
        df: pd.DataFrame, grouping_columns: Collection[str], threshold: int
    ) -> pd.DataFrame:
        """Calls :func:`~tmlt.core.utils.pandas_truncation.truncate_large_groups`."""
        return pandas_truncation.truncate_large_groups(
            df.copy(), list(grouping_columns), threshold
        )

    def drop_large_groups(
        df: pd.DataFrame, grouping_columns: Collection[str], threshold: int
    ) -> pd.DataFrame:
        """Calls :func:`~tmlt.core.utils.pandas_truncation.drop_large_groups`."""
        return pandas_truncation.drop_large_groups(
            df.copy(), list(grouping_columns), threshold
        )

    def limit_keys_per_group(
        df: pd.DataFrame,
        grouping_columns: Collection[str],
        key_columns: Collection[str],
        threshold: int,
    ) -> pd.DataFrame:
        """Calls :func:`~tmlt.core.utils.pandas_truncation.limit_keys_per_group`."""
        return pandas_truncation.limit_keys_per_group(
            df.copy(), list(grouping_columns), list(key_columns), threshold
        )

    return TruncationBackend(
        name="pandas",
        truncate_large_groups=truncate_large_groups,
        drop_large_groups=drop_large_groups,
        limit_keys_per_group=limit_keys_per_group,
    )


################################################################################
# Edge case corpus
################################################################################


@dataclass(frozen=True)
class EdgeCase:
    """A hand-written frame exercising one corner of the truncation contract.

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
        grouping: The grouping columns to truncate by.
        keys: The key columns for
            :func:`~tmlt.core.utils.truncation.limit_keys_per_group`.
        thresholds: The thresholds worth exercising for this case.
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
        notes="Randomly generated by truncation_testing.random_frame.",
    )


################################################################################
# Distances
################################################################################

# Sentinels standing in for values that are not usable as dictionary keys, or
# that must not be conflated with each other.
_NULL = "\x00tmlt-null"
_NAN = "\x00tmlt-nan"


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
