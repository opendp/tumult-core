"""Backend-neutral frame construction and conversion for the parity harness.

This module is part of the frozen harness API; see
:mod:`test.unit.backend_testing` for the freeze contract. It holds three
things:

* :class:`Backend`, the value the repo-wide ``backend`` fixture yields, and
  :class:`BackendLike`, the structural type every helper here accepts.
* The null taxonomy (:func:`is_null_value`) the whole harness is written
  against.
* The pandas-to-Spark construction path, which exists because
  ``spark.createDataFrame(pandas_df)`` is lossy in ways that matter to a parity
  suite.

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
  implementations' rendering assumes.
* A pandas ``float64`` column cannot hold SQL ``NULL``. Following the module
  contract of the pandas implementations, ``NaN`` in a ``float64``/``float32``
  column is a genuine NaN value, and SQL ``NULL`` in a floating point column is
  expressed with the nullable ``Float64``/``Float32`` dtypes and ``pd.NA``.
  Neither the corpus nor the generator ever uses ``np.nan`` to mean ``NULL``.
  An ``object`` column is the only pandas column that can hold both, which is
  what a Spark double column does.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import datetime
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Protocol, Tuple

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

_UTC = datetime.timezone.utc

_SESSION_TIMEZONE_KEY = "spark.sql.session.timeZone"

# Session timezone settings that render timestamps as UTC wall clocks.
_UTC_TIMEZONES = frozenset(
    {"UTC", "Etc/UTC", "Etc/GMT", "GMT", "UCT", "Universal", "Z", "Zulu", "+00:00"}
)


################################################################################
# Backend identity
################################################################################


class BackendLike(Protocol):
    """Anything that names a backend.

    Every helper in this package that branches on the backend reads only
    ``name``, so a test can pass the :class:`Backend` the ``backend`` fixture
    yields, or any richer per-suite backend object carrying the same field
    (:class:`~test.unit.utils.truncation_testing.TruncationBackend`, for
    instance). Names are lowercase and stable: ``"pandas"`` and ``"spark"``.
    """

    @property
    def name(self) -> str:
        """The backend's name."""
        ...  # pragma: no cover


#: The backend names the repo-wide ``backend`` fixture is parametrized over, in
#: fixture order. A suite that needs its own parametrization should read this
#: rather than spelling the names out, so that adding a backend is one edit.
BACKEND_NAMES: Tuple[str, ...] = ("spark", "pandas")


@dataclass(frozen=True)
class Backend:
    """One backend under test, as yielded by the repo-wide ``backend`` fixture.

    This is deliberately thin: it names the backend and, for Spark, carries the
    session that frames must be built with. It is *not* a dispatch table -- a
    suite that wants one builds it from ``name`` (see
    :func:`~test.unit.utils.truncation_testing.make_spark_backend`).

    The Spark session is carried rather than requested as a fixture because
    that is what keeps the pandas half of every parametrized test free of a
    JVM: the fixture resolves ``spark`` lazily, only for the ``"spark"``
    parameter, so a test's static fixture closure never mentions it and the
    ``test-nojvm`` lane can still run the pandas half.

    Attributes:
        name: The backend's name, one of :data:`BACKEND_NAMES`.
        spark: The Spark session, for the Spark backend; None for pandas.
    """

    name: str
    spark: Optional[SparkSession] = None

    def require_spark(self) -> SparkSession:
        """Returns this backend's Spark session, or raises if it has none.

        Returns:
            The Spark session.

        Raises:
            RuntimeError: If this backend carries no Spark session.
        """
        if self.spark is None:
            raise RuntimeError(
                f"The {self.name} backend carries no Spark session; only the "
                "spark backend does."
            )
        return self.spark


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

    Timestamps are only comparable across backends when Spark renders them as
    the same wall clock the pandas backend sees, which requires a UTC session
    timezone (see the module docstring). :func:`spark_df_from_pandas` and
    :func:`df_for` refuse to build a frame with timestamps outside this context
    manager, rather than quietly producing a shifted one.

    Args:
        spark: The Spark session to configure.
        timezone: The session timezone to set. Only UTC (or an alias of it)
            makes the backends comparable; the argument exists so that tests
            can deliberately set a different timezone.

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
            "in backend_testing.utc_session_timezone(spark)."
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
# Backend-neutral frame construction
################################################################################


def df_for(
    pandas_df: pd.DataFrame,
    backend: BackendLike,
    spark: Optional[SparkSession] = None,
) -> Any:
    """Returns a frame holding ``pandas_df``'s data, in a backend's own type.

    This is the input half of the harness: a test writes its fixture data once,
    as a pandas dataframe, and hands it to whichever backend it is running
    against. :func:`to_pandas` is the output half.

    For the pandas backend this is the *identity*: the very frame passed in is
    returned, not a copy. A backend under test that mutates its input therefore
    mutates the caller's frame, exactly as it would in production; copy first if
    that matters to the test.

    For the Spark backend the frame is built from Python row tuples under a
    schema derived by :func:`spark_schema_from_pandas`, never by handing the
    pandas frame to ``createDataFrame``, so that NaNs, nulls, and dtypes survive
    (see the module docstring). Call :func:`spark_df_from_pandas` directly, with
    an explicit ``schema``, when the derived one is not what is wanted -- an
    all-null object column, for instance, is derived as a string column,
    because it carries no type information at all.

    Args:
        pandas_df: The data to build a frame from.
        backend: The backend whose frame type is wanted.
        spark: The Spark session to build with. Defaults to the session carried
            by ``backend``, which is what the ``backend`` fixture supplies;
            pass one explicitly only when the backend carries none. Ignored by
            the pandas backend.

    Returns:
        ``pandas_df`` itself for the pandas backend, or a
        :class:`~pyspark.sql.DataFrame` for the Spark backend.

    Raises:
        ValueError: If ``backend`` is not a known backend.
        RuntimeError: If the Spark backend was given no session, or if the
            frame has timestamps and the session timezone is not UTC.
    """
    if backend.name == "pandas":
        return pandas_df
    if backend.name == "spark":
        session = spark if spark is not None else _spark_of(backend)
        return spark_df_from_pandas(session, pandas_df)
    raise ValueError(f"Unknown backend {backend.name}")


def to_pandas(value: Any, backend: BackendLike) -> pd.DataFrame:
    """Returns a backend's frame as a pandas dataframe.

    This is the output half of the harness: whatever a backend returns is
    brought back to pandas so that one assertion can compare results across
    backends. :func:`df_for` is the input half. (Not to be confused with
    :meth:`~test.unit.backend_testing.corpus.EdgeCase.to_pandas`, which renders
    a corpus case; this one converts a backend's *output*.)

    For the pandas backend this is the *identity*: the very frame passed in is
    returned, not a copy. For the Spark backend it is ``toPandas()``, which is
    lossy in three ways a comparison has to account for, and which is why the
    comparison helpers in
    :mod:`test.unit.backend_testing.comparison` canonicalize rather than
    compare dtypes:

    * A nullable ``LongType`` column with a null comes back as ``float64``.
    * A null in a floating point column comes back as ``NaN``, conflating the
      two. Frames where that distinction matters carry
      :data:`~test.unit.backend_testing.corpus.ROW_ID_COLUMN` and are compared
      by surviving row id instead.
    * Timestamps are rendered using ``spark.sql.session.timeZone``, so a frame
      with timestamps is only meaningful inside :func:`utc_session_timezone`.

    Args:
        value: The frame to convert: a :class:`~pandas.DataFrame` for the
            pandas backend, a :class:`~pyspark.sql.DataFrame` for Spark.
        backend: The backend ``value`` came from.

    Returns:
        The pandas rendering of ``value``.

    Raises:
        ValueError: If ``backend`` is not a known backend.
        TypeError: If ``value`` is not the frame type ``backend`` produces.
    """
    if backend.name == "pandas":
        if not isinstance(value, pd.DataFrame):
            raise TypeError(
                f"The pandas backend produces pandas dataframes, got {type(value)}."
            )
        return value
    if backend.name == "spark":
        if not isinstance(value, DataFrame):
            raise TypeError(
                f"The spark backend produces Spark dataframes, got {type(value)}."
            )
        return value.toPandas()
    raise ValueError(f"Unknown backend {backend.name}")


def _spark_of(backend: BackendLike) -> SparkSession:
    """Returns the Spark session a backend carries, or raises.

    Args:
        backend: The backend to read.

    Returns:
        The session.

    Raises:
        RuntimeError: If the backend carries no session.
    """
    session = getattr(backend, "spark", None)
    if session is None:
        raise RuntimeError(
            f"The {backend.name} backend carries no Spark session, and none was "
            "passed. Pass spark= explicitly, or use the backend fixture, which "
            "supplies one."
        )
    return session
