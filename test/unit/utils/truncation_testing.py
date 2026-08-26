"""Shared helpers for the truncation test suites.

This module has no ``test_`` prefix, so pytest never collects it. It is imported
by the parity, differential, and property test modules for
:mod:`~tmlt.core.utils.truncation` and its pandas counterpart, and holds the
pieces that are specific to *truncation*:

* :class:`TruncationBackend`, plus :func:`make_spark_backend` and
  :func:`make_pandas_backend`, which put the Spark and pandas truncation
  utilities behind a single pandas-in/pandas-out API.
* :data:`TRUNCATION_FUNCTIONS` and :func:`apply_truncation`, the one dispatch
  point for tests that are parametrized over the three function names.

Everything backend-neutral -- the :data:`EDGE_CASES` corpus,
:func:`random_frame`, the comparison helpers, the Spark construction path, and
:func:`utc_session_timezone` -- lives in :mod:`test.unit.backend_testing`, the
repo-wide parity harness, and is re-exported here so that this module stays the
single import for the truncation suites. See that package for the null
canonicalization semantics every comparison below depends on, and for the API
freeze the re-exported names are under.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

from dataclasses import dataclass
from test.unit.backend_testing import (
    BACKEND_NAMES,
    CJK,
    COLUMN_KINDS,
    DEFAULT_DTYPE_MENU,
    E_ACUTE,
    E_COMBINING_ACUTE,
    EDGE_CASES,
    EDGE_CASES_BY_ID,
    EMOJI,
    ROW_ID_COLUMN,
    SIMPLE_DTYPE_MENU,
    Backend,
    BackendLike,
    ColumnKind,
    EdgeCase,
    RandomLike,
    assert_frames_equal_as_multisets,
    assert_no_conflating_values,
    df_for,
    exact_value,
    frame_row_ids,
    grouped_symdiff_distance,
    is_null_value,
    label_value,
    multiset_symdiff,
    normalize_value,
    normalized_rows,
    python_rows_from_pandas,
    random_frame,
    spark_df_from_case,
    spark_df_from_pandas,
    spark_schema_from_pandas,
    to_pandas,
    utc_session_timezone,
)
from typing import Any, Callable, Collection, Optional, Sequence, Tuple

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType

from tmlt.core.utils import pandas_truncation, truncation

#: The truncation-specific names this module defines, plus every name it
#: re-exports from :mod:`test.unit.backend_testing`, so that the truncation
#: suites have one import to reach for.
__all__ = [
    "BACKEND_NAMES",
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
    "Backend",
    "BackendLike",
    "ColumnKind",
    "EdgeCase",
    "RandomLike",
    "TruncationBackend",
    "apply_truncation",
    "assert_frames_equal_as_multisets",
    "assert_no_conflating_values",
    "df_for",
    "exact_value",
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
    "to_pandas",
    "utc_session_timezone",
]


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
