"""Differential tests of the pandas structural transformations against Spark.

Every test here runs a pandas transformation and its Spark twin on the *same*
data, described by the *same* schema, and asserts that the two agree. The
inputs are the curated :data:`~test.unit.backend_testing.EDGE_CASES` corpus and
:func:`~test.unit.backend_testing.random_frame`, swept with fixed seeds.

What "agree" means, per transformation:

* :class:`~tmlt.core.transformations.pandas_transformations.select.Select` and
  :class:`~tmlt.core.transformations.pandas_transformations.rename.Rename` do
  not touch values, so the pandas result is compared to the *input frame*
  exactly -- dtypes, row order, and the null flavor of every cell -- and to the
  Spark result as a multiset of rows and in order. (Cross-backend, only the
  multiset comparison can be exact about values: ``toPandas()`` widens nullable
  integer columns and turns a null in a floating point column into a NaN, which
  is why the harness compares canonicalized values.)
* :class:`~tmlt.core.transformations.pandas_transformations.map.Map` is given a
  user function that renders every one of its row's values as a string, so the
  comparison covers what each backend *handed the function* as well as what it
  did with the result.

Two limits on the corpus are worth stating, because they are properties of the
code under test rather than choices:

* Only the cases the pandas column descriptors can describe are used. Binary
  columns, pandas' string extension dtype, and object columns holding floats
  have no :class:`~tmlt.core.domains.pandas_domains.PandasColumnDescriptor`, so
  no transformation over them can be constructed in the first place.
* The Map suites exclude timestamps. Spark's ``Map`` sends every row through
  ``sdf.rdd``, and pyspark's Python-side conversion of a ``TimestampType``
  goes through :func:`time.mktime`, in the *process's local timezone* and
  raising ``OverflowError: mktime argument out of range`` outside the range of
  the platform's ``time_t``. The corpus draws timestamps from 1700 onwards, so
  Spark's own Map cannot round-trip them at all; nothing about the pandas
  implementation is being avoided here.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import math
import random
from test.unit.backend_testing import (
    Backend,
    EdgeCase,
    assert_frames_equal_as_multisets,
    is_null_value,
    normalized_rows,
    random_frame,
    spark_df_from_case,
    to_pandas,
)
from test.unit.transformations.pandas_transformations.structural_testing import (
    describable_cases,
    labelled_value,
    pandas_domain_for_case,
    spark_domain_for_case,
)
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import DataFrame, SparkSession

from tmlt.core.domains.pandas_domains import (
    PandasRowDomain,
    PandasStringColumnDescriptor,
    PandasTableDomain,
)
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkRowDomain,
    SparkStringColumnDescriptor,
)
from tmlt.core.metrics import SymmetricDifference
from tmlt.core.transformations.pandas_transformations.map import (
    Map,
    RowToRowTransformation,
)
from tmlt.core.transformations.pandas_transformations.rename import Rename
from tmlt.core.transformations.pandas_transformations.select import Select
from tmlt.core.transformations.spark_transformations.map import Map as SparkMap
from tmlt.core.transformations.spark_transformations.map import (
    RowToRowTransformation as SparkRowToRowTransformation,
)
from tmlt.core.transformations.spark_transformations.rename import Rename as SparkRename
from tmlt.core.transformations.spark_transformations.select import Select as SparkSelect

#: Seed for the randomized sweeps.
SWEEP_SEED = 20260812

#: Number of random frames each sweep draws.
SWEEP_FRAMES = 12

#: The column kinds the sweeps draw from: every kind the pandas descriptors can
#: describe.
SWEEP_DTYPE_MENU: Tuple[str, ...] = (
    "int64",
    "Int64",
    "string",
    "float64",
    "Float64",
    "float32",
    "date",
    "timestamp",
)

#: The same, without timestamps, which Spark's own Map cannot round-trip; see
#: the module docstring.
MAP_SWEEP_DTYPE_MENU: Tuple[str, ...] = tuple(
    kind for kind in SWEEP_DTYPE_MENU if kind != "timestamp"
)

#: The name the Map suites give the column their user function adds.
LABEL_COLUMN = "row_label"

_SPARK_BACKEND = Backend(name="spark")


################################################################################
# Fixtures and helpers
################################################################################


def _sweep_cases(menu: Sequence[str]) -> List[EdgeCase]:
    """Returns the randomly generated cases of one sweep.

    Args:
        menu: The column kinds to draw from.
    """
    rng = random.Random(SWEEP_SEED)
    return [
        random_frame(
            rng,
            dtype_menu=menu,
            n_rows=12,
            n_payload_columns=2,
            case_id=f"random-{index}",
        )
        for index in range(SWEEP_FRAMES)
    ]


def _timestamp_free(cases: Sequence[EdgeCase]) -> List[EdgeCase]:
    """Returns the cases with no timestamp column.

    Args:
        cases: The cases to filter.
    """
    return [case for case in cases if not case.has_timestamps]


def _case_ids(cases: Sequence[EdgeCase]) -> List[str]:
    """Returns the ids of the given cases, for use as pytest ids.

    Args:
        cases: The cases to name.
    """
    return [case.id for case in cases]


def _as_doubles(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Returns a frame with the given numeric columns put through a double.

    This is the round trip ``toPandas()`` has already performed on the Spark
    side of every comparison here, applied to both sides so that they are
    compared on equal terms. It gives up exactly what that round trip does:

    * A null in a numeric column comes back as ``NaN`` -- a nullable
      ``LongType`` column comes back as ``float64`` in the first place -- so
      the two are merged onto ``None``.
    * An integer past 2^53 is not representable as a double, so
      ``9223372036854775807`` comes back as ``9.223372036854776e+18``. Every
      value is therefore compared as a double.

    Neither is a distinction this comparison could have made, whichever way it
    was keyed. That the pandas transformations *do* make them is asserted on the
    pandas side alone, where no round trip is involved: by
    :func:`_assert_columns_preserved_exactly` for Select and Rename, and by
    comparing the labels -- which each backend's own user function computed from
    the values it was handed, before any round trip -- for Map.

    Args:
        frame: The frame to canonicalize.
        columns: The numeric columns to canonicalize.
    """
    canonicalized = frame.copy()
    for column in columns:
        canonicalized[column] = pd.Series(
            [
                None if is_null_value(value) or _is_nan(value) else float(value)
                for value in frame[column]
            ],
            dtype=object,
        )
    return canonicalized


def _is_nan(value: Any) -> bool:
    """Returns whether a value is a float NaN.

    Args:
        value: The value to check.
    """
    return isinstance(value, (float, np.floating)) and math.isnan(float(value))


def _numeric_columns(frame: pd.DataFrame) -> List[str]:
    """Returns the names of a frame's numeric columns.

    Args:
        frame: The frame to inspect.
    """
    return [
        str(name)
        for name in frame.columns
        if pd.api.types.is_numeric_dtype(frame[name].dtype)
    ]


def _assert_same_result(
    spark_result: DataFrame, pandas_result: pd.DataFrame, ordered: bool = True
) -> None:
    """Asserts that a Spark result and a pandas one hold the same rows.

    Args:
        spark_result: The Spark transformation's output.
        pandas_result: The pandas transformation's output.
        ordered: Whether the two are also required to be in the same row order.
            Every transformation here is row-wise, so they are, but the
            assertion is separated from the multiset one so that a failure says
            which of the two properties broke.
    """
    spark_pandas = to_pandas(spark_result, _SPARK_BACKEND)
    assert list(spark_pandas.columns) == list(pandas_result.columns)
    numeric = _numeric_columns(pandas_result)
    left = _as_doubles(spark_pandas, numeric)
    right = _as_doubles(pandas_result, numeric)
    assert_frames_equal_as_multisets(left, right)
    if ordered:
        columns = [str(name) for name in pandas_result.columns]
        assert normalized_rows(left, columns) == normalized_rows(right, columns)


def _assert_columns_preserved_exactly(
    result: pd.DataFrame, source: pd.DataFrame, columns: Dict[str, str]
) -> None:
    """Asserts a result's columns are a source frame's, unchanged.

    This is the single-backend half of the comparison, and is exact where the
    cross-backend half cannot be: it checks the dtype and every value, with the
    three null flavors kept apart.

    Args:
        result: The transformation's output.
        source: The frame it was given.
        columns: Mapping from the result's column names to the source's.
    """
    assert list(result.columns) == list(columns)
    assert list(result.index) == list(range(len(source)))
    for result_column, source_column in columns.items():
        assert result[result_column].dtype == source[source_column].dtype
        assert_frames_equal_as_multisets(
            result[[result_column]].rename(columns={result_column: source_column}),
            source[[source_column]].reset_index(drop=True),
            normalize=False,
        )
        assert [labelled_value(value) for value in result[result_column]] == [
            labelled_value(value) for value in source[source_column]
        ]


def _selections(case: EdgeCase) -> List[List[str]]:
    """Returns the column selections a case is exercised with.

    Args:
        case: The case being selected from.
    """
    columns = list(case.columns)
    return [columns, columns[::-1], columns[:1], columns[1:]]


def _rename_mapping(case: EdgeCase) -> Dict[str, str]:
    """Returns the renaming a case is exercised with: every column suffixed.

    Args:
        case: The case being renamed.
    """
    return {column: f"{column}_renamed" for column in case.columns}


def _label_function(columns: Sequence[str]) -> Any:
    """Returns a user function rendering a row as one string.

    The same function object is given to both backends, so it must read a
    :class:`~pyspark.sql.Row` and a :class:`dict` alike, which indexing by
    column name does.

    Args:
        columns: The columns to render, in order.
    """

    def label(row: Any) -> Dict[str, Any]:
        return {LABEL_COLUMN: "|".join(labelled_value(row[c]) for c in columns)}

    return label


################################################################################
# Select
################################################################################


@pytest.mark.parametrize(
    "case", describable_cases(), ids=_case_ids(describable_cases())
)
def test_select_matches_spark_on_the_corpus(utc_spark: SparkSession, case: EdgeCase):
    """Select keeps the same columns and rows as its Spark twin.

    Args:
        utc_spark: The Spark session.
        case: The corpus case to run.
    """
    _check_select(utc_spark, case)


@pytest.mark.parametrize(
    "case",
    _sweep_cases(SWEEP_DTYPE_MENU),
    ids=_case_ids(_sweep_cases(SWEEP_DTYPE_MENU)),
)
def test_select_matches_spark_on_random_frames(utc_spark: SparkSession, case: EdgeCase):
    """Select agrees with its Spark twin over a sweep of random frames.

    Args:
        utc_spark: The Spark session.
        case: The generated case to run.
    """
    _check_select(utc_spark, case)


def _check_select(spark: SparkSession, case: EdgeCase) -> None:
    """Runs both Selects over a case and compares them.

    Args:
        spark: The Spark session.
        case: The case to run.
    """
    pandas_domain = pandas_domain_for_case(case)
    assert isinstance(pandas_domain, PandasTableDomain)
    spark_domain = spark_domain_for_case(case)
    for columns in _selections(case):
        pandas_frame = case.to_pandas()
        pandas_result = Select(
            input_domain=pandas_domain, metric=SymmetricDifference(), columns=columns
        )(pandas_frame)
        spark_result = SparkSelect(
            input_domain=spark_domain, metric=SymmetricDifference(), columns=columns
        )(spark_df_from_case(spark, case))
        _assert_same_result(spark_result, pandas_result)
        _assert_columns_preserved_exactly(
            pandas_result, pandas_frame, {column: column for column in columns}
        )


################################################################################
# Rename
################################################################################


@pytest.mark.parametrize(
    "case", describable_cases(), ids=_case_ids(describable_cases())
)
def test_rename_matches_spark_on_the_corpus(utc_spark: SparkSession, case: EdgeCase):
    """Rename produces the same frame as its Spark twin.

    Args:
        utc_spark: The Spark session.
        case: The corpus case to run.
    """
    _check_rename(utc_spark, case)


@pytest.mark.parametrize(
    "case",
    _sweep_cases(SWEEP_DTYPE_MENU),
    ids=_case_ids(_sweep_cases(SWEEP_DTYPE_MENU)),
)
def test_rename_matches_spark_on_random_frames(utc_spark: SparkSession, case: EdgeCase):
    """Rename agrees with its Spark twin over a sweep of random frames.

    Args:
        utc_spark: The Spark session.
        case: The generated case to run.
    """
    _check_rename(utc_spark, case)


def _check_rename(spark: SparkSession, case: EdgeCase) -> None:
    """Runs both Renames over a case and compares them.

    Args:
        spark: The Spark session.
        case: The case to run.
    """
    pandas_domain = pandas_domain_for_case(case)
    assert isinstance(pandas_domain, PandasTableDomain)
    mapping = _rename_mapping(case)
    pandas_frame = case.to_pandas()
    pandas_result = Rename(
        input_domain=pandas_domain,
        metric=SymmetricDifference(),
        rename_mapping=mapping,
    )(pandas_frame)
    spark_result = SparkRename(
        input_domain=spark_domain_for_case(case),
        metric=SymmetricDifference(),
        rename_mapping=mapping,
    )(spark_df_from_case(spark, case))
    _assert_same_result(spark_result, pandas_result)
    _assert_columns_preserved_exactly(
        pandas_result, pandas_frame, {new: old for old, new in mapping.items()}
    )


################################################################################
# Map
################################################################################


@pytest.mark.parametrize(
    "case",
    _timestamp_free(describable_cases()),
    ids=_case_ids(_timestamp_free(describable_cases())),
)
def test_map_matches_spark_on_the_corpus(utc_spark: SparkSession, case: EdgeCase):
    """Map hands its function the same values as its Spark twin, and agrees.

    Args:
        utc_spark: The Spark session.
        case: The corpus case to run.
    """
    _check_map(utc_spark, case)


@pytest.mark.parametrize(
    "case",
    _sweep_cases(MAP_SWEEP_DTYPE_MENU),
    ids=_case_ids(_sweep_cases(MAP_SWEEP_DTYPE_MENU)),
)
def test_map_matches_spark_on_random_frames(utc_spark: SparkSession, case: EdgeCase):
    """Map agrees with its Spark twin over a sweep of random frames.

    Args:
        utc_spark: The Spark session.
        case: The generated case to run.
    """
    _check_map(utc_spark, case)


def _check_map(spark: SparkSession, case: EdgeCase) -> None:
    """Runs both Maps over a case with a labelling function and compares them.

    The function is augmenting and adds one string column holding a rendering of
    every value in the row, so the comparison covers what each backend handed
    the function -- a missing value as against a NaN, an int as against a float,
    a date as against a timestamp -- and not only what came back.

    Args:
        spark: The Spark session.
        case: The case to run.
    """
    pandas_domain = pandas_domain_for_case(case)
    assert isinstance(pandas_domain, PandasTableDomain)
    spark_domain = spark_domain_for_case(case)
    label = _label_function(case.columns)

    pandas_frame = case.to_pandas()
    pandas_result = Map(
        metric=SymmetricDifference(),
        row_transformer=RowToRowTransformation(
            input_domain=PandasRowDomain(pandas_domain.schema),
            output_domain=PandasRowDomain(
                {
                    **pandas_domain.schema,
                    LABEL_COLUMN: PandasStringColumnDescriptor(),
                }
            ),
            trusted_f=label,
            augment=True,
        ),
    )(pandas_frame)
    spark_result = SparkMap(
        metric=SymmetricDifference(),
        row_transformer=SparkRowToRowTransformation(
            input_domain=SparkRowDomain(spark_domain.schema),
            output_domain=SparkRowDomain(
                {
                    **spark_domain.schema,
                    LABEL_COLUMN: SparkStringColumnDescriptor(),
                }
            ),
            trusted_f=label,
            augment=True,
        ),
    )(spark_df_from_case(spark, case))
    _assert_same_result(spark_result, pandas_result)
    # The labels are what the two backends' functions saw, so comparing them
    # exactly -- as strings, in row order -- is the sharpest assertion here.
    spark_labels = list(to_pandas(spark_result, _SPARK_BACKEND)[LABEL_COLUMN])
    assert spark_labels == list(pandas_result[LABEL_COLUMN])


def test_map_output_domain_matches_spark(utc_spark: SparkSession):
    """The two Maps describe their output with the same schema.

    Args:
        utc_spark: The Spark session, which this test does not otherwise need.
    """
    case = _timestamp_free(describable_cases())[0]
    pandas_domain = pandas_domain_for_case(case)
    assert isinstance(pandas_domain, PandasTableDomain)
    transformation = Map(
        metric=SymmetricDifference(),
        row_transformer=RowToRowTransformation(
            input_domain=PandasRowDomain(pandas_domain.schema),
            output_domain=PandasRowDomain(pandas_domain.schema),
            trusted_f=lambda row: {},
            augment=True,
        ),
    )
    assert isinstance(transformation.output_domain, PandasTableDomain)
    assert SparkDataFrameDomain(
        {
            column: descriptor.to_spark_descriptor()
            for column, descriptor in transformation.output_domain.schema.items()
        }
    ) == spark_domain_for_case(case)
