"""Cross-backend parity for the table-level metrics.

:mod:`tmlt.core.metrics` describes a table with either
:class:`~tmlt.core.domains.spark_domains.SparkDataFrameDomain` or
:class:`~tmlt.core.domains.pandas_domains.PandasTableDomain`, and the distance
between two tables must not depend on which. This module is that assertion:
for a corpus of frame pairs it computes each metric's distance under both
backends and requires the two numbers to be equal.

Why not reuse the older pandas branches
=======================================

:class:`~tmlt.core.domains.pandas_domains.PandasDataFrameDomain` -- the
DataFrame domain built out of numpy element domains -- has had a
:class:`~tmlt.core.metrics.SymmetricDifference` branch all along, and it does
not agree with Spark. It counts rows with ``Counter(map(tuple, df.values))``,
which is wrong twice over:

* ``DataFrame.values`` builds one homogeneous array, so an ``int64`` column
  beside a ``float64`` one is upcast to ``float64`` and two integers that
  differ only past ``2**53`` become one row.
* A tuple holding a ``NaN`` is not equal to itself, so a NaN-bearing row is
  counted as differing from its own copy.

The :class:`~tmlt.core.domains.pandas_domains.PandasTableDomain` branches
compare rows with :func:`tmlt.core.utils.pandas_grouping.row_keys` instead,
which is null-safe, per-column, and groups values exactly as Spark does. Both
hazards have a test of their own below, each contrasting the two domains on one
frame pair.

Layout
======

* The corpus: every :data:`~test.unit.backend_testing.EDGE_CASES` frame a
  pandas domain can describe, mutated four ways (unchanged, a row dropped, a
  row added, a value flipped to its null or NaN sibling).
* The parity tests, which take the ``spark`` fixture and so carry the ``spark``
  marker; ``-m "not spark"`` deselects them.
* The pandas-only tests, which run in the ``test-nojvm`` lane: the two
  ``.values`` and NaN hazards, the exact distance types, the empty frames, and
  the mixed-backend dictionary rejection.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

from contextlib import contextmanager
from dataclasses import dataclass
from test.unit.backend_testing import (
    EDGE_CASES,
    ROW_ID_COLUMN,
    Backend,
    ColumnSpec,
    EdgeCase,
    domain_for,
    floating_array,
    spark_df_from_pandas,
    utc_session_timezone,
)
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import pandas as pd
import pytest
import sympy as sp
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DataType,
    DateType,
    DoubleType,
    FloatType,
    LongType,
    StringType,
    TimestampType,
)

from tmlt.core.domains.collections import DictDomain
from tmlt.core.domains.numpy_domains import NumpyFloatDomain, NumpyIntegerDomain
from tmlt.core.domains.pandas_domains import (
    PandasDataFrameDomain,
    PandasDateColumnDescriptor,
    PandasFloatColumnDescriptor,
    PandasIntegerColumnDescriptor,
    PandasSeriesDomain,
    PandasStringColumnDescriptor,
    PandasTableDomain,
    PandasTimestampColumnDescriptor,
)
from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkDateColumnDescriptor,
    SparkIntegerColumnDescriptor,
    SparkTimestampColumnDescriptor,
)
from tmlt.core.exceptions import UnsupportedCombinationError
from tmlt.core.metrics import (
    AbsoluteDifference,
    AddRemoveKeys,
    HammingDistance,
    OnColumn,
    OnColumns,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.utils.exact_number import ExactNumber

PANDAS = Backend("pandas")

################################################################################
# Describing the corpus
################################################################################

# The column kind (see test.unit.backend_testing.domains) each Spark type is
# described by. BinaryType is deliberately absent: neither descriptor family
# has a binary descriptor, so a case with a binary column cannot be described
# by either backend's domain.
_KIND_FOR_SPARK_TYPE: Dict[type, str] = {
    LongType: "int64",
    DoubleType: "float64",
    FloatType: "float32",
    StringType: "string",
    DateType: "date",
    TimestampType: "timestamp",
}

# The pandas dtypes a kind's descriptor accepts, as far as the corpus uses
# them. This is deliberately a restatement rather than a read of
# PandasColumnDescriptor.accepted_dtypes: a case whose pandas rendering and
# Spark rendering disagree about a column's type must be skipped here, not
# quietly described by whichever of the two the accepted dtypes happen to
# allow.
_DTYPES_FOR_KIND: Dict[str, Tuple[str, ...]] = {
    "int64": ("int64", "Int64"),
    "float64": ("float64", "Float64"),
    "float32": ("float32", "Float32"),
    "string": ("object",),
    "date": ("object",),
    "timestamp": ("datetime64[ns]",),
}

#: The corpus cases no pandas domain can describe, and why. Pinned by
#: ``test_undescribable_cases_are_the_expected_ones``, so that adding a
#: descriptor family -- a binary one, say -- fails here until this list is
#: updated and the case starts being exercised.
UNDESCRIBABLE_CASES: Dict[str, str] = {
    "binary-values": "no backend has a binary column descriptor",
    "bytearray-binary-values": "no backend has a binary column descriptor",
    "object-column-with-nan-and-null": (
        "an object column holding floats is a Spark double column, but no "
        "pandas float descriptor accepts the object dtype"
    ),
    "pandas-string-dtype": (
        "PandasStringColumnDescriptor accepts only the object dtype, not "
        "pandas' nullable string dtype"
    ),
}


def _column_spec(case: EdgeCase, name: str) -> Optional[ColumnSpec]:
    """Returns the column spec describing one column of a case.

    Args:
        case: The case the column belongs to.
        name: The column's name.

    Returns:
        The spec, or None if no descriptor describes the column.
    """
    spark_type: DataType = case.spark_schema[name].dataType
    kind = _KIND_FOR_SPARK_TYPE.get(type(spark_type))
    if kind is None or case.pandas_dtypes[name] not in _DTYPES_FOR_KIND[kind]:
        return None
    if name == ROW_ID_COLUMN:
        # Row ids are unique and never null, and a non-null integer column is
        # the only kind of column OnColumn can be applied to at all: a nullable
        # integer descriptor has no numpy domain.
        return (kind, {"allow_null": False})
    return kind


def _schema_for(case: EdgeCase) -> Optional[Dict[str, ColumnSpec]]:
    """Returns the schema spec for a case, or None if it cannot be described.

    Args:
        case: The case to describe.

    Returns:
        One column spec per column, in column order, or None.
    """
    schema: Dict[str, ColumnSpec] = {}
    for name in case.columns:
        spec = _column_spec(case, name)
        if spec is None:
            return None
        schema[name] = spec
    return schema


DESCRIBABLE_CASES: Tuple[EdgeCase, ...] = tuple(
    case for case in EDGE_CASES if _schema_for(case) is not None
)

################################################################################
# Mutating the corpus
################################################################################

# The pandas dtypes that can hold a missing value without changing dtype. A
# numpy float column holds np.nan rather than a null, which is the NaN sibling
# this mutation is after; the others hold pd.NA, None, or NaT.
_NULLABLE_DTYPES = (
    "object",
    "Int64",
    "Int32",
    "Float64",
    "Float32",
    "float64",
    "float32",
    "datetime64[ns]",
    "string",
)


def _drop_first_row(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Returns the frame without its first row, or None if it has none."""
    if not len(df):
        return None
    return df.iloc[1:].reset_index(drop=True)


def _duplicate_first_row(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Returns the frame with its first row appended again, or None if empty."""
    if not len(df):
        return None
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)


def _null_first_value(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Returns the frame with one value replaced by its null or NaN sibling.

    The first column that can hold a missing value in its own dtype is the one
    changed, so the frame's dtypes -- and therefore its Spark schema -- are
    unchanged. In a numpy float column the value written is a ``NaN`` rather
    than a null, which is exactly the sibling worth flipping there.

    Args:
        df: The frame to mutate.

    Returns:
        The mutated frame, or None if no column can hold a missing value.
    """
    if not len(df):
        return None
    for position, name in enumerate(df.columns):
        if name == ROW_ID_COLUMN or str(df[name].dtype) not in _NULLABLE_DTYPES:
            continue
        mutated = df.copy()
        mutated.iloc[0, position] = None
        return mutated
    return None


@dataclass(frozen=True)
class FramePair:
    """Two frames of one case's schema, to compare the backends over.

    Attributes:
        id: A pytest test id.
        case: The corpus case the frames were built from.
        left: The first frame.
        right: The second frame.
    """

    id: str
    case: EdgeCase
    left: pd.DataFrame
    right: pd.DataFrame

    @property
    def schema(self) -> Dict[str, ColumnSpec]:
        """Returns the case's schema spec."""
        schema = _schema_for(self.case)
        assert schema is not None
        return schema


#: The mutations applied to each case. ``unchanged`` is the baseline every
#: metric must call zero; the other three are the smallest edits that move each
#: metric: a row dropped, a row added, and a value flipped to its null or NaN
#: sibling.
MUTATIONS: Tuple[Tuple[str, Callable[[pd.DataFrame], Optional[pd.DataFrame]]], ...] = (
    ("unchanged", lambda df: df),
    ("drop-first-row", _drop_first_row),
    ("duplicate-first-row", _duplicate_first_row),
    ("null-first-value", _null_first_value),
)


def _frame_pairs() -> List[FramePair]:
    """Returns every (case, mutation) pair the parity tests run over.

    Returns:
        The pairs, in corpus order.
    """
    pairs = []
    for case in DESCRIBABLE_CASES:
        left = case.to_pandas()
        for name, mutate in MUTATIONS:
            right = mutate(left)
            if right is None:
                continue
            pairs.append(FramePair(f"{case.id}-{name}", case, left, right))
    return pairs


FRAME_PAIRS: Tuple[FramePair, ...] = tuple(_frame_pairs())

################################################################################
# Helpers
################################################################################


@contextmanager
def _session_timezone_for(case: EdgeCase, spark: SparkSession) -> Iterator[None]:
    """Puts Spark in UTC for the cases that need it, and does nothing else.

    A naive pandas timestamp and its Spark counterpart only denote the same
    wall clock inside a UTC session, and building such a frame outside one
    raises. Cases without timestamps are left alone, so that they are not
    quietly depending on a session setting either.

    Args:
        case: The case about to be built.
        spark: The Spark session.

    Yields:
        Nothing.
    """
    if case.has_timestamps:
        with utc_session_timezone(spark):
            yield
    else:
        yield


def _spark_frames(
    pair: FramePair, spark: SparkSession
) -> Tuple[Any, Any, SparkDataFrameDomain]:
    """Returns a pair's frames and domain for the Spark backend.

    The frames are built under the *case's* explicit Spark schema rather than
    one inferred from the mutated pandas frames, so that a mutation cannot
    change a column's type -- nulling out the last date of a column would
    otherwise leave it with no type information at all.

    Args:
        pair: The pair to build.
        spark: The Spark session to build with.

    Returns:
        The two Spark frames and the domain they belong to.
    """
    domain = domain_for(pair.schema, Backend("spark", spark))
    assert isinstance(domain, SparkDataFrameDomain)
    left = spark_df_from_pandas(spark, pair.left, pair.case.spark_schema)
    right = spark_df_from_pandas(spark, pair.right, pair.case.spark_schema)
    return left, right, domain


def _pandas_domain(pair: FramePair) -> PandasTableDomain:
    """Returns the pandas domain a pair's frames belong to.

    Args:
        pair: The pair to describe.

    Returns:
        The domain.
    """
    domain = domain_for(pair.schema, PANDAS)
    assert isinstance(domain, PandasTableDomain)
    return domain


def _float64_frame(values: List[float], mask: List[bool]) -> pd.DataFrame:
    """Returns a one-column frame holding both NaNs and nulls.

    A ``Float64`` column is the only non-object pandas column that can hold
    both, and it can only be built this way: every Series constructor and
    ``astype`` reads ``np.nan`` as missing and would turn the NaNs into
    ``pd.NA``.

    Args:
        values: The column's values, with anything at a masked position
            ignored.
        mask: True at the positions that are null.

    Returns:
        The frame, with one column ``v``.
    """
    return pd.DataFrame({"v": floating_array(values, mask)})


################################################################################
# The corpus itself
################################################################################


def test_undescribable_cases_are_the_expected_ones() -> None:
    """Only the cases :data:`UNDESCRIBABLE_CASES` names are skipped.

    Every other corpus case is exercised by the parity tests below. Pinning the
    skipped set here means a new descriptor family, or a new case, cannot
    quietly shrink what this module covers.
    """
    skipped = {case.id for case in EDGE_CASES if _schema_for(case) is None}
    assert skipped == set(UNDESCRIBABLE_CASES)
    assert len(DESCRIBABLE_CASES) == len(EDGE_CASES) - len(UNDESCRIBABLE_CASES)


def test_frame_pairs_cover_every_mutation() -> None:
    """Each mutation is exercised, and each pair's frames have one schema.

    A mutation that silently produced None everywhere -- or one that changed a
    frame's dtypes, and so its Spark schema -- would leave the parity tests
    passing over nothing.
    """
    for name, _ in MUTATIONS:
        assert any(pair.id.endswith(name) for pair in FRAME_PAIRS), name
    for pair in FRAME_PAIRS:
        assert list(pair.left.dtypes) == list(pair.right.dtypes), pair.id


def test_add_remove_keys_cases_all_produce_pairs() -> None:
    """Every case :data:`ADD_REMOVE_KEYS_CASE_IDS` names is really exercised.

    The list is written by hand, so a renamed case, or one whose key column
    became a float, would otherwise drop out of that parity test in silence.
    """
    exercised = {pair.case.id for pair in ADD_REMOVE_KEYS_PAIRS}
    assert exercised == set(ADD_REMOVE_KEYS_CASE_IDS)


################################################################################
# Parity across backends
################################################################################


@pytest.mark.parametrize("pair", FRAME_PAIRS, ids=lambda pair: pair.id)
def test_symmetric_difference_agrees_across_backends(
    pair: FramePair, spark: SparkSession
) -> None:
    """The two backends give the same symmetric difference.

    Args:
        pair: The frames to compare.
        spark: The Spark session.
    """
    metric = SymmetricDifference()
    with _session_timezone_for(pair.case, spark):
        sdf1, sdf2, spark_domain = _spark_frames(pair, spark)
        spark_distance = metric.distance(sdf1, sdf2, spark_domain)
    pandas_distance = metric.distance(pair.left, pair.right, _pandas_domain(pair))
    assert pandas_distance == spark_distance
    assert type(pandas_distance) is ExactNumber
    assert type(spark_distance) is ExactNumber


@pytest.mark.parametrize("pair", FRAME_PAIRS, ids=lambda pair: pair.id)
def test_hamming_distance_agrees_across_backends(
    pair: FramePair, spark: SparkSession
) -> None:
    """The two backends give the same Hamming distance.

    The pairs of unequal size are the interesting half: both backends must call
    those infinite, though the Spark branch returns a bare sympy infinity where
    the pandas one returns the equal :class:`.ExactNumber`.

    Args:
        pair: The frames to compare.
        spark: The Spark session.
    """
    metric = HammingDistance()
    with _session_timezone_for(pair.case, spark):
        sdf1, sdf2, spark_domain = _spark_frames(pair, spark)
        spark_distance = metric.distance(sdf1, sdf2, spark_domain)
    pandas_distance = metric.distance(pair.left, pair.right, _pandas_domain(pair))
    assert pandas_distance == spark_distance
    if len(pair.left) == len(pair.right):
        assert type(pandas_distance) is ExactNumber
        assert type(spark_distance) is ExactNumber
    else:
        assert pandas_distance == ExactNumber(sp.oo)


@pytest.mark.parametrize(
    "pair",
    [pair for pair in FRAME_PAIRS if pair.case.has_row_id],
    ids=lambda pair: pair.id,
)
def test_on_column_agrees_across_backends(pair: FramePair, spark: SparkSession) -> None:
    """The two backends give the same OnColumn (and OnColumns) distance.

    The column is the corpus's row id, which is the one column of a case that
    is certainly a non-null integer -- a nullable integer descriptor has no
    numpy domain, so the inner metric could not be applied to one at all.

    Args:
        pair: The frames to compare.
        spark: The Spark session.
    """
    metric = OnColumn(ROW_ID_COLUMN, SumOf(AbsoluteDifference()))
    columns_metric = OnColumns([metric])
    with _session_timezone_for(pair.case, spark):
        sdf1, sdf2, spark_domain = _spark_frames(pair, spark)
        assert metric.supports_domain(spark_domain)
        spark_distance = metric.distance(sdf1, sdf2, spark_domain)
        spark_columns_distance = columns_metric.distance(sdf1, sdf2, spark_domain)
    pandas_domain = _pandas_domain(pair)
    assert metric.supports_domain(pandas_domain)
    pandas_distance = metric.distance(pair.left, pair.right, pandas_domain)
    assert pandas_distance == spark_distance
    assert columns_metric.distance(pair.left, pair.right, pandas_domain) == (
        spark_columns_distance
    )


#: The cases :class:`.AddRemoveKeys` parity runs over, and what each one's key
#: column contributes. Deliberately a handful rather than the whole corpus:
#: this metric filters both dataframes of the dictionary once per shared key,
#: so a Spark run of it costs an order of magnitude more than the other
#: metrics', and the corpus's key columns repeat each other's types. The other
#: metrics still run over every case.
ADD_REMOVE_KEYS_CASE_IDS: Dict[str, str] = {
    "nulls-in-grouping-and-key-columns": "string keys, some null",
    "nullable-int64-with-na": "Int64 keys, some pd.NA",
    "int64-extremes": "int64 keys at the extremes of the type",
    "dates-with-year-padding": "date keys, including a null",
    "all-null-rows": "every key null, so one key holds the whole frame",
}

# "timestamps-wall-clocks" is deliberately absent: the Spark branch of this
# metric collects its keys as Python values and passes them back to
# ``Column.eqNullSafe``, and py4j converts a naive datetime with
# ``time.mktime``, which raises OverflowError for the case's pre-epoch
# timestamps. That is a limitation of the Spark implementation, not of the
# pandas one, so there is nothing to compare; the pandas branch is exercised
# on timestamp keys by test_add_remove_keys_handles_timestamp_keys below.


def _add_remove_keys_pairs() -> List[FramePair]:
    """Returns the pairs :class:`.AddRemoveKeys` is exercised over.

    A pair qualifies when its case is named in :data:`ADD_REMOVE_KEYS_CASE_IDS`
    and its key column is not a floating point one, which the metric rejects.
    The mutations that cannot change a key set or a key's rows are left out.

    Returns:
        The qualifying pairs.
    """
    pairs = []
    for pair in FRAME_PAIRS:
        if pair.case.id not in ADD_REMOVE_KEYS_CASE_IDS:
            continue
        if pair.id.endswith("unchanged") or pair.id.endswith("duplicate-first-row"):
            continue
        spec = pair.schema[_key_column(pair.case)]
        kind = spec if isinstance(spec, str) else spec[0]
        if kind.startswith("float"):
            continue
        pairs.append(pair)
    return pairs


def _key_column(case: EdgeCase) -> str:
    """Returns the column a case's :class:`.AddRemoveKeys` metric keys on.

    A case's ``keys`` are the columns that act as keys within a group, which is
    what this metric's key column is; they are also the more interesting of the
    two, since the corpus's grouping columns are all strings and integers while
    its key columns include dates and timestamps.

    Args:
        case: The case to read.

    Returns:
        The key column's name.
    """
    return case.keys[0]


ADD_REMOVE_KEYS_PAIRS: Tuple[FramePair, ...] = tuple(_add_remove_keys_pairs())


@pytest.mark.parametrize("pair", ADD_REMOVE_KEYS_PAIRS, ids=lambda pair: pair.id)
def test_add_remove_keys_agrees_across_backends(
    pair: FramePair, spark: SparkSession
) -> None:
    """The two backends give the same AddRemoveKeys distance.

    The dictionary holds the same case's frame twice, under the same key
    column, with only one of the two entries mutated: that exercises the union
    of the per-dataframe key sets, and a key whose rows changed in one
    dataframe but not the other.

    Args:
        pair: The frames to compare.
        spark: The Spark session.
    """
    key_column = _key_column(pair.case)
    metric = AddRemoveKeys({1: key_column, 2: key_column})
    with _session_timezone_for(pair.case, spark):
        sdf1, sdf2, spark_element_domain = _spark_frames(pair, spark)
        spark_domain = DictDomain({1: spark_element_domain, 2: spark_element_domain})
        assert metric.supports_domain(spark_domain)
        spark_distance = metric.distance(
            {1: sdf1, 2: sdf1}, {1: sdf2, 2: sdf1}, spark_domain
        )
    pandas_element_domain = _pandas_domain(pair)
    pandas_domain = DictDomain({1: pandas_element_domain, 2: pandas_element_domain})
    assert metric.supports_domain(pandas_domain)
    pandas_distance = metric.distance(
        {1: pair.left, 2: pair.left}, {1: pair.right, 2: pair.left}, pandas_domain
    )
    assert pandas_distance == spark_distance
    assert type(pandas_distance) is ExactNumber


def test_na_and_nan_in_one_float_column_agree_across_backends(
    spark: SparkSession,
) -> None:
    """A ``Float64`` column's nulls and NaNs are told apart, as Spark tells them apart.

    This is the pandas expression of what a Spark double column does: a null
    and a NaN are different values. It needs a
    :class:`pandas.arrays.FloatingArray` built from values and mask, because
    every other route into a nullable float column reads ``np.nan`` as missing
    and would collapse the two.

    Args:
        spark: The Spark session.
    """
    nan = float("nan")
    left = _float64_frame([1.0, nan, 2.0], [False, False, True])
    right = _float64_frame([1.0, 2.0, 2.0], [False, True, True])
    schema: Dict[str, ColumnSpec] = {"v": "float64"}
    pandas_domain = domain_for(schema, PANDAS)
    spark_domain = domain_for(schema, Backend("spark", spark))
    assert isinstance(spark_domain, SparkDataFrameDomain)
    sdf1 = spark_df_from_pandas(spark, left, spark_domain.spark_schema)
    sdf2 = spark_df_from_pandas(spark, right, spark_domain.spark_schema)

    # One NaN was replaced by a null: one row leaves, one arrives.
    assert SymmetricDifference().distance(left, right, pandas_domain) == ExactNumber(2)
    assert SymmetricDifference().distance(
        left, right, pandas_domain
    ) == SymmetricDifference().distance(sdf1, sdf2, spark_domain)
    assert HammingDistance().distance(
        left, right, pandas_domain
    ) == HammingDistance().distance(sdf1, sdf2, spark_domain)


def test_int64_beside_float64_agrees_across_backends(spark: SparkSession) -> None:
    """Integers that differ past 2**53 are different rows, as they are in Spark.

    ``DataFrame.values`` would upcast the integer column to ``float64`` to sit
    beside the float one, and ``2**53`` and ``2**53 + 1`` are one ``float64``.
    The legacy
    :class:`~tmlt.core.domains.pandas_domains.PandasDataFrameDomain` branch does
    exactly that, and is asserted here to show what the table domain fixes.

    Args:
        spark: The Spark session.
    """
    left = pd.DataFrame({"n": [2**53], "x": [0.5]})
    right = pd.DataFrame({"n": [2**53 + 1], "x": [0.5]})
    schema: Dict[str, ColumnSpec] = {"n": "int64", "x": "float64"}
    pandas_domain = domain_for(schema, PANDAS)
    spark_domain = domain_for(schema, Backend("spark", spark))
    assert isinstance(spark_domain, SparkDataFrameDomain)
    sdf1 = spark_df_from_pandas(spark, left, spark_domain.spark_schema)
    sdf2 = spark_df_from_pandas(spark, right, spark_domain.spark_schema)

    assert SymmetricDifference().distance(left, right, pandas_domain) == ExactNumber(2)
    assert SymmetricDifference().distance(sdf1, sdf2, spark_domain) == ExactNumber(2)

    # What the older domain makes of the same two frames, for contrast: the
    # upcast merges the two integers, so it reports no difference at all.
    legacy_domain = PandasDataFrameDomain(
        {
            "n": PandasSeriesDomain(NumpyIntegerDomain()),
            "x": PandasSeriesDomain(NumpyFloatDomain()),
        }
    )
    assert SymmetricDifference().distance(left, right, legacy_domain) == ExactNumber(0)


################################################################################
# pandas-only: the hazards, the types, and the rejections
################################################################################


def test_nan_rows_are_equal_to_themselves() -> None:
    """A frame holding a NaN is at distance zero from its own copy.

    Row tuples would say otherwise -- ``nan != nan``, so every NaN-bearing row
    differs from its copy -- and the legacy
    :class:`~tmlt.core.domains.pandas_domains.PandasDataFrameDomain` branch is
    asserted here saying exactly that, for contrast. It is the bug this branch
    exists to not have; if it is ever fixed, this assertion is the one to
    update.
    """
    frame = pd.DataFrame({"v": [1.0, float("nan"), float("nan")]})
    domain = domain_for({"v": "float64"}, PANDAS)
    assert SymmetricDifference().distance(frame, frame.copy(), domain) == ExactNumber(0)
    assert HammingDistance().distance(frame, frame.copy(), domain) == ExactNumber(0)

    legacy_domain = PandasDataFrameDomain(
        {"v": PandasSeriesDomain(NumpyFloatDomain(allow_nan=True))}
    )
    assert SymmetricDifference().distance(
        frame, frame.copy(), legacy_domain
    ) == ExactNumber(4)


def test_none_and_nan_are_different_rows_in_an_object_column() -> None:
    """In an object column, a null and a NaN are two different rows.

    An object column is the only pandas column that can hold both, and
    :func:`tmlt.core.utils.pandas_grouping.row_keys` keeps them apart, as
    Spark's ``NULL`` and ``NaN`` are apart.

    This one is pandas-only rather than a parity test: a Spark column that can
    hold both is a double column, and no pandas float descriptor accepts an
    object column, so the pair has no Spark counterpart to compare against.
    Only a string descriptor describes an object column, and it treats the NaN
    as one more flavor of null.
    """
    domain = PandasTableDomain({"v": PandasStringColumnDescriptor(allow_null=True)})
    nulls = pd.DataFrame({"v": pd.Series([None, None], dtype=object)})
    nans = pd.DataFrame({"v": pd.Series([float("nan"), float("nan")], dtype=object)})
    mixed = pd.DataFrame({"v": pd.Series([None, float("nan")], dtype=object)})

    assert SymmetricDifference().distance(nulls, nulls.copy(), domain) == ExactNumber(0)
    assert SymmetricDifference().distance(nans, nans.copy(), domain) == ExactNumber(0)
    assert SymmetricDifference().distance(nulls, nans, domain) == ExactNumber(4)
    assert SymmetricDifference().distance(nulls, mixed, domain) == ExactNumber(2)


def test_empty_frames_are_at_distance_zero() -> None:
    """Two empty frames differ by nothing, and an empty frame by everything.

    An empty frame is where a grouping implementation with an unguarded reduce
    or an ``argsort`` over no rows falls over.
    """
    domain = domain_for({"g": "string", "n": "int64"}, PANDAS)
    empty = pd.DataFrame(
        {"g": pd.Series([], dtype=object), "n": pd.Series([], dtype="int64")}
    )
    one_row = pd.DataFrame({"g": pd.Series(["a"], dtype=object), "n": [1]})

    assert SymmetricDifference().distance(empty, empty.copy(), domain) == ExactNumber(0)
    assert HammingDistance().distance(empty, empty.copy(), domain) == ExactNumber(0)
    assert SymmetricDifference().distance(empty, one_row, domain) == ExactNumber(1)
    assert HammingDistance().distance(empty, one_row, domain) == ExactNumber(sp.oo)

    keys_domain = DictDomain({1: domain})
    metric = AddRemoveKeys({1: "g"})
    assert metric.distance({1: empty}, {1: empty.copy()}, keys_domain) == ExactNumber(0)
    assert metric.distance({1: empty}, {1: one_row}, keys_domain) == ExactNumber(1)


def test_distances_are_exact_numbers() -> None:
    """Every distance is an :class:`.ExactNumber`, not a Python int.

    The metrics' own ``validate`` methods are what a distance is checked
    against, and they are called on the way out of every branch here; a branch
    returning a bare int would still compare equal, and would still be wrong.
    """
    domain = domain_for({"n": ("int64", {"allow_null": False})}, PANDAS)
    left = pd.DataFrame({"n": [1, 2, 3]})
    right = pd.DataFrame({"n": [1, 2, 4]})

    symmetric = SymmetricDifference().distance(left, right, domain)
    hamming = HammingDistance().distance(left, right, domain)
    on_column = OnColumn("n", SumOf(AbsoluteDifference())).distance(left, right, domain)
    assert type(symmetric) is ExactNumber
    assert type(hamming) is ExactNumber
    assert type(on_column) is ExactNumber
    assert symmetric == ExactNumber(2)
    assert hamming == ExactNumber(1)
    assert on_column == ExactNumber(1)

    keys_domain = DictDomain({1: domain})
    add_remove = AddRemoveKeys({1: "n"}).distance({1: left}, {1: right}, keys_domain)
    assert type(add_remove) is ExactNumber
    assert add_remove == ExactNumber(2)


def test_add_remove_keys_null_keys_are_one_key() -> None:
    """Rows with a null key are one key, not dropped.

    A pandas ``groupby`` drops them by default, which would make a change to a
    null-keyed row invisible to the metric; the key enumeration here goes
    through :func:`tmlt.core.utils.pandas_grouping.group_indices` instead.
    """
    domain = DictDomain(
        {
            1: PandasTableDomain(
                {
                    "k": PandasStringColumnDescriptor(allow_null=True),
                    "v": PandasIntegerColumnDescriptor(),
                }
            )
        }
    )
    metric = AddRemoveKeys({1: "k"})
    left = pd.DataFrame(
        {"k": pd.Series([None, None, "a"], dtype=object), "v": [1, 2, 3]}
    )
    right = pd.DataFrame(
        {"k": pd.Series([None, None, "a"], dtype=object), "v": [1, 9, 3]}
    )
    # The null key's rows changed, so it is both added and removed; the "a" key
    # is untouched.
    assert metric.distance({1: left}, {1: right}, domain) == ExactNumber(2)
    assert metric.distance({1: left}, {1: left.copy()}, domain) == ExactNumber(0)


def test_add_remove_keys_handles_timestamp_keys() -> None:
    """A timestamp key column works, at Spark's microsecond resolution.

    This has no Spark counterpart to be compared against: that branch collects
    its keys and hands them back to ``Column.eqNullSafe``, where py4j converts
    a naive datetime with ``time.mktime`` and raises for a pre-epoch one. The
    pandas branch has no such range limit, and two timestamps that differ only
    below a microsecond -- a distinction Spark's ``TimestampType`` does not
    have -- are one key here, as they would be there.
    """
    domain = DictDomain(
        {1: PandasTableDomain({"t": PandasTimestampColumnDescriptor()})}
    )
    metric = AddRemoveKeys({1: "t"})
    left = pd.DataFrame(
        {"t": pd.to_datetime(pd.Series(["1700-01-01 00:00:00.000001"]))}
    )
    # One nanosecond later, which is the same microsecond and so the same key.
    right = pd.DataFrame({"t": left["t"] + pd.Timedelta(1, unit="ns")})
    assert metric.distance({1: left}, {1: right}, domain) == ExactNumber(0)

    later = pd.DataFrame({"t": left["t"] + pd.Timedelta(1, unit="us")})
    assert metric.distance({1: left}, {1: later}, domain) == ExactNumber(2)


def test_add_remove_keys_rejects_a_mixed_backend_dictionary() -> None:
    """A dictionary mixing Spark and pandas dataframes is refused, loudly.

    Keys of the two backends are values of unrelated Python types that nothing
    here relates, so a mixed dictionary would report every key as both added
    and removed -- a distance twice as large as the truth, which is safe, and a
    stability guarantee derived from nonsense, which is not.

    No Spark session is needed: a domain is a description, and the mixture is
    caught before either dataframe is looked at.
    """
    domain = DictDomain(
        {
            1: PandasTableDomain({"k": PandasIntegerColumnDescriptor()}),
            2: SparkDataFrameDomain({"k": SparkIntegerColumnDescriptor()}),
        }
    )
    metric = AddRemoveKeys({1: "k", 2: "k"})
    assert not metric.supports_domain(domain)
    with pytest.raises(UnsupportedCombinationError, match="mixing Spark and pandas"):
        metric.distance({}, {}, domain)


def test_add_remove_keys_rejects_float_key_columns() -> None:
    """A floating point key column is refused for pandas as it is for Spark."""
    domain = DictDomain({1: PandasTableDomain({"k": PandasFloatColumnDescriptor()})})
    assert not AddRemoveKeys({1: "k"}).supports_domain(domain)


def test_supports_domain_accepts_the_pandas_table_domain() -> None:
    """The table-level metrics all report the pandas table domain as supported.

    The domains that were supported before still are -- these branches are
    additions, not replacements -- so the older pandas domains are checked here
    too.
    """
    domain = domain_for({"n": ("int64", {"allow_null": False})}, PANDAS)
    metric = OnColumn("n", SumOf(AbsoluteDifference()))
    assert SymmetricDifference().supports_domain(domain)
    assert HammingDistance().supports_domain(domain)
    assert metric.supports_domain(domain)
    assert OnColumns([metric]).supports_domain(domain)

    series_domain = PandasSeriesDomain(NumpyIntegerDomain())
    legacy_domain = PandasDataFrameDomain({"n": series_domain})
    assert SymmetricDifference().supports_domain(series_domain)
    assert SymmetricDifference().supports_domain(legacy_domain)
    assert HammingDistance().supports_domain(series_domain)
    assert HammingDistance().supports_domain(legacy_domain)
    # OnColumn has never accepted the older DataFrame domain, and still does
    # not: it reads a column descriptor, which that domain has no notion of.
    assert not metric.supports_domain(legacy_domain)


@pytest.mark.parametrize(
    "descriptors",
    [
        (
            PandasIntegerColumnDescriptor(allow_null=True),
            SparkIntegerColumnDescriptor(allow_null=True),
        ),
        (PandasDateColumnDescriptor(), SparkDateColumnDescriptor()),
        (PandasTimestampColumnDescriptor(), SparkTimestampColumnDescriptor()),
    ],
    ids=["nullable-int", "date", "timestamp"],
)
def test_on_column_treats_columns_without_a_numpy_domain_alike(
    descriptors: Tuple[Any, Any],
) -> None:
    """A column with no numpy domain is refused the same way on both backends.

    :class:`.OnColumn` hands its inner metric a
    :class:`~tmlt.core.domains.pandas_domains.PandasSeriesDomain` built from
    the column descriptor's numpy domain, and three descriptors have none. The
    Spark branch has always raised :class:`RuntimeError` from
    :meth:`supports_domain` there rather than returning False, so the pandas
    branch does too; this pins them together, and needs no Spark session.

    Args:
        descriptors: The pandas descriptor and the Spark descriptor to check.
    """
    pandas_descriptor, spark_descriptor = descriptors
    metric = OnColumn("a", SumOf(AbsoluteDifference()))
    for domain in (
        PandasTableDomain({"a": pandas_descriptor}),
        SparkDataFrameDomain({"a": spark_descriptor}),
    ):
        with pytest.raises(RuntimeError, match="NumPy"):
            metric.supports_domain(domain)
    # A column the metric knows nothing about is still just unsupported.
    assert not metric.supports_domain(
        PandasTableDomain({"b": PandasIntegerColumnDescriptor()})
    )
