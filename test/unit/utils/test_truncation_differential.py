"""Differential tests of :mod:`~tmlt.core.utils.pandas_truncation` against Spark.

Every test in this module runs both :mod:`~tmlt.core.utils.truncation` and
:mod:`~tmlt.core.utils.pandas_truncation` on the same data and asserts that the
two keep the same rows. The inputs come from
:mod:`test.unit.utils.truncation_testing`: the curated :data:`EDGE_CASES`
corpus, which covers the corners where the two implementations could plausibly
disagree, and :func:`random_frame`, which is swept over with fixed seeds.

Two comparison modes are used, for the reasons given in
:mod:`test.unit.utils.truncation_testing`:

* Cases whose dtypes do not survive a Spark round trip unambiguously carry a
  unique ``row_id`` column, and are compared by the *set of surviving row ids*.
  This sidesteps ``toPandas()`` conflating ``NULL`` with ``NaN`` and widening
  nullable integers to floats. The ``row_id`` column is part of the frame on
  both sides, so it affects the two implementations identically.
* Cases without a ``row_id`` -- the ones with duplicate rows, which exercise the
  per-duplicate salt -- are compared as whole frames with
  :func:`~tmlt.core.utils.testing.assert_dataframe_equal`. There the surviving
  *multiset* of rows is the whole point, since which copy of a duplicate row
  survives is not observable.

The Spark session is put in UTC for every test here (naive timestamps are
otherwise rendered in a timezone the pandas implementation cannot know about),
and its shuffle partition count is lowered, which only affects how much work
Spark does.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import math
import random
import struct
from dataclasses import replace
from decimal import Decimal
from test.unit.utils.truncation_testing import (
    DEFAULT_DTYPE_MENU,
    EDGE_CASES,
    EDGE_CASES_BY_ID,
    SIMPLE_DTYPE_MENU,
    TRUNCATION_FUNCTIONS,
    EdgeCase,
    apply_truncation,
    frame_row_ids,
    label_value,
    random_frame,
    spark_df_from_case,
)
from typing import Any, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as sf
from pyspark.sql.types import (
    DataType,
    DoubleType,
    FloatType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from tmlt.core.utils import pandas_truncation, truncation
from tmlt.core.utils.pandas_truncation import (
    _hash_columns,
    _java_double_to_string,
    _java_float_to_string,
)
from tmlt.core.utils.testing import Case, assert_dataframe_equal, parametrize

#: Seed for the small randomized sweep.
SMALL_SWEEP_SEED = 20260809

#: Seed for the large randomized sweep. Deliberately different from
#: :data:`SMALL_SWEEP_SEED`, so the two sweeps do not test the same frames.
LARGE_SWEEP_SEED = 987654321

#: Number of frames in the small (unmarked) sweep.
SMALL_SWEEP_FRAMES = 25

#: Number of frames in the large (slow) sweep.
LARGE_SWEEP_FRAMES = 250

#: The dtypes the sweeps draw from: every supported kind except float32.
#:
#: The sweeps assert that Spark and pandas render every value identically, and
#: for float32 that is not achievable against a JVM older than 19: Java's
#: ``Float.toString`` emits more digits than are needed for roughly a tenth of
#: all floats (JDK-4511638), where its ``Double.toString`` does so for a
#: fraction of a percent of doubles with many significant digits -- which the
#: generator already avoids by drawing short decimal literals. There is no
#: comparable rule for floats: on the JVM this suite runs against,
#: ``-743887011840.0`` renders as ``-7.4388701E11`` in Spark and as the
#: shortest round-tripping ``-7.43887E11`` here. float32 rendering is covered
#: by the curated ``float32-column`` case and by
#: :func:`test_float_formatter_matches_spark_cast`, which classifies such
#: differences rather than requiring equality.
SWEEP_DTYPE_MENU: Tuple[str, ...] = tuple(
    kind for kind in DEFAULT_DTYPE_MENU if kind != "float32"
)

#: The magnitude at and above which this JVM stops rendering the doubles the
#: generator draws as the shortest decimal that round-trips. See
#: :func:`_jvm_stable_double`.
JVM_STABLE_DOUBLE_LIMIT = 1e17

#: Number of random bit patterns each formatter classification test draws.
FORMATTER_SAMPLE_SIZE = 2000

#: Seed for the float64 formatter classification test.
DOUBLE_FORMATTER_SEED = 11223344

#: Seed for the float32 formatter classification test.
FLOAT_FORMATTER_SEED = 44332211

################################################################################
# Running and comparing the two implementations
################################################################################


def _spark_result(
    sdf: DataFrame, case: EdgeCase, function: str, threshold: int
) -> pd.DataFrame:
    """Returns the output of a Spark truncation function, as a pandas frame.

    Args:
        sdf: The Spark rendering of ``case``.
        case: The case being run, which supplies the grouping and key columns.
        function: The name of the truncation function to run.
        threshold: The truncation threshold.

    Returns:
        The result of the Spark function, converted with ``toPandas()``.
    """
    result = apply_truncation(
        truncation, function, sdf, case.grouping, case.keys, threshold
    )
    return result.toPandas()


def _pandas_result(case: EdgeCase, function: str, threshold: int) -> pd.DataFrame:
    """Returns the output of a pandas truncation function.

    A fresh frame is built for each call, so that a function mutating its input
    cannot make a later call look correct.

    Args:
        case: The case to run, which supplies the frame, the grouping columns,
            and the key columns.
        function: The name of the truncation function to run.
        threshold: The truncation threshold.

    Returns:
        The result of the pandas function.
    """
    return apply_truncation(
        pandas_truncation,
        function,
        case.to_pandas(),
        case.grouping,
        case.keys,
        threshold,
    )


def _survivor_row_ids(df: pd.DataFrame) -> Set[int]:
    """Returns the set of row ids in a result frame."""
    return set(frame_row_ids(df))


def _assert_agrees(
    sdf: DataFrame, case: EdgeCase, function: str, threshold: int
) -> None:
    """Asserts that both implementations keep the same rows of a case.

    Args:
        sdf: The Spark rendering of ``case``, built by the caller so that one
            frame can be reused across functions and thresholds.
        case: The case to run.
        function: The name of the truncation function to run.
        threshold: The truncation threshold.
    """
    spark_result = _spark_result(sdf, case, function, threshold)
    pandas_result = _pandas_result(case, function, threshold)
    context = f"case {case.id}, {function}, threshold {threshold}"
    if case.has_row_id:
        pandas_ids = _survivor_row_ids(pandas_result)
        spark_ids = _survivor_row_ids(spark_result)
        # Row ids are unique in the input and truncation only ever selects
        # rows, so a result with fewer distinct ids than rows has duplicated
        # rows -- which comparing sets of ids would otherwise hide.
        for name, result, ids in (
            ("pandas", pandas_result, pandas_ids),
            ("Spark", spark_result, spark_ids),
        ):
            assert len(result) == len(ids), (
                f"{context}: the {name} result has {len(result)} rows but only "
                f"{len(ids)} distinct row ids, so it duplicated rows."
            )
        assert pandas_ids == spark_ids, (
            f"{context}: kept different rows. Only pandas kept row ids "
            f"{sorted(pandas_ids - spark_ids)}; only Spark kept "
            f"{sorted(spark_ids - pandas_ids)}. Input rows: {case.rows}"
        )
        return
    try:
        assert_dataframe_equal(pandas_result, spark_result)
    except AssertionError as error:
        raise AssertionError(f"{context}: {error}") from error


################################################################################
# Curated corpus
################################################################################


def _corpus_cases() -> List[Case]:
    """Returns the (curated case, threshold) parametrizations of the corpus.

    Returns:
        One :class:`~tmlt.core.utils.testing.Case` per case and threshold.
    """
    return [
        Case(f"{case.id}-threshold-{threshold}")(case_id=case.id, threshold=threshold)
        for case in EDGE_CASES
        for threshold in case.thresholds
    ]


@parametrize(Case(function)(function=function) for function in TRUNCATION_FUNCTIONS)
@parametrize(*_corpus_cases())
def test_truncation_matches_spark(
    utc_spark: SparkSession, function: str, case_id: str, threshold: int
) -> None:
    """Each truncation function keeps the rows its Spark version keeps."""
    case = EDGE_CASES_BY_ID[case_id]
    sdf = spark_df_from_case(utc_spark, case)
    _assert_agrees(sdf, case, function, threshold)


################################################################################
# Randomized sweeps
################################################################################


def _jvm_stable_double(value: float) -> float:
    """Returns a double near ``value`` that both backends are known to render alike.

    Java's pre-19 ``Double.toString`` does not always produce the shortest
    decimal that round-trips, which is what the pandas formatter produces; the
    two therefore hash some doubles differently on the JVM this suite runs
    against. Measured over 40000 values drawn the way
    :func:`~test.unit.utils.truncation_testing.random_frame` draws them, every
    divergence was at a magnitude of 1e17 or more (about 5% of those), and none
    at all occurred below it. Values at or above that magnitude are therefore
    rebuilt from their leading digits at a magnitude below one, which keeps
    them arbitrary-looking while putting them back in the range where the two
    renderings agree.

    Args:
        value: The generated value.

    Returns:
        ``value`` itself if it is already in the stable range, and a scaled-down
        value with the same leading digits otherwise.
    """
    if not math.isfinite(value) or abs(value) < JVM_STABLE_DOUBLE_LIMIT:
        return value
    digits = "".join(map(str, Decimal(repr(abs(value))).as_tuple().digits))
    return math.copysign(float(f"0.{digits[:12]}"), value)


def _make_comparable(case: EdgeCase) -> EdgeCase:
    """Returns a generated case whose two renderings hold the same values.

    Two kinds of generated value are repaired:

    * A NaN in a ``Float32``/``Float64`` column, which the two renderings of a
      case cannot agree on: the Spark side keeps it as a NaN, while
      ``astype("Float64")`` reads it as missing, so the pandas side gets pd.NA
      and the backends are handed different data. Such a NaN becomes a null.
      Genuine NaNs still reach both sides through the plain ``float64`` columns,
      where neither side can express a null.
    * A double that this JVM does not render as the shortest round-tripping
      decimal (see :func:`_jvm_stable_double`).

    Object columns are repaired too, since the ``object_float`` column kind puts
    doubles in one -- but a NaN there is kept, because an object column holds a
    NaN and a null as themselves and so hands both sides the same data. Values
    of any other type in an object column are left alone.

    Args:
        case: The generated case to repair.

    Returns:
        The repaired case, or ``case`` itself if it has no floating point
        columns.
    """
    nullable = {
        index
        for index, name in enumerate(case.columns)
        if case.pandas_dtypes[name] in ("Float32", "Float64")
    }
    floating = {
        index
        for index, name in enumerate(case.columns)
        if case.pandas_dtypes[name]
        in ("Float32", "Float64", "float32", "float64", "object")
    }
    if not floating:
        return case
    rows = []
    for row in case.rows:
        values = list(row)
        for index in floating:
            value = values[index]
            if not isinstance(value, float):
                continue
            if math.isnan(value) and index in nullable:
                values[index] = None
            else:
                values[index] = _jvm_stable_double(value)
        rows.append(tuple(values))
    return replace(case, rows=tuple(rows))


def _sweep_frame(index: int, rng: random.Random) -> EdgeCase:
    """Returns the ``index``-th frame of a sweep.

    The shape of the frame is derived from ``index`` and its values from
    ``rng``, so a failing frame can be rebuilt from the sweep's seed and the
    index alone. Every third frame has no ``row_id`` and a high duplicate rate,
    which is what exercises the per-duplicate salt; the others carry a
    ``row_id`` and draw from :data:`SWEEP_DTYPE_MENU`, rotating which dtype lands
    on the grouping and key columns. The generated values are then repaired by
    :func:`_make_comparable`.

    Args:
        index: The frame's position in the sweep.
        rng: The seeded source of randomness for the frame's values.

    Returns:
        The generated case.
    """
    with_row_id = index % 3 != 0
    menu = SWEEP_DTYPE_MENU if with_row_id else SIMPLE_DTYPE_MENU
    rotation = index % len(menu)
    case = random_frame(
        rng,
        dtype_menu=menu[rotation:] + menu[:rotation],
        n_rows=6 + 3 * (index % 7),
        n_groups=1 + index % 4,
        dup_rate=0.3 if with_row_id else 0.6,
        n_grouping_columns=1 + index % 2,
        n_key_columns=1 + (index // 2) % 2,
        n_payload_columns=index % 3,
        n_key_values=2 + index % 3,
        with_row_id=with_row_id,
        case_id=f"sweep-{index}",
    )
    return _make_comparable(case)


def _run_sweep(spark: SparkSession, seed: int, frames: int) -> None:
    """Compares both implementations on a seeded sequence of random frames.

    Args:
        spark: A Spark session with a UTC session timezone.
        seed: The seed of the first frame; frame ``i`` uses ``seed + i``.
        frames: The number of frames to generate.
    """
    thresholds = (1, 2, 3, 0)
    for index in range(frames):
        case = _sweep_frame(index, random.Random(seed + index))
        sdf = spark_df_from_case(spark, case)
        threshold = thresholds[index % len(thresholds)]
        for function in TRUNCATION_FUNCTIONS:
            _assert_agrees(sdf, case, function, threshold)


def test_random_sweep_small(utc_spark: SparkSession) -> None:
    """Both implementations agree on a small sweep of random frames."""
    _run_sweep(utc_spark, SMALL_SWEEP_SEED, SMALL_SWEEP_FRAMES)


@pytest.mark.slow
def test_random_sweep_large(utc_spark: SparkSession) -> None:
    """Both implementations agree on a large sweep of random frames."""
    _run_sweep(utc_spark, LARGE_SWEEP_SEED, LARGE_SWEEP_FRAMES)


################################################################################
# Nulls and NaNs in one column
################################################################################

# These cases cannot go through the corpus machinery: the rows they need are
# identical except for a NaN or a null in a double column, so neither a row_id
# (which would make the rows distinct, and so change what is being tested) nor
# toPandas() (which reads a null in a double column as a NaN) can tell the
# surviving rows apart. They are compared through collect() instead, which
# keeps the two distinct.

#: Three rows holding a NaN and three holding a null, alike in every other way.
#: Spark's duplicate-row salt numbers each triple 1, 2, 3, because it partitions
#: by every column and a NaN and a null are different partitions there.
_NAN_AND_NULL_FIELDS: Tuple[StructField, ...] = (
    StructField("g", StringType(), True),
    StructField("v", DoubleType(), True),
)

_NAN_AND_NULL_ROWS: Tuple[Tuple[Any, ...], ...] = (
    ("G", float("nan")),
    ("G", float("nan")),
    ("G", float("nan")),
    ("G", None),
    ("G", None),
    ("G", None),
)

#: Two rows whose combined hashes collide without any SHA-256 collision: a null
#: column contributes nothing to the combined hash, exactly as ``concat_ws``
#: skips it, so ('G', NaN, 'A', NULL) and ('G', NULL, 'nan', 'A') both hash the
#: string ``h('G'),h('nan'),h('A')``. Which of them survives is decided by the
#: ordering of the value columns, where Spark puts the null first and the NaN
#: last.
_HASH_TIE_FIELDS: Tuple[StructField, ...] = (
    StructField("g", StringType(), True),
    StructField("a", DoubleType(), True),
    StructField("b", StringType(), True),
    StructField("c", StringType(), True),
)

_HASH_TIE_ROWS: Tuple[Tuple[Any, ...], ...] = (
    ("G", float("nan"), "A", None),
    ("G", None, "nan", "A"),
)


def _threshold_cases(thresholds: Sequence[int]) -> List[Case]:
    """Returns one parametrization per threshold.

    Args:
        thresholds: The thresholds to run.

    Returns:
        One :class:`~tmlt.core.utils.testing.Case` per threshold.
    """
    return [
        Case(f"threshold-{threshold}")(threshold=threshold) for threshold in thresholds
    ]


def _spark_row_labels(sdf: DataFrame) -> Tuple[Tuple[str, ...], ...]:
    """Returns the sorted labelled rows of a Spark frame.

    ``collect()`` is used rather than ``toPandas()``, which reads a null in a
    double column as a NaN and so cannot tell the two apart.

    Args:
        sdf: The frame to label.

    Returns:
        One tuple of labels per row, sorted.
    """
    columns = sdf.columns
    return tuple(
        sorted(
            tuple(label_value(row[column]) for column in columns)
            for row in sdf.collect()
        )
    )


def _pandas_row_labels(df: pd.DataFrame) -> Tuple[Tuple[str, ...], ...]:
    """Returns the sorted labelled rows of a pandas frame.

    Args:
        df: The frame to label.

    Returns:
        One tuple of labels per row, sorted.
    """
    return tuple(
        sorted(
            tuple(label_value(value) for value in row)
            for row in df.itertuples(index=False, name=None)
        )
    )


def _both_frames(
    spark: SparkSession,
    fields: Sequence[StructField],
    rows: Sequence[Tuple[Any, ...]],
) -> Tuple[DataFrame, pd.DataFrame]:
    """Returns the Spark and pandas renderings of the given rows.

    The pandas columns are all object columns, which is the only pandas dtype
    that can hold a NaN and a null at once, as a Spark double column does.

    Args:
        spark: The Spark session to build the Spark frame with.
        fields: The fields of the Spark schema.
        rows: The rows, as Python tuples.

    Returns:
        The Spark frame and the pandas frame.
    """
    names = [field.name for field in fields]
    sdf = spark.createDataFrame(list(rows), StructType(list(fields)))
    df = pd.DataFrame(
        {
            name: pd.Series([row[index] for row in rows], dtype=object)
            for index, name in enumerate(names)
        },
        columns=names,
    )
    return sdf, df


@parametrize(*_threshold_cases((1, 2, 3, 4, 6)))
def test_truncate_salts_duplicate_nan_and_null_rows_like_spark(
    utc_spark: SparkSession, threshold: int
) -> None:
    """The duplicate-row salt numbers the NaN rows and the null rows apart.

    A pandas groupby puts a NaN and a null in one group, which would number
    these six rows 1 to 6 where Spark numbers them 1, 2, 3 and 1, 2, 3 -- so
    the rows would be hashed with different salts and different rows kept.
    """
    sdf, df = _both_frames(utc_spark, _NAN_AND_NULL_FIELDS, _NAN_AND_NULL_ROWS)
    assert _pandas_row_labels(
        pandas_truncation.truncate_large_groups(df, ["g"], threshold)
    ) == _spark_row_labels(truncation.truncate_large_groups(sdf, ["g"], threshold))


@parametrize(*_threshold_cases((1, 2, 3, 4)))
def test_drop_separates_nan_and_null_groups_like_spark(
    utc_spark: SparkSession, threshold: int
) -> None:
    """A NaN group and a null group are two groups, of three rows each."""
    sdf, df = _both_frames(utc_spark, _NAN_AND_NULL_FIELDS, _NAN_AND_NULL_ROWS)
    assert _pandas_row_labels(
        pandas_truncation.drop_large_groups(df, ["v"], threshold)
    ) == _spark_row_labels(truncation.drop_large_groups(sdf, ["v"], threshold))


@parametrize(*_threshold_cases((1, 2, 3)))
def test_limit_keys_separates_nan_and_null_keys_like_spark(
    utc_spark: SparkSession, threshold: int
) -> None:
    """A NaN key and a null key are two keys, as they are in Spark."""
    sdf, df = _both_frames(utc_spark, _NAN_AND_NULL_FIELDS, _NAN_AND_NULL_ROWS)
    assert _pandas_row_labels(
        pandas_truncation.limit_keys_per_group(df, ["g"], ["v"], threshold)
    ) == _spark_row_labels(
        truncation.limit_keys_per_group(sdf, ["g"], ["v"], threshold)
    )


@parametrize(*_threshold_cases((1, 2)))
def test_colliding_hashes_are_broken_in_sparks_order(
    utc_spark: SparkSession, threshold: int
) -> None:
    """Rows whose combined hashes collide are ordered the way Spark orders them.

    The tie is broken by the value columns, where Spark's ascending order puts
    the null first and the NaN last, while pandas' ``na_position`` puts them in
    the same place.
    """
    sdf, df = _both_frames(utc_spark, _HASH_TIE_FIELDS, _HASH_TIE_ROWS)
    hashes = set(_hash_columns(df, ["g", "a", "b", "c"]))
    assert len(hashes) == 1, "the two rows are supposed to hash identically"
    assert _pandas_row_labels(
        pandas_truncation.truncate_large_groups(df, ["g"], threshold)
    ) == _spark_row_labels(truncation.truncate_large_groups(sdf, ["g"], threshold))


################################################################################
# Floating point formatter
################################################################################

# Doubles whose rendering exercises a specific rule: the signed zeros, the
# boundaries of Java's plain-notation window, the extremes of the type, and
# values whose shortest repr has a trailing zero or needs all 17 digits.
CURATED_DOUBLES: Tuple[float, ...] = (
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.001,
    -0.001,
    0.0009,
    9999999.999,
    1e7,
    -1e7,
    1e16,
    1e-3,
    5e-324,
    -5e-324,
    2.2250738585072014e-308,
    1.7976931348623157e308,
    -1.7976931348623157e308,
    0.1,
    0.2,
    0.3,
    1.0 / 3.0,
    9999999.999999998,
    123456789012345.6,
    5152716558868863.0,
    1e23,
    1.0000000000000002,
    1234567890.12345,
)

# The float32 counterparts: signed zeros, the same window boundaries, the
# largest finite value, the smallest normal, and the smallest subnormal (whose
# shortest rendering, 1.4E-45, needs two digits where one would round-trip).
CURATED_FLOATS: Tuple[float, ...] = (
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.1,
    0.001,
    0.0009,
    1e7,
    -1e7,
    1e-3,
    1.401298464324817e-45,
    1.1754943508222875e-38,
    3.4028234663852886e38,
    -3.4028234663852886e38,
    16777216.0,
    0.3333333432674408,
)


def _random_bit_pattern_floats(
    rng: random.Random, count: int, float_code: str, int_code: str, bits: int
) -> List[float]:
    """Returns finite values drawn uniformly over a float type's bit patterns.

    Sampling bit patterns rather than decimal literals is what makes this a
    worst case for the formatter: almost every value drawn this way needs all
    of the type's significant digits (16 or 17 for a double), which is exactly
    the population where Java's pre-19 ``toString`` emits more digits than are
    needed.

    Args:
        rng: The seeded source of randomness.
        count: How many values to return.
        float_code: The ``struct`` format code of the float type.
        int_code: The ``struct`` format code of the same-width integer type.
        bits: The width of the type, in bits.

    Returns:
        The sampled values.
    """
    values: List[float] = []
    while len(values) < count:
        value = struct.unpack(float_code, struct.pack(int_code, rng.getrandbits(bits)))[
            0
        ]
        if math.isfinite(value):
            values.append(value)
    return values


def _random_doubles(rng: random.Random, count: int) -> List[float]:
    """Returns finite doubles drawn uniformly over 64-bit patterns."""
    return _random_bit_pattern_floats(rng, count, "<d", "<Q", 64)


def _random_floats(rng: random.Random, count: int) -> List[float]:
    """Returns finite float32 values drawn uniformly over 32-bit patterns.

    The sampled values are Python floats that are exactly representable in
    float32.
    """
    return _random_bit_pattern_floats(rng, count, "<f", "<I", 32)


def _spark_cast_strings(
    spark: SparkSession, values: Sequence[float], spark_type: DataType
) -> List[str]:
    """Returns Spark's string cast of each value, in the order given.

    This is the rendering :func:`~tmlt.core.utils.truncation._hash_column`
    hashes for floating point columns, so it is what the pandas formatters have
    to reproduce.

    Args:
        spark: The Spark session to compute the casts with.
        values: The values to render. They must be exactly representable in
            ``spark_type``.
        spark_type: The Spark type to store the values as.

    Returns:
        One string per value.
    """
    schema = StructType(
        [
            StructField("i", LongType(), False),
            StructField("v", spark_type, True),
        ]
    )
    rows = [(index, float(value)) for index, value in enumerate(values)]
    collected = (
        spark.createDataFrame(rows, schema)
        .select("i", sf.col("v").cast("string").alias("s"))
        .collect()
    )
    strings = [""] * len(values)
    for row in collected:
        strings[row["i"]] = row["s"]
    return strings


def _classify_rendering(
    value: float, spark_string: str, rendered: str, round_trip: float
) -> bool:
    """Returns whether a rendering diverges from Spark's, or raises if it is wrong.

    The two renderings agree on this JVM, or they differ only in the ways Java
    18 and earlier are known to differ from the shortest-round-trip rendering
    that Java 19 specifies (JDK-4511638): usually extra digits, and for the
    smallest subnormals a different choice among equally short candidates
    (``1e-323`` renders as ``1.0E-323`` before Java 19 and as the closer
    ``9.9E-324`` after it). Both denote the same value, which is therefore what
    is checked; a rendering from Spark that is *shorter* would mean the pandas
    side is padding, and fails.

    Args:
        value: The value being rendered.
        spark_string: Spark's rendering of the value.
        rendered: The pandas implementation's rendering of the value.
        round_trip: ``rendered`` parsed back at the precision of the value's
            type.

    Returns:
        True if the renderings differ, False if they are identical.
    """
    if spark_string == rendered:
        return False
    assert round_trip == value, (
        f"{value!r} was rendered as {rendered}, which is a different value "
        f"(Spark rendered it as {spark_string})."
    )
    assert len(spark_string) >= len(rendered), (
        f"{value!r} was rendered as {rendered}, but Spark rendered it as the "
        f"shorter string {spark_string}. Java's pre-19 renderings are never "
        "shorter than the shortest one that round-trips."
    )
    return True


@pytest.mark.slow
def test_double_formatter_matches_spark_cast(utc_spark: SparkSession) -> None:
    """The float64 formatter renders doubles the way Spark casts them.

    Renderings that differ are classified rather than accepted blindly: they
    must round-trip to the same double and be shorter than Spark's, which is
    the only difference a pre-Java-19 ``Double.toString`` can produce.
    """
    values = list(CURATED_DOUBLES) + _random_doubles(
        random.Random(DOUBLE_FORMATTER_SEED), FORMATTER_SAMPLE_SIZE
    )
    spark_strings = _spark_cast_strings(utc_spark, values, DoubleType())
    diverged = 0
    for value, spark_string in zip(values, spark_strings):
        rendered = _java_double_to_string(value)
        diverged += _classify_rendering(value, spark_string, rendered, float(rendered))
    # Sampling bit patterns hits the worst case for the pre-19 renderings, and
    # even there they are a fraction of a percent of all values; a formatter
    # that had the layout rules wrong would diverge on far more than that.
    assert diverged < len(values) // 10, (
        f"{diverged} of {len(values)} doubles rendered differently from Spark, "
        "which is far more than Java's extra-digit renderings can explain."
    )


@pytest.mark.slow
def test_float_formatter_matches_spark_cast(utc_spark: SparkSession) -> None:
    """The float32 formatter renders floats the way Spark casts them.

    As for doubles, differing renderings must round-trip to the same float32
    and be shorter than Spark's. Java's pre-19 ``Float.toString`` emits extra
    digits for a much larger share of floats than its double counterpart does.
    """
    values = list(CURATED_FLOATS) + _random_floats(
        random.Random(FLOAT_FORMATTER_SEED), FORMATTER_SAMPLE_SIZE
    )
    values = [float(np.float32(value)) for value in values]
    spark_strings = _spark_cast_strings(utc_spark, values, FloatType())
    diverged = 0
    for value, spark_string in zip(values, spark_strings):
        rendered = _java_float_to_string(np.float32(value))
        diverged += _classify_rendering(
            value, spark_string, rendered, float(np.float32(rendered))
        )
    assert diverged < len(values) // 2, (
        f"{diverged} of {len(values)} floats rendered differently from Spark, "
        "which is far more than Java's extra-digit renderings can explain."
    )
