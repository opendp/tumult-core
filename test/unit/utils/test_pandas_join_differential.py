"""Differential tests of :mod:`~tmlt.core.utils.pandas_join` against Spark.

Every test here runs :func:`tmlt.core.utils.join.join` and
:func:`tmlt.core.utils.pandas_join.join` on the same data and asserts that the
two produce the same result. Spark is the specification: where the two backends
could plausibly disagree, whatever Spark does is what the pandas implementation
has to do, and these tests are what pins that down.

Why the assertions compare row *labels*
=======================================

The interesting question about a join is which row of the left frame ended up
beside which row of the right frame, and that is exactly the question a Spark
round trip cannot destroy. Every frame here therefore carries a unique label
column -- ``l`` on the left, ``r`` on the right -- and the assertions compare
the multiset of ``(l, r)`` pairs, with ``None`` standing for "no counterpart".

Comparing the *join column* instead would fail on the round trip rather than on
the join: ``toPandas()`` turns a ``NULL`` in a Spark double column into ``NaN``
and widens a nullable long to ``float64``, which are precisely the distinctions
the null-key cases exist to test. The values the join column takes are asserted
on instead in :mod:`test.unit.utils.test_pandas_join`, which does not go through
Spark at all.

The null-key matrix
===================

:data:`NULL_KEY_CASES` sweeps the pandas dtypes that can carry a missing value
past a join, crossed with every join type and both settings of
``nulls_are_equal``. The two flavors of missing value are deliberately mixed in
one frame wherever a dtype can hold both:

* A ``NULL`` -- ``None``, ``pd.NA`` or ``NaT`` -- never equals another ``NULL``
  under Spark's ``=``, and always does under ``<=>``.
* A float ``NaN`` is a *value*. ``NaN = NaN`` is true in Spark, so a NaN key
  joins to a NaN key under both operators, whatever ``nulls_are_equal`` says.

The second of those is the finding these tests exist to protect: a pandas
``merge`` matches ``NaN`` keys as well, but for the opposite reason (it treats
them as missing values that happen to compare equal), and it matches ``None``
keys too, which Spark does not.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

from contextlib import nullcontext
from dataclasses import dataclass
from test.unit.backend_testing import (
    Backend,
    assert_frames_equal_as_multisets,
    df_for,
    floating_array,
    to_pandas,
    utc_session_timezone,
)
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

from tmlt.core.utils import join as spark_join
from tmlt.core.utils import pandas_join
from tmlt.core.utils.testing import Case, parametrize

#: Every join type :func:`~tmlt.core.utils.pandas_join.join` accepts.
JOIN_TYPES = ("inner", "left", "right", "outer")

#: The label columns the assertions compare.
LABEL_COLUMNS = ["l", "r"]

_NAN = float("nan")


################################################################################
# The null-key matrix
################################################################################


@dataclass(frozen=True)
class NullKeyCase:
    """One column of join keys, on each side, holding missing values.

    Attributes:
        name: The case's name, used as the test id.
        left: The left frame's join column. Its length fixes the left frame's.
        right: The right frame's join column.
        timestamps: Whether the case's frames hold timestamps, and so must be
            built inside :func:`~test.unit.backend_testing.utc_session_timezone`.
    """

    name: str
    left: pd.Series
    right: pd.Series
    timestamps: bool = False

    def frames(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Returns the left and right frames, each with its label column."""
        left = pd.DataFrame(
            {
                "k": self.left.reset_index(drop=True),
                "l": [f"l{index}" for index in range(len(self.left))],
            }
        )
        right = pd.DataFrame(
            {
                "k": self.right.reset_index(drop=True),
                "r": [f"r{index}" for index in range(len(self.right))],
            }
        )
        return left, right


#: The join-key columns swept by the null-key matrix.
#:
#: Every case puts a matching pair of ordinary values on the two sides, a
#: missing value that appears on both sides, and a missing value with no
#: counterpart, so that each join type has something to keep and something to
#: drop.
NULL_KEY_CASES = [
    NullKeyCase(
        name="object-string-none",
        left=pd.Series(["a", None, "b", None], dtype=object),
        right=pd.Series(["a", None, "a", "c"], dtype=object),
    ),
    NullKeyCase(
        # A numpy float column cannot hold a null at all: every missing value
        # in it is a NaN, and so a value that joins to itself.
        name="float64-nan",
        left=pd.Series([1.0, _NAN, 2.0, _NAN], dtype=np.float64),
        right=pd.Series([1.0, _NAN, _NAN, 3.0], dtype=np.float64),
    ),
    NullKeyCase(
        # The one pandas dtype that can hold a NaN and a null side by side.
        name="Float64-nan-and-na",
        left=pd.Series(
            floating_array([1.0, _NAN, 2.0, 0.0], [False, False, False, True])
        ),
        right=pd.Series(
            floating_array([1.0, _NAN, 0.0, 3.0], [False, False, True, False])
        ),
    ),
    NullKeyCase(
        name="Int64-na",
        left=pd.Series(pd.array([1, None, 2, None], dtype="Int64")),
        right=pd.Series(pd.array([1, None, None, 3], dtype="Int64")),
    ),
    NullKeyCase(
        name="datetime64-nat",
        left=pd.Series(pd.to_datetime(["2020-01-01", None, "2020-01-02", None])),
        right=pd.Series(pd.to_datetime(["2020-01-01", None, None, "2020-01-03"])),
        timestamps=True,
    ),
    NullKeyCase(
        # What a Spark double column looks like in pandas: an object column is
        # the only one that can hold a NULL and a NaN at once without a mask.
        name="object-double-nan-and-none",
        left=pd.Series([1.0, _NAN, 2.0, None], dtype=object),
        right=pd.Series([1.0, _NAN, None, 3.0], dtype=object),
    ),
]


def _label_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Returns just the label columns of a join result, as objects.

    Args:
        df: The join result, from either backend.
    """
    return pd.DataFrame(
        {name: df[name].astype(object) for name in LABEL_COLUMNS},
    )


def _joined_labels(
    left: pd.DataFrame,
    right: pd.DataFrame,
    how: str,
    nulls_are_equal: bool,
    spark: Optional[SparkSession],
) -> pd.DataFrame:
    """Returns the ``(l, r)`` label pairs of a join, on one backend.

    Args:
        left: The left frame.
        right: The right frame.
        how: The join type.
        nulls_are_equal: Whether nulls join to each other.
        spark: The session to run the Spark implementation with, or None to run
            the pandas one.
    """
    if spark is None:
        return _label_pairs(
            pandas_join.join(
                left, right, on=["k"], how=how, nulls_are_equal=nulls_are_equal
            )
        )
    backend = Backend(name="spark", spark=spark)
    result = spark_join.join(
        df_for(left, backend),
        df_for(right, backend),
        on=["k"],
        how=how,
        nulls_are_equal=nulls_are_equal,
    )
    return _label_pairs(to_pandas(result, backend))


@parametrize(
    Case(f"{case.name}-{how}-{'eqnull' if nulls_are_equal else 'eq'}")(
        case=case, how=how, nulls_are_equal=nulls_are_equal
    )
    for case in NULL_KEY_CASES
    for how in JOIN_TYPES
    for nulls_are_equal in (False, True)
)
def test_null_keys_match_spark(
    spark: SparkSession, case: NullKeyCase, how: str, nulls_are_equal: bool
) -> None:
    """The two backends pair up the same rows when join keys are missing.

    Args:
        spark: The Spark session.
        case: The join-key column to sweep.
        how: The join type.
        nulls_are_equal: Whether nulls join to each other.
    """
    left, right = case.frames()
    timezone = utc_session_timezone(spark) if case.timestamps else nullcontext()
    with timezone:
        expected = _joined_labels(left, right, how, nulls_are_equal, spark)
    actual = _joined_labels(left, right, how, nulls_are_equal, None)
    assert_frames_equal_as_multisets(actual, expected, normalize=False)


@parametrize(
    Case(case.name)(case=case) for case in NULL_KEY_CASES if case.name == "float64-nan"
)
def test_nan_keys_join_to_each_other(spark: SparkSession, case: NullKeyCase) -> None:
    """NaN is a value, not a null: a NaN key joins to a NaN key in Spark.

    This is the empirical finding the whole null-key design rests on, asserted
    here on its own rather than only as one cell of the matrix. ``NaN = NaN``
    is true in Spark, so the pairing is the same whether or not nulls are
    treated as equal, and the pandas implementation has to reproduce it.

    Args:
        spark: The Spark session.
        case: The float64 case, whose NaNs are values.
    """
    left, right = case.frames()
    spark_pairs = _joined_labels(left, right, "inner", False, spark)
    pandas_pairs = _joined_labels(left, right, "inner", False, None)
    assert_frames_equal_as_multisets(pandas_pairs, spark_pairs, normalize=False)
    # l1 and l3 are NaN keys; r1 and r2 are too. All four pairings appear.
    nan_pairs = {
        (row.l, row.r) for row in spark_pairs.itertuples() if row.l in ("l1", "l3")
    }
    assert nan_pairs == {
        ("l1", "r1"),
        ("l1", "r2"),
        ("l3", "r1"),
        ("l3", "r2"),
    }
    # And the answer does not depend on nulls_are_equal, since there are no
    # nulls in a numpy float column to be equal.
    assert_frames_equal_as_multisets(
        _joined_labels(left, right, "inner", True, None),
        pandas_pairs,
        normalize=False,
    )


def test_null_keys_never_match_under_plain_equality(spark: SparkSession) -> None:
    """A NULL key matches nothing under Spark's ``=``, in either backend.

    The companion of :func:`test_nan_keys_join_to_each_other`: the two flavors
    of missing value behave in opposite ways, and this pins the other one.

    Args:
        spark: The Spark session.
    """
    left = pd.DataFrame({"k": ["a", None], "l": ["l0", "l1"]})
    right = pd.DataFrame({"k": ["a", None], "r": ["r0", "r1"]})
    spark_pairs = _joined_labels(left, right, "inner", False, spark)
    assert_frames_equal_as_multisets(
        _joined_labels(left, right, "inner", False, None),
        spark_pairs,
        normalize=False,
    )
    assert {(row.l, row.r) for row in spark_pairs.itertuples()} == {("l0", "r0")}


################################################################################
# Fanout, overlap and empty frames
################################################################################


@parametrize(Case(how)(how=how) for how in JOIN_TYPES)
def test_duplicate_key_fanout_matches_spark(spark: SparkSession, how: str) -> None:
    """Repeated keys fan out to the same number of rows in both backends.

    Args:
        spark: The Spark session.
        how: The join type.
    """
    left = pd.DataFrame({"k": ["a", "a", "b", "c"], "l": ["l0", "l1", "l2", "l3"]})
    right = pd.DataFrame(
        {"k": ["a", "a", "a", "b", "d"], "r": ["r0", "r1", "r2", "r3", "r4"]}
    )
    expected = _joined_labels(left, right, how, False, spark)
    actual = _joined_labels(left, right, how, False, None)
    assert_frames_equal_as_multisets(actual, expected, normalize=False)
    assert len(actual) == len(expected)


@parametrize(Case(how)(how=how) for how in JOIN_TYPES)
def test_overlapping_columns_match_spark(spark: SparkSession, how: str) -> None:
    """Suffixed overlap columns hold the same values, in the same order.

    Args:
        spark: The Spark session.
        how: The join type.
    """
    left = pd.DataFrame({"a": ["a1", "a2"], "b": ["b1", "b2"], "c": ["c1", "c2"]})
    right = pd.DataFrame({"b": ["b1", "b3"], "c": ["c9", "c8"], "d": ["d1", "d2"]})
    backend = Backend(name="spark", spark=spark)
    expected = to_pandas(
        spark_join.join(
            df_for(left, backend), df_for(right, backend), on=["b"], how=how
        ),
        backend,
    )
    actual = pandas_join.join(left, right, on=["b"], how=how)
    assert list(actual.columns) == list(expected.columns)
    assert_frames_equal_as_multisets(actual, expected, normalize=False)


@parametrize(
    Case(f"{how}-{which}")(how=how, which=which)
    for how in JOIN_TYPES
    for which in ("left", "right", "both")
)
def test_empty_frames_match_spark(spark: SparkSession, how: str, which: str) -> None:
    """An empty input on either side gives the same result in both backends.

    Args:
        spark: The Spark session.
        how: The join type.
        which: Which side (or sides) to empty.
    """
    left = pd.DataFrame({"k": ["a", "b"], "l": ["l0", "l1"]})
    right = pd.DataFrame({"k": ["a", "c"], "r": ["r0", "r1"]})
    if which in ("left", "both"):
        left = left.iloc[:0]
    if which in ("right", "both"):
        right = right.iloc[:0]
    expected = _joined_labels(left, right, how, False, spark)
    actual = _joined_labels(left, right, how, False, None)
    assert_frames_equal_as_multisets(actual, expected, normalize=False)


def test_left_anti_matches_spark(spark: SparkSession) -> None:
    """``left_anti`` keeps the same rows in both backends.

    Args:
        spark: The Spark session.
    """
    left = pd.DataFrame({"k": ["a", "b", None, "a"], "l": ["l0", "l1", "l2", "l3"]})
    right = pd.DataFrame({"k": ["a", None], "r": ["r0", "r1"]})
    backend = Backend(name="spark", spark=spark)
    expected = to_pandas(
        spark_join.join(
            df_for(left, backend),
            df_for(right, backend),
            on=["k"],
            how="left_anti",
        ),
        backend,
    )
    actual = pandas_join.join(left, right, on=["k"], how="left_anti")
    assert list(actual.columns) == list(expected.columns)
    assert_frames_equal_as_multisets(actual, expected, normalize=False)


@parametrize(
    Case(f"{how}-{'eqnull' if nulls_are_equal else 'eq'}")(
        how=how, nulls_are_equal=nulls_are_equal
    )
    for how in JOIN_TYPES
    for nulls_are_equal in (False, True)
)
def test_multiple_join_columns_match_spark(
    spark: SparkSession, how: str, nulls_are_equal: bool
) -> None:
    """A row is matched only when *every* join column agrees.

    A null in one join column is enough to stop a row matching under Spark's
    ``=``, however well the other columns agree, which is the case a per-column
    sentinel has to get right.

    Args:
        spark: The Spark session.
        how: The join type.
        nulls_are_equal: Whether nulls join to each other.
    """
    left = pd.DataFrame(
        {
            "j": ["a", "a", None, None],
            "k": ["x", None, "x", None],
            "l": ["l0", "l1", "l2", "l3"],
        }
    )
    right = pd.DataFrame(
        {
            "j": ["a", "a", None, None],
            "k": ["x", None, "x", None],
            "r": ["r0", "r1", "r2", "r3"],
        }
    )
    expected = _joined_labels(left, right, how, nulls_are_equal, spark)
    actual = _joined_labels(left, right, how, nulls_are_equal, None)
    assert_frames_equal_as_multisets(actual, expected, normalize=False)


def test_signed_zero_keys_match_spark(spark: SparkSession) -> None:
    """``-0.0`` and ``0.0`` are one join key in both backends.

    Args:
        spark: The Spark session.
    """
    left = pd.DataFrame({"k": [0.0, -0.0], "l": ["l0", "l1"]})
    right = pd.DataFrame({"k": [0.0], "r": ["r0"]})
    expected = _joined_labels(left, right, "inner", False, spark)
    actual = _joined_labels(left, right, "inner", False, None)
    assert_frames_equal_as_multisets(actual, expected, normalize=False)
    assert len(actual) == 2
