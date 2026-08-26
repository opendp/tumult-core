"""Unit tests for :mod:`tmlt.core.utils.pandas_join`.

Nothing here starts a Spark session: these are the assertions that state what
the pandas implementation does, rather than that it agrees with Spark, and they
run in the ``test-nojvm`` lane. The agreement itself is
:mod:`test.unit.utils.test_pandas_join_differential`'s job, and the expected
values written out here were taken from it.

The two halves worth reading as specifications are :class:`TestNullKeys`, which
writes out which rows pair up when a join key is missing, and
:class:`TestDtypes`, which writes out the dtype every output column comes back
with. The second is the reason this module exists at all: a pandas ``merge``
that has to invent a missing value widens ``int64`` to ``float64``, and an
``int64`` above :math:`2^{53}` does not survive that.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import re
from test.unit.backend_testing import assert_frames_equal_as_multisets
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pytest

from tmlt.core.domains.pandas_domains import (
    PandasColumnDescriptor,
    PandasDateColumnDescriptor,
    PandasFloatColumnDescriptor,
    PandasIntegerColumnDescriptor,
    PandasStringColumnDescriptor,
    PandasTableDomain,
    PandasTimestampColumnDescriptor,
)
from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.utils.join import domain_after_join as spark_domain_after_join
from tmlt.core.utils.pandas_grouping import row_keys
from tmlt.core.utils.pandas_join import domain_after_join, join
from tmlt.core.utils.testing import Case, parametrize

#: Every join type :func:`~tmlt.core.utils.pandas_join.join` accepts.
JOIN_TYPES = ("inner", "left", "right", "outer")

#: The join types :func:`~tmlt.core.utils.pandas_join.domain_after_join` accepts.
DOMAIN_JOIN_TYPES = ("inner", "left", "right", "outer")

_NAN = float("nan")

#: An integer that a ``float64`` cannot represent: casting it to a float and
#: back gives ``2**53``, one less. Any join that widens an integer column to
#: hold a missing value loses it.
UNREPRESENTABLE_INTEGER = 2**53 + 1


def _pairs(df: pd.DataFrame) -> Set[Tuple[Optional[str], Optional[str]]]:
    """Returns the set of ``(l, r)`` label pairs of a join result.

    Args:
        df: A join result whose frames carried label columns.
    """
    return {
        (
            None if pd.isna(row.l) else row.l,
            None if pd.isna(row.r) else row.r,
        )
        for row in df.itertuples()
    }


################################################################################
# Null keys
################################################################################


class TestNullKeys:
    """What a missing join key does, written out.

    Every expectation here was read off a live Spark join by
    :mod:`test.unit.utils.test_pandas_join_differential`; this class is what
    keeps it asserted in a lane with no JVM.
    """

    LEFT = pd.DataFrame({"k": ["a", None, "b"], "l": ["l0", "l1", "l2"]})
    RIGHT = pd.DataFrame({"k": ["a", None, "c"], "r": ["r0", "r1", "r2"]})

    @parametrize(
        Case("inner")(how="inner", nulls_are_equal=False, expected={("l0", "r0")}),
        Case("left")(
            how="left",
            nulls_are_equal=False,
            expected={("l0", "r0"), ("l1", None), ("l2", None)},
        ),
        Case("right")(
            how="right",
            nulls_are_equal=False,
            expected={("l0", "r0"), (None, "r1"), (None, "r2")},
        ),
        Case("outer")(
            how="outer",
            nulls_are_equal=False,
            expected={
                ("l0", "r0"),
                ("l1", None),
                ("l2", None),
                (None, "r1"),
                (None, "r2"),
            },
        ),
        Case("inner-eqnull")(
            how="inner",
            nulls_are_equal=True,
            expected={("l0", "r0"), ("l1", "r1")},
        ),
        Case("left-eqnull")(
            how="left",
            nulls_are_equal=True,
            expected={("l0", "r0"), ("l1", "r1"), ("l2", None)},
        ),
        Case("outer-eqnull")(
            how="outer",
            nulls_are_equal=True,
            expected={
                ("l0", "r0"),
                ("l1", "r1"),
                ("l2", None),
                (None, "r2"),
            },
        ),
    )
    def test_null_keys(
        self,
        how: str,
        nulls_are_equal: bool,
        expected: Set[Tuple[Optional[str], Optional[str]]],
    ) -> None:
        """A NULL key matches nothing, unless nulls are declared equal.

        Args:
            how: The join type.
            nulls_are_equal: Whether nulls join to each other.
            expected: The expected ``(l, r)`` label pairs.
        """
        result = join(
            self.LEFT,
            self.RIGHT,
            on=["k"],
            how=how,
            nulls_are_equal=nulls_are_equal,
        )
        assert _pairs(result) == expected

    @parametrize(
        Case("eq")(nulls_are_equal=False), Case("eqnull")(nulls_are_equal=True)
    )
    def test_nan_keys_are_values(self, nulls_are_equal: bool) -> None:
        """A NaN key joins to a NaN key, whatever ``nulls_are_equal`` says.

        Spark's ``NaN = NaN`` is true -- a NaN is a value, not a null -- so
        this pairing does not depend on how nulls are treated.

        Args:
            nulls_are_equal: Whether nulls join to each other.
        """
        left = pd.DataFrame({"k": [1.0, _NAN], "l": ["l0", "l1"]})
        right = pd.DataFrame({"k": [1.0, _NAN], "r": ["r0", "r1"]})
        result = join(
            left, right, on=["k"], how="inner", nulls_are_equal=nulls_are_equal
        )
        assert _pairs(result) == {("l0", "r0"), ("l1", "r1")}

    @parametrize(
        Case("eq")(nulls_are_equal=False, expected={("l0", "r0")}),
        Case("eqnull")(nulls_are_equal=True, expected={("l0", "r0"), ("l1", "r1")}),
    )
    def test_nan_and_null_are_different_keys(
        self,
        nulls_are_equal: bool,
        expected: Set[Tuple[Optional[str], Optional[str]]],
    ) -> None:
        """A NaN key never joins to a NULL key, even under ``<=>``.

        An object column is the only pandas column that can hold both, which is
        what a Spark double column does.

        Args:
            nulls_are_equal: Whether nulls join to each other.
            expected: The expected ``(l, r)`` label pairs.
        """
        left = pd.DataFrame(
            {"k": pd.Series([_NAN, None], dtype=object), "l": ["l0", "l1"]}
        )
        right = pd.DataFrame(
            {"k": pd.Series([_NAN, None], dtype=object), "r": ["r0", "r1"]}
        )
        result = join(
            left, right, on=["k"], how="inner", nulls_are_equal=nulls_are_equal
        )
        assert _pairs(result) == expected

    def test_null_key_value_survives_an_outer_join(self) -> None:
        """An unmatched NULL key comes back as a NULL, not as a NaN.

        A pandas ``merge`` fills the columns of an unmatched row with ``NaN``
        even in an object column, where a NaN is a value rather than a null.
        The join has to tell the two apart.
        """
        left = pd.DataFrame({"k": pd.Series([None], dtype=object), "l": ["l0"]})
        right = pd.DataFrame({"k": pd.Series(["a"], dtype=object), "r": ["r0"]})
        result = join(left, right, on=["k"], how="outer")
        keys = sorted(result["k"], key=lambda value: (value is not None, repr(value)))
        assert keys[0] is None
        assert keys[1] == "a"
        assert [value is None for value in result["r"]].count(True) == 1

    def test_a_null_in_one_column_blocks_the_whole_row(self) -> None:
        """A row with a null in any join column matches nothing under ``=``."""
        left = pd.DataFrame({"j": ["a", "a"], "k": ["x", None], "l": ["l0", "l1"]})
        right = pd.DataFrame({"j": ["a", "a"], "k": ["x", None], "r": ["r0", "r1"]})
        assert _pairs(join(left, right, on=["j", "k"], how="inner")) == {("l0", "r0")}
        assert _pairs(
            join(left, right, on=["j", "k"], how="inner", nulls_are_equal=True)
        ) == {("l0", "r0"), ("l1", "r1")}

    def test_signed_zeros_are_one_key(self) -> None:
        """``-0.0`` and ``0.0`` are the same join key, as they are in Spark."""
        left = pd.DataFrame({"k": [0.0, -0.0], "l": ["l0", "l1"]})
        right = pd.DataFrame({"k": [0.0], "r": ["r0"]})
        assert _pairs(join(left, right, on=["k"], how="inner")) == {
            ("l0", "r0"),
            ("l1", "r0"),
        }

    def test_binary_keys_are_compared_by_content(self) -> None:
        """``bytes`` and a ``bytearray`` of the same bytes are one join key.

        Spark compares binary values by content, and a ``bytearray`` is not
        even hashable, so a merge on the raw values could not do this at all.
        """
        left = pd.DataFrame(
            {"k": pd.Series([b"ab", bytearray(b"cd")], dtype=object), "l": ["l0", "l1"]}
        )
        right = pd.DataFrame(
            {"k": pd.Series([bytearray(b"ab"), b"cd"], dtype=object), "r": ["r0", "r1"]}
        )
        assert _pairs(join(left, right, on=["k"], how="inner")) == {
            ("l0", "r0"),
            ("l1", "r1"),
        }

    def test_timestamp_keys_join_at_microsecond_resolution(self) -> None:
        """Two timestamps a nanosecond apart are one join key.

        Spark's ``TimestampType`` has microsecond resolution, so it cannot tell
        them apart; a pandas ``datetime64[ns]`` column can, and must not.
        """
        timestamp = pd.Timestamp("2020-01-01 00:00:00.000000")
        left = pd.DataFrame({"k": pd.Series([timestamp]), "l": ["l0"]})
        right = pd.DataFrame(
            {"k": pd.Series([timestamp + pd.Timedelta(1, "ns")]), "r": ["r0"]}
        )
        assert _pairs(join(left, right, on=["k"], how="inner")) == {("l0", "r0")}

    def test_matched_rows_keep_the_left_key(self) -> None:
        """A matched row's join column holds the left frame's value.

        Two values can be one join key without being the same value; Spark
        keeps the left one, and so does this.
        """
        left = pd.DataFrame({"k": [-0.0], "l": ["l0"]})
        right = pd.DataFrame({"k": [0.0], "r": ["r0"]})
        result = join(left, right, on=["k"], how="inner")
        assert np.copysign(1.0, result["k"][0]) == -1.0


################################################################################
# Dtypes
################################################################################


class TestDtypes:
    """Which dtype each output column comes back with.

    A column that the join can leave unmatched comes back in the nullable
    extension dtype for its values; every other column keeps the dtype it went
    in with. That rule is what
    :func:`~tmlt.core.utils.pandas_join.domain_after_join` computes on domains,
    so the two agree by construction.
    """

    def test_unrepresentable_integer_survives_a_left_join(self) -> None:
        """An ``int64`` above 2**53 survives a join that leaves it unmatched.

        This is the canonical failure a plain ``merge`` produces: it widens the
        right frame's ``int64`` column to ``float64`` to hold the unmatched
        row's missing value, and ``2**53 + 1`` is not a ``float64``.
        """
        left = pd.DataFrame({"k": ["a", "b"]})
        right = pd.DataFrame({"k": ["a"], "big": [UNREPRESENTABLE_INTEGER]})
        assert right["big"].dtype == np.dtype("int64")

        result = join(left, right, on=["k"], how="left")

        assert result["big"].dtype == pd.Int64Dtype()
        assert result.loc[result["k"] == "a", "big"].iloc[0] == UNREPRESENTABLE_INTEGER
        assert result.loc[result["k"] == "b", "big"].isna().all()

    def test_unrepresentable_integer_survives_an_outer_join(self) -> None:
        """The same, on the left frame's side of an outer join."""
        left = pd.DataFrame({"k": ["a"], "big": [UNREPRESENTABLE_INTEGER]})
        right = pd.DataFrame({"k": ["b"]})
        result = join(left, right, on=["k"], how="outer")
        assert result["big"].dtype == pd.Int64Dtype()
        assert set(result.loc[result["big"].notna(), "big"]) == {
            UNREPRESENTABLE_INTEGER
        }

    @parametrize(
        Case("inner")(how="inner", left_dtype="int64", right_dtype="int64"),
        Case("left")(how="left", left_dtype="int64", right_dtype="Int64"),
        Case("right")(how="right", left_dtype="Int64", right_dtype="int64"),
        Case("outer")(how="outer", left_dtype="Int64", right_dtype="Int64"),
    )
    def test_integer_columns(self, how: str, left_dtype: str, right_dtype: str) -> None:
        """An integer column widens to ``Int64`` exactly when it can be missing.

        Args:
            how: The join type.
            left_dtype: The expected dtype of the left frame's payload column.
            right_dtype: The expected dtype of the right frame's payload column.
        """
        left = pd.DataFrame({"k": ["a", "b"], "x": [1, 2]})
        right = pd.DataFrame({"k": ["a", "c"], "y": [3, 4]})
        result = join(left, right, on=["k"], how=how)
        assert result["x"].dtype == pd.api.types.pandas_dtype(left_dtype)
        assert result["y"].dtype == pd.api.types.pandas_dtype(right_dtype)

    @parametrize(
        Case("float")(column=pd.Series([1.5, 2.5]), inner="float64", left="Float64"),
        Case("bool")(column=pd.Series([True, False]), inner="bool", left="boolean"),
        Case("object")(
            column=pd.Series(["x", "y"], dtype=object), inner="object", left="object"
        ),
        Case("datetime")(
            column=pd.Series(pd.to_datetime(["2020-01-01", "2020-01-02"])),
            inner="datetime64[ns]",
            left="datetime64[ns]",
        ),
        Case("Int64")(
            column=pd.Series(pd.array([1, 2], dtype="Int64")),
            inner="Int64",
            left="Int64",
        ),
    )
    def test_payload_dtypes(self, column: pd.Series, inner: str, left: str) -> None:
        """Each dtype's nullable counterpart, when a join can leave it missing.

        Args:
            column: The right frame's payload column.
            inner: The dtype an inner join gives it back in.
            left: The dtype a left join gives it back in.
        """
        left_frame = pd.DataFrame({"k": ["a", "b"]})
        right_frame = pd.DataFrame({"k": ["a", "c"], "y": column.to_numpy()})
        right_frame["y"] = column.reset_index(drop=True)
        assert join(left_frame, right_frame, on=["k"], how="inner")[
            "y"
        ].dtype == pd.api.types.pandas_dtype(inner)
        assert join(left_frame, right_frame, on=["k"], how="left")[
            "y"
        ].dtype == pd.api.types.pandas_dtype(left)

    def test_a_nan_value_is_not_turned_into_a_null(self) -> None:
        """A NaN in a float column stays a NaN when the column widens.

        Widening ``float64`` to ``Float64`` has to keep the NaNs unmasked, or a
        NaN the caller put there would come back as a NULL.
        """
        left = pd.DataFrame({"k": ["a", "b"]})
        right = pd.DataFrame({"k": ["a"], "y": [_NAN]})
        result = join(left, right, on=["k"], how="left")
        assert result["y"].dtype == pd.Float64Dtype()
        matched = result.loc[result["k"] == "a", "y"]
        # The NaN is a value: it is not masked.
        assert not matched.isna().any()
        assert np.isnan(matched.to_numpy(dtype=np.float64)).all()
        # The unmatched row is a null, and so is masked.
        assert result.loc[result["k"] == "b", "y"].isna().all()

    def test_object_columns_are_filled_with_none(self) -> None:
        """An unmatched object column holds ``None``, not a float ``NaN``.

        In an object column a NaN is a value, so filling with one would invent
        a value the caller never supplied.
        """
        left = pd.DataFrame({"k": ["a", "b"]})
        right = pd.DataFrame({"k": ["a"], "y": pd.Series(["z"], dtype=object)})
        result = join(left, right, on=["k"], how="left")
        unmatched = result.loc[result["k"] == "b", "y"].iloc[0]
        assert unmatched is None

    @parametrize(
        Case("left")(how="left", nulls_are_equal=False, expected="int64"),
        Case("right")(how="right", nulls_are_equal=False, expected="Int64"),
        Case("inner")(how="inner", nulls_are_equal=False, expected="int64"),
        Case("inner-eqnull")(how="inner", nulls_are_equal=True, expected="int64"),
        Case("outer")(how="outer", nulls_are_equal=False, expected="Int64"),
    )
    def test_join_column_dtype(
        self, how: str, nulls_are_equal: bool, expected: str
    ) -> None:
        """The join column's dtype follows the same rule as the payload ones.

        The left frame's key is a non-nullable ``int64`` and the right's a
        nullable ``Int64``, so the two sides disagree and the join type decides.

        Args:
            how: The join type.
            nulls_are_equal: Whether nulls join to each other.
            expected: The expected dtype of the join column.
        """
        left = pd.DataFrame({"k": np.array([1, 2], dtype=np.int64), "l": ["a", "b"]})
        right = pd.DataFrame({"k": pd.array([1, 3], dtype="Int64"), "r": ["c", "d"]})
        result = join(left, right, on=["k"], how=how, nulls_are_equal=nulls_are_equal)
        assert result["k"].dtype == pd.api.types.pandas_dtype(expected)

    def test_join_column_downcasts_when_nulls_cannot_survive(self) -> None:
        """A nullable key comes back non-nullable when the join drops its nulls.

        Both sides are ``Int64``, but an inner join with ``nulls_are_equal``
        off keeps no null key at all, and the output domain says so.
        """
        left = pd.DataFrame({"k": pd.array([1, None], dtype="Int64")})
        right = pd.DataFrame({"k": pd.array([1, None], dtype="Int64")})
        assert join(left, right, on=["k"], how="inner")["k"].dtype == np.dtype("int64")
        assert (
            join(left, right, on=["k"], how="inner", nulls_are_equal=True)["k"].dtype
            == pd.Int64Dtype()
        )

    def test_output_frames_are_in_their_output_domain(self) -> None:
        """The output frame validates against the domain the join computes."""
        left_domain = PandasTableDomain(
            {
                "k": PandasStringColumnDescriptor(),
                "x": PandasIntegerColumnDescriptor(),
            }
        )
        right_domain = PandasTableDomain(
            {
                "k": PandasStringColumnDescriptor(),
                "y": PandasFloatColumnDescriptor(),
            }
        )
        left = pd.DataFrame({"k": ["a", "b"], "x": [1, 2]})
        right = pd.DataFrame({"k": ["a", "c"], "y": [1.5, 2.5]})
        assert left in left_domain
        assert right in right_domain
        for how in JOIN_TYPES:
            output_domain = domain_after_join(
                left_domain, right_domain, on=["k"], how=how
            )
            output_domain.validate(join(left, right, on=["k"], how=how))


################################################################################
# Columns, ordering and validation
################################################################################


class TestColumns:
    """Column naming, ordering and the checks that reject a bad join."""

    def test_overlap_columns_are_suffixed(self) -> None:
        """Overlapping non-join columns get ``_left`` and ``_right``."""
        left = pd.DataFrame({"a": ["a1"], "b": ["b1"], "c": ["c1"]})
        right = pd.DataFrame({"b": ["b1"], "c": ["c9"], "d": ["d1"]})
        result = join(left, right, on=["b"], how="inner")
        assert list(result.columns) == ["b", "a", "c_left", "c_right", "d"]
        assert result["c_left"][0] == "c1"
        assert result["c_right"][0] == "c9"

    def test_natural_join_columns_come_first(self) -> None:
        """With no ``on``, every shared column is joined on, in left order."""
        left = pd.DataFrame({"a": ["a1"], "b": ["b1"], "c": ["c1"]})
        right = pd.DataFrame({"b": ["b1"], "c": ["c1"], "d": ["d1"]})
        result = join(left, right)
        assert list(result.columns) == ["b", "c", "a", "d"]

    def test_left_anti_keeps_only_left_columns(self) -> None:
        """``left_anti`` returns the left frame's columns, join columns first."""
        left = pd.DataFrame({"a": ["a1", "a2"], "b": ["b1", "b2"]})
        right = pd.DataFrame({"b": ["b1"], "d": ["d1"]})
        result = join(left, right, on=["b"], how="left_anti")
        assert list(result.columns) == ["b", "a"]
        assert list(result["a"]) == ["a2"]

    @parametrize(Case(how)(how=how) for how in ("inner", "left", "left_anti"))
    def test_empty_frames(self, how: str) -> None:
        """An empty input gives an empty output with the right columns.

        Args:
            how: The join type.
        """
        left = pd.DataFrame({"k": pd.Series([], dtype=object), "x": []})
        right = pd.DataFrame({"k": ["a"], "y": [1]})
        result = join(left, right, on=["k"], how=how)
        assert len(result) == 0
        if how == "left_anti":
            assert list(result.columns) == ["k", "x"]
        else:
            assert list(result.columns) == ["k", "x", "y"]

    @parametrize(
        Case("no-columns")(
            left_columns=["a"],
            right_columns=["b"],
            on=None,
            how="inner",
            message="Join must involve at least one column.",
        ),
        Case("missing-left")(
            left_columns=["a"],
            right_columns=["a", "b"],
            on=["b"],
            how="inner",
            message="Join column 'b' not in the left table.",
        ),
        Case("missing-right")(
            left_columns=["a", "b"],
            right_columns=["a"],
            on=["b"],
            how="inner",
            message="Join column 'b' not in the right table.",
        ),
        Case("duplicate-on")(
            left_columns=["a"],
            right_columns=["a"],
            on=["a", "a"],
            how="inner",
            message="Join columns (`on`) contain duplicates.",
        ),
        Case("bad-how")(
            left_columns=["a"],
            right_columns=["a"],
            on=["a"],
            how="cross",
            message=(
                "Join type (`how`) must be one of 'left', 'right', 'inner', "
                "'outer', or 'left_anti', not 'cross'."
            ),
        ),
        Case("collision")(
            left_columns=["a", "b", "b_right"],
            right_columns=["a", "b"],
            on=["a"],
            how="inner",
            message="Name collision, ['b_right'] would appear more than once",
        ),
    )
    def test_validation(
        self,
        left_columns: List[str],
        right_columns: List[str],
        on: Optional[List[str]],
        how: str,
        message: str,
    ) -> None:
        """A bad join is rejected with the same message the Spark join gives.

        Args:
            left_columns: The left frame's columns.
            right_columns: The right frame's columns.
            on: The columns to join on.
            how: The join type.
            message: The expected error message.
        """
        left = pd.DataFrame({name: ["x"] for name in left_columns})
        right = pd.DataFrame({name: ["x"] for name in right_columns})
        with pytest.raises(ValueError, match=re.escape(message)):
            join(left, right, on=on, how=how)

    def test_mismatched_dtypes_are_rejected(self) -> None:
        """Join columns holding different kinds of value are rejected."""
        left = pd.DataFrame({"k": [1]})
        right = pd.DataFrame({"k": ["a"]})
        with pytest.raises(ValueError, match="different data types"):
            join(left, right, on=["k"], how="inner")

    def test_compatible_integer_dtypes_are_accepted(self) -> None:
        """``int64`` and ``Int64`` describe the same values, and so may join."""
        left = pd.DataFrame({"k": np.array([1], dtype=np.int64)})
        right = pd.DataFrame({"k": pd.array([1], dtype="Int64")})
        assert len(join(left, right, on=["k"], how="inner")) == 1


################################################################################
# Mixed datetime64 units
################################################################################

_PANDAS_2 = int(pd.__version__.split(".")[0]) >= 2

pandas_2_only = pytest.mark.skipif(
    not _PANDAS_2, reason="pandas 1 has no datetime64 unit other than nanoseconds"
)


def _timestamps(values: List[str], unit: str) -> pd.Series:
    """Returns a ``datetime64`` column in a given unit.

    Args:
        values: The timestamps, as ISO 8601 strings.
        unit: The ``datetime64`` unit the column is in.
    """
    return pd.Series(np.array(values, dtype=f"datetime64[{unit}]"))


@pandas_2_only
@parametrize(Case(how)(how=how) for how in JOIN_TYPES)
def test_mixed_datetime_units_keep_their_values(how: str) -> None:
    """Joining across units neither rounds nor rewrites a value.

    The output join column takes its value from whichever side contributed the
    row, and used to take it in the *left* frame's dtype: a right-only
    ``12:00:00.500`` came back as ``12:00:00`` against a left column of
    seconds. Spark compares two ``TimestampType`` columns whatever the frames
    were built from, so this has to as well.

    Args:
        how: The join type.
    """
    left = pd.DataFrame({"t": _timestamps(["2021-06-01T12:00:00"], "s"), "l": [1]})
    right = pd.DataFrame(
        {"t": _timestamps(["2021-06-01T12:00:00.500"], "ms"), "r": [2]}
    )

    result = join(left, right, on=["t"], how=how)

    # The finer of the two units, so that neither side's values are rounded.
    assert result["t"].dtype == np.dtype("datetime64[ms]")
    expected = {
        "inner": [],
        "left": ["2021-06-01T12:00:00.000"],
        "right": ["2021-06-01T12:00:00.500"],
        "outer": ["2021-06-01T12:00:00.000", "2021-06-01T12:00:00.500"],
    }[how]
    assert list(result["t"]) == [pd.Timestamp(value) for value in expected]


@pandas_2_only
def test_mixed_datetime_units_match_at_the_finer_one() -> None:
    """Two values equal in the finer unit are one key, and unequal ones are not."""
    left = pd.DataFrame(
        {"t": _timestamps(["2021-06-01T12:00:00", "2021-06-02T00:00:00"], "s")}
    )
    right = pd.DataFrame(
        {"t": _timestamps(["2021-06-01T12:00:00.000", "2021-06-02T00:00:00.001"], "ms")}
    )
    assert list(join(left, right, on=["t"], how="inner")["t"]) == [
        pd.Timestamp("2021-06-01T12:00:00")
    ]


@pandas_2_only
@parametrize(Case("coarse-left")(swap=False), Case("coarse-right")(swap=True))
def test_datetime_value_outside_the_finer_unit_is_named(swap: bool) -> None:
    """A value the finer unit cannot hold is reported, not an AssertionError.

    Args:
        swap: Whether the coarse frame is the right one rather than the left.
    """
    coarse = pd.DataFrame({"t": _timestamps(["9999-12-31T00:00:00"], "s")})
    fine = pd.DataFrame({"t": _timestamps(["2021-06-01T12:00:00"], "ns")})
    left, right = (fine, coarse) if swap else (coarse, fine)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "'t' cannot be joined on: it is datetime64[s] in the "
            f"{'right' if swap else 'left'} dataframe"
        ),
    ):
        join(left, right, on=["t"], how="outer")


################################################################################
# Categorical columns
################################################################################


@parametrize(Case(how)(how=how) for how in JOIN_TYPES)
def test_categorical_keys_with_different_categories_are_rejected(how: str) -> None:
    """Two categorical keys must have the same categories, whatever the join type.

    Their kinds are both "category", so this passed validation and then failed
    -- as a bare ``TypeError`` from inside pandas, and only for the join types
    that can take a value from the right frame -- while the output key was
    being built.

    Args:
        how: The join type.
    """
    left = pd.DataFrame(
        {"k": pd.Series(["a", "b"], dtype=pd.CategoricalDtype(["a", "b"]))}
    )
    right = pd.DataFrame(
        {"k": pd.Series(["a"], dtype=pd.CategoricalDtype(["a", "b", "c"]))}
    )
    with pytest.raises(ValueError, match="categorical with different categories"):
        join(left, right, on=["k"], how=how)


def test_categorical_keys_with_reordered_categories_join() -> None:
    """The categories are a set: listing them in another order is one dtype."""
    left = pd.DataFrame(
        {"k": pd.Series(["a", "b"], dtype=pd.CategoricalDtype(["a", "b"])), "l": [1, 2]}
    )
    right = pd.DataFrame(
        {"k": pd.Series(["b", "a"], dtype=pd.CategoricalDtype(["b", "a"])), "r": [3, 4]}
    )
    result = join(left, right, on=["k"], how="outer")
    assert dict(zip(result["k"], result["r"])) == {"a": 4, "b": 3}


def test_a_categorical_join_fill_is_a_null() -> None:
    """An unmatched categorical payload comes back as a null, not as a NaN.

    A categorical has one way to spell a missing entry -- the code ``-1``,
    which reads back as ``np.nan`` -- so that is what a null in one is. The
    grouping this module joins by used to call it a NaN, which here is a
    *value*, and gave it a group of its own rather than the null group.
    """
    left = pd.DataFrame({"k": [1, 2], "p": pd.Series(["a", "b"], dtype="category")})
    right = pd.DataFrame({"k": [1, 3], "r": [9, 9]})

    result = join(left, right, on=["k"], how="outer")

    assert result["p"].isna().tolist() == [False, False, True]
    explicit = pd.DataFrame({"p": pd.Series(["a", "b", None], dtype="category")})
    assert list(row_keys(result, ["p"])) == list(row_keys(explicit, ["p"]))


def test_nulls_are_equal_applies_to_a_categorical_key() -> None:
    """A missing categorical key is a null, so ``nulls_are_equal`` governs it."""
    left = pd.DataFrame({"k": pd.Series(["a", None], dtype="category"), "l": [1, 2]})
    right = pd.DataFrame({"k": pd.Series(["a", None], dtype="category"), "r": [3, 4]})
    assert list(join(left, right, on=["k"], how="inner")["l"]) == [1]
    assert list(
        join(left, right, on=["k"], how="inner", nulls_are_equal=True)["l"]
    ) == [
        1,
        2,
    ]


################################################################################
# Immutability
################################################################################


@parametrize(
    Case(f"{how}-{'eqnull' if nulls_are_equal else 'eq'}")(
        how=how, nulls_are_equal=nulls_are_equal
    )
    for how in (*JOIN_TYPES, "left_anti")
    for nulls_are_equal in (False, True)
)
def test_inputs_are_unchanged(how: str, nulls_are_equal: bool) -> None:
    """A join never writes to either of its inputs.

    Args:
        how: The join type.
        nulls_are_equal: Whether nulls join to each other.
    """
    left = pd.DataFrame({"k": ["a", None, "b"], "x": [1, 2, 3], "s": ["p", "q", "r"]})
    right = pd.DataFrame({"k": ["a", None], "y": [1.5, _NAN]})
    left_before, right_before = left.copy(deep=True), right.copy(deep=True)
    left_columns, right_columns = list(left.columns), list(right.columns)

    join(left, right, on=["k"], how=how, nulls_are_equal=nulls_are_equal)

    assert list(left.columns) == left_columns
    assert list(right.columns) == right_columns
    assert_frames_equal_as_multisets(left, left_before, normalize=False)
    assert_frames_equal_as_multisets(right, right_before, normalize=False)
    assert dict(left.dtypes) == dict(left_before.dtypes)
    assert dict(right.dtypes) == dict(right_before.dtypes)


@parametrize(Case(how)(how=how) for how in (*JOIN_TYPES, "left_anti"))
def test_result_does_not_share_state_with_its_inputs(how: str) -> None:
    """Writing to a join result never reaches back into an input.

    A :class:`~tmlt.core.domains.pandas_domains.PandasTableDomain` carrier is
    immutable by convention, and a result that shared a buffer with an input
    would make honoring that convention the *caller's* problem.

    Args:
        how: The join type.
    """
    left = pd.DataFrame({"k": ["a", "b", "c"], "x": [1, 2, 3]})
    right = pd.DataFrame({"k": ["a", "b"], "y": [4, 5]})
    result = join(left, right, on=["k"], how=how)
    for name in result.columns:
        if name != "k":
            result.iloc[0, result.columns.get_loc(name)] = 999
    assert list(left["x"]) == [1, 2, 3]
    assert list(right["y"]) == [4, 5]


################################################################################
# domain_after_join
################################################################################

#: A schema exercising every descriptor class and every flag combination that
#: the join's output descriptors depend on.
LEFT_SCHEMA: Dict[str, PandasColumnDescriptor] = {
    "int_key": PandasIntegerColumnDescriptor(allow_null=True, size=64),
    "str_key": PandasStringColumnDescriptor(allow_null=False),
    "float_key": PandasFloatColumnDescriptor(
        allow_nan=True, allow_inf=False, allow_null=True, size=64
    ),
    "left_only": PandasDateColumnDescriptor(allow_null=False),
    "shared": PandasTimestampColumnDescriptor(allow_null=True),
}

RIGHT_SCHEMA: Dict[str, PandasColumnDescriptor] = {
    "int_key": PandasIntegerColumnDescriptor(allow_null=False, size=64),
    "str_key": PandasStringColumnDescriptor(allow_null=True),
    "float_key": PandasFloatColumnDescriptor(
        allow_nan=False, allow_inf=True, allow_null=False, size=64
    ),
    "shared": PandasTimestampColumnDescriptor(allow_null=False),
    "right_only": PandasIntegerColumnDescriptor(allow_null=True, size=32),
}

#: The ``on`` arguments swept over the schemas above.
JOIN_COLUMN_SETS: List[Optional[List[str]]] = [
    ["int_key"],
    ["str_key"],
    ["float_key"],
    ["int_key", "str_key", "float_key"],
    None,
]


def _spark_domain(schema: Dict[str, PandasColumnDescriptor]) -> SparkDataFrameDomain:
    """Returns the Spark domain describing the same values as a pandas schema.

    Args:
        schema: The pandas schema to convert.
    """
    return SparkDataFrameDomain(
        {name: descriptor.to_spark_descriptor() for name, descriptor in schema.items()}
    )


@parametrize(
    Case(f"{index}-{how}-{'eqnull' if nulls_are_equal else 'eq'}")(
        on=on, how=how, nulls_are_equal=nulls_are_equal
    )
    for index, on in enumerate(JOIN_COLUMN_SETS)
    for how in DOMAIN_JOIN_TYPES
    for nulls_are_equal in (False, True)
)
def test_domain_after_join_matches_spark(
    on: Optional[List[str]], how: str, nulls_are_equal: bool
) -> None:
    """The pandas output domain describes the Spark output domain's values.

    The two are compared field by field, through
    :meth:`~tmlt.core.domains.pandas_domains.PandasColumnDescriptor.to_spark_descriptor`,
    which is the bridge the descriptor family provides for exactly this.

    Args:
        on: The columns to join on.
        how: The join type.
        nulls_are_equal: Whether nulls join to each other.
    """
    pandas_result = domain_after_join(
        PandasTableDomain(LEFT_SCHEMA),
        PandasTableDomain(RIGHT_SCHEMA),
        on=on,
        how=how,
        nulls_are_equal=nulls_are_equal,
    )
    spark_result = spark_domain_after_join(
        _spark_domain(LEFT_SCHEMA),
        _spark_domain(RIGHT_SCHEMA),
        on=on,
        how=how,
        nulls_are_equal=nulls_are_equal,
    )
    assert list(pandas_result.schema) == list(spark_result.schema)
    for name, descriptor in pandas_result.schema.items():
        assert descriptor.to_spark_descriptor() == spark_result.schema[name], name


@parametrize(
    Case("left-not-a-table-domain")(
        left=PandasTableDomain({}),
        right=PandasTableDomain({}),
        swap=True,
        message="Left join input domain must be a PandasTableDomain.",
    ),
    Case("right-not-a-table-domain")(
        left=PandasTableDomain({}),
        right=PandasTableDomain({}),
        swap=False,
        message="Right join input domain must be a PandasTableDomain.",
    ),
)
def test_domain_after_join_rejects_other_domains(
    left: PandasTableDomain,
    right: PandasTableDomain,
    swap: bool,
    message: str,
) -> None:
    """A domain of some other kind is rejected, naming which side it was.

    Args:
        left: The left domain (replaced by a Spark one when ``swap`` is set).
        right: The right domain (replaced when ``swap`` is not set).
        swap: Whether to replace the left domain rather than the right.
        message: The expected error message.
    """
    other: Any = SparkDataFrameDomain({})
    with pytest.raises(TypeError, match=re.escape(message)):
        domain_after_join(other if swap else left, right if swap else other)


def test_domain_after_join_rejects_left_anti() -> None:
    """``left_anti`` has no output domain, as in the Spark implementation."""
    domain = PandasTableDomain({"k": PandasStringColumnDescriptor()})
    with pytest.raises(ValueError, match="must be one of 'left', 'right', 'inner'"):
        domain_after_join(domain, domain, on=["k"], how="left_anti")


def test_domain_after_join_rejects_mismatched_types() -> None:
    """Join columns describing different kinds of value are rejected."""
    left = PandasTableDomain({"k": PandasStringColumnDescriptor()})
    right = PandasTableDomain({"k": PandasIntegerColumnDescriptor()})
    with pytest.raises(ValueError, match="different data types"):
        domain_after_join(left, right, on=["k"])
