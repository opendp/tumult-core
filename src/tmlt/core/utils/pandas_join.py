"""Joining pandas dataframes the way Spark joins them.

This is the pandas counterpart of :mod:`tmlt.core.utils.join`, and mirrors it
function for function: :func:`join` over frames and :func:`domain_after_join`
over domains, both taking the same arguments and both promising the same output
columns in the same order. The column algebra itself --
:func:`~tmlt.core.utils.join.natural_join_columns` and
:func:`~tmlt.core.utils.join.columns_after_join`, which decide what the output
columns are called and what order they come in -- is not reimplemented here but
imported from that module, since the two backends must not be able to drift
apart on it.

What a pandas ``merge`` gets wrong
==================================

A join is not a ``merge`` with the column names fixed up. Three things differ,
and each one silently changes a result:

*Null keys.* Spark's ``=`` never matches a ``NULL`` to a ``NULL``; its
``<=>`` -- what ``nulls_are_equal`` selects -- always does. A pandas ``merge``
always matches them, and there is no option that stops it. Worse, it decides
what a missing value *is* from the dtype, so a ``None``, a ``pd.NA`` and a
``NaT`` are one thing to it.

*NaN keys.* ``NaN = NaN`` is true in Spark: a NaN is a value, not a null, and a
NaN key joins to a NaN key under both operators, whether or not nulls are
declared equal. ``NaN <=> NULL``, by contrast, is false -- the two are never one
key. A pandas ``merge`` also matches NaN keys, but for the opposite reason, and
gets the ``NaN``/``NULL`` pairing wrong as a result. The corner is reachable in
any column that can hold both: a ``Float64`` with an unmasked NaN, and an
``object`` column, which is what a Spark double column looks like in pandas.

*Dtypes.* A ``merge`` that has to invent a missing value widens the column to
hold it: ``int64`` becomes ``float64``, ``bool`` becomes ``object``, and a null
in an object column becomes a float ``NaN`` -- which, per the paragraph above,
is a *value*. The widening is not merely untidy: ``2**53 + 1`` is an ``int64``
and is not a ``float64``, so a left join over a table of identifiers can quietly
change them.

How this module resolves them
=============================

Equality is delegated whole to :mod:`tmlt.core.utils.pandas_grouping`, whose
notion of when two values are one group is Spark's: nulls one group and NaNs
another, ``-0.0`` and ``0.0`` together, binary by content, timestamps at
microsecond resolution. Each join column's values are numbered so that the two
frames' numbers agree, and the merge runs on those numbers rather than on the
values. Under ``=``, a row whose key is ``NULL`` is then given a number of its
own that nothing else can carry, which is exactly what makes it match nothing
while leaving it available to an outer or left join as an unmatched row.

Dtypes are fixed *before* the merge rather than after: a column the join can
leave unmatched is converted to its nullable extension dtype first, so the merge
has a missing value to write and never has to widen anything. Which columns
those are depends only on the join type, so the output dtypes are a property of
the join and not of the data -- and they are the canonical dtypes of the domain
:func:`domain_after_join` computes.

Supported dtypes:
    Whatever :func:`tmlt.core.utils.pandas_grouping.group_codes` groups may be
    joined on, which is every dtype except one holding a value with no Python
    hash. Join columns on the two sides must describe the same kind of value,
    but need not have the same dtype: an ``int64`` column and an ``Int64`` one
    hold the same integers, and a left join produces the second from the first.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from typeguard import typechecked

from tmlt.core.domains.base import Domain
from tmlt.core.domains.pandas_domains import (
    PandasColumnDescriptor,
    PandasDateColumnDescriptor,
    PandasDtype,
    PandasFloatColumnDescriptor,
    PandasIntegerColumnDescriptor,
    PandasStringColumnDescriptor,
    PandasTableColumnsDescriptor,
    PandasTableDomain,
    PandasTimestampColumnDescriptor,
)

# The column algebra is backend-neutral -- these functions manipulate lists of
# names, join types and flags, and nothing else -- so it is imported rather than
# mirrored: the two backends' output columns, their order and their nullability
# have to be identical, and sharing the code is the only way to guarantee that
# they stay so.
from tmlt.core.utils.join import (
    _join_allows_null,
    _join_flag,
    _side_unmatchable,
    _validate_join_columns,
    columns_after_join,
    natural_join_columns,
)
from tmlt.core.utils.misc import get_nonconflicting_string
from tmlt.core.utils.pandas_grouping import (
    _missing_is_null,
    _null_and_nan_masks,
    group_codes,
    row_keys,
)

#: The join types :func:`domain_after_join` accepts.
DOMAIN_JOIN_TYPES = ("left", "right", "inner", "outer")

#: The nullable extension dtype holding the same values as each numpy dtype
#: that cannot hold a null. Every other dtype -- ``object``, ``datetime64``, and
#: the extension dtypes themselves -- already has somewhere to put one.
_NUMPY_TO_NULLABLE: Dict[np.dtype, PandasDtype] = {
    np.dtype("bool"): pd.BooleanDtype(),
    np.dtype("int8"): pd.Int8Dtype(),
    np.dtype("int16"): pd.Int16Dtype(),
    np.dtype("int32"): pd.Int32Dtype(),
    np.dtype("int64"): pd.Int64Dtype(),
    np.dtype("uint8"): pd.UInt8Dtype(),
    np.dtype("uint16"): pd.UInt16Dtype(),
    np.dtype("uint32"): pd.UInt32Dtype(),
    np.dtype("uint64"): pd.UInt64Dtype(),
    np.dtype("float32"): pd.Float32Dtype(),
    np.dtype("float64"): pd.Float64Dtype(),
}

#: The inverse of :data:`_NUMPY_TO_NULLABLE`.
_NULLABLE_TO_NUMPY: Dict[PandasDtype, np.dtype] = {
    nullable: numpy_dtype for numpy_dtype, nullable in _NUMPY_TO_NULLABLE.items()
}

#: The name :func:`_shared_ids` gives the column it keys.
_KEY_COLUMN = "value"


################################################################################
# Validation
################################################################################


@typechecked
def _validate_join(
    left_schema: PandasTableColumnsDescriptor,
    right_schema: PandasTableColumnsDescriptor,
    on: Optional[List[str]],
    how: str,
) -> None:
    """Check for any problems in the join of two described tables.

    This is :func:`tmlt.core.utils.join._validate_join` over pandas
    descriptors: the shared column-name checks of
    :func:`tmlt.core.utils.join._validate_join_columns` plus the requirement
    that the join columns describe the same kind of value. The kinds are named
    by
    :meth:`~tmlt.core.domains.pandas_domains.PandasColumnDescriptor.to_spark_descriptor`,
    so that a mismatch is reported in the same words on both backends.

    Args:
        left_schema: Descriptors of the left table's columns.
        right_schema: Descriptors of the right table's columns.
        on: Columns to join on. If None, join on all columns with the same name.
        how: Join type.
    """
    _validate_join_columns(list(left_schema), list(right_schema), on=on, how=how)
    if on is None:
        on = natural_join_columns(list(left_schema), list(right_schema))
    for column in on:
        left_dtype = left_schema[column].to_spark_descriptor().data_type
        right_dtype = right_schema[column].to_spark_descriptor().data_type
        if left_dtype != right_dtype:
            raise ValueError(
                f"'{column}' has different data types in left "
                f"({str(left_dtype).replace('()', '')}) and right "
                f"({str(right_dtype).replace('()', '')}) domains."
            )


def _dtype_kind(dtype: PandasDtype) -> str:
    """Returns a name for the kind of value a dtype holds.

    Two columns may be joined on when their kinds are equal. The kind
    deliberately forgets whether a dtype is nullable, since ``int64`` and
    ``Int64`` hold the same integers and a join produces the second from the
    first, and it forgets a ``datetime64``'s unit, since those denote the same
    instants at different resolutions. Two join columns whose units differ are
    brought to one unit before anything is compared; see
    :func:`_reconciled_units`.

    Note:
        An ``object`` column is one kind, whatever it holds: pandas keeps
        strings, dates and Python floats in object columns alike, and telling
        them apart would mean walking the values. Joining a column of strings
        to a column of dates is therefore accepted here and simply matches
        nothing, where the Spark implementation -- which has a schema to read --
        rejects it. :func:`domain_after_join`, which does have descriptors,
        rejects it too.

    Args:
        dtype: The dtype to name.
    """
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return (
            "timezone-aware datetime64"
            if isinstance(dtype, pd.DatetimeTZDtype)
            else "datetime64"
        )
    base = _base_dtype(dtype)
    if base == np.dtype(object):
        return "object"
    return str(base)


def _validate_join_dtypes(
    left: pd.DataFrame, right: pd.DataFrame, on: List[str]
) -> None:
    """Raises if a join column holds different kinds of value on the two sides.

    This is what :func:`join` does in place of the descriptor comparison in
    :func:`_validate_join`, which it has no descriptors to make. See
    :func:`_dtype_kind` for what counts as the same kind.

    Two categorical join columns are held to more than their kind: their
    categories have to be the same. An output join column takes its values from
    whichever side contributed the row, which pandas will not do between two
    categoricals with different categories -- and only for the join types that
    can take a value from the right frame, so without this a join validated,
    ran as an inner or left join, and raised a bare ``TypeError`` from inside
    pandas as an outer or right one.

    Args:
        left: The left dataframe.
        right: The right dataframe.
        on: The columns to join on, already known to be in both frames.
    """
    for column in on:
        left_dtype, right_dtype = left[column].dtype, right[column].dtype
        left_kind, right_kind = _dtype_kind(left_dtype), _dtype_kind(right_dtype)
        if left_kind != right_kind:
            raise ValueError(
                f"'{column}' has different data types in left ({left_kind}) and "
                f"right ({right_kind}) dataframes."
            )
        if isinstance(left_dtype, pd.CategoricalDtype) and left_dtype != right_dtype:
            raise ValueError(
                f"'{column}' is categorical with different categories in left "
                f"({list(left_dtype.categories)}) and right "
                f"({list(right_dtype.categories)}) dataframes. Give both columns "
                "the same categories, or convert them with .astype(object)."
            )


#: The naive ``datetime64`` units pandas supports, coarsest first. Only ``ns``
#: exists on pandas 1; pandas 2 added the other three.
_DATETIME64_UNITS = ("s", "ms", "us", "ns")


def _datetime64_unit(dtype: PandasDtype) -> Optional[str]:
    """Returns a naive ``datetime64`` dtype's unit, or None for anything else.

    A unit pandas cannot produce -- ``D``, or a sub-nanosecond one -- is
    reported as None, so that a column carrying it is left exactly as it is
    rather than converted on a guess.

    Args:
        dtype: The dtype to read.
    """
    if not isinstance(dtype, np.dtype) or dtype.kind != "M":
        return None
    unit = str(np.datetime_data(dtype)[0])
    return unit if unit in _DATETIME64_UNITS else None


def _in_unit(column: pd.Series, unit: str, side: str, name: str) -> pd.Series:
    """Returns a ``datetime64`` column in ``unit``, or raises if it does not fit.

    Args:
        column: The column to convert.
        unit: The ``datetime64`` unit to convert it to, finer than its own.
        side: Which frame the column came from, for the error message.
        name: The column's name, for the error message.

    Raises:
        ValueError: If a value of ``column`` is outside the range ``unit`` can
            represent.
    """
    target = np.dtype(f"datetime64[{unit}]")
    own_unit = _datetime64_unit(column.dtype)
    message = (
        f"'{name}' cannot be joined on: it is datetime64[{own_unit}] in the"
        f" {side} dataframe, which has to be compared in the finer unit"
        f" datetime64[{unit}] of the other one, and it holds a value outside the"
        f" range datetime64[{unit}] can represent."
    )
    try:
        converted = column.astype(target)
    except (ValueError, OverflowError) as error:
        # pandas raises OutOfBoundsDatetime, a ValueError, for a value it
        # cannot widen; numpy would wrap one silently, which the round trip
        # below is what catches.
        raise ValueError(message) from error
    if not converted.astype(column.dtype).equals(column):
        raise ValueError(message)
    return converted


def _reconciled_units(
    left: pd.DataFrame, right: pd.DataFrame, on: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns the two frames with each join column in one ``datetime64`` unit.

    On pandas 2 a ``datetime64`` column may be in seconds, milliseconds,
    microseconds or nanoseconds, and the two sides of a join need not agree.
    They denote the same instants, and Spark -- where both sides are a
    ``TimestampType`` of microseconds -- simply compares them, so the join must
    too. Everything downstream keys, merges and rebuilds a join column in *one*
    dtype, and taking the left frame's would silently rewrite the values the
    right frame alone contributed: a right-only ``12:00:00.500`` came back as
    ``12:00:00`` against a left column of seconds.

    Both sides are therefore brought to the finer of the two units before
    anything is compared, which is also the unit the output join column comes
    back in. Only the join columns are touched, and neither input frame is
    modified.

    Args:
        left: The left dataframe.
        right: The right dataframe.
        on: The columns to join on, already known to hold the same kind of
            value on both sides.

    Raises:
        ValueError: If a value cannot be represented in the finer unit, naming
            the column and the two units.
    """
    replacements: Tuple[Dict[str, pd.Series], Dict[str, pd.Series]] = ({}, {})
    for column in on:
        left_unit = _datetime64_unit(left[column].dtype)
        right_unit = _datetime64_unit(right[column].dtype)
        if left_unit is None or right_unit is None or left_unit == right_unit:
            continue
        units = (left_unit, right_unit)
        ranks = tuple(_DATETIME64_UNITS.index(unit) for unit in units)
        coarser = 0 if ranks[0] < ranks[1] else 1
        replacements[coarser][column] = _in_unit(
            (left, right)[coarser][column],
            units[1 - coarser],
            ("left", "right")[coarser],
            column,
        )
    return (
        _with_columns(left, replacements[0]),
        _with_columns(right, replacements[1]),
    )


def _with_columns(
    frame: pd.DataFrame, replacements: Dict[str, pd.Series]
) -> pd.DataFrame:
    """Returns a frame with some of its columns replaced, leaving it unmodified.

    Args:
        frame: The frame to rebuild.
        replacements: The columns to put in place of the frame's own.
    """
    if not replacements:
        return frame
    return pd.DataFrame(
        {name: replacements.get(name, frame[name]) for name in frame.columns},
        index=frame.index,
    )


################################################################################
# Domains
################################################################################


@typechecked
def domain_after_join(
    left_domain: Domain,
    right_domain: Domain,
    on: Optional[List[str]] = None,
    how: str = "inner",
    nulls_are_equal: bool = False,
) -> PandasTableDomain:
    r"""Returns the domain of the join of two pandas dataframes.

    This is :func:`tmlt.core.utils.join.domain_after_join` over
    :class:`~tmlt.core.domains.pandas_domains.PandasTableDomain`\ s, and
    computes the same descriptors: each output column's descriptor, read through
    :meth:`~tmlt.core.domains.pandas_domains.PandasColumnDescriptor.to_spark_descriptor`,
    equals the one the Spark implementation computes for the same join.

    Also does input validation. Checks:

        - All checks from :func:`~tmlt.core.utils.join.columns_after_join`.
        - ``how`` is one of "left", "right", "inner", or "outer".
        - Join columns have the same data type.
        - Left and right domains are PandasTableDomains.

    .. note::

        This takes into account extra metadata about the columns, such as
        whether nulls/infs are allowed, and what kind of join is performed.

        See :ref:`NaNs, nulls, and infs <special-values>` for more information
        about comparisons involving special values.

    Args:
        left_domain: Domain of the left dataframe.
        right_domain: Domain of the right dataframe.
        on: Columns to join on. If None, join on all columns with the same
            name.
        how: Join type. Must be one of "left", "right", "inner", "outer". This
            defaults to "inner".
        nulls_are_equal: If True, treats null values as equal. Defaults to False.
    """
    if not isinstance(left_domain, PandasTableDomain):
        raise TypeError("Left join input domain must be a PandasTableDomain.")
    if not isinstance(right_domain, PandasTableDomain):
        raise TypeError("Right join input domain must be a PandasTableDomain.")
    if on is None:
        on = natural_join_columns(
            left_columns=list(left_domain.schema),
            right_columns=list(right_domain.schema),
        )
    if how not in DOMAIN_JOIN_TYPES:
        raise ValueError(
            "Join type (`how`) must be one of 'left', 'right', 'inner', or 'outer', not"
            f" '{how}'."
        )
    _validate_join(left_domain.schema, right_domain.schema, on=on, how=how)
    output_columns = columns_after_join(
        left_columns=list(left_domain.schema),
        right_columns=list(right_domain.schema),
        on=on,
    )
    output_descriptors: Dict[str, PandasColumnDescriptor] = {}
    for output_column, (left_column, right_column) in output_columns.items():
        left_descriptor = left_domain.schema.get(left_column, None)  # type: ignore
        right_descriptor = right_domain.schema.get(right_column, None)  # type: ignore
        if left_descriptor is None:
            assert right_descriptor is not None
            output_descriptors[output_column] = dataclasses.replace(  # type: ignore
                right_descriptor,
                allow_null=right_descriptor.allow_null
                or _side_unmatchable("right", how),
            )
            continue
        if right_descriptor is None:
            assert left_descriptor is not None
            output_descriptors[output_column] = dataclasses.replace(  # type: ignore
                left_descriptor,
                allow_null=left_descriptor.allow_null or _side_unmatchable("left", how),
            )
            continue
        assert left_descriptor is not None
        assert right_descriptor is not None
        # The only remaining case is when the output column is a join column.
        assert output_column in on

        # All column types are nullable
        allow_null = _join_allows_null(
            left_descriptor.allow_null,
            right_descriptor.allow_null,
            how,
            nulls_are_equal,
        )
        new_descriptor: PandasColumnDescriptor
        if isinstance(left_descriptor, PandasIntegerColumnDescriptor):
            assert isinstance(right_descriptor, PandasIntegerColumnDescriptor)
            assert left_descriptor.size == right_descriptor.size
            new_descriptor = PandasIntegerColumnDescriptor(
                allow_null=allow_null, size=left_descriptor.size
            )
        elif isinstance(left_descriptor, PandasFloatColumnDescriptor):
            assert isinstance(right_descriptor, PandasFloatColumnDescriptor)
            allow_nan = _join_flag(
                how, left_descriptor.allow_nan, right_descriptor.allow_nan
            )
            allow_inf = _join_flag(
                how, left_descriptor.allow_inf, right_descriptor.allow_inf
            )
            assert left_descriptor.size == right_descriptor.size
            new_descriptor = PandasFloatColumnDescriptor(
                allow_nan=allow_nan,
                allow_inf=allow_inf,
                allow_null=allow_null,
                size=left_descriptor.size,
            )
        elif isinstance(
            left_descriptor,
            (
                PandasStringColumnDescriptor,
                PandasDateColumnDescriptor,
                PandasTimestampColumnDescriptor,
            ),
        ):
            descriptor_class = left_descriptor.__class__
            assert isinstance(right_descriptor, descriptor_class)
            new_descriptor = descriptor_class(allow_null=allow_null)
        else:
            raise NotImplementedError(
                f"Unsupported column descriptor {left_descriptor}."
            )
        output_descriptors[output_column] = new_descriptor
    return PandasTableDomain(output_descriptors)


################################################################################
# Dtypes
################################################################################


def _can_hold_null(dtype: PandasDtype) -> bool:
    """Returns whether a column of this dtype has anywhere to put a null.

    A numpy integer, float or boolean array has no mask, so it cannot hold one:
    in particular a NaN in a ``float64`` column is a NaN and not a null, which
    is the taxonomy :mod:`tmlt.core.utils.pandas_grouping` groups by and
    :class:`~tmlt.core.domains.pandas_domains.PandasFloatColumnDescriptor`
    validates against.

    Args:
        dtype: The dtype to classify.
    """
    return not (isinstance(dtype, np.dtype) and dtype in _NUMPY_TO_NULLABLE)


def _nullable_dtype(dtype: PandasDtype) -> PandasDtype:
    """Returns the dtype holding this dtype's values, plus nulls.

    Args:
        dtype: The dtype to widen.
    """
    if isinstance(dtype, np.dtype):
        return _NUMPY_TO_NULLABLE.get(dtype, dtype)
    return dtype


def _base_dtype(dtype: PandasDtype) -> PandasDtype:
    """Returns the dtype holding this dtype's values, without nulls.

    This is the inverse of :func:`_nullable_dtype` where one exists, and the
    identity everywhere else: an ``object`` or ``datetime64`` column can hold a
    null whatever the domain says, and there is no narrower dtype to move it to.

    Args:
        dtype: The dtype to narrow.
    """
    return _NULLABLE_TO_NUMPY.get(dtype, dtype)


def _target_dtype(dtype: PandasDtype, allow_null: bool) -> PandasDtype:
    """Returns the dtype of a column described as holding nulls, or not.

    Args:
        dtype: The dtype whose values the column holds.
        allow_null: Whether the column may hold a null.
    """
    return _nullable_dtype(dtype) if allow_null else _base_dtype(dtype)


def _payload_dtype(dtype: PandasDtype, unmatchable: bool) -> PandasDtype:
    """Returns the dtype a non-join output column comes back in.

    A column that came in nullable stays nullable, whether or not this join
    puts a null in it, and a column the join can leave unmatched is widened
    until it is. That is exactly what :func:`domain_after_join` does to a
    non-join column's descriptor -- ``allow_null`` is ORed with whether the
    join type can leave the column's side unmatched, and never cleared -- so
    the dtype and the descriptor's canonical dtype agree.

    Args:
        dtype: The input column's dtype.
        unmatchable: Whether the join can leave this column's side without a
            matching row.
    """
    return _nullable_dtype(dtype) if unmatchable else dtype


def _as_dtype(column: pd.Series, dtype: PandasDtype) -> pd.Series:
    """Returns a column converted to a dtype, without inventing missing values.

    ``astype`` is not usable for widening a numpy float column: it reads every
    NaN as a missing value and masks it, turning a NaN the caller put there into
    a null. The mask is built explicitly instead, with nothing in it.

    Args:
        column: The column to convert.
        dtype: The dtype to convert it to. It must hold the same values as the
            column's own, or the conversion raises.
    """
    if column.dtype == dtype:
        return column
    if (
        isinstance(dtype, (pd.Float32Dtype, pd.Float64Dtype))
        and isinstance(column.dtype, np.dtype)
        and column.dtype.kind == "f"
    ):
        values = column.to_numpy(dtype=_NULLABLE_TO_NUMPY[dtype], copy=True)
        return pd.Series(
            pd.arrays.FloatingArray(values, np.zeros(len(values), dtype=bool)),
            index=column.index,
            name=column.name,
        )
    return column.astype(dtype)


def _null_mask(column: pd.Series) -> np.ndarray:
    """Returns the positions at which a column holds a null.

    A null is a ``None``, a ``pd.NA`` or a ``NaT``; a float NaN is a value, and
    is not marked. The two are told apart with
    :func:`tmlt.core.utils.pandas_grouping._is_null`, the same predicate the
    grouping this module joins by uses, so that the rows this marks are exactly
    the rows whose group is the null group.

    Only the positions ``pandas.Series.isna`` reports -- which over-approximates
    the nulls, since it also reports NaNs -- are examined one at a time, which
    is what :func:`tmlt.core.utils.pandas_grouping._null_and_nan_masks` does;
    its null mask is this one. A column whose every missing entry is a null,
    such as a categorical one, is not examined at all; see
    :func:`tmlt.core.utils.pandas_grouping._missing_is_null`.

    Args:
        column: The column to inspect.
    """
    missing = column.isna().to_numpy()
    if not missing.any() or _missing_is_null(column.dtype):
        return missing
    return _null_and_nan_masks(column.to_numpy(dtype=object))[0]


################################################################################
# Join keys
################################################################################


def _shared_ids(
    left_column: pd.Series, right_column: pd.Series
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Numbers two columns' values so that the two frames' numbers agree.

    Two rows -- in the same frame or in different ones -- get the same number
    exactly when Spark's ``<=>`` holds between their values, which is what
    :func:`tmlt.core.utils.pandas_grouping.group_codes` decides.

    Two columns of one dtype are numbered *together*, by grouping the
    concatenation of the two: there is then only one numbering, so there is
    nothing to reconcile, and it is one vectorized pass over both frames' rows.
    Columns of different dtypes -- an ``int64`` joined to an ``Int64``, which
    hold the same integers -- cannot be concatenated without pandas choosing a
    common dtype for them, so each is numbered on its own and the two numberings
    are reconciled through the *distinct* values'
    :func:`~tmlt.core.utils.pandas_grouping.row_keys`, which are comparable
    across frames and across dtypes. That costs a Python-level step per distinct
    value, where the concatenation costs none.

    Args:
        left_column: The left frame's join column.
        right_column: The right frame's join column.

    Returns:
        The left column's numbers, the right column's numbers, and how many
        distinct numbers were handed out.
    """
    if left_column.dtype == right_column.dtype:
        codes = group_codes(pd.concat([left_column, right_column], ignore_index=True))
        left_count = len(left_column)
        return (
            codes[:left_count],
            codes[left_count:],
            (int(codes.max()) + 1) if len(codes) else 0,
        )
    shared: Dict[Any, int] = {}
    left_codes = group_codes(left_column)
    right_codes = group_codes(right_column)
    left_lookup = _shared_lookup(left_column, left_codes, shared)
    right_lookup = _shared_lookup(right_column, right_codes, shared)
    return left_lookup[left_codes], right_lookup[right_codes], len(shared)


def _shared_lookup(
    column: pd.Series, codes: np.ndarray, shared: Dict[Any, int]
) -> np.ndarray:
    """Returns the shared number of each of a column's group codes.

    Args:
        column: The column the codes were computed from.
        codes: The column's codes, dense and numbered in order of first
            appearance, as
            :func:`~tmlt.core.utils.pandas_grouping.group_codes` returns them.
        shared: The numbering built so far, extended in place.

    Returns:
        An int64 array with one entry per code, indexed by the code.
    """
    if not len(codes):
        return np.zeros(0, dtype=np.int64)
    # The codes are dense, so the uniques are 0, 1, ... and this is the position
    # of the first row carrying each of them.
    first_occurrences = np.unique(codes, return_index=True)[1]
    keys = row_keys(
        column.to_frame(name=_KEY_COLUMN).iloc[first_occurrences], [_KEY_COLUMN]
    )
    return np.array(
        [shared.setdefault(key, len(shared)) for key in keys], dtype=np.int64
    )


def _join_ids(
    left_column: pd.Series, right_column: pd.Series, nulls_are_equal: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """Numbers two join columns so that equal numbers mean a matching key.

    Under ``<=>`` this is exactly :func:`_shared_ids`. Under ``=`` it is that
    numbering with every null-keyed row given a number of its own, which no
    other row anywhere carries -- so the row matches nothing, while remaining an
    ordinary unmatched row that a left, right or outer join keeps.

    Args:
        left_column: The left frame's join column.
        right_column: The right frame's join column.
        nulls_are_equal: If True, a null key matches another null key.

    Returns:
        The left column's numbers and the right column's numbers.
    """
    left_ids, right_ids, group_count = _shared_ids(left_column, right_column)
    if nulls_are_equal:
        return left_ids, right_ids
    left_nulls = _null_mask(left_column)
    right_nulls = _null_mask(right_column)
    left_null_count = int(left_nulls.sum())
    left_ids[left_nulls] = group_count + np.arange(left_null_count)
    right_ids[right_nulls] = (
        group_count + left_null_count + np.arange(int(right_nulls.sum()))
    )
    return left_ids, right_ids


class _NameSource:
    """Hands out column names that collide with nothing already in use."""

    def __init__(self, used: List[str]):
        """Constructor.

        Args:
            used: The names already in use.
        """
        self._used = list(used)

    def take(self) -> str:
        """Returns a fresh name, and marks it as used."""
        name = get_nonconflicting_string(self._used)
        self._used.append(name)
        return name


################################################################################
# Joining
################################################################################


def join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: Optional[List[str]] = None,
    how: str = "inner",
    nulls_are_equal: bool = False,
) -> pd.DataFrame:
    """Returns the join of two pandas dataframes.

    This is :func:`tmlt.core.utils.join.join` over pandas frames, and answers
    the same question the same way: which rows pair up, what the output columns
    are called, and what order they come in are all Spark's, not pandas'. See
    the module docstring for where the two would otherwise differ.

    The output columns' dtypes are a property of the join rather than of the
    data: a column the join type can leave unmatched comes back in the nullable
    extension dtype for its values (``Int64``, ``Float64``, ``boolean``), and
    every other column comes back with the dtype it went in with. These are the
    canonical dtypes of the domain :func:`domain_after_join` computes for the
    same join, so an output frame is always in its output domain.

    A join column that is a ``datetime64`` in different units on the two sides
    -- which pandas 2 allows -- is the one exception to "the dtype it went in
    with": both sides are compared, and the output column comes back, in the
    *finer* of the two units, so that no value the join keeps is rounded. See
    :func:`_reconciled_units`.

    Neither input is modified, and neither shares mutable state with the result.

    Raises:
        ValueError: If a join column is missing, holds different kinds of value
            on the two sides, or holds a value that the finer of its two
            ``datetime64`` units cannot represent.

    Example:
        ..
            >>> import pandas as pd
            >>> from tmlt.core.utils.misc import print_pandas
            >>> left = pd.DataFrame({"a": [1, 4], "b": [2, 5], "c": [3, 6]})
            >>> right = pd.DataFrame({"b": [2, 5], "c": [7, 9], "d": [8, 10]})

        >>> print_pandas(left)
           a  b  c
        0  1  2  3
        1  4  5  6
        >>> print_pandas(right)
           b  c   d
        0  2  7   8
        1  5  9  10
        >>> print_pandas(join(left, right, on=["b"]))
           b  a  c_left  c_right   d
        0  2  1       3        7   8
        1  5  4       6        9  10

    Args:
        left: Left dataframe.
        right: Right dataframe.
        on: Columns to join on. If None, join on all columns with the same
            name.
        how: Join type. Must be one of "left", "right", "inner", "outer", or
            "left_anti". This defaults to "inner".
        nulls_are_equal: If True, treats null values as equal. Defaults to False.
    """
    left_columns, right_columns = list(left.columns), list(right.columns)
    if on is None:
        on = natural_join_columns(left_columns, right_columns)
    _validate_join_columns(left_columns, right_columns, on=on, how=how)
    _validate_join_dtypes(left, right, on)
    left, right = _reconciled_units(left, right, on)
    ids = {
        column: _join_ids(left[column], right[column], nulls_are_equal) for column in on
    }
    if how == "left_anti":
        return _left_anti_join(left, right, on, ids)
    return _join_on_ids(left, right, on, how, nulls_are_equal, ids)


def _left_anti_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: List[str],
    ids: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Returns the rows of ``left`` whose join key is in no row of ``right``.

    The right frame's keys are deduplicated before the merge, so a left row is
    tested once rather than once per matching right row, and only the ids and
    the row positions go through the merge, so no column of ``left`` is touched
    and no dtype can change.

    Args:
        left: Left dataframe.
        right: Right dataframe.
        on: Columns to join on.
        ids: Each join column's numbering, as :func:`_join_ids` returns it.
    """
    output_columns = columns_after_join(
        left_columns=list(left.columns),
        right_columns=list(right.columns),
        on=on,
        how="left_anti",
    )
    names = _NameSource(list(left.columns) + list(right.columns))
    id_names = {column: names.take() for column in on}
    position_name, indicator_name = names.take(), names.take()

    probe = pd.DataFrame({id_names[column]: ids[column][0] for column in on})
    probe[position_name] = np.arange(len(left))
    right_ids = pd.DataFrame(
        {id_names[column]: ids[column][1] for column in on}
    ).drop_duplicates()
    matched = probe.merge(
        right_ids,
        on=[id_names[column] for column in on],
        how="left",
        indicator=indicator_name,
    )
    positions = np.sort(
        matched.loc[matched[indicator_name] == "left_only", position_name].to_numpy()
    )
    kept = left.iloc[positions]
    return pd.DataFrame(
        {
            output_column: kept[left_column].reset_index(drop=True)
            for output_column, (left_column, _) in output_columns.items()
        }
    )


def _join_on_ids(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: List[str],
    how: str,
    nulls_are_equal: bool,
    ids: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Returns the join of two dataframes, merged on their key numberings.

    The two Spark join implementations --
    :func:`tmlt.core.utils.join._join_where_nulls_are_equal` and
    :func:`tmlt.core.utils.join._join_where_nulls_are_not_equal` -- collapse
    into one here, because the difference between ``=`` and ``<=>`` has already
    been made by :func:`_join_ids` before any merging happens.

    Args:
        left: Left dataframe.
        right: Right dataframe.
        on: Columns to join on.
        how: Join type, one of "left", "right", "inner" or "outer".
        nulls_are_equal: If True, treats null values as equal.
        ids: Each join column's numbering, as :func:`_join_ids` returns it.
    """
    output_columns = columns_after_join(
        left_columns=list(left.columns),
        right_columns=list(right.columns),
        on=on,
        how=how,
    )
    # Which side the join can leave without a matching row, and so which side's
    # columns need somewhere to put a missing value before the merge rather
    # than after it.
    left_unmatchable = _side_unmatchable("left", how)
    right_unmatchable = _side_unmatchable("right", how)

    names = _NameSource(list(left.columns) + list(right.columns) + list(output_columns))
    id_names = {column: names.take() for column in on}
    left_key_names = {column: names.take() for column in on}
    right_key_names = {column: names.take() for column in on}
    indicator_name = names.take()

    merge_left: Dict[str, Any] = {}
    merge_right: Dict[str, Any] = {}
    for output_column, (left_column, right_column) in output_columns.items():
        if output_column in on:
            continue
        if right_column is None:
            assert left_column is not None
            merge_left[output_column] = _prepared(left[left_column], left_unmatchable)
        else:
            merge_right[output_column] = _prepared(
                right[right_column], right_unmatchable
            )
    for column in on:
        merge_left[left_key_names[column]] = _prepared(left[column], left_unmatchable)
        merge_right[right_key_names[column]] = _prepared(
            right[column], right_unmatchable
        )
        merge_left[id_names[column]] = ids[column][0]
        merge_right[id_names[column]] = ids[column][1]

    merged = pd.DataFrame(merge_left).merge(
        pd.DataFrame(merge_right),
        on=[id_names[column] for column in on],
        how=how,
        indicator=indicator_name,
    )
    indicator = merged[indicator_name]
    left_only = (indicator == "left_only").to_numpy()
    right_only = (indicator == "right_only").to_numpy()

    result: Dict[str, pd.Series] = {}
    for output_column, (left_column, right_column) in output_columns.items():
        if output_column in on:
            result[output_column] = _combined_key(
                merged[left_key_names[output_column]],
                merged[right_key_names[output_column]],
                right_only,
                _key_dtype(
                    left[output_column].dtype,
                    right[output_column].dtype,
                    how,
                    nulls_are_equal,
                ),
            )
        elif right_column is None:
            assert left_column is not None
            result[output_column] = _finished(
                merged[output_column],
                _payload_dtype(left[left_column].dtype, left_unmatchable),
                right_only,
            )
        else:
            result[output_column] = _finished(
                merged[output_column],
                _payload_dtype(right[right_column].dtype, right_unmatchable),
                left_only,
            )
    return pd.DataFrame(result)


def _prepared(column: pd.Series, unmatchable: bool) -> pd.Series:
    """Returns a column ready to be merged, indexed from zero.

    A column on a side the join can leave unmatched is widened to its nullable
    dtype first, so that the merge writes a missing value into a column that
    already has room for one. Doing it afterwards would be too late: the merge
    would have widened ``int64`` to ``float64`` on its own, and an integer above
    :math:`2^{53}` does not survive the round trip.

    Args:
        column: The column to prepare.
        unmatchable: Whether the join can leave this column's side without a
            matching row.
    """
    column = column.reset_index(drop=True)
    if not unmatchable:
        return column
    return _as_dtype(column, _nullable_dtype(column.dtype))


def _finished(column: pd.Series, dtype: PandasDtype, missing: np.ndarray) -> pd.Series:
    """Returns a merged column as its output dtype, with real nulls in it.

    A merge fills an object column's unmatched rows with a float ``NaN``, which
    in an object column is a *value* rather than a null; those positions are
    replaced with ``None``. Every other dtype's fill is already the right thing,
    because :func:`_prepared` gave the column somewhere to put it.

    Args:
        column: The merged column.
        dtype: The dtype the output column should have.
        missing: The rows this column's side did not contribute to.
    """
    column = column.reset_index(drop=True)
    if column.dtype == np.dtype(object) and missing.any():
        column = column.copy()
        column[missing] = None
    return _as_dtype(column, dtype)


def _combined_key(
    left_key: pd.Series,
    right_key: pd.Series,
    right_only: np.ndarray,
    dtype: PandasDtype,
) -> pd.Series:
    """Returns an output join column, taken from whichever side matched.

    The left frame's value is kept wherever the left frame contributed a row,
    which is what the Spark implementation's ``when(left is not null, left)``
    amounts to; only a row that came from the right frame alone takes the right
    frame's value. Two values can be one join key without being the same value
    -- ``-0.0`` and ``0.0``, or two timestamps a nanosecond apart -- and this is
    what decides which of them the output holds.

    Args:
        left_key: The left frame's key column, as it came out of the merge.
        right_key: The right frame's key column, as it came out of the merge.
        right_only: The rows the left frame did not contribute to.
        dtype: The dtype the output column should have.
    """
    left_key = _as_dtype(
        left_key.reset_index(drop=True), _nullable_dtype(left_key.dtype)
    )
    right_key = _as_dtype(
        right_key.reset_index(drop=True), _nullable_dtype(right_key.dtype)
    )
    combined = left_key.where(~right_only, right_key) if right_only.any() else left_key
    return _as_dtype(combined, dtype)


def _key_dtype(
    left_dtype: PandasDtype,
    right_dtype: PandasDtype,
    how: str,
    nulls_are_equal: bool,
) -> PandasDtype:
    """Returns the dtype an output join column comes back in.

    Whether the column can hold a null is decided by the same
    :func:`~tmlt.core.utils.join._join_allows_null` that gives a join column its
    ``allow_null`` in :func:`domain_after_join`, reading each side's dtype for
    whether it could hold one in the first place. The values themselves are the
    left frame's wherever it has any, so the left frame's dtype is the one that
    is widened or narrowed.

    Args:
        left_dtype: The left frame's join column dtype.
        right_dtype: The right frame's join column dtype.
        how: The join type.
        nulls_are_equal: If True, treats null values as equal.
    """
    allow_null = _join_allows_null(
        _can_hold_null(left_dtype), _can_hold_null(right_dtype), how, nulls_are_equal
    )
    return _target_dtype(left_dtype, allow_null)
