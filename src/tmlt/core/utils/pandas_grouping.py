"""Grouping pandas DataFrames the way Spark groups them.

Spark's notion of when two values belong to the same group is not pandas'.
The differences are few, individually obscure, and each one silently changes
a result: a ``groupby`` puts ``NULL`` and ``NaN`` in one group and then drops
it, a ``bytearray`` has no Python hash at all so ``pd.factorize`` raises, and
a ``datetime64[ns]`` column splits a Spark group in two over nanoseconds
Spark cannot see. This module resolves all of them once, so that every pandas
component owing its Spark counterpart the same answer -- the truncation
utilities in :mod:`tmlt.core.utils.pandas_truncation`, and the metrics,
tables and joins built alongside them -- groups values identically.

``_group_key`` is the specification: it maps a value to the key Spark
groups and orders it by, and its docstring enumerates the divergences in
full, the sign of zero included (there pandas already agrees, and it is the
hashing layer that does not). Everything else here computes that same
identity faster, with one vectorized branch per dtype, and the public
functions expose it:

* :func:`group_codes` and :func:`group_ids` number the groups,
* :func:`row_keys` names them,
* :func:`group_indices` and :func:`distinct_rows` apply them.

Supported dtypes:
    No dtype is rejected. Boolean and categorical columns, which the hashing
    functions in :mod:`tmlt.core.utils.pandas_truncation` refuse because
    Spark cannot hash them, group perfectly well here. The one unsupported
    input is a value with no Python hash -- a ``dict`` or a ``list`` inside
    an ``object`` column -- which has no group key and raises
    :class:`NotImplementedError`. This is exactly the contract of
    :func:`tmlt.core.utils.pandas_truncation.drop_large_groups`, the one
    truncation function that groups without hashing.

    Timezone-aware timestamps are grouped by the instant they denote, as
    Spark's ``TimestampType`` groups them. The hashing functions reject them
    instead, because a naive column's wall clock is what those can reproduce.

The ordering layer:
    Spark's hash-based row *ordering* -- which rows survive a truncation --
    lives in :mod:`tmlt.core.utils.pandas_truncation`, along with the Java
    floating point rendering and the SHA-256 digest pipeline it is built
    from. That module imports this one; nothing here depends on it, so
    grouping never pays for rendering machinery it does not use.

Performance:
    Values are compared once per *distinct* value of a column, so grouping
    cost scales with column cardinality rather than row count. The dtypes
    with no vectorized branch fall back to building one
    ``_group_key`` per row, which is a Python-level pass over the column.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import enum
import math
from typing import Any, Collection, Dict, Iterator, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_SUPPORTED_FLOAT_DTYPES = (np.dtype("float32"), np.dtype("float64"))

# The three classes of value Spark's ascending order puts in this order: nulls
# first, then every ordinary value, then NaNs.
_NULL_ORDER = 0
_VALUE_ORDER = 1
_NAN_ORDER = 2

#: Object-column kinds, as reported by
#: ``pandas.api.types.infer_dtype(skipna=True)``, whose values are all
#: renderable and all faithfully factorized by ``pd.factorize``. Kinds like
#: ``mixed`` (which covers bytearrays, unhashable by ``pd.factorize``) and
#: ``mixed-integer-float`` (where ``pd.factorize`` merges ``1`` with ``1.0``,
#: which render differently) are deliberately absent.
_HOMOGENEOUS_OBJECT_KINDS = frozenset({"string", "bytes", "empty"})


def _is_null(value: Any) -> bool:
    """Returns whether a value is a null value, as opposed to a float NaN."""
    return value is None or value is pd.NA or value is pd.NaT


def _missing_is_null(dtype: Any) -> bool:
    """Returns whether every missing entry of such a column is a null.

    A categorical column stores a missing entry as the code ``-1`` and hands it
    back as ``np.nan``, which in a float or an object column is a *value* here.
    There is no other way to spell a missing value in a categorical -- pandas
    does not allow a NaN to be a category -- so in one of those a NaN is a
    null, and the null group of a grouping and the ``nulls_are_equal`` of a
    join both have to see it as one.

    Args:
        dtype: The dtype to classify.
    """
    return isinstance(dtype, pd.CategoricalDtype)


def _column_values(column: pd.Series) -> Iterator[Any]:
    """Returns the values of a column, with the precision of its dtype."""
    if column.dtype == np.dtype("float32"):
        # Iterating a numpy float32 series yields Python floats, which are
        # double precision and would be rendered with too many digits.
        return iter(column.to_numpy(dtype=np.float32))
    if isinstance(column.dtype, pd.Float32Dtype):
        return iter(column.array)
    if _missing_is_null(column.dtype):
        # Yielded as the None every other dtype's missing entry becomes, so
        # that _group_key reads it as the null it is rather than as a NaN.
        values = column.astype(object).to_numpy(dtype=object, copy=True)
        values[column.isna().to_numpy()] = None
        return iter(values)
    return iter(column)


def _object_kind(column: pd.Series) -> str:
    """Returns the inferred kind of an object column's values.

    Returns:
        The value of ``pandas.api.types.infer_dtype(column, skipna=True)``.
    """
    return pd.api.types.infer_dtype(column, skipna=True)


def _null_and_nan_masks(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the null and NaN masks of an object array.

    ``pandas.isna`` marks ``None``, ``pd.NA``, ``pd.NaT`` and float ``NaN``
    alike, but these functions treat a float ``NaN`` in an object column as a
    value that hashes to ``b"nan"`` and sorts last, and only the first three
    as nulls. The ambiguous positions -- which are few in every realistic
    frame -- are therefore resolved one at a time with :func:`_is_null`.

    Args:
        values: The object array to inspect.

    Returns:
        A boolean mask of the null positions and a boolean mask of the NaN
        positions. The two never overlap.
    """
    missing = pd.isna(values)
    null_mask = np.zeros(len(values), dtype=bool)
    nan_mask = np.zeros(len(values), dtype=bool)
    for position in np.flatnonzero(missing):
        if _is_null(values[position]):
            null_mask[position] = True
        else:
            nan_mask[position] = True
    return null_mask, nan_mask


class _ColumnClass(enum.Enum):
    """The column classes the vectorized paths dispatch on.

    :func:`tmlt.core.utils.pandas_truncation._digest_codes`,
    :func:`_group_codes` and
    :func:`tmlt.core.utils.pandas_truncation._order_keys` all dispatch on
    :func:`_column_class`, so a column takes the corresponding branch in each
    of them and the three cannot drift apart: a dtype is either vectorized
    everywhere or falls back everywhere.
    """

    NULLABLE_FLOAT = enum.auto()  #: pd.Float32Dtype or pd.Float64Dtype
    NULLABLE_INT = enum.auto()  #: a pandas nullable integer dtype
    STRING = enum.auto()  #: a pandas string dtype
    NUMPY_INT = enum.auto()  #: a numpy signed or unsigned integer dtype
    NUMPY_FLOAT = enum.auto()  #: numpy float32 or float64
    DATETIME = enum.auto()  #: datetime64, timezone-naive
    HOMOGENEOUS_OBJECT = enum.auto()  #: object, of a faithfully factorizable kind
    FALLBACK = enum.auto()  #: everything else: the per-value paths


def _column_class(column: pd.Series) -> _ColumnClass:
    """Classifies a column for the vectorized paths.

    Returns:
        The class whose branch the vectorized functions take for this column.
    """
    dtype = column.dtype
    if isinstance(dtype, (pd.Float32Dtype, pd.Float64Dtype)):
        return _ColumnClass.NULLABLE_FLOAT
    if pd.api.types.is_integer_dtype(dtype) and not isinstance(dtype, np.dtype):
        return _ColumnClass.NULLABLE_INT
    if isinstance(dtype, pd.StringDtype):
        return _ColumnClass.STRING
    if not isinstance(dtype, np.dtype):
        # Extension dtypes with no vectorized path, e.g. the categorical and
        # boolean columns drop_large_groups accepts without validation.
        return _ColumnClass.FALLBACK
    if dtype.kind in "iu":
        return _ColumnClass.NUMPY_INT
    if dtype in _SUPPORTED_FLOAT_DTYPES:
        return _ColumnClass.NUMPY_FLOAT
    if pd.api.types.is_datetime64_dtype(dtype):
        return _ColumnClass.DATETIME
    if dtype == np.dtype(object) and _object_kind(column) in _HOMOGENEOUS_OBJECT_KINDS:
        return _ColumnClass.HOMOGENEOUS_OBJECT
    return _ColumnClass.FALLBACK


def _codes_with_sentinels(values: Any, *masks: np.ndarray) -> np.ndarray:
    """Factorizes ``values``, giving each mask's positions a code of their own.

    Args:
        values: The values to factorize, as ``pd.factorize`` accepts them.
        masks: Disjoint boolean masks, each marking positions that are one
            class of their own, distinct from every value and from the other
            masks' classes. Together they must cover every position
            ``pd.factorize`` marks as missing, or the missing-value sentinel
            would leak into the result as a class of its own.

    Returns:
        A non-negative int64 array aligned with ``values``.
    """
    return _factorization_with_sentinels(pd.factorize(values), *masks)


def _factorization_with_sentinels(
    factorization: Tuple[np.ndarray, Sequence[Any]], *masks: np.ndarray
) -> np.ndarray:
    """Writes each mask's sentinel code into a factorization's codes.

    This is the sentinel step of :func:`_codes_with_sentinels`, split out so
    that the canonical factorizations a :class:`_FactorizeMemo` shares can
    take it without being factorized again. The codes must be the caller's
    own -- fresh from ``pd.factorize``, or copied out of the memo -- because
    the sentinels are written into them in place.

    Args:
        factorization: The ``(codes, uniques)`` pair whose codes are written.
        masks: Disjoint boolean masks, as :func:`_codes_with_sentinels`
            takes them.

    Returns:
        A non-negative int64 array aligned with the codes.
    """
    codes, uniques = factorization
    codes = codes.astype(np.int64, copy=False)
    for offset, mask in enumerate(masks):
        codes[mask] = len(uniques) + offset
    return codes


def _nullable_int_values(column: pd.Series) -> np.ndarray:
    """Returns a nullable integer column's values, with nulls reading as 0.

    Unsigned values above ``2**63 - 1`` would not survive a cast to int64, so
    unsigned dtypes are materialized as ``uint64``. The caller separates the
    nulls out again with the column's own null mask.

    Returns:
        An int64 or uint64 array aligned with ``column``.
    """
    target: Any = (
        np.uint64 if pd.api.types.is_unsigned_integer_dtype(column.dtype) else np.int64
    )
    return column.to_numpy(target, na_value=0)


def _microsecond_keys(column: pd.Series) -> np.ndarray:
    """Returns int64 keys grouping and ordering a datetime column like Spark.

    Two rows share a key exactly when their values agree at Spark's
    microsecond resolution, and the keys ascend in Spark's timestamp order.
    A nanosecond column is floored to microseconds, merging the
    sub-microsecond distinctions Spark cannot see; numpy's cast floors toward
    negative infinity, like ``Timestamp.floor``. A column in a coarser unit
    ('s', 'ms' or 'us', which pandas 2 allows) already carries no
    sub-microsecond precision, so its own representation is the key:
    converting it to nanoseconds, as ``to_numpy("datetime64[ns]")`` would,
    silently wraps values outside the nanosecond range, such as 9999-12-31
    in a microsecond column. ``NaT`` keeps its own sentinel value.

    Returns:
        An int64 array aligned with ``column``. Only equality and relative
        order are meaningful; the unit is the column's own.
    """
    values = column.to_numpy()
    if values.dtype == np.dtype("datetime64[ns]"):
        values = values.astype("datetime64[us]")
    return values.view("int64")


def _canonical_array(column: pd.Series, klass: _ColumnClass) -> np.ndarray:
    """Returns the array a column's class canonically factorizes.

    The canonical array is the exact array the class's branches in
    :func:`tmlt.core.utils.pandas_truncation._validate_column`,
    :func:`_group_codes` and
    :func:`tmlt.core.utils.pandas_truncation._digest_codes` all factorize --
    the *identical* input, so one factorization serves all three, which is
    what lets a :class:`_FactorizeMemo` share it.
    :func:`tmlt.core.utils.pandas_truncation._order_keys` ranks the same
    array, and derives those ranks from the shared factorization rather than
    building another one (see
    :func:`tmlt.core.utils.pandas_truncation._dense_ranks_from_factorization`).
    The float and datetime classes have no canonical array: their consumers
    factorize *different* derived arrays (bit patterns versus values, raw
    versus microsecond-floored), which must never be merged.

    Args:
        column: The column whose canonical array is built.
        klass: The column's :func:`_column_class`, which every caller has
            already computed. It must be one of the four classes named here.

    Returns:
        The array the class's consumers factorize.
    """
    if klass is _ColumnClass.NULLABLE_INT:
        return _nullable_int_values(column)
    if klass is _ColumnClass.NUMPY_INT:
        return column.to_numpy()
    if klass is _ColumnClass.STRING:
        return column.to_numpy(object, na_value=None)
    if klass is _ColumnClass.HOMOGENEOUS_OBJECT:
        return column.to_numpy()
    raise AssertionError(f"No canonical factorization for {klass}")


def _canonical_factorization(
    column: pd.Series, klass: _ColumnClass
) -> Tuple[np.ndarray, Sequence[Any]]:
    """Returns ``pd.factorize`` of a column's :func:`_canonical_array`.

    Args:
        column: The column to factorize.
        klass: The column's :func:`_column_class`, as
            :func:`_canonical_array` takes it.

    Returns:
        The ``(codes, uniques)`` pair as ``pd.factorize`` returns it, with
        the column's nulls carrying the missing-value sentinel; the callers'
        masks are what separate those out.

    Raises:
        TypeError: From ``pd.factorize``, for a homogeneous object column
            holding a value it cannot hash, such as a bytearray. The
            callers' fallbacks expect exactly this.
    """
    return pd.factorize(_canonical_array(column, klass))


class _FactorizeMemo:
    """A per-call cache of canonical factorizations and null/NaN masks.

    Within one call of a hashing truncation function, the same full-frame
    column is factorized by several consumers -- validation's UTF-8
    encodability check, :func:`_group_codes`, the
    :func:`tmlt.core.utils.pandas_truncation._digest_codes` behind
    :func:`tmlt.core.utils.pandas_truncation.limit_keys_per_group`'s refined
    budget test and behind the row hashing itself, and the dense ranks of
    :func:`tmlt.core.utils.pandas_truncation._order_keys` -- and for the
    classes :func:`_canonical_array` covers, those are factorizations of the
    identical array. The public functions pass one memo down so that the
    factorization, the most expensive per-column step, runs once per column
    however many of those consumers ask for it; the object columns'
    :func:`_null_and_nan_masks`, recomputed by the same consumers, are shared
    the same way. :func:`tmlt.core.utils.pandas_truncation._order_keys` needs
    that factorization's codes in the ascending order of its uniques, and
    derives them from the shared factorization without a second pass over the
    rows (see
    :func:`tmlt.core.utils.pandas_truncation._dense_ranks_from_factorization`).

    The memo is keyed by column name, so a memo must only ever see columns
    of the one frame its call is truncating -- in particular never the fast
    paths' frames of selected or representative rows, whose columns share
    names with the full frame's but hold fewer rows. The row-hashing call
    sites therefore pass it only on the branches where the frame they hash
    *is* the full frame.
    """

    def __init__(self) -> None:
        """Constructor."""
        self._factorizations: Dict[Any, Optional[Tuple[np.ndarray, Sequence[Any]]]] = {}
        self._masks: Dict[Any, Tuple[np.ndarray, np.ndarray]] = {}

    def factorization(
        self, column: pd.Series, klass: _ColumnClass
    ) -> Tuple[np.ndarray, Sequence[Any]]:
        """Returns the column's canonical factorization, computed once per call.

        Some consumers write sentinel codes into the codes they are handed,
        so the stored codes are copied out on every request and the caller
        owns its copy -- a copy costs microseconds where the factorization
        it saves costs milliseconds. The uniques are only ever read, and are
        returned as stored.

        Raises:
            TypeError: When the column has no faithful factorization (a
                homogeneous object column holding a value ``pd.factorize``
                cannot hash). The failure is remembered, so the failing
                factorization is not run a second time either.
        """
        name = column.name
        if name not in self._factorizations:
            try:
                self._factorizations[name] = _canonical_factorization(column, klass)
            except TypeError:
                self._factorizations[name] = None
        factorization = self._factorizations[name]
        if factorization is None:
            raise TypeError(f"no faithful factorization for column {name}")
        codes, uniques = factorization
        return codes.copy(), uniques

    def null_and_nan_masks(self, column: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the column's :func:`_null_and_nan_masks`, computed once per call.

        The masks are shared, not copied: every consumer only reads them.
        """
        name = column.name
        if name not in self._masks:
            self._masks[name] = _null_and_nan_masks(column.to_numpy())
        return self._masks[name]


def _memoized_factorization(
    column: pd.Series, klass: _ColumnClass, memo: Optional[_FactorizeMemo]
) -> Tuple[np.ndarray, Sequence[Any]]:
    """Returns the canonical factorization, through the memo when one is given.

    Either way the caller owns the returned codes and may write into them:
    without a memo they are fresh from ``pd.factorize``, and the memo copies
    its shared codes out (see :meth:`_FactorizeMemo.factorization`).
    """
    if memo is None:
        return _canonical_factorization(column, klass)
    return memo.factorization(column, klass)


def _memoized_null_and_nan_masks(
    column: pd.Series, memo: Optional[_FactorizeMemo]
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns an object column's null and NaN masks, through the memo if given.

    The memo's masks are shared between its consumers, all of which only
    read them.
    """
    if memo is None:
        return _null_and_nan_masks(column.to_numpy())
    return memo.null_and_nan_masks(column)


def _group_key(value: Any) -> Tuple[int, Any]:
    """Returns the key Spark groups and orders a value by.

    Spark's window partitioning and ordering differ from what a pandas
    ``groupby`` or ``sort_values`` does in four ways, all of which this key
    encodes:

    * A null and a NaN are different partitions, and ascending order puts nulls
      first and NaNs last, while pandas puts both in the same group and, with
      ``na_position``, in the same place. This is reachable in an ``object``
      column, which can hold both.
    * ``-0.0`` and ``0.0`` are one partition, and tie in an ordering, even
      though they hash differently. Two Python floats already behave that way.
    * Binary values are compared by content, and a ``bytearray`` is not even
      hashable, so binary values are keyed by their bytes.
    * Timestamps have microsecond resolution. Values are hashed with
      sub-microsecond precision discarded, so grouping and ordering have to
      discard it too, or a ``datetime64[ns]`` column would split a Spark
      partition in two.

    Returns:
        A hashable key whose natural order is Spark's ascending order.
    """
    if _is_null(value):
        return (_NULL_ORDER, 0)
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return (_NAN_ORDER, 0)
        return (_VALUE_ORDER, float(value))
    if isinstance(value, (bytes, bytearray)):
        return (_VALUE_ORDER, bytes(value))
    if isinstance(value, pd.Timestamp) and value.nanosecond:
        # Flooring to microseconds must not construct another Timestamp:
        # for values within a microsecond of Timestamp.min, floor("us")
        # lands below the nanosecond bound and raises OverflowError. The
        # stdlib datetime has no such bound, and dropping the nanosecond
        # field is the same floor -- the wall-clock fields already carry
        # the microsecond part, with the nanoseconds a non-negative
        # remainder on top, pre-epoch values included. A plain datetime
        # also compares and hashes equal to an equal Timestamp, so this
        # key unifies with the keys of the datetime.datetime values an
        # object column can hold alongside Timestamps.
        return (_VALUE_ORDER, value.to_pydatetime(warn=False))
    if pd.api.types.is_scalar(value) and pd.isna(value):
        # An NA-like value outside the branches above -- Decimal("NaN"), or a
        # raw np.datetime64("NaT") in an object column -- compares unequal to
        # itself, so keying it by value would make every occurrence a
        # partition of its own and let an oversized group of them slip past
        # drop_large_groups. Such values are keyed like the float NaNs they
        # behave as, which is also where :func:`_null_and_nan_masks` puts
        # them on the vectorized paths.
        return (_NAN_ORDER, 0)
    # pd.factorize over the keys, and the set/sort in the ordering fallback,
    # need the key to be hashable; a value with no Python hash -- a dict or
    # a list in an object column -- would otherwise surface as a bare
    # TypeError from inside pandas. Spark cannot hold such a value either,
    # so it is reported as the unsupported value it is, in the same form
    # _render_value uses. (A bytearray, equally unhashable, never reaches
    # this probe: it was keyed by its bytes above.)
    try:
        hash(value)
    except TypeError as error:
        raise NotImplementedError(
            f"Unsupported data type {type(value).__name__}"
        ) from error
    return (_VALUE_ORDER, value)


def _dense_codes(codes: Sequence[np.ndarray]) -> np.ndarray:
    """Combines several code arrays into one dense code per distinct combination.

    Args:
        codes: The code arrays to combine, at least one.

    Returns:
        An int64 array of codes in ``range(number of distinct combinations)``,
        numbered in order of first appearance.
    """
    # The host series exists only to give groupby a frame-shaped anchor; the
    # keys carry all the information.
    return (
        pd.Series(0, index=range(len(codes[0])))
        .groupby(list(codes), sort=False, dropna=False)
        .ngroup()
        .to_numpy()
    )


def _fallback_group_codes(column: pd.Series) -> np.ndarray:
    """Returns group codes by building every row's :func:`_group_key`.

    This is the exact, per-value path for the columns the vectorized cases in
    :func:`_group_codes` cannot handle faithfully.

    Returns:
        A non-negative int64 array aligned with ``column``.
    """
    keys = pd.Series(
        [_group_key(value) for value in _column_values(column)], dtype=object
    )
    # Every key is a tuple, so pd.factorize marks nothing as missing and no
    # sentinel masks are needed.
    return _codes_with_sentinels(keys)


def _group_codes(
    column: pd.Series, memo: Optional[_FactorizeMemo] = None
) -> np.ndarray:
    """Returns one dense code per Spark partition key of a column.

    Two rows share a code exactly when :func:`_group_key` gives them the same
    key, so grouping by these codes forms the partitions Spark would form.
    Unlike :func:`tmlt.core.utils.pandas_truncation._digest_codes`, this
    factorization must be exact in both directions: an over-split (``0.0``
    versus ``-0.0``, ``bytes`` versus ``bytearray``) would change which rows
    share a group.

    Args:
        column: The column to code.
        memo: A per-call memo sharing the canonical factorizations and masks
            with the other consumers, or None to compute everything here.

    Returns:
        A non-negative int64 array aligned with ``column``.
    """
    dtype = column.dtype
    klass = _column_class(column)
    if klass is _ColumnClass.NULLABLE_FLOAT:
        float_dtype = np.float32 if isinstance(dtype, pd.Float32Dtype) else np.float64
        floats = column.to_numpy(float_dtype, na_value=np.nan)
        # Factorizing the values, not the bit patterns, makes 0.0 and -0.0
        # one partition, as _group_key does. NaNs come out as missing
        # alongside the nulls and the two are then separated by their masks.
        null_mask = column.isna().to_numpy()
        return _codes_with_sentinels(floats, np.isnan(floats) & ~null_mask, null_mask)
    if klass is _ColumnClass.NULLABLE_INT:
        return _factorization_with_sentinels(
            _memoized_factorization(column, klass, memo), column.isna().to_numpy()
        )
    if klass is _ColumnClass.STRING:
        # The nulls are one partition.
        return _factorization_with_sentinels(
            _memoized_factorization(column, klass, memo), column.isna().to_numpy()
        )
    if klass is _ColumnClass.NUMPY_INT:
        return _factorization_with_sentinels(
            _memoized_factorization(column, klass, memo)
        )
    if klass is _ColumnClass.NUMPY_FLOAT:
        # The NaNs are one partition.
        floats = column.to_numpy()
        return _codes_with_sentinels(floats, np.isnan(floats))
    if klass is _ColumnClass.DATETIME:
        # NaT keeps its own sentinel value and so its own partition.
        return _codes_with_sentinels(_microsecond_keys(column))
    if klass is _ColumnClass.HOMOGENEOUS_OBJECT:
        null_mask, nan_mask = _memoized_null_and_nan_masks(column, memo)
        try:
            # A pandas groupby would put NaNs and nulls in one group; Spark
            # makes them two partitions.
            return _factorization_with_sentinels(
                _memoized_factorization(column, klass, memo), nan_mask, null_mask
            )
        except TypeError:
            return _fallback_group_codes(column)
    return _fallback_group_codes(column)


def _first_occurrences(codes: np.ndarray) -> np.ndarray:
    """Returns the position of each code's first occurrence, in code order.

    ``codes`` must be first-occurrence dense -- ``pd.factorize`` or
    :func:`_dense_codes` output, numbered ``0, 1, ...`` in order of first
    appearance. A position is then a first occurrence exactly when its code
    exceeds every earlier code, and first occurrences appear in code order,
    so this equals ``np.unique(codes, return_index=True)[1]`` without the
    O(n log n) sort.

    Returns:
        An int64 array with one position per distinct code.
    """
    if not len(codes):
        return np.zeros(0, dtype=np.int64)
    is_first = np.empty(len(codes), dtype=bool)
    is_first[0] = True
    is_first[1:] = codes[1:] > np.maximum.accumulate(codes)[:-1]
    return np.flatnonzero(is_first)


def _group_ids(codes: Sequence[np.ndarray], n_rows: int) -> np.ndarray:
    """Returns one dense group id per row, treating no columns as one group.

    Args:
        codes: One array per grouping column, as :func:`_group_codes` returns
            them, possibly none at all.
        n_rows: The number of rows, which fixes the result's length when
            there are no grouping columns.

    Returns:
        A non-negative int64 array aligned with the frame. Every consumer is
        label-agnostic, so a single column's codes already are its ids.
    """
    if not codes:
        return np.zeros(n_rows, dtype=np.int64)
    if len(codes) == 1:
        return codes[0]
    return _dense_codes(codes)


def _require_columns(df: pd.DataFrame, columns: Collection[str]) -> None:
    """Raises KeyError when a named column is not a column of ``df``.

    The truncation functions call this before their threshold early returns,
    so that an unknown column raises whatever the threshold is.
    """
    for column in columns:
        if column not in df.columns:
            raise KeyError(column)


def _reindexed_from_zero(selection: pd.DataFrame) -> pd.DataFrame:
    """Returns a frame selected out of another, reindexed from zero.

    ``selection`` must be a fresh mask or ``iloc`` selection: such a selection
    is already a copy, so replacing its index in place costs nothing, where
    ``reset_index`` would copy the whole frame a second time.
    """
    selection.index = pd.RangeIndex(len(selection))
    return selection


def _first_occurrence_codes(codes: np.ndarray) -> np.ndarray:
    """Renumbers codes as ``0, 1, ...`` in order of first appearance.

    The internal code arrays are dense only in the sense of being small
    non-negative integers: :func:`_group_codes` numbers a column's nulls and
    NaNs above its values, so a column with no nulls at all can still leave a
    gap. The public functions promise contiguous ids instead, which index a
    table of groups directly and make ``np.bincount`` a group-size count, so
    they renumber before returning. The input carries no missing values, so
    ``pd.factorize`` never produces its sentinel.

    Returns:
        An int64 array aligned with ``codes``.
    """
    return pd.factorize(codes)[0].astype(np.int64, copy=False)


def group_codes(column: pd.Series) -> np.ndarray:
    """Returns one id per Spark group of a column's values.

    Two rows get the same id exactly when Spark would put them in the same
    group: nulls form one group and NaNs another, ``-0.0`` and ``0.0`` share
    one, binary values group by content, and timestamps group at Spark's
    microsecond resolution. The ids are ``0, 1, ...`` in order of first
    appearance.

    Example:
        >>> import numpy as np
        >>> import pandas as pd
        >>> group_codes(pd.Series([1.0, -0.0, 0.0, np.nan, 1.0]))
        array([0, 1, 1, 2, 0])
        >>> group_codes(pd.Series(["a", None, "b", None], dtype=object))
        array([0, 1, 2, 1])

    Args:
        column: The column to group.

    Returns:
        A non-negative int64 array aligned with ``column``.

    Raises:
        NotImplementedError: If a value has no Python hash, such as a ``dict``
            or a ``list`` in an ``object`` column. Spark cannot hold such a
            value either.
    """
    return _first_occurrence_codes(_group_codes(column))


def group_ids(df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Returns one id per Spark group of a frame's rows.

    Two rows get the same id exactly when they agree on every named column
    under :func:`group_codes`' notion of agreement. The ids are ``0, 1, ...``
    in order of first appearance. A repeated column name is grouped by once,
    as Spark accepts a repeated partitioning column, and no columns at all
    make the whole frame one group.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"a": [1, 1, 2, 2], "b": ["x", "y", "x", "x"]})
        >>> group_ids(df, ["a", "b"])
        array([0, 1, 2, 2])
        >>> group_ids(df, ["a"])
        array([0, 0, 1, 1])
        >>> group_ids(df, [])
        array([0, 0, 0, 0])

    Args:
        df: The frame whose rows are grouped.
        columns: The columns defining the groups.

    Returns:
        A non-negative int64 array aligned with ``df``.

    Raises:
        KeyError: If a named column is not a column of ``df``.
        NotImplementedError: If a value has no Python hash, as in
            :func:`group_codes`.
    """
    _require_columns(df, columns)
    unique = list(dict.fromkeys(columns))
    return _first_occurrence_codes(
        _group_ids([_group_codes(df[column]) for column in unique], len(df))
    )


def row_keys(df: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> pd.Series:
    """Returns one hashable key per row, naming the row's Spark group.

    Two rows carry equal keys exactly when :func:`group_ids` gives them the
    same id, so the keys name the groups that function numbers: they are
    hashable (a ``bytearray``, which is not, is keyed by its bytes) and
    usable as dictionary keys, which is what :func:`group_indices` returns
    them as. Their internal structure is not part of the contract; compare
    them with each other, and do not read values back out of them.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"v": [None, float("nan"), None]}, dtype=object)
        >>> keys = row_keys(df)
        >>> keys[0] == keys[2]  # two nulls are one Spark group
        True
        >>> keys[0] == keys[1]  # a null and a NaN are not
        False

    Args:
        df: The frame whose rows are keyed.
        columns: The columns defining the groups, or None for every column of
            ``df``.

    Returns:
        An object-dtype series of tuples, aligned with ``df``'s index.

    Raises:
        KeyError: If a named column is not a column of ``df``.
        NotImplementedError: If a value has no Python hash, as in
            :func:`group_codes`.
    """
    names = list(df.columns) if columns is None else list(columns)
    _require_columns(df, names)
    keyed = [
        [_group_key(value) for value in _column_values(df[name])] for name in names
    ]
    # The keys are written into an object array one at a time: assigning a
    # list of equal-length tuples to an object array's slice would have numpy
    # read it as a two-dimensional array instead.
    keys = np.empty(len(df), dtype=object)
    for position, key in enumerate(zip(*keyed) if names else [()] * len(df)):
        keys[position] = key
    return pd.Series(keys, index=df.index, dtype=object)


def distinct_rows(
    df: pd.DataFrame, columns: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """Returns the rows of a frame that are distinct under Spark's semantics.

    This is the pandas counterpart of Spark's ``DataFrame.distinct`` (with no
    ``columns``) and ``DataFrame.dropDuplicates`` (with them). Rows are
    compared with :func:`group_ids`, so a null and a NaN are two different
    rows where ``pandas.DataFrame.drop_duplicates`` would have to be told
    which of them it is looking at.

    Every column of ``df`` is returned whatever ``columns`` says, with its
    original dtype; only which rows survive depends on the argument. The first
    row of each group is the one kept -- Spark keeps an arbitrary one -- and
    the survivors are returned in input order, reindexed from zero.

    Example:
        ..
            >>> import pandas as pd
            >>> from tmlt.core.utils.misc import print_pandas
            >>> dataframe = pd.DataFrame(
            ...     {
            ...         "A": ["a1", "a1", "a2", "a2"],
            ...         "B": ["b1", "b1", "b1", "b2"],
            ...     }
            ... )

        >>> # Example input
        >>> print_pandas(dataframe)
            A   B
        0  a1  b1
        1  a1  b1
        2  a2  b1
        3  a2  b2
        >>> print_pandas(distinct_rows(dataframe))
            A   B
        0  a1  b1
        1  a2  b1
        2  a2  b2
        >>> print_pandas(distinct_rows(dataframe, ["A"]))
            A   B
        0  a1  b1
        1  a2  b1

    Args:
        df: The frame to deduplicate.
        columns: The columns deciding which rows are duplicates, or None for
            every column of ``df``.

    Returns:
        The surviving rows, with every column of ``df``.

    Raises:
        KeyError: If a named column is not a column of ``df``.
        NotImplementedError: If a value has no Python hash, as in
            :func:`group_codes`.
    """
    names = list(df.columns) if columns is None else list(columns)
    return _reindexed_from_zero(df.iloc[_first_occurrences(group_ids(df, names))])


def group_indices(
    df: pd.DataFrame, columns: Sequence[str]
) -> Dict[Tuple[Any, ...], np.ndarray]:
    """Returns the positions of the rows of each Spark group of a frame.

    This is the null-safe counterpart of ``df.groupby(columns).indices``,
    which merges nulls with NaNs and, by default, drops them altogether.
    The groups come out in order of first appearance, and each one's
    positions in ascending order, so ``positions[0]`` is the group's first
    row.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"g": ["a", "b", "a"]})
        >>> groups = group_indices(df, ["g"])
        >>> [positions.tolist() for positions in groups.values()]
        [[0, 2], [1]]
        >>> groups[row_keys(df, ["g"])[1]].tolist()
        [1]
        >>> missing = pd.DataFrame({"g": [None, float("nan"), None]}, dtype=object)
        >>> [positions.tolist() for positions in group_indices(missing, ["g"]).values()]
        [[0, 2], [1]]

    Args:
        df: The frame whose rows are grouped.
        columns: The columns defining the groups. No columns at all make the
            whole frame one group.

    Returns:
        One entry per group, keyed by the group's :func:`row_keys` key, whose
        value is the positions of that group's rows in ``df``.

    Raises:
        KeyError: If a named column is not a column of ``df``.
        NotImplementedError: If a value has no Python hash, as in
            :func:`group_codes`.
    """
    ids = group_ids(df, columns)
    if not len(ids):
        return {}
    # A stable sort by id lists each group's rows in ascending position order,
    # and the ids are first-occurrence dense, so the groups come out in order
    # of first appearance.
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    starts = np.flatnonzero(np.concatenate(([True], sorted_ids[1:] != sorted_ids[:-1])))
    positions = np.split(order, starts[1:])
    keys = row_keys(df.iloc[_first_occurrences(ids)], columns)
    return dict(zip(keys, positions))
