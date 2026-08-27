"""Functions for truncating pandas DataFrames.

These functions are pandas counterparts of the Spark truncation utilities in
:mod:`tmlt.core.utils.truncation`. They implement the same algorithms, using the
same SHA-256 based row ordering, so that for every column type supported by both
backends the two implementations keep exactly the same rows.

Compatibility with :mod:`tmlt.core.utils.truncation`:
    Supported dtypes
        A column is hashable by these functions if its dtype is one of:

        * any numpy or pandas nullable integer dtype (``int8`` through
          ``int64``, ``uint8`` through ``uint64``, ``Int8`` through ``UInt64``),
          which corresponds to Spark's ``IntegerType`` and ``LongType``;
        * ``float32`` or ``Float32`` (Spark ``FloatType``), and ``float64`` or
          ``Float64`` (Spark ``DoubleType``);
        * ``datetime64`` without a timezone (Spark ``TimestampType``), in the
          ``ns`` unit or, on pandas 2, any of the coarser ``s``/``ms``/``us``
          units -- whose values may lie far outside the ``ns`` range and are
          hashed at their own precision, never through a narrowing cast;
        * ``object`` and the pandas string dtypes, whose values may be
          :class:`str` (Spark ``StringType``), :class:`bytes` or
          :class:`bytearray` (Spark ``BinaryType``),
          :class:`datetime.date` (Spark ``DateType``),
          :class:`datetime.datetime` (Spark ``TimestampType``), any of the
          numeric types above, or a null value.

        Every other dtype raises :class:`NotImplementedError`, as do boolean
        columns and unsupported values inside ``object`` columns. Because an
        empty ``object`` column carries no values, unsupported *value* types
        cannot be detected in that case.

        Which columns are hashed differs by function, and so does when an
        unsupported dtype is reported: :func:`truncate_large_groups` hashes
        every column, :func:`limit_keys_per_group` hashes only the grouping and
        key columns, and :func:`drop_large_groups` hashes nothing and therefore
        never rejects a dtype -- though a value that is not hashable in the
        Python sense, such as a ``dict`` or a ``list`` inside an ``object``
        column, has no group key and raises :class:`NotImplementedError` even
        there.

    Strings with unpaired surrogates
        A Python ``str`` can hold an unpaired surrogate code point --
        ``os.fsdecode``, ``surrogateescape`` decoding, and JSON with escaped
        surrogates all produce them -- and such a string has no UTF-8
        encoding. Spark coerces surrogates to U+FFFD when the value is
        ingested, so two strings differing only in their surrogates are one
        value to Spark: one partition and one sort position. Coercing here
        would silently rewrite caller data, so the functions that hash
        strings reject them with :class:`NotImplementedError` instead of
        silently keeping different rows. (:func:`drop_large_groups` hashes
        nothing and accepts them, grouping them by their own code points,
        exactly as it accepts the boolean columns Spark cannot hold.)

    Nulls and NaNs in float columns
        Spark distinguishes ``NULL`` from ``NaN``, and hashes them differently.
        A numpy ``float32``/``float64`` column cannot represent ``NULL``, so
        ``NaN`` values in such columns are hashed the way Spark hashes ``NaN``.
        To express ``NULL`` in a float column, use the pandas nullable
        ``Float32``/``Float64`` dtypes with ``pd.NA``. An ``object`` column can
        hold both. The two are also different group keys, as they are in Spark,
        even though a pandas ``groupby`` would put them in the same group.

    Dates and timestamps
        Timestamps are hashed and grouped as their wall-clock value, with
        sub-microsecond precision discarded, so they hash identically to Spark
        timestamps whenever the naive pandas values represent wall clocks in
        Spark's session timezone (``spark.sql.session.timeZone``). Timezone-aware
        columns raise :class:`NotImplementedError`; convert them with
        :meth:`~pandas.Series.dt.tz_convert` followed by
        :meth:`~pandas.Series.dt.tz_localize` first.

    Floating-point rendering and the JVM version
        Spark renders ``float`` and ``double`` values with the JVM's
        ``Float.toString``/``Double.toString``, and hashes the result. This
        module reimplements the rendering specified by Java 19 and later, which
        is the shortest decimal that round-trips to the value. Java 18 and
        earlier sometimes render the same value differently, usually with more
        digits than are needed (`JDK-4511638
        <https://bugs.openjdk.org/browse/JDK-4511638>`_). Those renderings
        denote the same value but hash differently, so against a Spark running
        on such a JVM some values hash differently here. Sampling uniformly
        over bit patterns, this affects roughly 0.2% of ``double`` values and
        roughly 10% of ``float`` values -- those needing many significant
        digits, plus the smallest subnormals. Java 19 and later are unaffected.

    Signed zeros in duplicate rows
        :func:`truncate_large_groups` salts identical rows so that they are not
        kept or dropped as a block, mirroring Spark's ``row_number`` over a
        window partitioned by every column. Spark partitions that window with
        ``-0.0`` and ``0.0`` compared equal, but hashes each value as stored,
        where the two render differently. Two rows that are identical except
        for the sign of a zero therefore share a salt partition while hashing
        differently, and which of them Spark salts first is whatever its
        shuffle produced: Spark need not keep the same row twice in a row on
        such data. This module salts in input order and so always gives one
        deterministic answer among those Spark is entitled to give, which is
        the one case where the two implementations may keep different rows
        without either being wrong. The differential tests cannot pin this
        down, and collapse such pairs into genuine duplicates instead.

Row order:
    All three functions return their surviving rows in input order: a
    surviving row precedes another in the result exactly when it did in the
    input. The hash order decides only *which* rows survive, never how they
    are returned. (Spark makes no ordering promise at all; its output order
    is whatever the shuffle produced.)

Performance:
    Values are rendered and hashed once per *distinct* value of a column, so
    hashing cost scales with column cardinality rather than row count. The
    exception is floating point columns, whose Java rendering has no
    vectorized equivalent: a float column costs one rendering per distinct
    bit pattern, which for a near-all-distinct million-row float column is
    seconds, not milliseconds.

Fast paths:
    Both hashing functions restrict hashing and sorting to the rows that can
    actually be truncated. Let ``G`` be the set of groups whose size exceeds
    the threshold, and let ``S`` be the rows belonging to a group in ``G``.
    The fast path computes the truncation on ``S`` alone and keeps every row
    outside ``S``. This is exact:

    1. Rows outside ``S`` all survive the full path. A group of size
       ``m <= threshold`` contributes its first ``min(m, threshold) = m``
       rows in the hash order -- i.e. all of them -- regardless of what that
       order is. So restricting attention to ``S`` cannot change their fate.
    2. ``S`` is a union of whole groups, by construction.
    3. The duplicate-row salt is group-local (the salt-locality step). The
       salt is a cumulative count over a partition of *every* column. Two
       rows in the same all-columns partition agree on every column, and the
       grouping columns are a subset of the frame's columns (they must be,
       or indexing raises), so they agree on the grouping columns and lie in
       the same group. Each all-columns partition is therefore contained in
       a single group, which by (2) is either wholly inside ``S`` or wholly
       outside it. The count follows frame order, and taking a subsequence
       preserves relative order, so every row in ``S`` gets the same salt it
       would have got from the full frame. The partition itself is intrinsic
       to the values, so computing it on ``S``'s rows directly gives the
       same partition as restricting a full-frame computation.
    4. The digest depends only on a row's own values and its salt, both
       unchanged by (3).
    5. The order and the grouping are computed on the full frame and then
       restricted (the restriction step). The order keys and the grouping
       columns' codes are evaluated over all rows *before* ``S`` is chosen,
       and the resulting arrays are indexed with ``S``'s positions. The keys
       are therefore literally the same numbers the full path would compare,
       so the order induced on ``S`` is the full path's order restricted to
       ``S``, and the stable sort breaks ties by position in ``S``, which
       preserves relative input order exactly as it does in the full path.
       In particular this holds even for mixed-type object columns, where
       the ordering falls back to a type-name key: that fallback is also
       decided once, over the whole frame.
    6. Rank is a within-group prefix. For a group ``g`` in ``G``, the rows of
       ``g`` in the restricted order are exactly the rows of ``g`` in the
       full order, so the first ``threshold`` of them are the same rows.

    Hence the surviving multiset is identical. With rows returned in input
    order, the surviving *frame* is identical, which is what
    ``test_fast_path_matches_full_path`` asserts.

    For :func:`limit_keys_per_group` the same argument applies with "rank"
    replaced by ``dense_rank`` over distinct (group, key) pairs, plus one
    extra step: the budget test uses a *refinement* of Spark's pair identity,
    so the per-group pair count is an over-estimate. An over-estimate can
    only put a group *into* ``G`` that did not need to be there (correct,
    just slower); it can never leave a group out, because refined count
    ``>=`` true count means refined count ``<= threshold`` implies true count
    ``<= threshold``, and such a group keeps all of its keys and hence all of
    its rows.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import datetime
import hashlib
import math
from decimal import ROUND_HALF_EVEN, Context, Decimal
from typing import (
    Any,
    Callable,
    Collection,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np
import pandas as pd

from tmlt.core.utils.pandas_grouping import (
    _NAN_ORDER,
    _NULL_ORDER,
    _SUPPORTED_FLOAT_DTYPES,
    _VALUE_ORDER,
    _canonical_array,
    _column_class,
    _column_values,
    _ColumnClass,
    _dense_codes,
    _FactorizeMemo,
    _first_occurrences,
    _group_codes,
    _group_ids,
    _group_key,
    _is_null,
    _memoized_factorization,
    _memoized_null_and_nan_masks,
    _microsecond_keys,
    _nullable_int_values,
    _object_kind,
    _reindexed_from_zero,
    _require_columns,
)

_UNSUPPORTED_EXTENSION_DTYPES = (
    pd.CategoricalDtype,
    pd.IntervalDtype,
    pd.PeriodDtype,
    pd.SparseDtype,
)

#: Code marking a null value in a digest-code array. A null has no digest and
#: is skipped by the combiner, so it cannot share a code with any real value.
#: It coincides with the sentinel ``pd.factorize`` uses for missing values.
_NULL_DIGEST_CODE = -1

#: Object-column kinds whose non-NA values are all renderable, so that
#: validation needs no per-value scan (the NA-like values, which the
#: ``skipna=True`` kind inference cannot see, are always checked separately,
#: and ``string`` keeps the batched UTF-8 encodability check of
#: :func:`_validate_string_uniques`).
#: ``datetime`` is deliberately absent, because a timezone-aware datetime must
#: still be rejected, and so is ``date``, which also covers columns mixing
#: dates with (possibly timezone-aware) datetimes; both keep a scan, at one
#: rendering per distinct value. ``floating`` also covers ``np.float16`` and
#: ``np.longdouble`` values that have no Spark rendering, and equal floats of
#: different widths may differ in renderability, so it keeps the full scan.
_RENDERABLE_OBJECT_KINDS = frozenset({"string", "bytes", "integer", "empty"})

#: Character budget for one batch of :func:`_validate_string_uniques`'s
#: join-and-encode check. Batching is what bounds the check's peak scratch
#: memory: the joined string and its UTF-8 encoding each cost at most four
#: bytes per character (UCS-4 string storage, four-byte UTF-8 sequences), so
#: one batch allocates at most ~8 bytes per budgeted character -- a few tens
#: of MiB -- no matter how much distinct-string content the column holds.
#: A single value longer than the whole budget forms a batch of its own,
#: degrading the bound only to the size of the largest single string, which
#: the column already stores. Tests lower this to exercise the batching.
_UTF8_VALIDATION_BATCH_CHARS = 4 * 1024 * 1024

#: Whether the fast paths that restrict hashing to the rows that can actually
#: be truncated are enabled. Tests set this to False to check that the fast
#: and full paths produce identical frames.
_FAST_PATH_ENABLED = True

#: The context for the decimal scaling in :func:`_two_significant_digits`.
#: Decimal operations otherwise consult the calling thread's active context,
#: so a caller that had lowered its precision would silently change the
#: renderings -- and therefore the digests, and hence which rows survive --
#: of the smallest subnormals. The precision exceeds the 767 significant
#: digits of the longest exact decimal expansion of a double, so the scaling
#: this context governs is always exact and the explicit half-even rounding
#: to two digits stays the only rounding that ever happens.
_EXACT_DECIMAL_CONTEXT = Context(prec=800)


def _layout_java_decimal(sign: str, digits: str, decimal_exponent: int) -> str:
    """Returns the Java rendering of ``sign`` ``0.<digits>`` * 10 ** exponent."""
    # Java uses plain notation for magnitudes in [1e-3, 1e7), and computerized
    # scientific notation everywhere else.
    if -2 <= decimal_exponent <= 7:
        if decimal_exponent <= 0:
            return sign + "0." + "0" * (-decimal_exponent) + digits
        if decimal_exponent >= len(digits):
            padding = "0" * (decimal_exponent - len(digits))
            return sign + digits + padding + ".0"
        return sign + digits[:decimal_exponent] + "." + digits[decimal_exponent:]
    mantissa = digits[0] + "." + (digits[1:] or "0")
    return sign + mantissa + "E" + str(decimal_exponent - 1)


def _shortest_digits(text: str) -> Tuple[str, int]:
    """Returns the significant digits and decimal exponent of a decimal string.

    The value is ``0.<digits> * 10 ** exponent``. Trailing zeros are stripped
    because they are not significant: for example ``repr(5152716558868863.0)``
    is ``'5152716558868863.0'``, whose final zero must not be counted.
    """
    as_tuple = Decimal(text).as_tuple()
    digits = "".join(map(str, as_tuple.digits)).rstrip("0") or "0"
    return digits, int(as_tuple.exponent) + len(as_tuple.digits)


def _two_significant_digits(exact: Decimal) -> Tuple[str, int]:
    """Returns the two-digit decimal closest to ``exact``, and its exponent.

    Java picks the shortest decimal that round-trips, except that when a single
    digit suffices it instead picks whichever decimal with one or two digits is
    closest to the exact value, breaking ties towards an even last digit. This
    only ever changes the result for tiny subnormal values, where the gap
    between adjacent floating point numbers is large relative to their
    magnitude: ``Double.MIN_VALUE`` renders as ``4.9E-324``, not ``5.0E-324``.

    The computation must not depend on the ambient decimal context, which a
    caller may have narrowed or armed with traps: the callers convert with
    ``Decimal.from_float``, which is exact like the constructor but exempt
    from the ``FloatOperation`` trap, and the scaling carries its own
    high-precision context (see :data:`_EXACT_DECIMAL_CONTEXT`) so that an
    installed low precision cannot round the exact value before the
    half-even rounding below does.
    """
    decimal_exponent = exact.adjusted() + 1
    scaled = exact.scaleb(2 - decimal_exponent, context=_EXACT_DECIMAL_CONTEXT)
    significand = int(scaled.to_integral_value(rounding=ROUND_HALF_EVEN))
    if significand >= 100:
        significand //= 10
        decimal_exponent += 1
    return (str(significand).rstrip("0") or "0"), decimal_exponent


def _java_double_to_string(value: float) -> str:
    """Returns the value as Java's ``Double.toString`` renders it.

    The argument must be finite: infinities and NaNs are special-cased by
    :func:`_render_value` before it reaches this function, exactly as the Spark
    implementation special-cases them before casting to a string.
    """
    if value == 0.0:
        return "-0.0" if math.copysign(1.0, value) < 0 else "0.0"
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    digits, decimal_exponent = _shortest_digits(repr(magnitude))
    if len(digits) == 1:
        digits, decimal_exponent = _two_significant_digits(
            Decimal.from_float(magnitude)
        )
    return _layout_java_decimal(sign, digits, decimal_exponent)


def _java_float_to_string(value: np.float32) -> str:
    """Returns the value as Java's ``Float.toString`` renders it.

    The argument must be finite, for the same reason as in
    :func:`_java_double_to_string`.
    """
    if value == np.float32(0.0):
        return "-0.0" if math.copysign(1.0, float(value)) < 0 else "0.0"
    sign = "-" if value < np.float32(0.0) else ""
    magnitude = np.float32(abs(float(value)))
    digits, decimal_exponent = _shortest_digits(
        np.format_float_positional(magnitude, unique=True, trim="-")
    )
    if len(digits) == 1:
        digits, decimal_exponent = _two_significant_digits(
            Decimal.from_float(float(magnitude))
        )
    return _layout_java_decimal(sign, digits, decimal_exponent)


def _sha256(data: bytes) -> str:
    """Returns the hex-encoded SHA-256 digest of the given bytes."""
    return hashlib.sha256(data).hexdigest()


def _render_value(value: Any) -> Optional[bytes]:
    """Renders a single value as the bytes Spark hashes for it.

    Returns:
        The rendering Spark's ``_hash_column`` would hash, or None if the value
        is null.
    """
    if _is_null(value):
        return None
    # isinstance(True, int) is True, so booleans must be rejected before the
    # integer branch is reached.
    if isinstance(value, (bool, np.bool_)):
        raise NotImplementedError("Unsupported data type bool")
    if isinstance(value, np.float32):
        # np.float32 is not a subclass of float, so it must be dispatched
        # before the general float branch.
        if np.isnan(value):
            return b"nan"
        if np.isinf(value):
            return b"-inf" if value < 0 else b"inf"
        return _java_float_to_string(value).encode("utf-8")
    if isinstance(value, (float, np.float64)):
        if math.isnan(value):
            return b"nan"
        if math.isinf(value):
            return b"-inf" if value < 0 else b"inf"
        return _java_double_to_string(float(value)).encode("utf-8")
    if isinstance(value, (int, np.integer)):
        return str(int(value)).encode("utf-8")
    if isinstance(value, str):
        try:
            return value.encode("utf-8")
        except UnicodeEncodeError as error:
            # A str holding an unpaired surrogate has no UTF-8 encoding to
            # hash (see the module docstring). Validation rejects such
            # strings before anything is hashed; converting here as well
            # keeps every other route from leaking a UnicodeEncodeError
            # where callers expect NotImplementedError.
            raise NotImplementedError(
                "Unsupported string value that cannot be encoded as UTF-8"
            ) from error
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    # datetime.datetime is a subclass of datetime.date, so it must be
    # dispatched first.
    if isinstance(value, datetime.datetime):
        if value.tzinfo is not None:
            raise NotImplementedError(
                "Unsupported data type timezone-aware datetime; convert "
                "timestamps to wall-clock values first, for example with "
                "series.dt.tz_convert('UTC').dt.tz_localize(None)"
            )
        rendered = (
            f"{value.year:04d}-{value.month:02d}-{value.day:02d} "
            f"{value.hour:02d}:{value.minute:02d}:{value.second:02d}"
        )
        # Spark prints as many fractional digits as are needed, and none at all
        # when the fraction is zero. Anything finer than a microsecond is
        # discarded.
        if value.microsecond:
            rendered += "." + f"{value.microsecond:06d}".rstrip("0")
        return rendered.encode("utf-8")
    if isinstance(value, datetime.date):
        return value.isoformat().encode("utf-8")
    raise NotImplementedError(f"Unsupported data type {type(value).__name__}")


def _hash_value(value: Any) -> Optional[str]:
    """Hashes a single value the way Spark's ``_hash_column`` hashes it.

    Returns:
        The hex-encoded SHA-256 digest of the value's Spark string rendering,
        or None if the value is null.
    """
    rendered = _render_value(value)
    return None if rendered is None else _sha256(rendered)


def _combine_digests(digests: Sequence[Optional[str]]) -> str:
    """Combines the per-value digests of one row into that row's digest.

    This mirrors Spark's ``_hash_columns``: the per-value digests are joined
    with commas, skipping nulls, and that string is hashed, and its digest
    hashed once more.

    This is the choke point every combined hash flows through, and the seam
    that four hash-collision regression tests in
    ``test.unit.utils.test_pandas_truncation`` replace with a constant. It
    must stay the single point every row's digest passes through: inlining it
    would leave those tests patching a function nothing calls.

    Returns:
        The hex-encoded SHA-256 digest for the given per-value digests.
    """
    # hashlib.sha256 is called directly rather than through _sha256: this runs
    # once per row, and the extra Python call is measurable at large sizes.
    sha256 = hashlib.sha256
    concatenated = sha256(
        ",".join(digest for digest in digests if digest is not None).encode("utf-8")
    ).hexdigest()
    return sha256(concatenated.encode("utf-8")).hexdigest()


def _combined_hash(values: Sequence[Any]) -> str:
    """Combines the hashes of ``values`` into a single hash.

    Returns:
        The hex-encoded SHA-256 digest for the given values.
    """
    return _combine_digests([_hash_value(value) for value in values])


def _validate_column(
    column: pd.Series, name: str, memo: Optional["_FactorizeMemo"] = None
) -> None:
    """Raises an error if the column has a dtype that cannot be hashed.

    The optional memo shares this validation's factorizations and null/NaN
    masks with the other consumers of the same call (see
    :class:`_FactorizeMemo`); without one, everything is computed here. The
    errors raised, and when they are raised, are identical either way: the
    memo only ever changes how often an array is factorized.
    """
    dtype = column.dtype
    message = f"Unsupported data type {dtype} for column {name}"
    if isinstance(dtype, pd.DatetimeTZDtype):
        raise NotImplementedError(
            f"{message}; convert timestamps to wall-clock values first, for "
            "example with series.dt.tz_convert('UTC').dt.tz_localize(None)"
        )
    # These have to be rejected up front: a categorical dtype whose categories
    # are integers, for example, passes the integer check below.
    if isinstance(dtype, _UNSUPPORTED_EXTENSION_DTYPES):
        raise NotImplementedError(message)
    if pd.api.types.is_bool_dtype(dtype):
        raise NotImplementedError(message)
    if pd.api.types.is_integer_dtype(dtype):
        return
    if dtype in _SUPPORTED_FLOAT_DTYPES or isinstance(
        dtype, (pd.Float32Dtype, pd.Float64Dtype)
    ):
        return
    if pd.api.types.is_datetime64_dtype(dtype):
        return
    if isinstance(dtype, pd.StringDtype):
        # A string dtype can hold nothing but str and NA, yet a str is not
        # always renderable: an unpaired surrogate has no UTF-8 encoding.
        _validate_string_uniques(
            _memoized_factorization(column, _ColumnClass.STRING, memo)[1], name
        )
        return
    if pd.api.types.is_object_dtype(dtype):
        # An object column has no type of its own, so its values have to be
        # checked. An empty one carries no values and so cannot be checked,
        # unlike the Spark schema it corresponds to. When the column's
        # inferred kind proves that every value it can hold is renderable,
        # the per-value scan is skipped -- except at the positions the
        # ``skipna=True`` kind inference skipped: an NA-like value with no
        # Spark rendering, such as ``np.float16("nan")``, is invisible to
        # every kind, so the values classified as NaNs are still rendered.
        # A genuine float NaN renders as ``b"nan"``; anything else raises
        # here exactly as it does on the full scan.
        kind = _object_kind(column)
        if kind in _RENDERABLE_OBJECT_KINDS:
            values = column.to_numpy()
            if kind == "string":
                # str values are renderable but not always encodable: an
                # unpaired surrogate has no UTF-8 encoding. The other kinds
                # in the accept list cannot hold a str. The ``string`` kind
                # is one of _HOMOGENEOUS_OBJECT_KINDS, so the distinct
                # values are its canonical factorization's uniques.
                _validate_string_uniques(
                    _memoized_factorization(
                        column, _ColumnClass.HOMOGENEOUS_OBJECT, memo
                    )[1],
                    name,
                )
            _render_nan_classified_values(
                values, _memoized_null_and_nan_masks(column, memo)[1]
            )
            return
        if kind in ("date", "datetime"):
            # Every value is a date or a (possibly timezone-aware) datetime,
            # where two equal values always render identically or both
            # raise -- a timezone-aware datetime never equals a naive one --
            # so rendering one value per distinct value is exactly the full
            # scan, at one rendering per *distinct* date rather than per row.
            # The first failing distinct value, in order of first appearance,
            # is the first failing value of the full scan, so the error is
            # the same one. NA-like values are invisible to the
            # factorization, as they are to the kind, and are checked as
            # above.
            values = column.to_numpy()
            for value in pd.factorize(values)[1]:
                _render_value(value)
            _render_nan_classified_values(
                values, _memoized_null_and_nan_masks(column, memo)[1]
            )
            return
        for value in _column_values(column):
            _render_value(value)
        return
    raise NotImplementedError(message)


def _render_nan_classified_values(values: np.ndarray, nan_mask: np.ndarray) -> None:
    """Renders every value of an object array that classifies as a NaN.

    This is the validation for the values ``infer_dtype(skipna=True)`` cannot
    see. The null-classified values need no check -- they render as ``None``
    whatever they are -- but a NaN-classified value is hashed as a value, so
    it must render: a float NaN renders as ``b"nan"``, and an NA-like value
    with no Spark rendering, such as ``np.float16("nan")`` or a stray
    ``np.datetime64("NaT")``, raises :class:`NotImplementedError`.

    Args:
        values: The object array whose NaN-classified values are rendered.
        nan_mask: The array's NaN mask, as
            :func:`tmlt.core.utils.pandas_grouping._null_and_nan_masks` returns
            it -- possibly shared through a :class:`_FactorizeMemo`.
    """
    for position in np.flatnonzero(nan_mask):
        _render_value(values[position])


def _validate_string_uniques(uniques: Sequence[str], name: str) -> None:
    """Raises unless every distinct string value can be encoded as UTF-8.

    A Python ``str`` can hold an unpaired surrogate -- ``os.fsdecode``,
    ``surrogateescape`` decoding, and JSON with escaped surrogates all
    produce them -- and such a string has no UTF-8 encoding to hash. Spark
    instead coerces surrogates to U+FFFD at ingest, which makes two strings
    that differ only in their surrogates one value to Spark: one partition,
    one sort position. Matching that here would mean rewriting caller data
    before rendering, grouping and ordering alike, so such strings are
    rejected up front instead (see the module docstring) -- from validation,
    which both the fast and the full path run before choosing which rows to
    hash, so the two paths cannot disagree about the error.

    The check costs one C-level join and encode per *batch* of the
    *distinct* values. Python's strict UTF-8 encoder rejects each surrogate
    code point by itself -- it never pairs adjacent ones -- so a
    concatenation encodes exactly when every value in it does, and splitting
    the values into batches changes nothing about what is accepted. The
    batches are capped at :data:`_UTF8_VALIDATION_BATCH_CHARS` characters so
    that the joined string and its encoding stay bounded scratch allocations
    however large the column's total distinct-string content is; only a
    failing batch is re-scanned one value at a time (see
    :func:`_encode_string_batch`). NA-like values are invisible to the
    factorization that produced the uniques, exactly as they are in
    :func:`_validate_column`'s date branch.

    Args:
        uniques: The distinct values of a column whose non-NA values are all
            ``str``, as the uniques of its canonical factorization.
        name: The column's name, for the error message.
    """
    # sum(map(len, ...)) runs the length scan at C speed, so the common case
    # -- everything fits one batch -- stays the single join and encode, with
    # one pass of len() calls on top and no per-value Python loop.
    if sum(map(len, uniques)) <= _UTF8_VALIDATION_BATCH_CHARS:
        _encode_string_batch(uniques, name)
        return
    batch: List[str] = []
    batch_chars = 0
    for value in uniques:
        # Flushing before the append keeps every batch within the budget;
        # only a value that alone exceeds it becomes an oversized batch of
        # one, and the column already stores that value.
        if batch and batch_chars + len(value) > _UTF8_VALIDATION_BATCH_CHARS:
            _encode_string_batch(batch, name)
            batch = []
            batch_chars = 0
        batch.append(value)
        batch_chars += len(value)
    if batch:
        _encode_string_batch(batch, name)


def _encode_string_batch(batch: Sequence[str], name: str) -> None:
    """Raises unless one batch of string values can be encoded as UTF-8.

    This is the join-and-encode step of :func:`_validate_string_uniques`. A
    batch fails exactly when some value in it fails (the strict encoder never
    pairs surrogates across values), so a failing batch is re-scanned one
    value at a time and the error chained from the offending value's own
    ``UnicodeEncodeError`` rather than from an offset inside the
    concatenation. The batch's own error is kept as the cause only for the
    impossible case where no single value fails, so a failed batch can never
    pass silently.

    Args:
        batch: The string values to encode, as one concatenation.
        name: The column's name, for the error message.
    """
    try:
        "".join(batch).encode("utf-8")
        return
    except UnicodeEncodeError as batch_error:
        error: UnicodeEncodeError = batch_error
        for value in batch:
            try:
                value.encode("utf-8")
            except UnicodeEncodeError as value_error:
                error = value_error
                break
        raise NotImplementedError(
            f"Unsupported string value in column {name}; the value cannot "
            "be encoded as UTF-8, which usually means it contains an "
            "unpaired surrogate"
        ) from error


def _digest_codes(
    column: pd.Series, memo: Optional[_FactorizeMemo] = None
) -> Optional[Tuple[np.ndarray, Sequence[Any]]]:
    """Returns a factorization of a column that never merges distinct renderings.

    The contract is one-directional: two rows sharing a code must render to
    the same bytes, but two rows that render alike may still get different
    codes. That makes over-splitting harmless and lets floating point columns
    be factorized by bit pattern, which is what keeps ``0.0`` and ``-0.0``
    apart.

    Args:
        column: The column to factorize.
        memo: A per-call memo sharing the canonical factorizations and masks
            with the other consumers, or None to compute everything here.

    Returns:
        The per-row codes, with :data:`_NULL_DIGEST_CODE` marking nulls,
        together with one representative value per non-negative code -- at the
        precision of the column's dtype, as :func:`_column_values` yields it.
        Returns None when the column's dtype has no faithful factorization, in
        which case the caller renders every value.
    """
    dtype = column.dtype
    klass = _column_class(column)
    if klass is _ColumnClass.NULLABLE_FLOAT:
        float_dtype, bits_dtype = (
            (np.float32, np.int32)
            if isinstance(dtype, pd.Float32Dtype)
            else (np.float64, np.int64)
        )
        bits = column.to_numpy(float_dtype, na_value=0.0).view(bits_dtype)
        codes, uniques = pd.factorize(bits)
        codes[column.isna().to_numpy()] = _NULL_DIGEST_CODE
        return codes, uniques.view(float_dtype)
    if klass is _ColumnClass.NULLABLE_INT:
        codes, uniques = _memoized_factorization(column, klass, memo)
        codes[column.isna().to_numpy()] = _NULL_DIGEST_CODE
        return codes, uniques
    if klass is _ColumnClass.STRING:
        # pd.factorize marks the None positions with -1, which is exactly
        # _NULL_DIGEST_CODE; a string dtype can hold no NaN value.
        return _memoized_factorization(column, klass, memo)
    if klass is _ColumnClass.NUMPY_INT:
        return _memoized_factorization(column, klass, memo)
    if klass is _ColumnClass.NUMPY_FLOAT:
        # Factorizing the bit pattern separates 0.0 from -0.0, which render
        # differently, and splits NaN payloads, which is a harmless
        # over-split. The representatives keep the column's dtype: a float32
        # rendered with the double formatter would gain digits the float
        # never had.
        bits_dtype = np.int32 if dtype == np.dtype("float32") else np.int64
        codes, uniques = pd.factorize(column.to_numpy().view(bits_dtype))
        return codes, uniques.view(dtype)
    if klass is _ColumnClass.DATETIME:
        # The factorization stays in the column's own unit: converting to
        # nanoseconds first would silently wrap values outside the nanosecond
        # range, which non-nanosecond columns (pandas 2) can hold.
        values = column.to_numpy()
        codes, uniques = pd.factorize(values.view("int64"))
        codes[column.isna().to_numpy()] = _NULL_DIGEST_CODE
        # Sub-microsecond precision is deliberately kept: two values
        # rendering alike may get different codes, which merely over-splits.
        return codes, [pd.Timestamp(value) for value in uniques.view(values.dtype)]
    if klass is _ColumnClass.HOMOGENEOUS_OBJECT:
        try:
            codes, uniques = _memoized_factorization(column, klass, memo)
        except TypeError:
            # A value pd.factorize cannot hash, such as a bytearray.
            return None
        representatives = list(uniques)
        # The null positions keep pd.factorize's missing-value code, which is
        # exactly _NULL_DIGEST_CODE, so only the NaN mask is needed here.
        _, nan_mask = _memoized_null_and_nan_masks(column, memo)
        # pd.factorize treats a float NaN in an object array as missing, but
        # here it is a value that hashes to sha256(b"nan"), unlike a null,
        # which contributes nothing; only the null positions may keep the
        # missing-value code.
        if nan_mask.any():
            codes[nan_mask] = len(representatives)
            representatives.append(float("nan"))
        return codes, representatives
    return None


class _ColumnDigests(NamedTuple):
    """The per-row digests of one column.

    Attributes:
        digests: An object array of hex digests aligned with the column,
            holding None wherever the column holds a null.
        has_null: Whether any digest is None. This is known for free from the
            digest codes, and saves the combiner a full null scan.
    """

    digests: np.ndarray
    has_null: bool


def _column_digests(
    column: pd.Series, memo: Optional[_FactorizeMemo] = None
) -> _ColumnDigests:
    """Hashes every value of a column, hashing each distinct value once.

    Args:
        column: The column to hash.
        memo: A per-call memo sharing the canonical factorizations and masks
            with the other consumers, or None to compute everything here. A
            memo may only be passed for a column of the frame its call is
            truncating, never for a selection of that frame's rows (see
            :class:`_FactorizeMemo`).

    Returns:
        The per-row digests, and whether any of them is None.
    """
    codes_and_values = _digest_codes(column, memo)
    if codes_and_values is None:
        # No faithful factorization exists, so every value is rendered, as it
        # was before deduplication.
        rendered = [_hash_value(value) for value in _column_values(column)]
        return _ColumnDigests(
            np.array(rendered, dtype=object),
            any(digest is None for digest in rendered),
        )
    codes, values = codes_and_values
    # Every distinct value is rendered, so an unsupported value in an object
    # column still raises exactly as it does on the fallback path above;
    # deduplication cannot hide an error.
    digests = np.empty(len(values) + 1, dtype=object)  # slot 0 is the null slot
    digests[0] = None
    digests[1:] = [_hash_value(value) for value in values]
    return _ColumnDigests(digests[codes + 1], bool((codes == _NULL_DIGEST_CODE).any()))


def _row_digests(columns: Sequence[_ColumnDigests], n_rows: int) -> np.ndarray:
    """Combines per-column digests into one digest per row.

    Args:
        columns: One entry per hashed column, as :func:`_column_digests`
            returns them.
        n_rows: The number of rows, which is what fixes the result's length
            when there are no columns at all.

    Returns:
        An object array of hex digests, one per row.
    """
    if not columns:
        return np.full(n_rows, _combine_digests(()), dtype=object)
    combine = _combine_digests
    arrays = [column.digests for column in columns]
    if not any(column.has_null for column in columns):
        # The null-filtering comprehension below costs about 0.25 s per
        # million rows; skip it when no column holds a null.
        return np.array([combine(row) for row in zip(*arrays)], dtype=object)
    return np.array(
        [
            combine([digest for digest in row if digest is not None])
            for row in zip(*arrays)
        ],
        dtype=object,
    )


def _hash_columns(df: pd.DataFrame, columns: List[str]) -> pd.Series:
    """Hashes the given columns of every row into a single value.

    The truncation functions inline these steps around their fast paths, so
    this composition survives as the reference implementation the golden
    vectors and the hash-agreement tests pin.

    Returns:
        A series of hex-encoded SHA-256 digests, aligned with ``df``.
    """
    for column in columns:
        _validate_column(df[column], column)
    hashes = _row_digests([_column_digests(df[column]) for column in columns], len(df))
    return pd.Series(hashes, index=df.index, dtype=object)


def _sorted_keys(keys: Set[Tuple[int, Any]]) -> List[Tuple[int, Any]]:
    """Returns a column's group keys in Spark's ascending order.

    Returns:
        The keys, sorted.
    """
    try:
        return sorted(keys)
    except TypeError:
        # A column holding values of several types has no Spark counterpart,
        # since a Spark column has a single type. Falling back to ordering such
        # values by type name keeps the sort deterministic rather than failing.
        return sorted(keys, key=lambda key: (key[0], type(key[1]).__name__, key[1]))


class _OrderKeys(NamedTuple):
    """Lexsort keys reproducing Spark's ascending order for one column.

    Attributes:
        classes: One of :data:`_NULL_ORDER`, :data:`_VALUE_ORDER` or
            :data:`_NAN_ORDER` per row, or None when every row holds an
            ordinary value and the class is therefore constant.
        values: A per-row key whose ascending order is Spark's, compared only
            between rows of the same class.
    """

    classes: Optional[np.ndarray]
    values: np.ndarray


def _order_classes(
    null_mask: Optional[np.ndarray], nan_mask: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """Returns the per-row order class, or None when every row is a value.

    Args:
        null_mask: The mask of null rows, or None when there can be none.
        nan_mask: The mask of NaN rows, or None when there can be none.

    Returns:
        An int8 array of order classes, or None when it would be constant.
    """
    has_nulls = null_mask is not None and null_mask.any()
    has_nans = nan_mask is not None and nan_mask.any()
    if not has_nulls and not has_nans:
        return None
    length = len(null_mask if null_mask is not None else nan_mask)  # type: ignore[arg-type]
    classes = np.full(length, _VALUE_ORDER, dtype=np.int8)
    if has_nulls:
        classes[null_mask] = _NULL_ORDER
    if has_nans:
        classes[nan_mask] = _NAN_ORDER
    return classes


def _dense_ranks(values: np.ndarray) -> np.ndarray:
    """Returns each value's dense rank in the ascending order of its uniques.

    The ranks are computed over the whole array, never over a subset, so
    restricting them to any subset of rows induces the same order there.
    Missing positions (the nulls and NaNs ``pd.factorize`` marks) rank zero;
    the caller's class key is what separates them from the values.

    Returns:
        An int64 array aligned with ``values``.
    """
    codes, _ = pd.factorize(values, sort=True)
    return np.where(codes < 0, 0, codes).astype(np.int64, copy=False)


def _dense_ranks_from_factorization(
    factorization: Tuple[np.ndarray, Sequence[Any]],
) -> np.ndarray:
    """Returns :func:`_dense_ranks` of a canonical factorization's own array.

    ``pd.factorize(values, sort=True)`` is ``pd.factorize(values)`` with its
    codes remapped to the ascending order of its uniques, so ranking the
    *distinct* values and fanning the ranks back out over the codes gives the
    identical result. That is what lets a :class:`_FactorizeMemo`'s shared
    factorization serve the ordering too, at one sort of the uniques instead
    of another pass over every row.

    Args:
        factorization: The ``(codes, uniques)`` pair, as
            :func:`tmlt.core.utils.pandas_grouping._canonical_factorization`
            returns it, with the missing positions carrying ``pd.factorize``'s
            sentinel.

    Returns:
        An int64 array aligned with the codes.

    Raises:
        TypeError: When the uniques have no order, exactly as
            :func:`_dense_ranks` raises it for the same values.
    """
    codes, uniques = factorization
    # Slot 0 is the missing-value slot, which ranks zero as it does in
    # _dense_ranks; the caller's class key is what separates it from the
    # values, one of which may rank zero too.
    ranks = np.zeros(len(uniques) + 1, dtype=np.int64)
    ranks[1:] = pd.factorize(uniques, sort=True)[0]
    return ranks[codes + 1]


def _memoized_dense_ranks(
    column: pd.Series, klass: _ColumnClass, memo: Optional[_FactorizeMemo]
) -> np.ndarray:
    """Returns the dense ranks of a column's canonical array, through the memo.

    Without a memo the array is ranked directly; with one, the ranks come out
    of the shared canonical factorization, so ordering costs no factorization
    of its own.

    Raises:
        TypeError: When the column has no faithful factorization, or its
            values have no order -- the two failures :func:`_dense_ranks`
            raises, and which :func:`_order_keys`'s object branch falls back
            on either way.
    """
    if memo is None:
        return _dense_ranks(_canonical_array(column, klass))
    return _dense_ranks_from_factorization(memo.factorization(column, klass))


def _fallback_order_keys(column: pd.Series) -> _OrderKeys:
    """Returns order keys for a column with no vectorized ordering.

    The ranks are those of every row's full :func:`_group_key`, in
    :func:`_sorted_keys` order, so mixed-type object columns keep the exact
    deterministic order (including the type-name fallback) they had when the
    per-value path was the only path. The class is part of the key, so no
    separate class array is needed.

    Returns:
        The order keys for the column.
    """
    keys = [_group_key(value) for value in _column_values(column)]
    ranks = {key: rank for rank, key in enumerate(_sorted_keys(set(keys)))}
    return _OrderKeys(None, np.array([ranks[key] for key in keys], dtype=np.int64))


def _order_keys(column: pd.Series, memo: Optional[_FactorizeMemo] = None) -> _OrderKeys:
    """Returns the sort keys ordering a column the way Spark orders it.

    The keys are absolute -- derived from the values themselves, or from
    dense ranks over the whole column -- so restricting them to a subset of
    the rows induces the same order on that subset. That is what lets the
    fast path compute them once, before deciding which rows to hash.

    Args:
        column: The column to order.
        memo: A per-call memo sharing the canonical factorizations and masks
            with the other consumers, or None to compute everything here.
            Because the ranks must be those of the whole column, only the
            frame being truncated may be ordered against its own memo (see
            :class:`_FactorizeMemo`).

    Returns:
        The class and value keys for the column.
    """
    dtype = column.dtype
    klass = _column_class(column)
    if klass is _ColumnClass.NULLABLE_FLOAT:
        float_dtype = np.float32 if isinstance(dtype, pd.Float32Dtype) else np.float64
        floats = column.to_numpy(float_dtype, na_value=np.nan)
        nans = np.isnan(floats)
        null_mask = column.isna().to_numpy()
        values = np.where(nans, float_dtype(0.0), floats)
        return _OrderKeys(_order_classes(null_mask, nans & ~null_mask), values)
    if klass is _ColumnClass.NULLABLE_INT:
        null_mask = column.isna().to_numpy()
        values = _nullable_int_values(column)
        return _OrderKeys(_order_classes(null_mask, None), values)
    if klass is _ColumnClass.STRING:
        null_mask = column.isna().to_numpy()
        return _OrderKeys(
            _order_classes(null_mask, None), _memoized_dense_ranks(column, klass, memo)
        )
    if klass is _ColumnClass.NUMPY_INT:
        # Any monotone key works; the raw integers are one.
        return _OrderKeys(None, column.to_numpy())
    if klass is _ColumnClass.NUMPY_FLOAT:
        floats = column.to_numpy()
        nan_mask = np.isnan(floats)
        # -0.0 and 0.0 compare equal in the value key, and the sort is
        # stable, which is exactly the tie _group_key gives them.
        values = np.where(nan_mask, floats.dtype.type(0.0), floats)
        return _OrderKeys(_order_classes(None, nan_mask), values)
    if klass is _ColumnClass.DATETIME:
        null_mask = column.isna().to_numpy()
        values = np.where(null_mask, np.int64(0), _microsecond_keys(column))
        return _OrderKeys(_order_classes(null_mask, None), values)
    if klass is _ColumnClass.HOMOGENEOUS_OBJECT:
        null_mask, nan_mask = _memoized_null_and_nan_masks(column, memo)
        try:
            values = _memoized_dense_ranks(column, klass, memo)
        except TypeError:
            return _fallback_order_keys(column)
        return _OrderKeys(_order_classes(null_mask, nan_mask), values)
    return _fallback_order_keys(column)


def _digest_order_key(digests: np.ndarray) -> np.ndarray:
    """Returns the sort key ordering a column of hex digests.

    Every digest is 64 ASCII characters, so a fixed-width bytes array orders
    them exactly as Python orders the strings, and compares them in C rather
    than through the Python object protocol.

    Returns:
        An ``S64`` array aligned with ``digests``.
    """
    return digests.astype("S64")


def _tie_break_keys(
    order_keys: Mapping[str, _OrderKeys], columns: Sequence[str], take: np.ndarray
) -> List[np.ndarray]:
    """Returns the tie-breaking lexsort keys for ``columns``, taken at ``take``.

    numpy's lexsort takes the last key as the primary one, so the keys run
    from the last tie-breaking column upward, each column contributing its
    value key and, above it, its class key when one exists. The caller
    supplies the digest key as the primary key.

    Args:
        order_keys: The per-column order keys, computed over the full frame.
        columns: The tie-breaking columns, from highest to lowest priority.
        take: The positions of the rows being sorted in the keys' frame.

    Returns:
        The lexsort keys, in increasing order of priority.
    """
    keys: List[np.ndarray] = []
    for column in reversed(list(columns)):
        order_key = order_keys[column]
        keys.append(order_key.values[take])
        if order_key.classes is not None:
            keys.append(order_key.classes[take])
    return keys


def _hash_sort_order(
    tie_keys: Callable[[], Sequence[np.ndarray]], digest_key: np.ndarray
) -> np.ndarray:
    """Returns the permutation sorting rows by digest, then by the tie keys.

    When every digest in the frame is distinct, the digest alone is a strict
    total order and the tie-breaking keys cannot matter, so the cheaper
    single-key sort is used and ``tie_keys`` is never called -- which is what
    lets callers defer building the order keys entirely in the common
    all-distinct case. The branch is exact, not probabilistic: it is taken
    only when no two rows share a digest, which an adjacent-duplicate check
    on the sorted digests decides. Duplicate digests do occur -- a null
    contributes nothing to the combined hash, so ``(NULL, "k1")`` and
    ``("k1", NULL)`` collide -- and then every key participates.

    Args:
        tie_keys: Returns the tie-breaking lexsort keys, in increasing order
            of priority. Called only when two rows share a digest.
        digest_key: The primary key, as :func:`_digest_order_key` returns it.

    Returns:
        The stable ascending permutation.
    """
    order = np.argsort(digest_key, kind="stable")
    sorted_key = digest_key[order]
    if not (sorted_key[1:] == sorted_key[:-1]).any():
        return order
    # numpy's lexsort takes the last key as the primary one, and is stable.
    return np.lexsort([*tie_keys(), digest_key])


def _prefix_ranks(ids: np.ndarray) -> np.ndarray:
    """Returns each element's one-based rank among prior elements of its id.

    This is the cumulative count Spark's windowed ``row_number`` produces
    once the rows stand in their final order.

    Returns:
        An int64 array aligned with ``ids``.
    """
    series = pd.Series(ids)
    return (series.groupby(series, sort=False).cumcount() + 1).to_numpy()


def _survivors_in_input_order(
    working_df: pd.DataFrame,
    selected: np.ndarray,
    kept_positions: np.ndarray,
) -> pd.DataFrame:
    """Returns the rows surviving a truncation, in input order.

    The survivors are every row the fast path left unselected, plus the
    selected rows at ``kept_positions``. Selecting with a mask returns them
    in input order; see the module docstring's "Row order" note.

    Args:
        working_df: The frame being truncated, with a default index.
        selected: The mask of rows the truncation considered.
        kept_positions: The positions of the considered rows that survived.

    Returns:
        The surviving frame, reindexed from zero.
    """
    survivors = np.zeros(len(working_df), dtype=bool)
    survivors[~selected] = True
    survivors[kept_positions] = True
    return _reindexed_from_zero(working_df.loc[survivors])


def truncate_large_groups(
    df: pd.DataFrame, grouping_columns: Collection[str], threshold: int
) -> pd.DataFrame:
    """Order rows by a hash function and keep at most ``threshold`` rows for each group.

    This is the pandas counterpart of
    :func:`tmlt.core.utils.truncation.truncate_large_groups`, and keeps the same
    rows as that function does.

    Example:
        ..
            >>> import pandas as pd
            >>> from tmlt.core.utils.misc import print_pandas
            >>> dataframe = pd.DataFrame(
            ...     {
            ...         "A": ["a1", "a2", "a3", "a3", "a3"],
            ...         "B": ["b1", "b1", "b2", "b2", "b3"],
            ...     }
            ... )

        >>> # Example input
        >>> print_pandas(dataframe)
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2
        4  a3  b3
        >>> print_pandas(truncate_large_groups(dataframe, ["A"], 3))
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2
        4  a3  b3
        >>> print_pandas(truncate_large_groups(dataframe, ["A"], 2))
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2
        >>> print_pandas(truncate_large_groups(dataframe, ["A"], 1))
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2

    Args:
        df: DataFrame to truncate.
        grouping_columns: Columns defining the groups.
        threshold: Maximum number of rows to include for each group.
    """
    starting_columns = list(df.columns)
    working_df = df.reset_index(drop=True)
    # One memo for the whole call: validation and the grouping columns' group
    # codes below factorize the same full-frame columns, and the memo runs
    # each canonical factorization once. Nothing computed on the fast path's
    # frame of selected rows may touch it -- those columns are different
    # (shorter) arrays under the same names.
    memo = _FactorizeMemo()
    for column in starting_columns:
        _validate_column(working_df[column], column, memo)
    n = len(working_df)
    # Spark accepts a repeated partitioning column; grouping by the same
    # column twice here would only be wasted work.
    grouping_unique = list(dict.fromkeys(grouping_columns))
    _require_columns(working_df, grouping_unique)
    if threshold <= 0 or n == 0:
        # Spark expresses the threshold as a filter, so a non-positive
        # threshold is an empty result, not an error.
        return working_df.iloc[:0].copy()
    group_code = {
        column: _group_codes(working_df[column], memo) for column in grouping_unique
    }
    group_ids = _group_ids([group_code[column] for column in grouping_unique], n)
    # The fast path: a group of size m <= threshold contributes its first
    # min(m, threshold) = m rows -- all of them -- whatever the hash order
    # turns out to be, so only the rows of oversized groups need hashing.
    # The module docstring's "Fast paths" section holds the full argument.
    sizes = np.bincount(group_ids)[group_ids]
    selected = sizes > threshold if _FAST_PATH_ENABLED else np.ones(n, dtype=bool)
    if not selected.any():
        # Every row survives. working_df is already private: the reset_index
        # at entry copied the data (under copy-on-write it is a lazy copy,
        # which is equally safe), so no second copy is needed.
        return working_df
    positions = np.flatnonzero(selected)
    # Taking every position would only copy the frame for nothing.
    sub = working_df if selected.all() else working_df.iloc[positions]
    # The memo's factorizations are of the full frame's columns, so everything
    # computed on sub may consult it only when sub is that very frame -- the
    # case the memo exists for, since that is where the most is hashed.
    sub_memo = memo if sub is working_df else None
    # Identical rows must hash differently, or they would be kept or dropped
    # as a block. Spark numbers them with row_number over a window partitioned
    # by every column, which is a cumulative count over identical rows. Each
    # all-columns partition lies within one group, so restricting the count to
    # the selected rows leaves every salt unchanged (the salt-locality step of
    # the "Fast paths" argument) -- and because the partition is intrinsic to
    # the values, the non-grouping columns' codes can be computed on the
    # selected rows directly. The salt also makes deduplicating whole rows
    # before hashing pointless: (values, salt) is unique per row by
    # construction.
    if starting_columns:
        salt = (
            sub.groupby(
                [
                    group_code[column][positions]
                    if column in group_code
                    else _group_codes(sub[column], sub_memo)
                    for column in starting_columns
                ],
                sort=False,
                dropna=False,
            ).cumcount()
            + 1
        ).to_numpy()
    else:
        # With no columns at all, every row is in the same partition, so the
        # full-frame count at position p is p itself.
        salt = positions + 1
    digests = _row_digests(
        [_column_digests(sub[column], sub_memo) for column in starting_columns]
        # The salt is derived here, not a column of the frame, so it has
        # nothing in the memo and must not be entered into it either.
        + [_column_digests(pd.Series(salt))],
        len(sub),
    )

    def tie_keys() -> List[np.ndarray]:
        # The order keys are computed over the full frame and then restricted
        # (the restriction step of the "Fast paths" argument). They matter
        # only when two digests collide, so they are built lazily.
        order_keys = {
            column: _order_keys(working_df[column], memo) for column in starting_columns
        }
        return _tie_break_keys(order_keys, starting_columns, positions)

    order = _hash_sort_order(tie_keys, _digest_order_key(digests))
    rank = _prefix_ranks(group_ids[positions][order])
    return _survivors_in_input_order(
        working_df, selected, positions[order[rank <= threshold]]
    )


def drop_large_groups(
    df: pd.DataFrame, grouping_columns: List[str], threshold: int
) -> pd.DataFrame:
    """Drop all rows for groups that have more than ``threshold`` rows.

    This is the pandas counterpart of
    :func:`tmlt.core.utils.truncation.drop_large_groups`, and keeps the same
    rows as that function does. It does not hash any values, so unlike the
    hashing functions it never rejects a column for its dtype; the one
    unsupported input is a value with no Python hash, such as a ``dict`` or
    a ``list`` inside an ``object`` column, which has no group key and
    raises :class:`NotImplementedError`.

    Example:
        ..
            >>> import pandas as pd
            >>> from tmlt.core.utils.misc import print_pandas
            >>> dataframe = pd.DataFrame(
            ...     {
            ...         "A": ["a1", "a2", "a3", "a3", "a3"],
            ...         "B": ["b1", "b1", "b2", "b2", "b3"],
            ...     }
            ... )

        >>> # Example input
        >>> print_pandas(dataframe)
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2
        4  a3  b3
        >>> print_pandas(drop_large_groups(dataframe, ["A"], 3))
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2
        4  a3  b3
        >>> print_pandas(drop_large_groups(dataframe, ["A"], 2))
            A   B
        0  a1  b1
        1  a2  b1
        >>> print_pandas(drop_large_groups(dataframe, ["A"], 1))
            A   B
        0  a1  b1
        1  a2  b1

    Args:
        df: DataFrame to truncate.
        grouping_columns: Columns defining the groups.
        threshold: Threshold for dropping groups. If more than ``threshold`` rows belong
            to the same group, all rows in that group are dropped.
    """
    working_df = df.reset_index(drop=True)
    grouping_unique = list(dict.fromkeys(grouping_columns))
    _require_columns(working_df, grouping_unique)
    group_ids = _group_ids(
        [_group_codes(working_df[column]) for column in grouping_unique],
        len(working_df),
    )
    sizes = np.bincount(group_ids)[group_ids]
    return _reindexed_from_zero(working_df.loc[sizes <= threshold])


class _RefinedPairs(NamedTuple):
    """The refined (group, digest, key) classes of the fast budget test.

    Attributes:
        codes: One dense code per row, as :func:`_dense_codes` returns them.
        first: The position of each class's first occurrence, in code order.
    """

    codes: np.ndarray
    first: np.ndarray


def _refined_pairs(
    working_df: pd.DataFrame,
    hashed_unique: List[str],
    group_code: Mapping[str, np.ndarray],
    memo: Optional[_FactorizeMemo] = None,
) -> Optional[_RefinedPairs]:
    """Returns the refined (group, digest, key) classes, or None.

    Rows sharing a refined code necessarily share their group key, their
    combined digest, and their key key. The refinement is only valid when
    EVERY hashed column contributed digest codes: a column that fell back to
    per-value rendering has none, and the refined identity would then be
    coarser than Spark's -- it would merge, for instance, an object column's
    1 and 1.0, which share a group key but render "1" and "1.0" -- which
    would UNDER-count a group's keys and skip a group that needed truncating.
    The collection therefore stops at the first such column (and is skipped
    entirely when the fast path is off), rather than factorizing columns
    whose codes would be discarded unused. ``None`` -- rather than some
    weaker refinement -- means a wrongly weakened guard crashes instead of
    silently merging every row into one class.

    Returns:
        The refined classes, or None when the refinement is unavailable.
    """
    if not _FAST_PATH_ENABLED:
        return None
    if not hashed_unique:
        # With no hashed columns every row is one refined class, mirroring
        # the way _group_ids treats no columns as one group; _dense_codes
        # needs at least one code array, so it cannot build this case itself.
        refined_codes = np.zeros(len(working_df), dtype=np.int64)
        return _RefinedPairs(refined_codes, _first_occurrences(refined_codes))
    digest_codes = []
    for column in hashed_unique:
        codes_and_values = _digest_codes(working_df[column], memo)
        if codes_and_values is None:
            return None
        digest_codes.append(codes_and_values[0])
    refined_codes = _dense_codes(
        [group_code[column] for column in hashed_unique] + digest_codes
    )
    return _RefinedPairs(refined_codes, _first_occurrences(refined_codes))


def limit_keys_per_group(
    df: pd.DataFrame,
    grouping_columns: Collection[str],
    key_columns: Collection[str],
    threshold: int,
) -> pd.DataFrame:
    """Order keys by a hash function and keep at most ``threshold`` keys for each group.

    This is the pandas counterpart of
    :func:`tmlt.core.utils.truncation.limit_keys_per_group`, and keeps the same
    rows as that function does.

    .. note::

        After truncation there may still be an unbounded number of rows per key, but
        at most ``threshold`` keys per group

    Example:
        ..
            >>> import pandas as pd
            >>> from tmlt.core.utils.misc import print_pandas
            >>> dataframe = pd.DataFrame(
            ...     {
            ...         "A": ["a1", "a2", "a3", "a3", "a3", "a4", "a4", "a4"],
            ...         "B": ["b1", "b1", "b2", "b2", "b3", "b1", "b2", "b3"],
            ...     }
            ... )

        >>> # Example input
        >>> print_pandas(dataframe)
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2
        4  a3  b3
        5  a4  b1
        6  a4  b2
        7  a4  b3
        >>> print_pandas(
        ...     limit_keys_per_group(
        ...         df=dataframe,
        ...         grouping_columns=["A"],
        ...         key_columns=["B"],
        ...         threshold=2,
        ...     )
        ... )
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b2
        3  a3  b2
        4  a3  b3
        5  a4  b2
        6  a4  b3
        >>> print_pandas(
        ...     limit_keys_per_group(
        ...         df=dataframe,
        ...         grouping_columns=["A"],
        ...         key_columns=["B"],
        ...         threshold=1,
        ...     )
        ... )
            A   B
        0  a1  b1
        1  a2  b1
        2  a3  b3
        3  a4  b3

    Args:
        df: DataFrame to truncate.
        grouping_columns: Columns defining the groups.
        key_columns: Column defining the keys.
        threshold: Maximum number of keys to include for each group.
    """
    hashed = [*grouping_columns, *key_columns]
    hashed_unique = list(dict.fromkeys(hashed))
    working_df = df.reset_index(drop=True)
    _require_columns(working_df, hashed_unique)
    # One memo for the whole call: validation, the group codes and the
    # refined pairs' digest codes below all factorize the same full-frame
    # columns, and the memo runs each canonical factorization once. Nothing
    # computed on the fast path's frames of selected or representative rows
    # may touch it -- those columns are different (shorter) arrays under the
    # same names.
    memo = _FactorizeMemo()
    for column in hashed_unique:
        _validate_column(working_df[column], column, memo)
    n = len(working_df)
    if threshold <= 0 or n == 0:
        # Spark expresses the threshold as a filter, so a non-positive
        # threshold is an empty result, not an error.
        return working_df.iloc[:0].copy()
    grouping_unique = list(dict.fromkeys(grouping_columns))
    key_unique = list(dict.fromkeys(key_columns))
    group_code = {
        column: _group_codes(working_df[column], memo) for column in hashed_unique
    }
    group_ids = _group_ids([group_code[column] for column in grouping_unique], n)
    refined = _refined_pairs(working_df, hashed_unique, group_code, memo)
    if refined is not None:
        # Counting refined classes per group can only OVER-count a group's
        # keys, which merely hashes a group that needed no hashing (see the
        # module docstring's "Fast paths" section).
        pairs_per_group = np.bincount(
            group_ids[refined.first], minlength=int(group_ids.max()) + 1
        )
        selected = pairs_per_group[group_ids] > threshold
    else:
        selected = np.ones(n, dtype=bool)
    if not selected.any():
        # Every group is within its key budget. working_df is already
        # private: the reset_index at entry copied the data (under
        # copy-on-write it is a lazy copy, which is equally safe), so no
        # second copy is needed.
        return working_df
    positions = np.flatnonzero(selected)
    # Rows in one refined class share every per-column digest, so the
    # combined digest can be computed once per class and fanned back out; on
    # frames with many rows per (group, key) pair this removes most of the
    # hashing. Both branches produce bit-identical digests; the cutoff is
    # purely economic. Every refined class lies within one group and selected
    # is a union of groups, so counting the selected class representatives
    # counts the classes. The class machinery costs about a third of what
    # combining costs per row, so it pays only when it removes at least a
    # third of the rows.
    if refined is not None and 3 * int(selected[refined.first].sum()) <= 2 * len(
        positions
    ):
        class_codes = pd.factorize(refined.codes[positions])[0]
        class_first = _first_occurrences(class_codes)
        representatives = working_df.iloc[positions[class_first]]
        # One row per class is a strict selection of the frame's rows, so the
        # memo -- whose factorizations are full-frame -- has no part in it.
        column_digests = {
            column: _column_digests(representatives[column]) for column in hashed_unique
        }
        class_digests = _row_digests(
            [column_digests[column] for column in hashed], len(class_first)
        )
        digests = class_digests[class_codes]
        # Factorizing the per-class digests and fanning the codes out gives
        # the same codes as factorizing per row, hashing one 64-character
        # string per class instead of per row.
        digest_codes = pd.factorize(class_digests)[0][class_codes]
    else:
        # Taking every position would only copy the frame for nothing.
        sub = working_df if selected.all() else working_df.iloc[positions]
        # The memo's factorizations are of the full frame's columns, so they
        # may be reused only when sub is that very frame -- which is the case
        # this branch is reached in whenever every group is over its budget.
        sub_memo = memo if sub is working_df else None
        column_digests = {
            column: _column_digests(sub[column], sub_memo) for column in hashed_unique
        }
        digests = _row_digests([column_digests[column] for column in hashed], len(sub))
        digest_codes = pd.factorize(digests)[0]
    # The hash only depends on the grouping and key columns, so all rows of a
    # (group, key) pair share it. Spark ranks the pairs with dense_rank; here
    # each pair is given an id, ranked once, and the surviving ids select rows.
    # Spark's dense_rank ranks by (hash, *key_columns), so the hash is part of
    # the pair's identity: pandas considers -0.0 and 0.0 equal keys, but they
    # hash differently and Spark counts them as two keys.
    pair_ids = _dense_codes(
        [group_code[column][positions] for column in grouping_unique]
        + [digest_codes]
        + [group_code[column][positions] for column in key_unique]
    )
    # One representative row per pair, in input order (matching the stable
    # drop_duplicates this replaces), sorted by (digest, *key_columns).
    pair_first = _first_occurrences(pair_ids)
    pair_positions = positions[pair_first]

    def tie_keys() -> List[np.ndarray]:
        # The order keys are computed over the full frame and then restricted
        # (the restriction step of the "Fast paths" argument). They matter
        # only when two pair digests collide, so they are built lazily.
        order_keys = {
            column: _order_keys(working_df[column], memo) for column in key_unique
        }
        return _tie_break_keys(order_keys, list(key_columns), pair_positions)

    ordered_pairs = pair_first[
        _hash_sort_order(tie_keys, _digest_order_key(digests[pair_first]))
    ]
    rank = _prefix_ranks(group_ids[positions][ordered_pairs])
    # The pair ids are 0, 1, ... and so index a mask of the surviving pairs
    # directly, which selects the rows belonging to those pairs.
    surviving = np.zeros(len(pair_first), dtype=bool)
    surviving[pair_ids[ordered_pairs[rank <= threshold]]] = True
    return _survivors_in_input_order(
        working_df, selected, positions[surviving[pair_ids]]
    )
