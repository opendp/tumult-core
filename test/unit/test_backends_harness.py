"""Self-tests for the backend-parity harness.

:mod:`test.unit.backend_testing` is not code under test, it is the *oracle*
that every parity suite is judged against: if its comparison helpers were to
call two different frames equal, or its conversion helpers to quietly change
data on the way into a backend, whole suites would pass while agreeing about
nothing. This module pins the harness's own behavior, so that a change to it
fails here rather than silently weakening its consumers.

Three things are checked:

* The comparison helpers separate the values the harness promises to keep
  apart, and merge the ones it promises to merge. Most of these tests assert a
  *failure*: a frame that differs only by ``None`` vs ``NaN`` must not compare
  equal.
* :func:`~test.unit.backend_testing.df_for` and
  :func:`~test.unit.backend_testing.to_pandas` round-trip the whole
  :data:`~test.unit.backend_testing.EDGE_CASES` corpus under both backends,
  which is the closest thing there is to a proof that a parity failure is the
  operation's fault rather than the harness's.
* The repo-wide ``backend`` fixture yields what this package says it does.

Tests taking the ``backend`` fixture run twice; only the Spark run requests a
Spark session, and only it carries the ``spark`` marker.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import datetime
import itertools
import random
from contextlib import contextmanager
from test.unit.backend_testing import (
    BACKEND_NAMES,
    EDGE_CASES,
    KIND_NAMES,
    ROW_ID_COLUMN,
    SIMPLE_DTYPE_MENU,
    Backend,
    EdgeCase,
    assert_frames_equal_as_multisets,
    df_for,
    domain_for,
    exact_value,
    frame_row_ids,
    is_null_value,
    normalize_value,
    random_frame,
    spark_schema_from_pandas,
    to_pandas,
    utc_session_timezone,
)
from typing import Any, Dict, Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType, FloatType, LongType

from tmlt.core.domains.pandas_domains import (
    PandasFloatColumnDescriptor,
    PandasTableDomain,
)
from tmlt.core.domains.spark_domains import SparkDataFrameDomain
from tmlt.core.utils.testing import Case, parametrize

#: Seed for the generated frames used here. Any seed would do; a fixed one
#: makes a failure reproducible.
SEED = 20260812

#: The flags each column kind takes, restated rather than imported from
#: :mod:`test.unit.backend_testing.domains`: this module is the oracle that
#: package is judged against, so a flag quietly dropped there has to fail here
#: instead of moving the oracle with it. Parametrizing over
#: :data:`~test.unit.backend_testing.KIND_NAMES` means a kind added there
#: without an entry here fails too.
_KIND_FLAG_NAMES: Dict[str, Tuple[str, ...]] = {
    "int32": ("allow_null",),
    "int64": ("allow_null",),
    "float32": ("allow_nan", "allow_inf", "allow_null"),
    "float64": ("allow_nan", "allow_inf", "allow_null"),
    "string": ("allow_null",),
    "date": ("allow_null",),
    "timestamp": ("allow_null",),
}

################################################################################
# Helpers
################################################################################


def _flag_combinations(names: Sequence[str]) -> List[Dict[str, bool]]:
    """Returns every assignment of True and False to the given flags.

    Args:
        names: The flag names.

    Returns:
        One dictionary per combination.
    """
    return [
        dict(zip(names, values))
        for values in itertools.product([True, False], repeat=len(names))
    ]


def _frame(dtypes: Dict[str, str], rows: Sequence[Tuple[Any, ...]]) -> pd.DataFrame:
    """Returns a dataframe with the given dtypes and rows.

    Each column is built as an object-dtype Series and then cast, so that
    pandas never infers a dtype of its own -- which would, for instance, turn a
    ``None`` in an integer column into a float ``NaN``, and so quietly write a
    different test than the one intended.

    Args:
        dtypes: The pandas dtype of each column, by name, in column order.
        rows: The rows, as tuples in the order given by ``dtypes``.

    Returns:
        The assembled dataframe.
    """
    data: Dict[str, pd.Series] = {}
    for index, name in enumerate(dtypes):
        values = [row[index] for row in rows]
        data[name] = pd.Series(values, dtype=object).astype(dtypes[name])
    return pd.DataFrame(data, columns=list(dtypes))


def _survives_spark_round_trip(df: pd.DataFrame) -> bool:
    """Returns whether a frame's values survive a Spark round trip unchanged.

    ``toPandas()`` has no way to return a null in a numeric column: a null in a
    ``LongType`` column comes back as ``NaN`` in a widened ``float64`` column,
    and a null in a floating point column comes back as ``NaN`` outright. Any
    other column type round-trips its values, nulls included.

    Frames this returns False for are exactly the ones the corpus gives a
    :data:`ROW_ID_COLUMN` so that they can be compared by surviving row instead
    of by value.

    Args:
        df: The frame that would be sent to Spark.

    Returns:
        Whether every value would come back as itself.
    """
    for field in spark_schema_from_pandas(df).fields:
        if isinstance(field.dataType, (LongType, DoubleType, FloatType)) and any(
            is_null_value(value) for value in df[field.name]
        ):
            return False
    return True


@contextmanager
def _session_timezone_for(case: EdgeCase, backend: Backend) -> Iterator[None]:
    """Puts Spark in UTC for the cases that need it, and does nothing else.

    A frame with naive timestamps may only be built while the Spark session
    renders timestamps as UTC wall clocks; every other case, and every pandas
    run, needs no setup at all.

    Args:
        case: The case about to be built.
        backend: The backend it is being built for.

    Yields:
        Nothing.
    """
    if backend.name == "spark" and case.has_timestamps:
        with utc_session_timezone(backend.require_spark()):
            yield
    else:
        yield


################################################################################
# Comparison: what must not compare equal
################################################################################


@parametrize(
    Case("none-vs-nan")(
        left=_frame({"v": "object"}, [(None,), (1.5,)]),
        right=_frame({"v": "object"}, [(float("nan"),), (1.5,)]),
        reason="a null and a NaN are different values in an object column",
    ),
    Case("na-vs-nan")(
        left=_frame({"v": "Float64"}, [(None,), (1.5,)]),
        right=_frame({"v": "float64"}, [(float("nan"),), (1.5,)]),
        reason="pd.NA means NULL and np.nan means NaN, whatever the dtype",
    ),
    Case("value-vs-null")(
        left=_frame({"v": "object"}, [("a",), ("b",)]),
        right=_frame({"v": "object"}, [("a",), (None,)]),
        reason="a dropped value is not a null",
    ),
    Case("different-multiplicity")(
        left=_frame({"v": "int64"}, [(1,), (1,), (2,)]),
        right=_frame({"v": "int64"}, [(1,), (2,), (2,)]),
        reason="rows are compared as a multiset, not as a set",
    ),
    Case("extra-row")(
        left=_frame({"v": "int64"}, [(1,), (2,)]),
        right=_frame({"v": "int64"}, [(1,), (2,), (3,)]),
        reason="a frame is not equal to a strict superset of itself",
    ),
    Case("different-values")(
        left=_frame({"v": "object"}, [("a",)]),
        right=_frame({"v": "object"}, [("A",)]),
        reason="string comparison is case sensitive",
    ),
)
def test_unequal_frames_are_not_equal(
    left: pd.DataFrame, right: pd.DataFrame, reason: str
) -> None:
    """Frames the harness must keep apart do not compare equal.

    These are the differences a comparison helper could plausibly swallow --
    the null flavors above all, since pandas itself displays several of them as
    ``NaN``. A helper that merged them would let a backend lose the
    distinction without failing a single test.

    Args:
        left: The first frame.
        right: The second frame.
        reason: Why the two must not be equal, for the failure message.
    """
    with pytest.raises(AssertionError, match="differ as multisets"):
        assert_frames_equal_as_multisets(left, right)
    with pytest.raises(AssertionError, match="differ as multisets"):
        assert_frames_equal_as_multisets(left, right, normalize=False)


@parametrize(
    Case("null-flavors")(
        left=_frame({"v": "object"}, [(None,)]),
        right=_frame({"v": "object"}, [(pd.NA,)]),
        reason="None and pd.NA are both nulls, and toPandas picks its own",
    ),
    Case("int-vs-float")(
        left=_frame({"v": "int64"}, [(1,), (2,)]),
        right=_frame({"v": "float64"}, [(1.0,), (2.0,)]),
        reason="a Spark round trip widens a nullable integer column to float",
    ),
    Case("signed-zeros")(
        left=_frame({"v": "float64"}, [(0.0,)]),
        right=_frame({"v": "float64"}, [(-0.0,)]),
        reason="0.0 == -0.0, and only the sign bit tells them apart",
    ),
)
def test_normalization_merges_what_a_round_trip_destroys(
    left: pd.DataFrame, right: pd.DataFrame, reason: str
) -> None:
    """Frames differing only in ways ``toPandas()`` erases compare equal.

    Each of these is a distinction the harness deliberately gives up, because
    keeping it would make every cross-backend comparison fail on the conversion
    rather than on the operation under test. ``normalize=False`` keeps all of
    them, which is what makes it the right mode for a single-backend assertion.

    Args:
        left: The first frame.
        right: The second frame.
        reason: Why the distinction is given up, for the failure message.
    """
    assert_frames_equal_as_multisets(left, right)
    with pytest.raises(AssertionError, match="differ as multisets"):
        assert_frames_equal_as_multisets(left, right, normalize=False)


def test_multiset_equality_ignores_row_order() -> None:
    """Reordering a frame's rows does not change what it is equal to."""
    rows = [(2, "b"), (1, "a"), (3, "c"), (1, "a")]
    frame = _frame({"k": "int64", "v": "object"}, rows)
    shuffled = _frame({"k": "int64", "v": "object"}, list(reversed(rows)))
    assert_frames_equal_as_multisets(frame, shuffled)
    assert_frames_equal_as_multisets(frame, shuffled, normalize=False)


def test_multiset_equality_ignores_column_order() -> None:
    """Reordering a frame's columns does not change what it is equal to."""
    frame = _frame({"k": "int64", "v": "object"}, [(1, "a"), (2, "b")])
    assert_frames_equal_as_multisets(frame, frame[["v", "k"]])


def test_multiset_equality_ignores_the_index() -> None:
    """A frame's pandas index is not part of its data.

    Backends renumber, and truncation leaves gaps in, the index; none of that
    is observable in the data the operation returned.
    """
    frame = _frame({"v": "int64"}, [(1,), (2,), (3,)])
    assert_frames_equal_as_multisets(frame, frame.iloc[::-1].reset_index(drop=True))
    assert_frames_equal_as_multisets(frame, frame[frame["v"] > 0])


def test_empty_frames_are_equal() -> None:
    """Two empty frames with the same columns are equal, whatever their dtypes."""
    left = _frame({"k": "int64", "v": "object"}, [])
    right = _frame({"k": "Int64", "v": "string"}, [])
    assert_frames_equal_as_multisets(left, right)


def test_a_frame_with_nans_equals_itself() -> None:
    """NaN does not make a frame unequal to itself, under either mode.

    ``nan != nan``, so a comparison keyed on raw values would call every
    NaN-bearing frame different from itself. Both key functions map all NaNs
    onto one key to avoid that.
    """
    frame = _frame(
        {"v": "float64", "w": "object"},
        [(float("nan"), float("nan")), (1.0, None), (float("inf"), pd.NA)],
    )
    assert_frames_equal_as_multisets(frame, frame.copy())
    assert_frames_equal_as_multisets(frame, frame.copy(), normalize=False)


def test_mismatched_columns_raise() -> None:
    """Comparing frames with different columns is an error, not a failure.

    A test that compares the wrong two frames has a bug in the test, and
    should say so rather than report a data difference.
    """
    left = _frame({"k": "int64"}, [(1,)])
    right = _frame({"j": "int64"}, [(1,)])
    with pytest.raises(ValueError, match="matching columns"):
        assert_frames_equal_as_multisets(left, right)


def test_the_failure_message_shows_the_differing_rows() -> None:
    """A failure names the rows on each side, so a diff needs no debugger."""
    left = _frame({"k": "int64", "v": "object"}, [(1, "a"), (2, "keep")])
    right = _frame({"k": "int64", "v": "object"}, [(3, "b"), (2, "keep")])
    with pytest.raises(AssertionError) as caught:
        assert_frames_equal_as_multisets(left, right)
    message = str(caught.value)
    assert "only in left" in message and "only in right" in message
    assert "'a'" in message and "'b'" in message
    assert "keep" not in message, "rows that match should not be reported"


@parametrize(
    Case("none-and-na")(left=None, right=pd.NA),
    Case("none-and-nat")(left=None, right=pd.NaT),
    Case("int-and-float")(left=1, right=1.0),
    Case("signed-zeros")(left=0.0, right=-0.0),
    Case("numpy-and-python-int")(left=np.int64(3), right=3),
    Case("bytes-and-bytearray")(left=b"ab", right=bytearray(b"ab")),
    Case("timestamp-and-datetime")(
        left=pd.Timestamp("2020-01-02 03:04:05"),
        right=datetime.datetime(2020, 1, 2, 3, 4, 5),
    ),
)
def test_normalize_value_merges(left: Any, right: Any) -> None:
    """:func:`normalize_value` gives these pairs the same key.

    Args:
        left: The first value.
        right: The second value.
    """
    assert normalize_value(left) == normalize_value(right)


@parametrize(
    Case("none-and-na")(left=None, right=pd.NA),
    Case("none-and-nat")(left=None, right=pd.NaT),
    Case("null-and-nan")(left=None, right=float("nan")),
    Case("int-and-float")(left=1, right=1.0),
    Case("int-and-bool")(left=1, right=True),
    Case("signed-zeros")(left=0.0, right=-0.0),
    Case("string-and-bytes")(left="a", right=b"a"),
)
def test_exact_value_separates(left: Any, right: Any) -> None:
    """:func:`exact_value` gives these pairs different keys.

    Args:
        left: The first value.
        right: The second value.
    """
    assert exact_value(left) != exact_value(right)


@parametrize(
    Case("numpy-and-python-int")(left=np.int64(3), right=3),
    Case("numpy-and-python-float")(left=np.float64(1.5), right=1.5),
    Case("bytes-and-bytearray")(left=b"ab", right=bytearray(b"ab")),
    Case("nan-and-nan")(left=float("nan"), right=float("nan")),
    Case("timestamp-and-datetime")(
        left=pd.Timestamp("2020-01-02 03:04:05"),
        right=datetime.datetime(2020, 1, 2, 3, 4, 5),
    ),
)
def test_exact_value_merges_what_pandas_makes_unobservable(
    left: Any, right: Any
) -> None:
    """:func:`exact_value` merges only the pairs a test cannot tell apart.

    Which of these two spellings pandas hands back is an artifact of how a
    Series was constructed, not of the data, so keeping them apart would make
    the strict mode useless rather than strict.

    Args:
        left: The first value.
        right: The second value.
    """
    assert exact_value(left) == exact_value(right)


################################################################################
# Conversion: the corpus round trip
################################################################################


@pytest.mark.parametrize("case", EDGE_CASES, ids=lambda case: case.id)
def test_round_trip_preserves_the_corpus(case: EdgeCase, backend: Backend) -> None:
    """Every edge case survives a trip through a backend and back.

    This is the harness's central claim: whatever a parity suite observes is
    the operation's doing, because handing a frame to a backend and taking it
    straight back changes nothing. Cases whose values cannot survive
    ``toPandas()`` -- a null in a numeric column, which comes back as ``NaN``
    -- are compared by surviving row instead, which is exactly what the corpus
    carries :data:`ROW_ID_COLUMN` for.

    Args:
        case: The edge case to round-trip.
        backend: The backend to round-trip it through.
    """
    original = case.to_pandas()
    with _session_timezone_for(case, backend):
        native = df_for(original, backend)
        result = to_pandas(native, backend)

    assert list(result.columns) == list(original.columns)
    assert len(result) == len(original)
    if _survives_spark_round_trip(original) or backend.name == "pandas":
        assert_frames_equal_as_multisets(original, result)
    else:
        assert case.has_row_id, (
            f"Case {case.id} does not survive a Spark round trip by value, so "
            f"it needs a {ROW_ID_COLUMN} column to be compared by."
        )
        assert sorted(frame_row_ids(result)) == sorted(frame_row_ids(original))


@parametrize(
    Case("simple-dtypes")(dtype_menu=SIMPLE_DTYPE_MENU, with_row_id=True),
    Case("simple-dtypes-with-duplicates")(
        dtype_menu=SIMPLE_DTYPE_MENU, with_row_id=False
    ),
)
def test_round_trip_preserves_generated_frames(
    dtype_menu: Tuple[str, ...], with_row_id: bool, backend: Backend
) -> None:
    """Generated frames round-trip by value, duplicate rows included.

    The generator's frames are drawn from a menu whose dtypes all survive a
    Spark round trip, so this asserts full value equality where
    :func:`test_round_trip_preserves_the_corpus` sometimes falls back to row
    ids. Running it without row ids is the point of the second case: the
    duplicate rows it then produces are the ones a set-based comparison would
    lose.

    Args:
        dtype_menu: The column kinds to draw from.
        with_row_id: Whether the frames get a unique row id column.
        backend: The backend to round-trip through.
    """
    rng = random.Random(SEED)
    for _ in range(5):
        case = random_frame(
            rng, dtype_menu, n_rows=30, dup_rate=0.4, with_row_id=with_row_id
        )
        original = case.to_pandas()
        with _session_timezone_for(case, backend):
            result = to_pandas(df_for(original, backend), backend)
        assert_frames_equal_as_multisets(original, result)


def test_pandas_conversion_is_the_identity() -> None:
    """The pandas backend's conversions return the frame passed in, not a copy.

    Documented as identity, and depended on: a test that wants to watch a
    backend mutate its input needs to know it is looking at the same object.
    """
    backend = Backend(name="pandas")
    frame = _frame({"v": "int64"}, [(1,), (2,)])
    assert df_for(frame, backend) is frame
    assert to_pandas(frame, backend) is frame


def test_df_for_builds_a_spark_frame(backend: Backend) -> None:
    """``df_for`` returns each backend's own frame type.

    Args:
        backend: The backend to build a frame for.
    """
    frame = _frame({"v": "int64"}, [(1,), (2,)])
    native = df_for(frame, backend)
    if backend.name == "spark":
        assert isinstance(native, DataFrame)
    else:
        assert isinstance(native, pd.DataFrame)


def test_df_for_accepts_an_explicit_session(backend: Backend) -> None:
    """A session passed to ``df_for`` is used in place of the backend's.

    A suite that has a session but no :class:`Backend` carrying one -- one
    using its own backend object, for instance -- passes it explicitly.

    Args:
        backend: The backend to build a frame for.
    """
    frame = _frame({"v": "int64"}, [(1,)])
    session = backend.spark
    assert to_pandas(df_for(frame, backend, spark=session), backend).equals(frame)


def test_df_for_without_a_session_raises() -> None:
    """Building a Spark frame with no session anywhere says so."""
    with pytest.raises(RuntimeError, match="carries no Spark session"):
        df_for(_frame({"v": "int64"}, [(1,)]), Backend(name="spark"))


@parametrize(
    Case("unknown-backend")(backend_name="duckdb"),
    Case("misspelled-backend")(backend_name="Spark"),
)
def test_an_unknown_backend_raises(backend_name: str) -> None:
    """An unrecognized backend name is an error, not a silent pandas fallback.

    Args:
        backend_name: The name to reject.
    """
    unknown = Backend(name=backend_name)
    with pytest.raises(ValueError, match="Unknown backend"):
        df_for(_frame({"v": "int64"}, [(1,)]), unknown)
    with pytest.raises(ValueError, match="Unknown backend"):
        to_pandas(_frame({"v": "int64"}, [(1,)]), unknown)


def test_to_pandas_rejects_the_wrong_frame_type() -> None:
    """Handing ``to_pandas`` the other backend's frame type is an error."""
    with pytest.raises(TypeError, match="pandas dataframes"):
        to_pandas("not a frame", Backend(name="pandas"))


def test_timestamps_outside_a_utc_session_raise(backend: Backend) -> None:
    """A Spark frame with timestamps refuses to be built in the wrong timezone.

    Producing a frame whose timestamps are silently shifted by the process's
    local timezone would make a parity suite fail somewhere else entirely, so
    the construction path refuses instead. The pandas backend has no session
    and so no such hazard.

    Args:
        backend: The backend to build a frame for.
    """
    frame = _frame({"t": "datetime64[ns]"}, [(datetime.datetime(2020, 1, 2, 3, 4, 5),)])
    if backend.name != "spark":
        assert df_for(frame, backend) is frame
        return
    with utc_session_timezone(backend.require_spark(), "America/New_York"):
        with pytest.raises(RuntimeError, match="UTC session"):
            df_for(frame, backend)


################################################################################
# The backend fixture
################################################################################


def test_the_backend_fixture_yields_a_known_backend(backend: Backend) -> None:
    """The fixture yields one of the names this package publishes.

    Args:
        backend: The backend under test.
    """
    assert isinstance(backend, Backend)
    assert backend.name in BACKEND_NAMES


def test_only_the_spark_backend_carries_a_session(backend: Backend) -> None:
    """A Spark session exists for the Spark backend and for nothing else.

    The pandas parameter carrying no session is what proves the fixture's lazy
    ``getfixturevalue`` is doing its job: a pandas run that had booted a JVM
    would have one to hand.

    Args:
        backend: The backend under test.
    """
    if backend.name == "spark":
        assert backend.spark is not None
        assert backend.require_spark() is backend.spark
    else:
        assert backend.spark is None
        with pytest.raises(RuntimeError, match="carries no Spark session"):
            backend.require_spark()


################################################################################
# Domains
################################################################################


def test_domain_for_builds_each_backends_domain(backend: Backend) -> None:
    """:func:`domain_for` returns the backend's own table domain.

    This replaces the placeholder test that asserted it raised
    :class:`NotImplementedError` while the pandas domains were in flight.

    Args:
        backend: The backend to ask for a domain.
    """
    domain = domain_for({"a": "int64", "b": "string"}, backend)
    expected_type = (
        SparkDataFrameDomain if backend.name == "spark" else PandasTableDomain
    )
    assert isinstance(domain, expected_type)
    assert list(domain.schema) == ["a", "b"]


@pytest.mark.parametrize("kind", KIND_NAMES)
def test_domain_for_descriptors_agree_across_backends(kind: str) -> None:
    """The two backends' descriptors describe the same values, flag for flag.

    :meth:`~tmlt.core.domains.pandas_domains.PandasColumnDescriptor.to_spark_descriptor`
    is the bridge between the two descriptor families, and this is what makes
    :func:`domain_for` a *parity* helper rather than two unrelated builders: a
    pandas domain and a Spark domain built from one spec must not differ in
    what they admit. Every flag combination of every kind is checked, so a new
    flag on either family fails here.

    This test needs no Spark session: a Spark domain is a description of a
    frame, not a frame.

    Args:
        kind: The column kind to check.
    """
    flag_names = _KIND_FLAG_NAMES[kind]
    for flags in _flag_combinations(flag_names):
        spec: Tuple[str, Dict[str, bool]] = (kind, flags)
        pandas_domain = domain_for({"a": spec}, Backend("pandas"))
        spark_domain = domain_for({"a": spec}, Backend("spark"))
        assert isinstance(pandas_domain, PandasTableDomain)
        assert isinstance(spark_domain, SparkDataFrameDomain)
        assert pandas_domain["a"].to_spark_descriptor() == spark_domain["a"], (
            f"{kind} with {flags} does not bridge to its Spark descriptor"
        )


def test_domain_for_rejects_unknown_kinds_flags_and_backends() -> None:
    """A spec that names nothing real is an error, not a surprising domain.

    A dtype spelling that is not a *type* -- ``object``, which may hold strings
    or dates -- is called out by name rather than lumped in with the typos, so
    that a test written from a frame's dtypes is told what to write instead.
    """
    with pytest.raises(ValueError, match="Unknown column kind 'int65'"):
        domain_for({"a": "int65"}, Backend("pandas"))
    with pytest.raises(ValueError, match="'string' or 'date'"):
        domain_for({"a": "object"}, Backend("pandas"))
    with pytest.raises(ValueError, match="has no flag"):
        domain_for({"a": ("string", {"allow_nan": True})}, Backend("pandas"))
    with pytest.raises(ValueError, match="kind name or a"):
        domain_for({"a": ("string", "allow_null", True)}, Backend("pandas"))  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="Unknown backend"):
        domain_for({"a": "int64"}, Backend("duckdb"))


def test_domain_for_flags_default_to_permissive() -> None:
    """Every flag defaults to True, and an override turns one off.

    The default matters: the corpus is full of nulls, NaNs and infinities, and
    a domain that rejected them would reject most of it before any operation
    under test ran.
    """
    domain = domain_for({"v": "float64"}, Backend("pandas"))
    assert isinstance(domain, PandasTableDomain)
    assert domain["v"] == PandasFloatColumnDescriptor(
        allow_nan=True, allow_inf=True, allow_null=True, size=64
    )
    strict = domain_for({"v": ("float64", {"allow_nan": False})}, Backend("pandas"))
    assert isinstance(strict, PandasTableDomain)
    assert strict["v"] == PandasFloatColumnDescriptor(
        allow_nan=False, allow_inf=True, allow_null=True, size=64
    )


def test_domain_for_accepts_nullable_dtype_spellings() -> None:
    """``Int64`` and ``int64`` are one kind, as the pandas descriptors are.

    A pandas integer descriptor accepts a column of either dtype, so the two
    spellings cannot name different domains; accepting both keeps a test from
    having to know which of them the data happens to use.
    """
    for pair in (("int64", "Int64"), ("float32", "Float32")):
        assert domain_for({"a": pair[0]}, Backend("pandas")) == domain_for(
            {"a": pair[1]}, Backend("pandas")
        )
    assert domain_for({"t": "datetime64[ns]"}, Backend("pandas")) == domain_for(
        {"t": "timestamp"}, Backend("pandas")
    )


################################################################################
# Generation
################################################################################


def test_random_frame_is_deterministic() -> None:
    """The same seed gives the same frame, so a failing sweep can be replayed."""
    first = random_frame(random.Random(SEED)).to_pandas()
    second = random_frame(random.Random(SEED)).to_pandas()
    assert_frames_equal_as_multisets(first, second, normalize=False)
    assert list(first.dtypes) == list(second.dtypes)


def test_random_frame_never_uses_nan_as_a_null() -> None:
    """Generated nulls are never ``np.nan`` in a non-object float column.

    The harness's whole null taxonomy rests on ``NaN`` meaning NaN. A generator
    that reached for ``np.nan`` to mean "missing" would produce frames the two
    backends legitimately disagree about, and the disagreement would look like
    an implementation bug.
    """
    rng = random.Random(SEED)
    for _ in range(20):
        frame = random_frame(rng, n_rows=15).to_pandas()
        for name in frame.columns:
            if str(frame[name].dtype) not in ("float64", "float32"):
                continue
            values: List[Any] = list(frame[name])
            assert not any(is_null_value(value) for value in values), (
                f"Column {name} of dtype {frame[name].dtype} holds a null; a "
                "plain float column can only express NaN."
            )
