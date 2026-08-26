"""The repo-wide backend-parity test harness.

Core is growing a pandas backend alongside its Spark one. Wherever both exist,
the interesting question is not "does this work" but "do the two agree", and
answering it needs one test body that can be handed either backend. This
package is the machinery for that: a way to build a frame for a backend, a way
to get its output back as pandas, a corpus of inputs that live where backends
disagree, and a definition of when two results count as the same.

Everything here is backend-*neutral*. Anything that knows what a particular
operation does belongs in the suite testing it, not in this package --
:mod:`test.unit.utils.truncation_testing`, which owns the truncation
functions' own dispatch table, is the pattern to follow.

The ``backend`` fixture
=======================

``test/conftest.py`` defines a repo-wide ``backend`` fixture, parametrized over
:data:`BACKEND_NAMES` and yielding a :class:`Backend`, so any test anywhere in
the suite can be run against both backends by taking it as an argument::

    def test_something(backend):
        result = some_operation(df_for(INPUT, backend))
        assert_frames_equal_as_multisets(to_pandas(result, backend), EXPECTED)

Two properties of that fixture are load-bearing and must survive any change to
it:

* Its Spark session is resolved *lazily*, so the ``pandas`` run of a test never
  starts a JVM even though its ``spark`` run does.
* Only its ``spark`` parameter carries the ``spark`` marker, so
  ``-m "not spark"`` deselects half of each parametrized test rather than all
  of it. That is what lets the ``test-nojvm`` lane exercise the pandas paths
  with pyspark installed but forbidden from starting (see
  ``TMLT_FORBID_JVM`` in ``test/conftest.py``).

A suite that needs a richer backend object than :class:`Backend` -- a dispatch
table of the functions it is testing, say -- should override ``backend`` in its
own ``conftest.py`` and build it from the fixture it overrides, so that the
parametrization, the laziness, and the marker keep living in one place.

API freeze
==========

**The names below are frozen.** Later work packages -- metrics parity, grouped
tables, joins, transformations, measurements -- write their tests against this
package, and they land in parallel. A signature change here is therefore not a
local edit; it breaks branches that are not yet merged and cannot be found by
grepping the repo. Under the freeze:

* Adding a name, or adding a keyword argument with a default, is fine.
* Renaming, removing, reordering parameters, or changing what a helper
  *guarantees* is not, and needs the harness's consumers updated in the same
  change.

The frozen surface is exactly this module's :data:`__all__`. Everything else,
including every underscore-prefixed name in the submodules, is an
implementation detail.

What the helpers guarantee
==========================

*Null canonicalization is the subtle part*, because the two backends do not
agree on what a null is:

* :func:`is_null_value` is the harness's null taxonomy: ``None``, ``pd.NA``,
  and ``pd.NaT`` are nulls; a float ``NaN`` is a *value*. It deliberately
  restates the taxonomy of the code under test rather than importing it, so
  that a regression there fails a test instead of moving the oracle with it.
* Consequently a ``NaN`` in a ``float64`` column means NaN, never NULL. A
  floating point column that needs to hold NULL uses the nullable ``Float64``
  dtype and ``pd.NA``; an ``object`` column is the only pandas column that can
  hold both, which is what a Spark double column does. The corpus and the
  generator never use ``np.nan`` to mean NULL, and neither should a test.
* :func:`normalize_value` collapses all three null flavors onto one sentinel
  and ``NaN`` onto another, so the two are never conflated but their flavors
  are. It also compares numbers by value across types, so ``1`` equals ``1.0``
  and ``0.0`` equals ``-0.0``. Both losses are deliberate: ``toPandas()``
  destroys exactly those distinctions, so a cross-backend comparison that kept
  them would fail on the round trip rather than on the operation. Use
  :func:`assert_no_conflating_values` to assert that a frame does not depend on
  a distinction this key throws away.
* :func:`exact_value` is the strict key that merges nothing -- the three null
  flavors differ from each other and from NaN, ``1`` differs from ``1.0``,
  ``0.0`` differs from ``-0.0`` -- for asserting that a *single* backend
  preserved its input.
* :func:`assert_frames_equal_as_multisets` compares with the first key by
  default and the second under ``normalize=False``. It ignores row order and
  dtypes, requires the same column *names*, and compares multiplicities, so a
  lost duplicate row is a failure.

The other guarantees:

* :func:`df_for` and :func:`to_pandas` are exact inverses for the pandas
  backend (both are the identity, returning the frame passed in, not a copy).
  For Spark they are not: ``to_pandas`` widens a nullable integer column to
  ``float64`` and turns a null in a floating point column into ``NaN``. Frames
  where that matters carry :data:`ROW_ID_COLUMN`, and are compared by
  surviving row id.
* :func:`df_for` never hands a pandas frame to ``createDataFrame``: it builds
  Spark frames from Python row tuples under an explicit schema, because the
  Arrow path silently turns ``NaN`` into ``NULL`` and changes dtypes.
* Naive timestamps only mean the same wall clock on both backends inside
  :func:`utc_session_timezone`; building a Spark frame with timestamps outside
  it raises rather than producing a frame that is quietly shifted.
* :func:`random_frame` is deterministic in its ``rng``, and only draws values
  that are comparable across backends at all -- see its docstring for the
  constraints, which are part of the contract.
* :func:`domain_for` is the domain counterpart of :func:`df_for`: one schema
  spec, two equivalent domains. Its flags default to *permissive*, because a
  domain is validated against on both arguments of every ``distance`` call and
  the corpus is deliberately full of nulls, NaNs and infinities. See
  :mod:`test.unit.backend_testing.domains` for the spec.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

from test.unit.backend_testing.comparison import (
    assert_frames_equal_as_multisets,
    assert_no_conflating_values,
    exact_value,
    grouped_symdiff_distance,
    label_value,
    multiset_symdiff,
    normalize_value,
    normalized_rows,
)
from test.unit.backend_testing.conversion import (
    BACKEND_NAMES,
    Backend,
    BackendLike,
    df_for,
    is_null_value,
    python_rows_from_pandas,
    spark_df_from_pandas,
    spark_schema_from_pandas,
    to_pandas,
    utc_session_timezone,
)
from test.unit.backend_testing.corpus import (
    CJK,
    E_ACUTE,
    E_COMBINING_ACUTE,
    EDGE_CASES,
    EDGE_CASES_BY_ID,
    EMOJI,
    ROW_ID_COLUMN,
    EdgeCase,
    frame_row_ids,
    spark_df_from_case,
)
from test.unit.backend_testing.domains import KIND_NAMES, ColumnSpec, domain_for
from test.unit.backend_testing.generation import (
    COLUMN_KINDS,
    DEFAULT_DTYPE_MENU,
    SIMPLE_DTYPE_MENU,
    ColumnKind,
    RandomLike,
    floating_array,
    random_frame,
)

#: The frozen harness API. See the module docstring for what the freeze means.
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
    "KIND_NAMES",
    "ROW_ID_COLUMN",
    "SIMPLE_DTYPE_MENU",
    "Backend",
    "BackendLike",
    "ColumnKind",
    "ColumnSpec",
    "EdgeCase",
    "RandomLike",
    "assert_frames_equal_as_multisets",
    "assert_no_conflating_values",
    "df_for",
    "domain_for",
    "exact_value",
    "floating_array",
    "frame_row_ids",
    "grouped_symdiff_distance",
    "is_null_value",
    "label_value",
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
