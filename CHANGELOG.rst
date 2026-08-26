.. _core-changelog:

Changelog
=========

.. _v0.19.1-ep-backend-3:

0.19.1+ep.backend.3 - 2026-08-13
--------------------------------
A fork build: official tmlt.core 0.19.1, everything in the Unreleased section
below -- the pandas backend work packages -- and the reuse and efficiency
changes listed here. Not an official Tumult Labs release, and never published
to PyPI.

Changed
~~~~~~~

- The column-name validation and the nullability algebra a join's output domain
  is computed with are shared between the two backends rather than copied.
  :mod:`tmlt.core.utils.join` grew ``_validate_join_columns``, ``_join_flag``,
  ``_join_allows_null`` and ``_side_unmatchable``, which
  :mod:`tmlt.core.utils.pandas_join` now calls; the pandas module's own copies
  are gone. The two backends have to agree about which joins are legal and about
  which output columns can hold a null, and sharing the code is what guarantees
  they keep agreeing. No behavior changed.
- The pandas grouped aggregations count without materializing a frame per group.
  :class:`~tmlt.core.utils.pandas_grouped_table.PandasGroupedTable` grew
  ``agg_by_position``, which hands an aggregation the *positions* of a group's
  rows rather than the rows themselves; it makes every promise ``agg`` makes
  about the output. The two count transformations in
  :mod:`tmlt.core.transformations.pandas_transformations.agg` use it, and
  ``CountDistinctGrouped`` additionally numbers the frame's rows once for the
  whole frame rather than once per group -- two rows are the same row, or not,
  wherever they sit. On 200k rows in 1000 groups the count is 65% cheaper and
  the distinct count 87% cheaper, for identical answers.
- :func:`tmlt.core.utils.pandas_join.join` numbers two join columns of the same
  dtype together, by grouping the concatenation of the two, instead of numbering
  each side and reconciling the two numberings through a Python key per distinct
  value. Join columns of *different* dtypes -- an ``int64`` joined to an
  ``Int64`` -- still take the reconciling path, since concatenating those would
  have pandas choose a common dtype for them. Joining on 100k distinct keys is
  92% cheaper for object keys and 96% for integers, with byte-identical
  numbering.
- An augmenting
  :class:`~tmlt.core.transformations.pandas_transformations.map.RowToRowTransformation`
  builds the domain it validates its function's output against once per set of
  row columns rather than once per row. It depends on the row's columns and not
  on its values, and every row of a frame has the same columns. Mapping 200k
  rows of 6 columns is 54% cheaper.
- Both backend arms of :meth:`.IfGroupedBy.distance` ask one helper for the
  groupby columns in the table's own column order, rather than each writing out
  the same comprehension over the domain's schema. No behavior changed.

Unreleased
----------

Added
~~~~~

- :class:`.Transformation`\s, :class:`.Measurement`\ s, and :class:`.Domain`\ s have a new ``format`` method, which renders a human-readable string showing the structure of the object to aid in visualization and debugging.
- Added :mod:`tmlt.core.utils.pandas_grouping`, which groups pandas dataframes the way
  Spark groups them (``group_codes``, ``group_ids``, ``row_keys``, ``distinct_rows``,
  and ``group_indices``). ``NULL`` and ``NaN`` are different groups, ``-0.0`` and
  ``0.0`` are one, binary values group by content, and timestamps group at Spark's
  microsecond resolution -- none of which a pandas ``groupby`` or ``drop_duplicates``
  gets right on its own. :mod:`tmlt.core.utils.pandas_truncation` is built on this
  module, which it was extracted from.
- Added :mod:`tmlt.core.utils.pandas_truncation`, pandas counterparts of the Spark
  truncation utilities in :mod:`tmlt.core.utils.truncation` (``truncate_large_groups``,
  ``drop_large_groups``, and ``limit_keys_per_group``). For the column types supported
  by the Spark implementations, the pandas functions hash values identically and keep
  exactly the same rows, so results are directly comparable across backends. The one
  exception is floating point columns: Spark renders those values with the JVM's
  ``Double.toString``/``Float.toString``, and a JVM older than 19 renders some values
  with more digits than the shortest that round-trips, which hashes differently. See
  the module documentation for details.
- Added :mod:`tmlt.core.utils.pandas_join`, the pandas counterpart of
  :mod:`tmlt.core.utils.join` (``join`` and ``domain_after_join``). It reproduces
  Spark's join semantics, which a pandas ``merge`` does not: a ``NULL`` key never
  matches another ``NULL`` key unless ``nulls_are_equal`` is set, while a ``NaN`` key
  always matches a ``NaN`` key (``NaN = NaN`` is true in Spark -- a NaN is a value,
  not a null) and never matches a ``NULL``. Output columns are also given the dtypes
  of the domain ``domain_after_join`` computes rather than whatever a merge widens
  them to, so an integer column that a left or outer join leaves unmatched comes back
  as ``Int64`` rather than as ``float64``, and values above :math:`2^{53}` survive.
- Added a column-descriptor family to :mod:`tmlt.core.domains.pandas_domains`:
  :class:`.PandasColumnDescriptor` with an integer, float, string, date, and timestamp
  subclass, collected in a :class:`.PandasTableDomain`. These describe a pandas
  DataFrame the way :class:`.SparkDataFrameDomain` describes a Spark one and carry the
  same information, so that both backends can describe the same table; each descriptor
  also fixes which dtypes a column it describes may have, since pandas -- unlike Spark
  -- can hold the same values in a numpy array or in a nullable extension array. The
  existing :class:`.PandasSeriesDomain` and :class:`.PandasDataFrameDomain`, which
  describe a DataFrame through the numpy domain of each column's elements, are
  unchanged.
- Added :class:`.PandasRowDomain`, the domain of the rows
  :class:`~tmlt.core.transformations.pandas_transformations.map.Map` applies a function
  to. A row is a :class:`dict`, and a missing value in one is ``None`` whatever marker
  its column stores -- ``pd.NA``, ``NaT``, or ``None`` -- while a NaN in a floating
  point column stays a NaN, since there it is a value rather than a missing value. The
  full per-dtype mapping is documented on
  :mod:`tmlt.core.transformations.pandas_transformations.map`.
- The table-level metrics in :mod:`tmlt.core.metrics` -- :class:`.SymmetricDifference`,
  :class:`.HammingDistance`, :class:`.OnColumn`, :class:`.OnColumns` and
  :class:`.AddRemoveKeys` -- now accept :class:`.PandasTableDomain` alongside
  :class:`.SparkDataFrameDomain`, and give the same distance for the same data under
  either backend. Rows and keys are compared with
  :mod:`tmlt.core.utils.pandas_grouping`, so a ``NULL`` and a ``NaN`` are different
  values, ``-0.0`` and ``0.0`` are one, and each column is compared in its own dtype.
  The branches for the older :class:`.PandasDataFrameDomain` and
  :class:`.PandasSeriesDomain` are unchanged; note that the former compares whole rows
  as Python tuples of ``DataFrame.values``, which merges integers differing only past
  ``2**53`` when a float column is present, and never finds a NaN-bearing row equal to
  itself. :class:`.AddRemoveKeys` requires a dictionary's dataframes to be all Spark or
  all pandas, and says so rather than silently reporting every key as changed.
- Added :mod:`tmlt.core.transformations.pandas_transformations`, holding
  :class:`~tmlt.core.transformations.pandas_transformations.select.Select`,
  :class:`~tmlt.core.transformations.pandas_transformations.rename.Rename`,
  :class:`~tmlt.core.transformations.pandas_transformations.map.Map` and
  :class:`~tmlt.core.transformations.pandas_transformations.map.RowToRowTransformation`
  over :class:`.PandasTableDomain`. Each mirrors its counterpart in
  :mod:`tmlt.core.transformations.spark_transformations`: it takes the same metrics,
  rejects the same arguments with the same errors, and has the same stability
  function. The pandas transformations additionally guarantee that they do not modify
  the frame they are given, and that the rows of their result are in the order they
  arrived in.
- Added the pandas grouped-table stack, the counterpart of the Spark one: a
  :class:`.PandasGroupedTable` in :mod:`tmlt.core.utils.pandas_grouped_table`, holding
  a frame together with an explicit frame of public group keys; a
  :class:`.PandasGroupedTableDomain` describing one; branches in
  :class:`.IfGroupedBy`, :class:`.SumOf` and :class:`.RootSumOfSquared` for the two new
  domains; and, in :mod:`tmlt.core.transformations.pandas_transformations`,
  :class:`~tmlt.core.transformations.pandas_transformations.groupby.GroupBy` (and its
  two constructor helpers) and the
  :class:`~tmlt.core.transformations.pandas_transformations.agg.CountGrouped` and
  :class:`~tmlt.core.transformations.pandas_transformations.agg.CountDistinctGrouped`
  aggregations. As in the Spark implementation, an aggregation produces exactly one
  row per declared group key, filling the keys with no rows and dropping the groups
  that were not declared; unlike it, the output is ordered by the group keys, so that
  an input's row order cannot be observed through its output's. Grouping goes through
  :mod:`tmlt.core.utils.pandas_grouping`, so the two backends agree about which rows
  share a group.
- Added :mod:`tmlt.core.transformations.pandas_transformations.join`, with
  :class:`~tmlt.core.transformations.pandas_transformations.join.PrivateJoin` and
  :class:`~tmlt.core.transformations.pandas_transformations.join.PrivateJoinOnKey` over
  :class:`.PandasTableDomain`\ s. These mirror their counterparts in
  :mod:`tmlt.core.transformations.spark_transformations.join`: same constructor
  checks, same output domain, and the same stability functions. They share that
  module's ``TruncationStrategy``, which names a strategy and is engine-neutral, and
  truncate through :mod:`tmlt.core.utils.pandas_truncation`.
- Added :mod:`tmlt.core.transformations.pandas_transformations.truncation`, holding
  :class:`~tmlt.core.transformations.pandas_transformations.truncation.LimitRowsPerGroup`,
  :class:`~tmlt.core.transformations.pandas_transformations.truncation.LimitKeysPerGroup`
  and
  :class:`~tmlt.core.transformations.pandas_transformations.truncation.LimitRowsPerKeyPerGroup`
  over :class:`.PandasTableDomain`. Each takes the same arguments as its counterpart in
  :mod:`tmlt.core.transformations.spark_transformations.truncation`, rejects the same
  ones with the same errors, and has the same stability function; the truncation itself
  is :mod:`tmlt.core.utils.pandas_truncation`, so the two backends keep the same rows.
  As with the other pandas transformations, the frame they are given is not modified,
  and the surviving rows are returned in the order they arrived in, reindexed from 0.
- Added :mod:`tmlt.core.transformations.pandas_transformations.add_remove_keys`, the
  pandas counterpart of
  :mod:`tmlt.core.transformations.spark_transformations.add_remove_keys`: the
  ``LimitRowsPerGroupValue``, ``LimitKeysPerGroupValue``,
  ``LimitRowsPerKeyPerGroupValue``, ``MapValue``, ``RenameValue`` and ``SelectValue``
  wrappers, which apply one pandas transformation to one table of a dictionary under
  :class:`.AddRemoveKeys` and augment the dictionary with the result. Each takes the
  same arguments as the Spark wrapper of the same name, rejects the same ones with
  the same errors, and has the same stability function. The wrappers for the
  operations the pandas backend has no transformation for -- filtering, public joins,
  flat maps, and dropping or replacing nulls, NaNs and infinities -- and the ones
  wrapping Spark's persistence machinery have no counterpart here.
- Added the pandas measurement layer, closing the counts-only pandas slice:
  :class:`tmlt.core.measurements.pandas_measurements.table.AddNoiseToColumn`, which
  adds an :class:`.AddNoiseToSeries`' noise to one aggregated column of a
  :class:`.PandasTableDomain` frame, and
  :mod:`tmlt.core.measurements.pandas_aggregations`, with
  :func:`~tmlt.core.measurements.pandas_aggregations.create_count_measurement` and
  :func:`~tmlt.core.measurements.pandas_aggregations.create_count_distinct_measurement`.
  Each factory has the same signature as its twin in
  :mod:`tmlt.core.measurements.aggregations` -- same parameter names, order and
  defaults, typed on the pandas domains and transformations -- and, over the same
  query, the same privacy function: the two backends spend identical budget and
  reject identical budget/mechanism combinations. Unlike the Spark ``count``
  factory, both pandas factories report an unusable ``groupby_transformation``
  with :class:`.UnsupportedDomainError` or :class:`.UnsupportedMetricError` rather
  than a bare assertion, following the Spark ``count_distinct`` factory. The
  pandas ``AddNoiseToColumn`` subclasses :class:`.Measurement` directly rather
  than mirroring ``SparkMeasurement``, since a pandas frame is eager and its noise
  cannot be redrawn by a later collect; it never modifies the frame it is given,
  reindexes its output from zero, and casts the noised column explicitly (integral
  for the geometric and discrete Gaussian mechanisms, floating point for the
  Laplace and Gaussian ones) so that an empty frame comes back with the same dtype
  a non-empty one would have.

Changed
~~~~~~~

- The block building an :class:`.AddNoiseToSeries` from a
  :class:`~tmlt.core.measurements.aggregations.NoiseMechanism` and a noise scale,
  which was duplicated between
  :func:`~tmlt.core.measurements.aggregations.create_count_measurement` and
  :func:`~tmlt.core.measurements.aggregations.create_count_distinct_measurement`,
  is now a private helper the two call and the pandas factories share. No
  behavior changed.
- Moved :class:`~tmlt.core.transformations.dictionary.TransformValue` from
  :mod:`tmlt.core.transformations.spark_transformations.add_remove_keys` to
  :mod:`tmlt.core.transformations.dictionary`. The class only constrains the domains
  and metrics of a dictionary and of the transformation applied to one of its values,
  none of which is Spark-specific, so it can be the base class of both backends'
  wrappers. It is re-exported under its old name, and the Spark wrappers derived from
  it are unchanged, so nothing that imports it moves.
- Both backends' ``GroupBy`` now put their group keys' columns in the input
  domain's order, whatever order they were given in, and the Spark grouped
  aggregations build their output schema by walking that domain's schema rather
  than iterating the ``groupby_columns`` frozenset, as the pandas ones already
  did. An aggregation emits its groupby columns in the group keys' order and
  declares them in the schema's order, and nothing validates that those agree,
  so this is a fix as well as a change: group keys built from a keyset whose
  columns are not in the input domain's order produced an output frame that the
  transformation's own output domain rejected, and Spark's *declared* order
  additionally depended on ``PYTHONHASHSEED`` and so varied from run to run.
  **This changes the column order of such an aggregation's result**, on both
  backends at once so that they stay in step; an aggregation whose keyset was
  already in the input domain's order is unaffected.
- :class:`~tmlt.core.utils.exact_number.ExactNumber` remembers the conversion of
  the integers and strings it is built from, up to a bounded number of them.
  Building the :class:`sympy.Expr` -- ``sympy.simplify`` above all -- dominates
  the cost of an :class:`~.ExactNumber`, and a privacy calculation builds tens
  of thousands of them from a handful of distinct values. A ``sympy.Expr`` is
  immutable, so sharing one between callers changes nothing but the cost;
  floats and booleans are deliberately not remembered, since each hashes and
  compares equal to an integer and would share its entry.
- Deleted the re-exports in
  :mod:`tmlt.core.transformations.pandas_transformations`' ``__init__``, which
  nothing imported, matching the Spark package's. Import the leaf modules, as
  everything already did.
- The two grouped-domain arms of :meth:`.SymmetricDifference.distance`,
  :meth:`.AggregationMetric.distance` and
  :meth:`.AggregationMetric.supports_domain`, and the two backend arms of
  :meth:`.AddRemoveKeys.distance`, are each one arm now: the algorithm was the
  same in both, and only how a group is asked whether it is empty, or how a
  dataframe's keys are enumerated and its rows selected, actually differed. No
  behavior changed.

Fixed
~~~~~

- Importing :mod:`tmlt.core.utils.cleanup` no longer starts a JVM at interpreter
  exit. Its ``atexit`` hook asked for a Spark session with ``getOrCreate``, which
  built one -- JVM included -- in any process that had not already made one, purely
  to drop a temporary database that such a process cannot have created. It now
  looks for a session the process already has, and returns when there is none.
  A process that does use Spark still has its temporary database dropped, but
  one detail of *how* the session is found changed: the hook takes the calling
  thread's active session if there is one and the process' instantiated session
  otherwise, rather than building a session. That fallback matters because the
  active session is thread-scoped while the hook runs on the main thread, so a
  session built on a worker thread is not found by the first lookup alone.
- :class:`tmlt.core.measurements.pandas_measurements.table.AddNoiseToColumn`
  resolves the dtype of the column it noises when it is built rather than when
  the column is written. A mechanism that was a *subclass* of one of the four
  it knows -- which the Spark twin handles by asking the mechanism for its
  ``output_type`` -- raised a bare :class:`KeyError` out of ``__call__``, after
  the noise had been drawn and the budget spent. Subclasses are accepted now,
  and a mechanism that cannot be resolved is reported when the measurement is
  constructed.
- The column descriptors in :mod:`tmlt.core.domains.pandas_domains` and a column
  they describe now agree about which of its values are null. The per-value
  check recognised ``float("nan")`` but not ``numpy.float32("nan")`` -- which is
  not a :class:`float`, where a ``numpy.float64`` is -- nor a raw
  ``numpy.datetime64("NaT")``, while :meth:`pandas.Series.isna`, which column
  validation goes through, makes no such distinction: an object column holding a
  ``float32`` NaN validated while the same value, handed to a map function's
  output row, did not. A NaN in a *float* column is still a value gated by
  ``allow_nan``, as before.
- :meth:`.PandasTimestampColumnDescriptor.valid_py_value` no longer accepts
  ``NaT`` when ``allow_null`` is False. ``NaTType`` subclasses
  :class:`datetime.datetime`, so a ``NaT`` was answered as an ordinary
  timezone-naive timestamp and the nullability branch was dead. It also rejects
  a value outside :attr:`pandas.Timestamp.min`/:attr:`~pandas.Timestamp.max`,
  which a described column's canonical ``datetime64[ns]`` dtype cannot hold:
  such a value passed validation and then failed, as a raw
  ``OutOfBoundsDatetime``, inside whatever went on to build the column. This is
  narrower than Spark's ``TimestampType``, which covers years 1 to 9999, and is
  documented on the descriptor as the engine limit it is.
- :func:`tmlt.core.utils.pandas_join.join` brings two ``datetime64`` join
  columns to the finer of their two units before comparing them, and returns the
  output join column in that unit. On pandas 2 the two sides need not be in the
  same unit; the output column was built in the left frame's dtype, so a
  right-only ``12:00:00.500`` came back as ``12:00:00`` against a left column of
  seconds, and a value outside the nanosecond range crashed with a
  pandas-internal ``AssertionError``. A value the finer unit cannot represent is
  now reported by name, with the column and both units.
- :func:`tmlt.core.utils.pandas_join.join` rejects two categorical join columns
  whose categories differ, instead of running as an inner or left join and
  raising a bare ``TypeError`` from inside pandas as an outer or right one.
- A categorical column's missing entries are nulls to
  :mod:`tmlt.core.utils.pandas_grouping` and to the join built on it. A
  categorical spells one as the code ``-1``, which reads back as ``np.nan`` --
  in a float or object column a *value* here, but in a categorical the only
  spelling there is, since pandas does not allow a NaN to be a category. They
  were grouped as NaNs, which gave a left join's fill of a categorical payload
  column a group of its own rather than the null group, and made
  ``nulls_are_equal`` inert for a categorical key.
- :class:`~tmlt.core.transformations.pandas_transformations.map.RowToRowTransformation`
  reports a map function that returns something other than a :class:`dict` --
  a :class:`~pyspark.sql.Row`, say, which the Spark implementation accepts --
  with an :class:`~.OutOfDomainError` naming what was returned and what is
  wanted, rather than with a bare assert that ``-O`` strips.

.. _v0.19.1:

0.19.1 - 2026-06-04
-------------------

Fixed
~~~~~
- Updated how internal helper functions hash columns to support infinite and nan values in PySpark Double and Float columns.

.. _v0.19.0:

0.19.0 - 2026-05-22
-------------------
This release adds support for PySpark 4 on Python 3.12 and truncation over multiple columns. It drops support for Python 3.9 on all platforms and older PySpark versions on Macs.

Added
~~~~~
- Added support for PySpark 4 on Python 3.12.

Changed
~~~~~~~
- Updated minimum randomgen version to 1.23 for Python 3.10.
- Dropped support for Python 3.9, as it has reached end-of-life.
- Dropped support for pyspark <3.5.0 on Macs after discovering that these configurations frequently crash. Older versions of the library may also be affected.
- Removed ``pytest`` and ``parameterized`` as dependencies.
  :mod:`tmlt.core.utils.testing` can now only be imported when the ``testing`` extra is installed;
  for most users, this module will not be used, and so the extra does not need to be installed.
- When returning intermediate values, the measurement creation functions in :mod:`tmlt.core.measurements.aggregations`
  now include the midpoints when returning grouped results, and use the user-specified names when returning scalar results.
- The truncation operations in :mod:`~tmlt.core.transformations.spark_transformations.truncation` support
  grouping by multiple columns, and :class:`~tmlt.core.metrics.IfGroupedBy` supports multiple grouping columns.
- :class:`~tmlt.core.transformations.spark_transformations.groupby.GroupBy` and :class:`tmlt.core.utils.grouped_dataframe.GroupedDataFrame`
  now accept None groupby keys to trigger a total aggregation. Empty dataframes can still be passed in, but when accessing the group keys
  they will always be null for a total aggregation (regardless of the way the object was constructed).

Fixed
~~~~~
- When returning intermediate values, :func:`~tmlt.core.measurements.aggregations.create_average_measurement` now names the sum column correctly by default (was previously `sum(None)`).
- :func:`~tmlt.core.utils.testing.assert_dataframe_equal` now detects that dataframes with no columns but different numbers of rows are not equivalent.

.. _v0.18.2:

0.18.2 - 2025-04-02
-------------------

Added
~~~~~
- Add LinkedIn announcement to CHANGELOG.rst.

.. _v0.18.1:

0.18.1 - 2025-03-17
-------------------

Changed
~~~~~~~
- We now support sympy versions >=1.10, <1.13.

.. _v0.18.0:

0.18.0 - 2025-01-14
-------------------
This release drops support for older versions of Python and Spark, improves the performance of bounds-finding, and makes additional minor miscellaneous changes.

Added
~~~~~
- :func:`~tmlt.core.utils.join.join` now supports ``left_anti`` joins. Note that the Core join transformations still do not support ``left_anti`` joins.

Changed
~~~~~~~
- The ``rng`` parameter to :func:`~tmlt.core.random.discrete_gaussian.sample_dgauss` has been removed, and it now always uses :func:`tmlt.core.random.rng.prng` as its random number generator.
- :class:`~tmlt.core.random.rng.RNGWrapper` has been moved into :mod:`tmlt.core.random.rng`.
- The parameter to :meth:`.RNGWrapper.randrange` has renamed from ``high`` to ``stop`` for consistency with the single-parameter version of :func:`random.randrange`.
- Refactor ``NoisyBounds`` to be more scalable. The new measurement is :class:`~.SparseVectorPrefixSums`, which is used in :func:`~.create_bounds_measurement` to construct the bounds measurement.
- Now requires PyArrow 18 or higher to remove any possibility of CVE-2024-52338.

Removed
~~~~~~~
- Python 3.8 and PySpark versions earlier than 3.3.1 are no longer supported.

Fixed
~~~~~
- Fixed a bug in ``NoisyBounds``, now :class:`~.SparseVectorPrefixSums`, that would try to select an upper bound larger than the maximum 64-bit integer, leading to an overflow.

Changed
~~~~~~~
- Improved performance of noise addition mechanisms under infinite budgets.

.. _v0.17.0:

0.17.0 - 2024-10-02
-------------------
This release changes the behavior of :class:`~tmlt.core.transformations.spark_transformations.map.RowToRowTransformation`, :class:`~.RowToRowsTransformation`, and :class:`~.RowsToRowsTransformation` (and thus :class:`~tmlt.core.transformations.spark_transformations.map.Map`, :class:`~.FlatMap`, and :class:`~.FlatMapByKey`) so that they catch many function outputs that would be invalid under their output domains.

.. note::

   Tumult Core 0.17 will be the last minor version to support Python 3.8 and PySpark versions below 3.3.1.
   If you are using Python 3.8 or one of these versions of PySpark, you will need to upgrade them in order to use Tumult Core 0.18.0.

Fixed
~~~~~
- :class:`~tmlt.core.transformations.spark_transformations.map.RowToRowTransformation`, :class:`~.RowToRowsTransformation`, and :class:`~.RowsToRowsTransformation` now all check that their outputs match their output domains, raising an exception if they do not.
  This should not impact correct Tumult Core programs, but may catch a few incorrect ones that were previously missed, and will improve the error messages produced in these cases.
- :class:`~tmlt.core.transformations.spark_transformations.map.RowToRowTransformation` and :class:`~.RowToRowsTransformation` now disallow mapping functions that produce values for the input columns when augmenting.

.. _v0.16.5:

0.16.5 - 2024-08-29
-------------------
This release fixes a bug in 0.16.3. CI problems meant 0.16.4 was unavailable.

Fixed
~~~~~
- Fixed an incorrect type declaration that caused typeguard errors.

.. _v0.16.3:

0.16.3 - 2024-08-22
-------------------
0.16.3 was yanked. The changes have been incorporated into 0.16.5.

This is a maintenance release that does not include user-visible changes.

.. _v0.16.2:

0.16.2 - 2024-08-14
-------------------

Fixed
~~~~~
- The :class:`~tmlt.core.transformations.spark_transformations.map.FlatMapByKey` transformation was incorrectly turning some NaNs into nulls and vice versa when converting the input dataframe into the input for the user-defined transformer function and when converting the output of that function back into a dataframe.
  This should no longer occur.

.. _v0.16.1:

0.16.1 - 2024-08-01
-------------------

Fixed
~~~~~
- Fixed bug in lower and upper bound tuple value ordering in :func:`~tmlt.core.measurements.aggregations.create_bounds_measurement`.
  The lower bound is now the first element and the upper bound is the second element.


.. _v0.16.0:

0.16.0 - 2024-07-29
-------------------

Added
~~~~~
- Added a way to construct a bounds measurement per-group using :func:`~tmlt.core.measurements.aggregations.create_bounds_measurement`.
- Added :class:`~tmlt.core.transformations.spark_transformations.map.FlatMapByKey`, a transformation for combining all records sharing a key under the ``IfGroupedBy("key", SymmetricDifference())`` metric into an arbitrary collection of other records with the same key using a user-defined function.
  In addition, added the :class:`~tmlt.core.transformations.spark_transformations.add_remove_keys.FlatMapByKeyValue` transformation, which performs this same operation on a table under an :class:`~tmlt.core.metrics.AddRemoveKeys` metric.
- Added :class:`~tmlt.core.transformations.spark_transformations.map.RowsToRowsTransformation`, a transformation mapping a set of records to another set of records using a user-defined function.

Changed
~~~~~~~
- Refactored bounds measurement to use a Pandas UDF. ``BoundSelection`` measurement was removed and equivalent ``NoisyBounds`` was added.
- Renamed ``create_bound_selection_measurement`` to :func:`~tmlt.core.measurements.aggregations.create_bounds_measurement`. The ``bound_column`` parameter was renamed to ``measure_column``.

Removed
~~~~~~~
- Removed support for Pandas 1.2 and 1.3 due to a known bug in Pandas versions below 1.4.

.. _v0.15.2:

0.15.2 - 2024-07-15
-------------------

Fixed
~~~~~
- Made :meth:`tmlt.core.utils.misc.get_nonconflicting_string` case-insensitive, since Spark is case insensitive by default.

.. _v0.15.1:

0.15.1 - 2024-07-05
-------------------

This release replaces Tumult Core 0.15.0, which was yanked.
Support for Pandas 2.0 has been reverted due to conflicts with PySpark.
Python 3.12 support should be considered experimental; a version with official support will be released once PySpark 4.0 becomes available.

.. _v0.15.0:

0.15.0 - 2024-06-26
-------------------

.. note:: Tumult Core 0.15.0 was yanked due to conflicts between PySpark and Pandas 2.0.

Added
~~~~~

- Added support for Python 3.12.

Removed
~~~~~~~

- Removed support for Python 3.7.

.. _v0.14.2:

0.14.2 - 2024-06-17
-------------------

Added
~~~~~

- Added support for left public joins to :class:`~.PublicJoin`, previously only inner joins were supported.

.. _v0.14.1:

0.14.1 - 2024-06-04
-------------------

Added
~~~~~

- Tumult Core now runs natively on Apple silicon, supporting Python 3.9 and above.

Removed
~~~~~~~

- Provided binary wheels for macOS now support only macOS 12 (Monterey) and above.

.. _v0.14.0:

0.14.0 - 2024-05-16
-------------------

Added
~~~~~
- :meth:`tmlt.core.utils.misc.get_materialized_df`, a utility function that materializes a Spark DataFrame. This is a public version of a previously internal function.

Fixed
~~~~~~~
- Stopped trying to set extra options for Java 11 and removed error when options are not set. Removed both ``check_java11()`` function and ``SparkConfigError`` exception.
- Updated minimum supported Spark version to 3.1.1 to prevent Java 11 error.

.. _v0.13.0:

0.13.0 - 2024-04-03
-------------------

Changed
~~~~~~~
- Updated :func:`~.calculate_noise_scale` to return a noise scale of 0 when both the
  ``d_in`` and ``d_out`` are infinite.
- Adjusted error messages related to spending privacy budgets in classes of type :class:`~.PrivacyBudget`.
- Moved InsufficientBudgetError from :mod:`~.interactive_measurements` to :mod:`~.measures`.
- Adjusted :meth:`tmlt.core.measurements.aggregations.create_variance_measurement` and :meth:`tmlt.core.measurements.aggregations.create_standard_deviation_measurement` to calculate sample variance and sample standard deviation instead of population variance and population standard deviation.
- In :class:`~tmlt.core.transformations.spark_transformations.groupby.GroupBy` and :class:`~.GroupedDataFrame` removed restriction on empty dataframes with non-empty columns.

Fixed
~~~~~
- SumGrouped now correctly handles the case with both empty input dataframes and empty group keys.
- SumGrouped, CountDistinct, and CountDistinctGrouped now always returns the correct output datatypes.
- :meth:`tmlt.core.domains.collections.DictDomain.validate` will no longer raise
  a ``TypeError`` when its dictionary keys cannot be sorted.

.. _v0.12.0:

0.12.0 - 2024-02-26
-------------------

Added
~~~~~
- Added a non-truncating truncation strategy with infinite stability.
- Added functions implementing various mechanisms to support slow scaling PRDP.

Changed
~~~~~~~
- Changed :func:`~tmlt.core.utils.truncation.truncate_large_groups` and
  :func:`~tmlt.core.utils.truncation.limit_keys_per_group` to use
  SHA-2 (256 bits) instead of Spark's default hash (Murmur3). This results in a minor
  performance hit, but these functions should be less likely to have collisions which
  could impact utility. **Note that this may change the output of transformations which
  use these functions.** In particular,
  :class:`~tmlt.core.transformations.spark_transformations.join.PrivateJoin`,
  :class:`~tmlt.core.transformations.spark_transformations.truncation.LimitRowsPerGroup`,
  :class:`~tmlt.core.transformations.spark_transformations.truncation.LimitKeysPerGroup`,
  and
  :class:`~tmlt.core.transformations.spark_transformations.truncation.LimitRowsPerKeyPerGroup`.
- Expanded the explanation of :class:`~.GroupingFlatMap`'s stability.
- Support all metrics for the flat map transformation.

Fixed
~~~~~
- Fixed missing minus sign in the documentation of the discrete Gaussian pmf.
- Fixed :func:`~.create_partition_selection_measurement` behavior when called
  with infinite budgets.
- Fixed :func:`~.create_partition_selection_measurement` crashing when called
  with very large budgets.


.. _v0.11.6:

0.11.6 - 2024-02-21
-------------------

0.11.6 was yanked. Those changes will be released in 0.12.0.


.. _v0.11.5:

0.11.5 - 2023-11-29
-------------------

Fixed
~~~~~
-  Addressed a serious security vulnerability in PyArrow: `CVE-2023-47248 <https://nvd.nist.gov/vuln/detail/CVE-2023-47248>`__.

   -  Python 3.8+ now requires PyArrow 14.0.1 or higher, which is the recommended fix and addresses the vulnerability.
   -  Python 3.7 uses the hotfix, as PyArrow 14.0.1 is not compatible with Python 3.7. Note that if you are using 3.7 the hotfix must be imported before your Spark code. Core imports the hotfix, so importing Core before Spark will also work.
   -  **It is strongly recommended to upgrade if you are using an older version of Core.**
   -  Also see the `GitHub Advisory entry <https://github.com/advisories/GHSA-5wvp-7f3h-6wmm>`__ for more information.

- Fixed a reference to an uninitialized variable that could cause :func:`~.arb_union` to crash the Python interpreter.

.. _v0.11.4:

0.11.4 - 2023-11-01
-------------------

Fixed a typo that prevented PyArrow from being installed on Python 3.8.

.. _v0.11.3:

0.11.3 - 2023-10-31
-------------------

Fixed a typo that prevented PySpark from being installed on Python 3.8.

.. _v0.11.2:

0.11.2 - 2023-10-27
-------------------

Added
~~~~~
- Added support for Python 3.11.

.. _v0.11.1:

0.11.1 - 2023-09-25
-------------------

Added
~~~~~
- Added documentation for known vulnerabilities related to Parallel Composition and the use of SymPy.

.. _v0.11.0:

0.11.0 - 2023-08-15
-------------------

Changed
~~~~~~~
- Replaced the `group_keys` for constructing :class:`~.SparkGroupedDataFrameDomain`\ s with `groupby_columns`.
- Modified :class:`~.SymmetricDifference` to define the distance
  between two elements of :class:`~.SparkGroupedDataFrameDomain`\ s to be infinite when the two elements have different `group_keys`.
- Updated maximum version for PySpark from 3.3.1 to 3.3.2.

.. _v0.10.2:

0.10.2 - 2023-07-18
-------------------

Changed
~~~~~~~
- Build wheels for macOS 11 instead of macOS 13.
- Updated dependency version for ``typing_extensions`` to 4.1.0

.. _v0.10.1:

0.10.1 - 2023-06-08
-------------------

Added
~~~~~
- Added support for Python 3.10.
- Added the :func:`~.arb_exp`, :func:`~.arb_const_pi`, :func:`~.arb_neg`, :func:`~.arb_product`, :func:`~.arb_sum`, :func:`~.arb_union`, :func:`~.arb_erf`, and :func:`~.arb_erfc` functions.
- Added a new error, :class:`~.DomainMismatchError`, which is raised when two or more domains should match but do not.
- Added a new error, :class:`~.UnsupportedMetricError`, which is raised when an unsupported metric is used.
- Added a new error, :class:`~.MetricMismatchError`, which is raised when two or more metrics should match but do not.
- Added a new error, :class:`~.UnsupportedMeasureError`, which is raised when an unsupported measure is used.
- Added a new error, :class:`~.MeasureMismatchError`, which is raised when two or more measures should match but do not.
- Added a new error, :class:`~.UnsupportedCombinationError`, which is raised when some combination of domain, metric, and measure is not supported (but each one is individually valid).
- Added a new error, :class:`~.UnsupportedNoiseMechanismError`, which is raised when a user tries to create a measurement with a noise mechanism that is not supported.
- Added a new error, :class:`~.UnsupportedSympyExprError`, which is raised when a user tries to create an :class:`~.ExactNumber` with an invalid SymPy expression.

Changed
~~~~~~~
- Restructured the repository to keep code under the ``src/`` directory.

.. _v0.10.0:

0.10.0 - 2023-05-17
-------------------

Added
~~~~~
- Added the `BoundSelection` spark measurement.

Changed
~~~~~~~
- Replaced many existing exceptions in Core with new classes that contain metadata about the inputs causing the exception.

Fixed
~~~~~
- Fixed bug in :func:`~tmlt.core.utils.truncation.limit_keys_per_group`.
- Fixed bug in :func:`~.gaussian`.
- :func:`~tmlt.core.utils.cleanup.cleanup` now emits a warning rather than an exception if it fails to get a Spark session.
  This should prevent unexpected exceptions in the ``atexit`` cleanup handler.

.. _v0.9.2:

0.9.2 - 2023-05-16
------------------

0.9.2 was yanked, as it contained breaking changes. Those changes will be released in 0.10.0.

.. _v0.9.1:

0.9.1 - 2023-04-20
------------------

Added
~~~~~
- Subclasses of :class:`~.Measure` now have equations defining the distance they represent.

.. _v0.9.0:

0.9.0 - 2023-04-14
------------------

Added
~~~~~

- :mod:`~.utils.join`, which contains utilities for validating join parameters, propogating domains through joins, and joining dataframes.

Changed
~~~~~~~

- :func:`~tmlt.core.utils.truncation.truncate_large_groups` does not clump identical records together in hash-based ordering.
- :class:`~.TransformValue` no longer fails when renaming the id column using :class:`~tmlt.core.transformations.spark_transformations.add_remove_keys.RenameValue`.

Fixed
~~~~~

- groupby no longer outputs nan values when both tables are views on the same original table
- private join no longer drops Nulls on non-join columns when join_on_nulls=False
- groupby average and variance no longer drops groups containing null values

.. _v0.8.3:

0.8.3 - 2023-03-08
------------------

Changed
~~~~~~~

- Functions in :mod:`~.aggregations` now support :class:`~.ApproxDP`.

.. _v0.8.2:

0.8.2 - 2023-03-02
------------------

Added
~~~~~
- Added :class:`~tmlt.core.transformations.spark_transformations.add_remove_keys.LimitKeysPerGroupValue` transformation

Changed
~~~~~~~
- Updated :class:`~tmlt.core.transformations.spark_transformations.truncation.LimitKeysPerGroup` to require an output metric, and to support the
  ``IfGroupedBy(grouping_column, SymmetricDifference())`` output metric. Dropped the ``use_l2`` parameter.

.. _v0.8.1:

0.8.1 - 2023-02-24
------------------

Added
~~~~~

- Added :class:`~tmlt.core.transformations.spark_transformations.truncation.LimitRowsPerKeyPerGroup` and :class:`~tmlt.core.transformations.spark_transformations.add_remove_keys.LimitRowsPerKeyPerGroupValue` transformations

Changed
~~~~~~~

- Faster implementation of :func:`~.discrete_gaussian_inverse_cmf`.

.. _v0.8.0:

0.8.0 - 2023-02-14
------------------

Added
~~~~~

- Added :class:`~tmlt.core.transformations.spark_transformations.add_remove_keys.LimitRowsPerGroupValue` transformation

Changed
~~~~~~~

- Updated :class:`~tmlt.core.transformations.spark_transformations.truncation.LimitRowsPerGroup` to require an output metric, and to support the
  ``IfGroupedBy(column, SymmetricDifference())`` output metric.
- Added a check so that :class:`~.TransformValue` can no longer be instantiated without
  subclassing.


.. _v0.7.0:

0.7.0 - 2023-02-02
------------------

Added
~~~~~

- Added measurement for adding Gaussian noise.

.. _v0.6.3:

0.6.3 - 2022-12-20
------------------

Changed
~~~~~~~

- On Linux, Core previously used `MPIR <https://en.wikipedia.org/wiki/MPIR_(mathematics_software)>`__ as a multi-precision arithmetic library to support `FLINT <https://flintlib.org/>`__ and `Arb <https://arblib.org/>`__.
  MPIR is no longer maintained, so Core now uses `GMP <https://gmplib.org/>`__ instead.
  This change does not affect macOS builds, which have always used GMP, and does not change Core's Python API.

Fixed
~~~~~

- Fixed a bug where PrivateJoin's privacy relation would only accept string keys in the d_in. It now accepts any type of key.


.. _v0.6.2:

0.6.2 - 2022-12-07
------------------

This is a maintenance release which introduces a number of documentation improvements, but has no publicly-visible API changes.

Fixed
~~~~~

- ``tmlt.core.utils.configuration.check_java11()`` now has the correct behavior when Java is not installed.

.. _v0.6.1:

0.6.1 - 2022-12-05
------------------

Added
~~~~~

-  Added approximate DP support to interactive mechanisms.
-  Added support for Spark 3.1 through 3.3, in addition to existing support for Spark 3.0.

Fixed
~~~~~

-  Validation for ``SparkedGroupDataFrameDomain``\ s used to fail with a Spark ``AnalysisException`` in some environments.
   That should no longer happen.

.. _v0.6.0:

0.6.0 - 2022-11-14
------------------

Added
~~~~~

-  Added new ``PrivateJoinOnKey`` transformation that works with ``AddRemoveKeys``.
-  Added inverse CDF methods to noise mechanisms.

.. _v0.5.1:

0.5.1 - 2022-11-03
------------------

Fixed
~~~~~

-  Domains and metrics make copies of mutable constructor arguments and return copies of mutable properties.

.. _v0.5.0:

0.5.0 - 2022-10-14
------------------

Changed
~~~~~~~

-  Core no longer depends on the ``python-flint`` package, and instead packages libflint and libarb itself.
   Binary wheels are available, and the source distribution includes scripting to build these dependencies from source.

Fixed
~~~~~

-  Equality checks on ``SparkGroupedDataFrameDomain``\ s used to occasionally fail with a Spark ``AnalysisException`` in some environments.
   That should no longer happen.
-  ``AddRemoveKeys`` now allows different names for the key column in each dataframe.

.. _v0.4.3:

0.4.3 - 2022-09-01
------------------

-  Core now checks to see if the user is running Java 11 or higher. If they are, Core either sets the appropriate Spark options (if Spark is not yet running) or raises an informative exception (if Spark is running and configured incorrectly).

.. _v0.4.2:

0.4.2 - 2022-08-24
------------------

Changed
~~~~~~~

-  Replaced uses of PySpark DataFrame’s ``intersect`` with inner joins. See https://issues.apache.org/jira/browse/SPARK-40181 for background.

.. _v0.4.1:

0.4.1 - 2022-07-25
------------------

Added
~~~~~

-  Added an alternate prng for non-intel architectures that don’t support RDRAND.
-  Add new metric ``AddRemoveKeys`` for multiple tables using ``IfGroupedBy(X, SymmetricDifference())``.
-  Add new ``TransformValue`` base class for wrapping transformations to support ``AddRemoveKeys``.
-  Add many new transformations using ``TransformValue``: ``FilterValue``, ``PublicJoinValue``, ``FlatMapValue``, ``MapValue``, ``DropInfsValue``, ``DropNaNsValue``, ``DropNullsValue``, ``ReplaceInfsValue``, ``ReplaceNaNsValue``, ``ReplaceNullsValue``, ``PersistValue``, ``UnpersistValue``, ``SparkActionValue``, ``RenameValue``, ``SelectValue``.

Changed
~~~~~~~

-  Fixed bug in ``ReplaceNulls`` to not allow replacing values for grouping column in ``IfGroupedBy``.
-  Changed ``ReplaceNulls``, ``ReplaceNaNs``, and ``ReplaceInfs`` to only support specific ``IfGroupedBy`` metrics.

.. _v0.3.2:

0.3.2 - 2022-06-23
------------------

Changed
~~~~~~~

-  Moved ``IMMUTABLE_TYPES`` from ``utils/testing.py`` to ``utils/type_utils.py`` to avoid importing nose when accessing ``IMMUTABLE_TYPES``.

.. _v0.3.1:

0.3.1 - 2022-06-23
------------------

Changed
~~~~~~~

-  Fixed ``copy_if_mutable`` so that it works with containers that can’t be deep-copied.
-  Reverted change from 0.3.0 “Add checks in ``ParallelComposition`` constructor to only permit L1/L2 over SymmetricDifference or AbsoluteDifference.”
-  Temporarily disabled flaky statistical tests.

.. _v0.3.0:

0.3.0 - 2022-06-22
------------------

Added
~~~~~

-  Added new transformations ``DropInfs`` and ``ReplaceInfs`` for handling infinities in data.
-  Added ``IfGroupedBy(X, SymmetricDifference())`` input metric.

   -  Added support for this metric to ``Filter``, ``Map``, ``FlatMap``, ``PublicJoin``, ``Select``, ``Rename``, ``DropNaNs``, ``DropNulls``, ``DropInfs``, ``ReplaceNulls``, ``ReplaceNaNs``, and ``ReplaceInfs``.

-  Added new truncation transformations for ``IfGroupedBy(X, SymmetricDifference())``: ``LimitRowsPerGroup``, ``LimitKeysPerGroup``
-  Added ``AddUniqueColumn`` for switching from ``SymmetricDifference`` to ``IfGroupedBy(X, SymmetricDifference())``.
-  Added a topic guide around NaNs, nulls and infinities.

Changed
~~~~~~~

-  Moved truncation transformations used by ``PrivateJoin`` to be functions (now in ``utils/truncation.py``).
-  Change ``GroupBy`` and ``PartitionByKeys`` to have an ``use_l2`` argument instead of ``output_metric``.
-  Fixed bug in ``AddUniqueColumn``.
-  Operations that group on null values are now supported.
-  Modify ``CountDistinctGrouped`` and ``CountDistinct`` so they work as expected with null values.
-  Changed ``ReplaceNulls``, ``ReplaceNaNs``, and ``ReplaceInfs`` to only support specific ``IfGroupedBy`` metrics.
-  Fixed bug in ``ReplaceNulls`` to not allow replacing values for grouping column in ``IfGroupedBy``.
-  ``PrivateJoin`` has a new parameter for ``__init__``: ``join_on_nulls``.
   When ``join_on_nulls`` is ``True``, the ``PrivateJoin`` can join null values between both dataframes.
-  Changed transformations and measurements to make a copy of mutable constructor arguments.
-  Add checks in ``ParallelComposition`` constructor to only permit L1/L2 over SymmetricDifference or AbsoluteDifference.

Removed
~~~~~~~

-  Removed old examples from ``examples/``.
   Future examples will be added directly to the documentation.

.. _v0.2.0:

0.2.0 - 2022-04-12 (internal release)
-------------------------------------

Added
~~~~~

-  Added ``SparkDateColumnDescriptor`` and ``SparkTimestampColumnDescriptor``, enabling support for Spark dates and timestamps.
-  Added two exception types, ``InsufficientBudgetError`` and ``InactiveAccountantError``, to PrivacyAccountants.
-  Future documentation will include any exceptions defined in this library.
-  Added ``cleanup.remove_all_temp_tables()`` function, which will remove all temporary tables created by Core.
-  Added new components ``DropNaNs``, ``DropNulls``, ``ReplaceNulls``, and ``ReplaceNaNs``.

.. _v0.1.1:

0.1.1 - 2022-02-24 (internal release)
-------------------------------------

Added
~~~~~

-  Added new implementations for SequentialComposition and ParallelComposition.
-  Added new spark transformations: Persist, Unpersist and SparkAction.
-  Added PrivacyAccountant.
-  Installation on Python 3.7.1 through 3.7.3 is now allowed.
-  Added ``DecorateQueryable``, ``DecoratedQueryable`` and ``create_adaptive_composition`` components.

Changed
~~~~~~~

-  Fixed a bug where ``create_quantile_measurement`` would always be created with PureDP as the output measure.
-  ``PySparkTest`` now runs ``tmlt.core.utils.cleanup.cleanup()`` during ``tearDownClass``.
-  Refactored noise distribution tests.
-  Remove sorting from ``GroupedDataFrame.apply_in_pandas`` and ``GroupedDataFrame.agg``.
-  Repartition DataFrames output by ``SparkMeasurement`` to prevent privacy violation.
-  Updated repartitioning in ``SparkMeasurement`` to use a random column.
-  Changed quantile implementation to use arblib.
-  Changed Laplace implementation to use arblib.

Removed
~~~~~~~

-  Removed ``ExponentialMechanism`` and ``PermuteAndFlip`` components.
-  Removed ``AddNoise``, ``AddLaplaceNoise``, ``AddGeometricNoise``, and ``AddDiscreteGaussianNoise`` from ``tmlt.core.measurements.pandas.series``.
-  Removed ``SequentialComposition``, ``ParallelComposition`` and corresponding Queryables from ``tmlt.core.measurements.composition``.
-  Removed ``tmlt.core.transformations.cache``.

.. _v0.1.0:

0.1.0 - 2022-02-14 (internal release)
-------------------------------------

Added
~~~~~

-  Initial release.
