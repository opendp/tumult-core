.. _core-changelog:

Changelog
=========

0.8.2 - 2023-03-02
------------------

Added
~~~~~
- Added `LimitKeysPerGroupValue` transformation

Changed
~~~~~~~
- Updated `LimitKeysPerGroup` to require an output metric, and to support the
  `IfGroupedBy(grouping_column, SymmetricDifference())` output metric. Dropped the 'use_l2' parameter.

0.8.1 - 2023-02-24
------------------

Added
~~~~~

- Added `LimitRowsPerKeyPerGroup` and `LimitRowsPerKeyPerGroupValue` transformations

Changed
~~~~~~~

- Faster implementation of discrete_gaussian_inverse_cmf.

0.8.0 - 2023-02-14
------------------

Added
~~~~~

- Added `LimitRowsPerGroupValue` transformation

Changed
~~~~~~~

- Updated `LimitRowsPerGroup` to require an output metric, and to support the
  `IfGroupedBy(column, SymmetricDifference())` output metric.
- Added a check so that `TransformValue` can no longer be instantiated without
  subclassing.


0.7.0 - 2023-02-02
------------------

Added
~~~~~

- Added measurement for adding Gaussian noise.

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


0.6.2 - 2022-12-07
------------------

This is a maintenance release which introduces a number of documentation improvements, but has no publicly-visible API changes.

Fixed
~~~~~

- :func:`~tmlt.core.utils.configuration.check_java11()` now has the correct behavior when Java is not installed.

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

0.6.0 - 2022-11-14
------------------

Added
~~~~~

-  Added new ``PrivateJoinOnKey`` transformation that works with ``AddRemoveKeys``.
-  Added inverse CDF methods to noise mechanisms.

0.5.1 - 2022-11-03
------------------

Fixed
~~~~~

-  Domains and metrics make copies of mutable constructor arguments and return copies of mutable properties.

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

0.4.3 - 2022-09-01
------------------

-  Core now checks to see if the user is running Java 11 or higher. If they are, Core either sets the appropriate Spark options (if Spark is not yet running) or raises an informative exception (if Spark is running and configured incorrectly).

0.4.2 - 2022-08-24
------------------

Changed
~~~~~~~

-  Replaced uses of PySpark DataFrame’s ``intersect`` with inner joins. See https://issues.apache.org/jira/browse/SPARK-40181 for background.

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

0.3.2 - 2022-06-23
------------------

Changed
~~~~~~~

-  Moved ``IMMUTABLE_TYPES`` from ``utils/testing.py`` to ``utils/type_utils.py`` to avoid importing nose when accessing ``IMMUTABLE_TYPES``.

0.3.1 - 2022-06-23
------------------

Changed
~~~~~~~

-  Fixed ``copy_if_mutable`` so that it works with containers that can’t be deep-copied.
-  Reverted change from 0.3.0 “Add checks in ``ParallelComposition`` constructor to only permit L1/L2 over SymmetricDifference or AbsoluteDifference.”
-  Temporarily disabled flaky statistical tests.

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

0.2.0 - 2022-04-12 (internal release)
-------------------------------------

Added
~~~~~

-  Added ``SparkDateColumnDescriptor`` and ``SparkTimestampColumnDescriptor``, enabling support for Spark dates and timestamps.
-  Added two exception types, ``InsufficientBudgetError`` and ``InactiveAccountantError``, to PrivacyAccountants.
-  Future documentation will include any exceptions defined in this library.
-  Added ``cleanup.remove_all_temp_tables()`` function, which will remove all temporary tables created by Core.
-  Added new components ``DropNaNs``, ``DropNulls``, ``ReplaceNulls``, and ``ReplaceNaNs``.

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

0.1.0 - 2022-02-14 (internal release)
-------------------------------------

Added
~~~~~

-  Initial release.
