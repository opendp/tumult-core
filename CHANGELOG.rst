.. _core-changelog:

Changelog
=========

🚨 Important Update: the Tumult Labs Team is Joining LinkedIn 🚨
----------------------------------------------------------------

The `Tumult Labs team has joined LinkedIn <https://www.linkedin.com/pulse/whats-next-us-tumult-labs-gerome-miklau-zmpye>`__! 🎉 As part of this transition, we are exploring options for the future of Tumult Core, including finding a new home for the project. 🏡
We greatly appreciate the community’s support and contributions. If your organization is interested in maintaining or adopting Tumult Core, please reach out! 📩
For now, the repository remains available, and we encourage users to continue engaging with the project. We’ll provide updates as soon as we have more to share.

— The Tumult Labs Team 💙

Unreleased
----------

Changed
~~~~~~~
- Dropped support for Python 3.9, as it has reached end-of-life.
- Dropped support for pyspark <3.5.0 on Macs after discovering that these configurations frequently crash. Older versions of the library may also be affected.

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
This release changes the behavior of :class:`~.RowToRowTransformation`, :class:`~.RowToRowsTransformation`, and :class:`~.RowsToRowsTransformation` (and thus :class:`~.Map`, :class:`~.FlatMap`, and :class:`~.FlatMapByKey`) so that they catch many function outputs that would be invalid under their output domains.

.. note::

   Tumult Core 0.17 will be the last minor version to support Python 3.8 and PySpark versions below 3.3.1.
   If you are using Python 3.8 or one of these versions of PySpark, you will need to upgrade them in order to use Tumult Core 0.18.0.

Fixed
~~~~~
- :class:`~.RowToRowTransformation`, :class:`~.RowToRowsTransformation`, and :class:`~.RowsToRowsTransformation` now all check that their outputs match their output domains, raising an exception if they do not.
  This should not impact correct Tumult Core programs, but may catch a few incorrect ones that were previously missed, and will improve the error messages produced in these cases.
- :class:`~.RowToRowTransformation` and :class:`~.RowToRowsTransformation` now disallow mapping functions that produce values for the input columns when augmenting.

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
- In :class:`~.GroupBy` and :class:`~.GroupedDataFrame` removed restriction on empty dataframes with non-empty columns.

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
- Changed :func:`~.truncate_large_groups` and :func:`~.limit_keys_per_group` to use
  SHA-2 (256 bits) instead of Spark's default hash (Murmur3). This results in a minor
  performance hit, but these functions should be less likely to have collisions which
  could impact utility. **Note that this may change the output of transformations which
  use these functions.** In particular, :class:`~.PrivateJoin`,
  :class:`~.LimitRowsPerGroup`, :class:`~.LimitKeysPerGroup`, and
  :class:`~.LimitRowsPerKeyPerGroup`.
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
- Fixed bug in :func:`~.limit_keys_per_group`.
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

- :func:`~.truncate_large_groups` does not clump identical records together in hash-based ordering.
- :class:`~.TransformValue` no longer fails when renaming the id column using :class:`~.RenameValue`.

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
- Added :class:`~.LimitKeysPerGroupValue` transformation

Changed
~~~~~~~
- Updated :class:`~.LimitKeysPerGroup` to require an output metric, and to support the
  ``IfGroupedBy(grouping_column, SymmetricDifference())`` output metric. Dropped the ``use_l2`` parameter.

.. _v0.8.1:

0.8.1 - 2023-02-24
------------------

Added
~~~~~

- Added :class:`~.LimitRowsPerKeyPerGroup` and :class:`~.LimitRowsPerKeyPerGroupValue` transformations

Changed
~~~~~~~

- Faster implementation of :func:`~.discrete_gaussian_inverse_cmf`.

.. _v0.8.0:

0.8.0 - 2023-02-14
------------------

Added
~~~~~

- Added :class:`~.LimitRowsPerGroupValue` transformation

Changed
~~~~~~~

- Updated :class:`~.LimitRowsPerGroup` to require an output metric, and to support the
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
