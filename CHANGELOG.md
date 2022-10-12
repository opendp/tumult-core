# Changelog

## Unreleased
### Changed
- Core no longer depends on the `python-flint` package, and instead packages libflint and libarb itself.
  Binary wheels are available, and the source distribution includes scripting to build these dependencies from source.

### Fixed
- Equality checks on `SparkGroupedDataFrameDomain`s used to occasionally fail with a Spark `AnalysisException` in some environments.
  That should no longer happen.
- `AddRemoveKeys` now allows different names for the key column in each dataframe.

## 0.4.3 - 2022-09-01
- Core now checks to see if the user is running Java 11 or higher. If they are, Core either sets the appropriate Spark options (if Spark is not yet running) or raises an informative exception (if Spark is running and configured incorrectly).

## 0.4.2 - 2022-08-24
### Changed
- Replaced uses of PySpark DataFrame's `intersect` with inner joins. See https://issues.apache.org/jira/browse/SPARK-40181 for background.

## 0.4.1 - 2022-07-25
### Added
- Added an alternate prng for non-intel architectures that don't support RDRAND.
- Add new metric `AddRemoveKeys` for multiple tables using `IfGroupedBy(X, SymmetricDifference())`.
- Add new `TransformValue` base class for wrapping transformations to support `AddRemoveKeys`.
- Add many new transformations using `TransformValue`: `FilterValue`, `PublicJoinValue`, `FlatMapValue`, `MapValue`, `DropInfsValue`, `DropNaNsValue`, `DropNullsValue`, `ReplaceInfsValue`, `ReplaceNaNsValue`, `ReplaceNullsValue`, `PersistValue`, `UnpersistValue`, `SparkActionValue`, `RenameValue`, `SelectValue`.


### Changed
- Fixed bug in `ReplaceNulls` to not allow replacing values for grouping column in `IfGroupedBy`
- Changed `ReplaceNulls`, `ReplaceNaNs`, and `ReplaceInfs` to only support specific `IfGroupedBy` metrics


## 0.3.2 - 2022-06-23
### Changed
- Moved `IMMUTABLE_TYPES` from utils/testing.py to utils/type_utils to avoid importing nose when accessing `IMMUTABLE_TYPES`

## 0.3.1 - 2022-06-23
### Changed
- Fixed `copy_if_mutable` so that it works with containers that can't be deep-copied.
- Reverted change from 0.3.0 "Add checks in `ParallelComposition` constructor to only permit L1/L2 over SymmetricDifference or AbsoluteDifference."
- Temporarily disabled flaky statistical tests.

## 0.3.0 - 2022-06-22
### Added
- Added new transformations `DropInfs` and `ReplaceInfs` for handling infinities in data.
- Added `IfGroupedBy(X, SymmetricDifference())` input metric.
  - Added support for this metric to `Filter`, `Map`, `FlatMap`, `PublicJoin`, `Select`, `Rename`, `DropNaNs`, `DropNulls`, `DropInfs`, `ReplaceNulls`, `ReplaceNaNs`, and `ReplaceInfs`.
- Added new truncation transformations for `IfGroupedBy(X, SymmetricDifference())`: `LimitRowsPerGroup`, `LimitKeysPerGroup`
- Added `AddUniqueColumn` for switching from `SymmetricDifference` to `IfGroupedBy(X, SymmetricDifference())`.
- Added a topic guide around NaNs, nulls and infinities.

### Changed
- Moved truncation transformations used by `PrivateJoin` to be functions (now in `utils/truncation.py`).
- Change `GroupBy` and `PartitionByKeys` to have an `use_l2` argument instead of `output_metric`.
- Fixed bug in `AddUniqueColumn`.
- Operations that group on null values are now supported.
- Modify `CountDistinctGrouped` and `CountDistinct` so they work as expected with null values.
- Changed `ReplaceNulls`, `ReplaceNaNs`, and `ReplaceInfs` to only support specific `IfGroupedBy` metrics.
- Fixed bug in `ReplaceNulls` to not allow replacing values for grouping column in `IfGroupedBy`.
- `PrivateJoin` has a new parameter for `__init__`: `join_on_nulls`.
  When `join_on_nulls` is `True`, the `PrivateJoin` can join null values between both dataframes.
- Changed transformations and measurements to make a copy of mutable constructor arguments.
- Add checks in `ParallelComposition` constructor to only permit L1/L2 over SymmetricDifference or AbsoluteDifference.

### Removed
- Removed old examples from `examples/`.
  Future examples will be added directly to the documentation.

## 0.2.0 - 2022-04-12 (internal release)
### Added
- Added `SparkDateColumnDescriptor` and `SparkTimestampColumnDescriptor`, enabling support for Spark dates and timestamps.
- Added two exception types, `InsufficientBudgetError` and `InactiveAccountantError`, to PrivacyAccountants.
- Future documentation will include any exceptions defined in this library.
- Added `cleanup.remove_all_temp_tables()` function, which will remove all temporary tables created by Core.
- Added new components `DropNaNs`, `DropNulls`, `ReplaceNulls`, and `ReplaceNaNs`.

## 0.1.1 - 2022-02-24 (internal release)
### Added
- Added new implementations for SequentialComposition and ParallelComposition.
- Added new spark transformations: Persist, Unpersist and SparkAction.
- Added PrivacyAccountant.
- Installation on Python 3.7.1 through 3.7.3 is now allowed.
- Added `DecorateQueryable`, `DecoratedQueryable` and `create_adaptive_composition` components.

### Changed
- Fixed a bug where `create_quantile_measurement` would always be created with PureDP as the output measure.
- `PySparkTest` now runs `tmlt.core.utils.cleanup.cleanup()` during `tearDownClass`.
- Refactored noise distribution tests.
- Remove sorting from `GroupedDataFrame.apply_in_pandas` and `GroupedDataFrame.agg`.
- Repartition DataFrames output by `SparkMeasurement` to prevent privacy violation.
- Updated repartitioning in `SparkMeasurement` to use a random column.
- Changed quantile implementation to use arblib.
- Changed Laplace implementation to use arblib.

### Removed
- Removed `ExponentialMechanism` and `PermuteAndFlip` components.
- Removed `AddNoise`, `AddLaplaceNoise`, `AddGeometricNoise`, and `AddDiscreteGaussianNoise` from `tmlt.core.measurements.pandas.series`.
- Removed `SequentialComposition`, `ParallelComposition` and corresponding Queryables from `tmlt.core.measurements.composition`.
- Removed `tmlt.core.transformations.cache`.

## 0.1.0 - 2022-02-14 (internal release)
### Added
- Initial release.
