# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## Unreleased
### Added
- Added support for `IfGroupedBy(X, SymmetricDifference())` to `DropNaNs`, `DropNulls`, `DropInfs`, `ReplaceNulls`, `ReplaceNaNs`, and `ReplaceInfs`

### Changed
- Fixed bug in `ReplaceNulls` to not allow replacing values for grouping column in `IfGroupedBy`
- Changed `ReplaceNulls`, `ReplaceNaNs`, and `ReplaceInfs` to only support specific `IfGroupedBy` metrics

## 0.3.0-alpha.4 - 2022-05-27
### Added
- Added new transformations `DropInfs` and `ReplaceInfs`.
- Added topic guide around NaNs, nulls and infs.

### Changed
- Fixed bug in `AddUniqueColumn`
- Modify `CountDistinctGrouped` and `CountDistinct` so it works as expected with null values.

## 0.3.0-alpha.3 - 2022-05-19
### Added
- Added `IfGroupedBy(X, SymmetricDifference())` input metric
  - Added support for this metric to `Filter`, `Map`, `FlatMap`, `PublicJoin`, `Select`, and `Rename`
- Added new truncation transformations for `IfGroupedBy(X, SymmetricDifference())`
  - `LimitRowsPerGroup`, `LimitKeysPerGroup`
- Added `AddUniqueColumn` for switching from `SymmetricDifference` to `IfGroupedBy(X, SymmetricDifference())`

### Changed
- `PrivateJoin` has a new parameter for `__init__`: `join_on_nulls`. When `join_on_nulls` is `True`, the `PrivateJoin` can join null values between both dataframes.
- Operations that group on null values are now supported.
- Moved truncation transformations used by `PrivateJoin` to be functions (now in `utils.truncation.py`).
- Change `GroupBy` and `PartitionByKeys` to have an `use_l2` argument instead of `output_metric` 

### Removed
- Removed old examples from `examples/`, future examples will be added directly to the documentation.

## 0.2.0 - 2022-04-12
### Added
- Added `SparkDateColumnDescriptor` and `SparkTimestampColumnDescriptor`, enabling support for Spark dates and timestamps.
- Added two exception types, `InsufficientBudgetError` and `InactiveAccountantError`, to PrivacyAccountants.
- Future documentation will include any exceptions defined in this library.
- Added `cleanup.remove_all_temp_tables()` function, which will remove all temporary tables created by Core.
- Added new components `DropNaNs`, `DropNulls`, `ReplaceNulls`, and `ReplaceNaNs`.

## 0.1.1 - 2022-02-24
### Added
- Added new implementations for SequentialComposition and ParallelComposition.
- Added new spark transformations: Persist, Unpersist and SparkAction.
- Added PrivacyAccountant.
- Installation on Python 3.7.1 through 3.7.3 is now allowed.
- Added `DecorateQueryable`, `DecoratedQueryable` and `create_adaptive_composition` components.

### Changed
- Fixed a bug where create_quantile_measurement would always be created with PureDP as the output measure.
- `PySparkTest` now runs `tmlt.core.utils.cleanup.cleanup()` during `tearDownClass`.
- Refactored noise distribution tests.
- Remove sorting from `GroupedDataFrame.apply_in_pandas` and `GroupedDataFrame.agg`.
- Repartition DataFrames output by `SparkMeasurement` to prevent privacy violation.
- Updated repartitioning in `SparkMeasurement` to use a random column.
- Changed quantile implementation to use arblib.
- Changed Laplace implementation to use arblib.

### Removed
- Removed `ExponentialMechanism` and `PermuteAndFlip` components.
- Removed `AddNoise`, `AddLaplaceNoise`, `AddGeometricNoise`, and `AddDiscreteGaussianNoise` from
  `tmlt.core.measurements.pandas.series`.
- Removed `SequentialComposition`, `ParallelComposition` and corresponding Queryables from
  `tmlt.core.measurements.composition`.
- Removed `tmlt.core.transformations.cache`.

## 0.1.0 - 2022-02-14
### Added
- Initial release
