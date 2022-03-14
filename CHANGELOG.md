# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).


## 0.1.1 - 2022-02-24
### Added
- Added new implementations for SequentialComposition and ParallelComposition.
- Added new spark transformations: Persist, Unpersist and SparkAction.
- Added PrivacyAccountant.
- Installation on Python 3.7.1 through 3.7.3 is now allowed.

### Changed
- Fixed a bug where create_quantile_measurement would always be created with PureDP as the output measure.
- `PySparkTest` now runs `tmlt.core.utils.cleanup.cleanup()` during `tearDownClass`.

## 0.1.0 - 2022-02-14
### Added
- Initial release
