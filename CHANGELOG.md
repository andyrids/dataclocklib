# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- Added | Changed | Deprecated | Removed | Fixed -->

## [0.1.8] - 2025-01-20

### Added

- Basic overview guide on documentation site
- Dynamic 'optimal' figure size calculation based on total wedge count.
- Dynamic chart annotation font scaling & spacing adjustment.
- Dynamic polar axis label font scaling based on number of rings.
- Configuration files (`dataclocklib/config/`) for default chart title & subtitle creation.
- Dataclock *kwargs* `**fg_kw` added, which aligns with `pyplot.subplots`.
  - Figure size (figsize) parameter will be overwritten and must be modified with the returned Figure object.

### Changed

- Moved custom types to `dataclocklib.typing`.
  - `ColorMap` type changed to `CmapNames`.
- Moved colorbar logic to `dataclocklib.utility.add_colorbar`.
- Dataclock arguments; `chart_title`, `chart_subtitle`, `chart_period`, `chart_source` are keyword only.


## [0.1.7] - 2025-01-15

### Added

- Extra unit tests:
  - Test aggregation values for different chart modes.
  - Test figure generation for different chart modes.
  - Test custom chart annotation text values.
- Error handling for empty DataFrame & wrong data type.

### Changed

- Parameter 'default_text' triggers default chart title and subtitle annotations if chart_title & chart_subtitle are None.
- Parameter 'chart_period' for optional annotation below subtitle for dataset reporting period.
- Raises ValueError if data[date_column] Series does not have not a 'datetime64[ns]' data type.
- Raises ValueError if data is an empty DataFrame.

### Fixed

- Ring & wedge value generation inefficiencies (~75% improvement).
- Redundant inner loop for wedge bar creation.
- Divide by zero error when passed an empty DataFrame.
- Leap year ring values changed from 53 to 52 in 'YEAR_WEEK' mode.

## [0.1.6] - 2025-01-14

### Changed

- Tutorial updates & improvements.

## [0.1.5] - 2025-01-10

### Added

- Jupyter Notebook Tutorial for documentation.

## [0.1.4] - 2025-01-09

### Added

- PyPI deployment.
- Pytest functions added.

## [0.1.3] - 2025-01-08

### Added

- README documentation.
- GitHub action for GitHub page deployment.

### Changed

- Astral uv workflow job added to actions.

## [0.1.2] - 2025-01-07

### Added

- Sphinx documentation.
- GitHub action for GitHub page deployment.

### Changed

- Matplotlib colormap use instead of custom colormap.

## [0.1.1] - 2025-01-06

### Added

- DOW_HOUR chart mode. Chart rings are Monday - Sunday and wedges are 24 hour periods.
- Pytest functionality for matplotlib chart generation.

### Changed

- Wedge labels rotate around polar axis.

## [0.1.0] - 2025-01-05

### Added

- Initial data clock chart.
