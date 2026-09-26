<!-- pyml disable MD024 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> [!NOTE]
>
> - `[SemVer] - yyyy-mm-dd` or `[Unreleased]` for release heading
> - `Added` for new features.
> - `Changed` for changes in existing functionality.
> - `Deprecated` for soon-to-be removed features.
> - `Removed` for now removed features.
> - `Fixed` for any bug fixes.
> - `Security` in case of vulnerabilities.

## [Unreleased]

### Changed

- Package metadata improved for discoverability: a more descriptive summary, expanded
  keywords (e.g. `radial-heatmap`, `time-series`, `visualization`) and the
  `Topic :: Scientific/Engineering :: Visualization`,
  `Topic :: Scientific/Engineering :: Information Analysis` and `Typing :: Typed` classifiers.

## [0.3.0] - 2026-09-26

### Breaking changes

- Python 3.10 and pandas below 3.0 are no longer supported; pin `dataclocklib<0.3` to stay on
  those versions.
- `assign_temporal_columns` raises `ModeError` (a `ValueError`) instead of `KeyError` for an
  unexpected mode value; catch `ValueError` or `ModeError`.
- A non-numeric `agg_column` (e.g. string, datetime or object dtype) with an aggregation other
  than `'count'` now raises `AggregationColumnError`; bool columns are still accepted.
- `'YEAR_WEEK'` and `'WEEK_DAY'` output differs at year boundaries (see Fixed), and the
  DataFrames returned by `dataclock`, `line_chart` and `aggregate_temporal_columns` are sorted
  in chronological ring order; select rows by label (`.loc`), not position (`.iloc`).

### Added

- `publish.yml` workflow: PyPI trusted publishing (OIDC) on `v*` tags, via a `pypi`
  environment, after checking that the tag matches the built version.
- Release and tagged docs builds pin the version to the git tag and refuse to build from a
  dirty working tree.
- `just docs`, `just docs-serve` and `just docs-clean` recipes.
- `.gitattributes` normalising line endings to LF (binary images and data files untouched).
- `py.typed` marker, so type checkers recognise the package's inline type hints.
- `add_colorbar` gains a trailing `vmin` keyword (default `1`), so the colour normalisation
  range is configurable; existing positional callers are unaffected.
- `dataclock` and `line_chart` now raise a clear `MissingDatetimeError` for `NaT` values in
  `date_column`, and `AggregationColumnError` for a non-numeric `agg_column` or one named
  `'ring'`/`'wedge'` (reserved names), each with a reason in the error message.

### Changed

- Build backend is now hatchling + hatch-vcs (was setuptools + setuptools_scm); the version
  still comes from the git tag. The sdist only contains the package source, README,
  CHANGELOG and licence files.
- Documentation is built with `just docs` (`sphinx-build -W`, warnings are errors), locally
  and in CI; the API reference uses fully-qualified `dataclocklib.*` names.
- CI tests on Windows and Linux with Python 3.11, 3.13 and 3.14, installs from `uv.lock`,
  builds and checks the distributions, and pins actions to commit SHAs.
- The GitLab package index is `explicit`: only `pkgdx` is resolved from it, every other
  package comes from PyPI.
- `just setup` installs all extras, and `just secrets-baseline` updates `.secrets.baseline`
  in place instead of regenerating it.
- Licence metadata uses the SPDX expression `GPL-3.0-or-later` (PEP 639); `LICENSE` and
  `COPYRIGHT` are included in distributions.
- Module docstrings carry an SPDX licence identifier instead of the full GPL notice.
- Minimum supported versions are now Python 3.11 and pandas 3.0.
- Lint, format, typing, markdown & secrets checks use pkgdx standards via prek hooks.
- CI runs the prek hooks instead of standalone Ruff steps.
- Chart functions refactored into smaller private helpers; the module-level `charts.config`
  object was removed along with them, so it is no longer part of the public API.
- `MissingDatetimeError` message now states that a naive datetime64 dtype is expected.
- `assign_temporal_columns` now raises `ModeError` (a `ValueError`) for an unexpected
  mode value; it previously raised a `KeyError` from the underlying `astype` call.
- `dataclock`'s docstring now documents that `fig_kw` always overrides `figsize` and
  `constrained_layout`; other keys, such as `dpi`, are passed through to `pyplot.subplots`.

### Removed

- Redundant Ruff and Pyright configuration from `pyproject.toml`.
- `docs/Makefile`, `docs/make.bat`, `readthedocs.yml` and the unused `docs/Contributing.rst`.
- The unused `[[tool.bumpversion.files]]` rule.

### Fixed

- Colour scale: `'count'` aggregations keep the 1..max scale, while every other aggregation
  now scales from the data minimum to the data maximum, so values below 1 are no longer
  clipped to the same colour. Empty bins stay white; a real zero or negative value is coloured.
- Colorbar ticks from `add_colorbar` are unique, and integer ticks are only used for integral
  data, so a float sum's top tick now equals the true maximum.
- Chart title, subtitle and period font sizes are no longer scaled twice, so large charts
  (e.g. `'WEEK_DAY'` over multiple years) render text at the intended size and no longer
  overlap or run outside the figure.
- Naive datetime64 columns with non-nanosecond resolution (us, ms, s) are now accepted.
- Matplotlib `Colormap.set_under` pending deprecation warning.
- Test data paths no longer depend on the pytest working directory.
- 'WEEK_DAY' rings use the ISO year, so days at the turn of a year are no longer
  merged into the wrong week (e.g. 2013-12-30 is now in ring 201401, not 201301). A
  calendar-year slice can therefore start or end with a partial ISO-week ring, whose days
  outside the data are empty.
- 'YEAR_WEEK' weeks no longer cross calendar years; early-January days in ISO week
  52/53 are placed in week 1 and late-December days in ISO week 1 in week 52.
- Rings are drawn in chronological order in every mode, regardless of input row order; the
  DataFrame returned by `aggregate_temporal_columns` is sorted in the same ring order.
- 'DAY_HOUR' mode documentation now states days 1 - 366 (previously 356).
- Wheels built without git metadata (e.g. from a source archive) were missing the
  `config/*.ini` files, so `dataclock()` failed at runtime.
- API documentation referenced functions that no longer exist and omitted `line_chart`
  and `add_wedge_labels`.
- The documentation build no longer executes the notebooks (which installed packages and
  downloaded data over the network); the stored notebook outputs are rendered instead.

## [0.2.0] - 2025-01-23

### Added

- PyPalettes library dependency, providing 2500+ palettes.
- Colormap reverse flag parameter added to dataclock function.
- Chart polar spine color parameter added to dataclock.
- Chart polar grid color parameter added to dataclock.

### Changed

- Wedge label logic moved to `dataclocklib.utility.add_wedge_labels`.
- Temporal aggregation logic moved to `dataclocklib.utility.aggregate_temporal_columns`.

## [0.1.8] - 2025-01-20

### Added

- Basic overview guide on documentation site
- Dynamic 'optimal' figure size calculation based on total wedge count.
- Dynamic chart annotation font scaling & spacing adjustment.
- Dynamic polar axis label font scaling based on number of rings.
- Configuration files (`dataclocklib/config/`) for default chart title & subtitle creation.
- Dataclock *kwargs* `**fg_kw` added, which aligns with `pyplot.subplots`.
  - Figure size (figsize) parameter will be overwritten and must be modified with the returned
    Figure object.

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

- Parameter 'default_text' triggers default chart title and subtitle annotations if chart_title &
  chart_subtitle are None.
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
