# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), except that, due
to internal constraints, the initial public version is [3.0.0]. Version 3 is in beta, where the API is liable to change.
Version 4 will be the first stable version.

## [Unreleased]

### Added

- Support for Python 3.13. (https://github.com/gchq/Vanguard/pull/511)

### Fixed

-

### Changed

-

### Removed

- [BREAKING CHANGE] Removed support for old versions of some dependencies; most importantly, we no longer support
  NumPy 1.x. (https://github.com/gchq/Vanguard/pull/511)

### Deprecated

-


## [3.0.1]

### Fixed

- Package name corrected to `vanguard-gp` in pyproject.toml. (https://github.com/gchq/Vanguard/pull/505)
- Dev dependencies option was installing `vanguard` (wrong package). (https://github.com/gchq/Vanguard/pull/505)


## [3.0.0] - [YANKED]

Yanked due to build failure.

### Added

- Decorator framework for Gaussian Processes, including:
  - Classification,
  - Distributed GPs,
  - Hierarchical GPs with Bayesian hyperparameters,
  - Multitask GPs (EXPERIMENTAL),
  - Variational inference,
  - Compositional input warping for GPs,
  - Input normalisation,
  - Higher-rank features (EXPERIMENTAL).
- Optimisation features:
  - Smart optimiser with early stopping,
  - Learning rate optimisation,
  - Learning rate scheduler integration.
- Synthetic datasets for testing.
- Worked notebook examples of major features.
- Detailed online documentation, complete with references.
- Near-complete unit test coverage.
- Comprehensive type hints, checked by [beartype].
- Support for Python 3.9-3.12.
- This changelog, to make it easier for users and contributors to see precisely what notable changes have been made
  between each release of the project.
- And various contributor-facing amenities:
  - Contributor guidelines,
  - Automatic [pre-commit] checks, including formatting and linting,
  - GH Actions to check unit tests and documentation build on each PR.
  - Test coverage badge.


[//]: # (## [M.m.p] - YYYY-mm-dd)

[//]: # (### Added)
[//]: # (This is where features that have been added should be noted.)

[//]: # (### Fixed)
[//]: # (This is where fixes should be noted.)

[//]: # (### Changed)
[//]: # (This is where changes from previous versions should be noted.)

[//]: # (### Removed)
[//]: # (This is where elements which have been removed should be noted.)

[//]: # (### Deprecated)
[//]: # (This is where existing but deprecated elements should be noted.)

[beartype]: https://pypi.org/project/beartype/
[pre-commit]: https://pre-commit.com/

[Unreleased]: https://github.com/gchq/Vanguard/compare/v3.0.1...HEAD
[3.0.1]: https://github.com/gchq/Vanguard/compare/v3.0.0...v3.0.1
[3.0.0]: https://github.com/gchq/Vanguard/releases/tag/v3.0.0
