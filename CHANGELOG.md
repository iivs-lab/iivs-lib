# Changelog

All notable changes to this project will be documented in this file.

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `iivs.dhm.lynceetec.phase`: load and save Lyncee Tec / Koala float32
  `.bin` phase images (`load_bin`, `save_bin`) with a typed
  `PhaseBinHeader` (and its `PhaseBinHeader.DTYPE`) and a `PhaseUnit`
  enum. `load_bin(..., unit=...)` returns values converted to the
  requested unit and `save_bin(..., data_unit=...)` converts the input
  before storing, both via the header's `height_per_radian`.
