# Changelog

All notable changes to this project will be documented in this file.

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `iivs.dhm.koala.phase`: read and write Koala (Lyncée Tec) float32 `.bin`
  phase images.
  - `PhaseBinHeader` — a typed, validated header (with
    `PhaseBinHeader.DTYPE`, `from_dtype` / `from_stream` / `from_file` /
    `to_dtype`, and `shape`, `pixel_count`, `field_of_view`,
    `pixel_size_um`, `height_scale_nm` conveniences) and the `PhaseUnit`
    enum.
  - `load_phase_bin(path, *, return_header=False)` — load a 2D image, optionally
    with its header; `read_phase_bin_header(path)` — read just the header cheaply.
  - `save_phase_bin(...)` — write a 2D image; the phase-to-height scale is given
    either as `height_scale`, or as `wavelength` + `refractive_delta`.
  - `validate_phase(data, *, on_nonfinite=...)` — validate a float32 phase
    image or stack.
  - `convert_phase_unit(data, *, source, target, height_scale)` — rescale a
    phase/height image between `PhaseUnit` representations.
  - `PhaseSequence` — read-only base type for phase image sequences (any
    source), implemented by `PhaseBinSequence`.
  - `PhaseBinSequence` — an ordered `kaparoo.data.sequences.FileFolderSequence`
    over a folder of `{index:05d}_phase.bin` images (item = image, metadata =
    source path), exposing the shared acquisition `header`, optional unit
    conversion, and a `validate` method.
- `iivs.dhm.koala.timestamp`: per-frame acquisition timing.
  - `Timestamp` — `elapsed_ms` / `interval_ms` for one frame, with
    `Timestamp.series_from_elapsed_times`.
  - `TimestampSequence` — read-only interface exposing `mean_interval_ms` and
    `mean_frame_rate`; implemented by `TimestampTxtSequence` (Koala
    `timestamps.txt`) and `TimestampFpsSequence` (synthesized from a frame
    rate).
- `iivs.dhm.koala.hologram`: read Koala (Lyncée Tec) uint8 holograms from a
  `.tif` folder or a single multi-frame `.raw` file.
  - `load_hologram_tif(path)` / `save_hologram_tif(path, data, *, overwrite=False)`
    and `validate_hologram(data)`.
  - `HologramSequence` — read-only base type for hologram sequences (any
    source), implemented by `HologramTifSequence` and `HologramRawSequence`.
  - `HologramTifSequence` — an ordered `FileFolderSequence` over a folder of
    `{index:05d}_holo.tif` images (item = image, metadata = source path),
    with a `validate` method.
  - `HologramRawHeader` / `read_hologram_raw_header(path)` — the 16-byte
    `.raw` header (width, height, bit depth, frame count).
  - `HologramRawSequence` — a `SingleFileSequence` over a `.raw` file's frames
    (held internally as a lazy, read-only `np.memmap`; each item is a writable
    frame copy, metadata is the frame index).
