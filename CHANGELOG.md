# Changelog

All notable changes to this project will be documented in this file.

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `iivs.dhm.data.phase`: read and write Koala (Lyncée Tec) float32 `.bin`
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
  - `PhaseSequence` — read-only base type for any phase image sequence;
    same-shape sources also mix in `data.sequence.FrameShapedMixin` for
    `frame_shape` (height, width).
  - `PhaseBinFolder` — an ordered `kaparoo.data.sequences.FileFolderSequence`
    over a folder of `{index:05d}_phase.bin` images (item = image, metadata =
    source path), exposing the shared acquisition `header`, optional unit
    conversion, and a `validate` method.
  - `PhaseBinList` — a `kaparoo.data.sequences.FileListSequence` over an
    explicit, arbitrary list of `.bin` files (any location, no naming or
    shared-header constraint); each file is read independently with per-file
    unit conversion.
- `iivs.dhm.data.folder`: `SequentialFileFolderSequence` — a
  `kaparoo.data.sequences.FileFolderSequence` base for `{index:05d}_<stem>.<ext>`
  folders. Subclasses set `FILE_STEM` / `FILE_EXT` / `LEVELS` / `DEFAULT_LEVEL`
  and a `_validate_content` hook; numbered discovery (`list_files`), `get_meta`,
  and the contiguity-checked `validate` / `validate_file` come for free. The
  `PhaseBinFolder`, `IntensityBinFolder`, and `HologramTifFolder` build on it.
- `iivs.dhm.data.sequence`: `FrameShapedMixin` — a mixin that forces a uniform
  `frame_shape`. There is no per-modality `Uniform*Sequence`: a same-shape
  source is `<Modality>Sequence` + `FrameShapedMixin`, so "a uniform phase
  sequence" is `isinstance(x, PhaseSequence) and isinstance(x, FrameShapedMixin)`.
- `iivs.dhm.data.binfile`: the shared Lyncée Tec Koala `.bin` format — the
  `KoalaBinHeader` base (geometry plus the packed 23-byte header machinery and
  the float32 pixel block I/O via `read_bin_pixels` / `write_bin`),
  specialized per modality by `PhaseBinHeader` and `IntensityBinHeader`.
- `iivs.dhm.data.image`: `validate_float32_image` — the shared, modality-agnostic
  float32 image/stack validator (dtype, dimensionality, and the `on_nonfinite`
  policy). `phase.validate_phase` and `intensity.validate_intensity` are
  domain-named aliases over it.
- `iivs.dhm.data.intensity`: read and write Koala float32 `.bin` intensity
  images (the amplitude/intensity reconstruction Koala exports alongside
  phase).
  - `IntensityBinHeader` — the typed header; intensity carries no height scale
    or unit, so Koala's `hconv = -1` / `unit = 0` sentinel is written on save
    and ignored on load.
  - `load_intensity_bin(path, *, return_header=False)` /
    `save_intensity_bin(...)` / `read_intensity_bin_header(path)`.
  - `validate_intensity(data, *, on_nonfinite=...)`.
  - `IntensitySequence` base type (same-shape sources add `FrameShapedMixin`),
    with concrete `IntensityBinFolder` (a `{index:05d}_intensity.bin` folder)
    and `IntensityBinList` (an arbitrary list of `.bin` files).
- `iivs.dhm.data.timestamp`: per-frame acquisition timing.
  - `Timestamp` — `elapsed_ms` / `interval_ms` for one frame, with
    `Timestamp.series_from_elapsed_times`.
  - `TimestampSequence` — read-only interface exposing `mean_interval_ms` and
    `mean_frame_rate`; implemented by `TimestampsTxtFile` (Koala
    `timestamps.txt`) and `TimestampsFixedFPS` (synthesized from a frame
    rate).
- `iivs.dhm.data.hologram`: read and write Koala (Lyncée Tec) uint8 holograms
  — `.tif` files (single or a folder/list), and read-only single multi-frame
  `.raw` files.
  - `load_hologram_tif(path)` / `save_hologram_tif(path, data, *, overwrite=False)`
    and `validate_hologram(data)`.
  - `HologramSequence` — read-only base type for any hologram sequence;
    same-shape sources also mix in `data.sequence.FrameShapedMixin` for
    `frame_shape` (height, width).
  - `HologramTifFolder` — an ordered `FileFolderSequence` over a folder of
    `{index:05d}_holo.tif` images (item = image, metadata = source path),
    with a `validate` method.
  - `HologramTifList` — a `FileListSequence` over an explicit,
    arbitrary list of `.tif` files (any location, no naming constraint), each
    decoded independently.
  - `HologramRawHeader` / `read_hologram_raw_header(path)` — the 16-byte
    `.raw` header (width, height, bit depth, frame count).
  - `HologramRawFile` — a `SingleFileSequence` over a `.raw` file's frames
    (held internally as a lazy, read-only `np.memmap`; each item is a writable
    frame copy, metadata is the frame index).
- `iivs.dhm.data.constants`: typical optical, geometric, and biophysical
  parameters for the lab's transmission setup — `DEFAULT_WAVELENGTH` (666 nm,
  in m) / `DEFAULT_WAVELENGTH_NM`, `DEFAULT_REFRACTIVE_DELTA` (0.5),
  `DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT` (2.0e-4 m³/kg, for dry mass), and
  `PIXEL_SIZE_20X` (in m) / `PIXEL_SIZE_20X_UM` (~285 nm; Koala 20X) as a
  header-less fallback for `PhaseBinHeader.pixel_size`.
- `iivs.dhm.analysis`: quantitative analysis derived from phase.
  - `opd.OPDConverter` — convert between phase and optical path difference
    (`OPD = phase * wavelength / (2*pi)`) at a fixed wavelength via
    `convert_to_opd` / `convert_to_phase`, with OPD in nm (the QPI
    convention) and the cached scale exposed as `opd_scale` (nm/rad); construct
    with m or `OPDConverter.from_wavelength_nm`. `opd.phase_to_opd` /
    `opd.opd_to_phase` are one-shot conveniences over it (as `json.dumps` is
    over `JSONEncoder`).
  - `drymass.DryMassCalculator` — integrate a background-corrected, optionally
    masked OPD (`calc_from_opd`, in nm) or phase (`calc_from_phase`) into a dry
    mass in pg (Barer), binding pixel size, specific refractive
    increment, and an injected `opd_converter` (for the phase path) once,
    precomputing the per-pixel factor (exposed as `drymass_scale`; sum in
    float64). Build it from a wavelength with
    `DryMassCalculator.from_wavelength`; `wavelength` / `wavelength_nm`
    shortcuts read the converter's. `drymass.calc_drymass` /
    `drymass.calc_drymass_from_phase` are one-shot conveniences over it.
    Segmentation and background estimation stay the caller's job.
