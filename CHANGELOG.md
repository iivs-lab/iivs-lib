# Changelog

All notable changes to this project will be documented in this file.

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `iivs.dhm.data.phase`: read and write Koala (Lyncée Tec) float32 phase
  images -- `.bin` (read + write) and the `Float/Txt` text export (read).
  - `PhaseBinHeader` — a typed, validated header (with
    `PhaseBinHeader.DTYPE`, `from_dtype` / `from_stream` / `from_file` /
    `to_dtype`, and `shape`, `pixel_count`, `field_of_view`,
    `pixel_size_um`, `height_scale_nm` conveniences) and the `PhaseUnit`
    enum.
  - `load_phase_bin(path, *, return_header=False)` — load a 2D image, optionally
    with its header; `read_phase_bin_header(path)` — read just the header cheaply.
  - `save_phase_bin(...)` — write a 2D image; the phase-to-height scale is given
    either as `height_scale`, or as `wavelength` + `refractive_delta`.
  - `convert_phase_unit(data, *, source, target, height_scale)` — rescale a
    phase/height image between `PhaseUnit` representations.
  - `PhaseSequence` — read-only modality base for any phase sequence, over
    both `PhaseFloatSequence` (quantitative float32, from `Float/{Bin,Txt}`)
    and `PhaseImageSequence` (the uint8 `Image/*.tif` preview). Same-shape
    sources also mix in `data.common.FrameShapedMixin` for `frame_shape`
    (height, width).
  - `PhaseBinFolder` — an ordered `kaparoo.data.sequences.FileFolderSequence`
    over a folder of `{index:05d}_phase.bin` images (item = image, metadata =
    source path), exposing the shared acquisition `header`, optional unit
    conversion, and a `validate` method.
  - `PhaseBinList` — a `kaparoo.data.sequences.FileListSequence` over an
    explicit, arbitrary list of `.bin` files (any location, no naming or
    shared-header constraint); each file is read independently with per-file
    unit conversion, and `header_at(index)` reads one file's header (the
    per-file twin of a folder's shared `header`).
  - `load_phase_txt` / `read_phase_txt_header`, with `PhaseTxtFolder` /
    `PhaseTxtList` — the text twins of the `.bin` readers over Koala's
    `Float/Txt` export (a 4-line header + float grid carrying the same
    quantitative phase and `PhaseBinHeader`).
  - `PhaseTifFolder` / `PhaseTifList` — the uint8 `Image/*.tif` display-preview
    sources (a `PhaseImageSequence`, *not* the quantitative `PhaseFloatSequence`;
    each item is the 8-bit preview, not the float phase). Decoding the
    LZW-compressed Koala previews needs the `iivs-lib[image]` extra
    (`imagecodecs`).
  - `PhaseNpyFolder` — a header-less `{index:05d}_phase.npy` float32 folder.
    `.npy` carries no Koala header, so `pixel_size`, `unit`, and `height_scale`
    (or `wavelength` + `refractive_delta`) are passed to the constructor and
    shared by every frame; arrays load via `numpy.load(allow_pickle=False)`
    (pickle disabled). `resolve_height_scale` is the shared scale-or-wavelength
    helper it and `save_phase_bin` use.
  - `PhaseBounds` with `read_phbounds` / `write_phbounds` — the Koala
    `phbounds.txt` display-bounds record (a `[nm]` tag then `min max`), and
    `PhaseFloatSequence.bounds_nm()` to recompute those bounds straight from the
    float source (global min/max in nanometers, per-file `height_scale`).
  - `save_phase_txt` / `save_phase_npy` single-image writers (the `.txt` / `.npy`
    twins of `save_phase_bin`), and `convert_phase_folder` / `convert_phase_list`
    to re-encode phase between the lossless `bin` / `txt` / `npy` formats. The
    `folder` form writes a new numbered folder under `root` (shared header); the
    `list` form rewrites each file in place -- a sibling with the new extension,
    keeping per-file metadata.
- `iivs.dhm.data.common`: the building blocks shared across the data
  modalities.
  - `KoalaBinHeader` — base for the packed 23-byte Lyncée Tec Koala `.bin`
    header (geometry + structural read), with the float32 pixel-block I/O
    `read_bin_pixels` / `write_bin`; specialized per modality by
    `PhaseBinHeader` and `IntensityBinHeader`.
  - `SequentialFileFolder` — a `kaparoo.data.sequences.FileFolderSequence` base
    for `{index:05d}_<stem>.<ext>` folders. Subclasses set `FILE_STEM` /
    `FILE_EXT` / `LEVELS` / `DEFAULT_LEVEL` and a `_validate_content` hook;
    numbered discovery (`list_files`), `get_meta`, the contiguity-checked
    `validate` / `validate_file`, and `FrameShapedMixin` come for free. Every
    modality `*Folder` builds on it. `numbered_name(index, *, stem, ext)` is
    the one source of that naming convention, shared by `expected_name` and the
    folder converters.
  - `FrameShapedMixin` — a mixin that forces a uniform `frame_shape`. There is
    no per-modality `Uniform*Sequence`: a same-shape source is
    `<Modality>Sequence` + `FrameShapedMixin`, so "a uniform phase sequence" is
    `isinstance(x, PhaseSequence) and isinstance(x, FrameShapedMixin)`. Numbered
    folders get it via `SequentialFileFolder`; single-file sources
    (`HologramRawFile`) mix it in directly.
  - `validate_float32_image` / `validate_uint8_image` — the shared,
    modality-agnostic image/stack validators (dtype, dimensionality, and -- for
    float32 -- the `on_nonfinite` policy). Both take `allow_stack` (default
    True; pass False to require a single 2-D image), used by the `save_*`
    writers. phase/intensity validate float32, holograms uint8.
  - `parse_txt_grid` and `write_txt_grid` — parse / atomically write a Koala
    `Float/Txt` body (whitespace-separated float rows ↔ a float32 `(H, W)`
    array); shared by the `.txt` readers and `save_*_txt` writers.
  - `KoalaTxtHeader` — the `Float/Txt` header reader and writer (the text twin
    of `KoalaBinHeader`), with `from_file` / `from_lines` / `to_lines`. `phase`
    and `intensity` subclass it to bridge their text header to/from the matching
    `*BinHeader` (`_from_geometry` / `_extra_lines`), sharing the line-count
    check and the `h/w` + `pixel size` regex.
  - `load_uint8_tif`, with the `ImageFileList` / `ImageFileFolder` codec bases
    and their `.tif` concretes `ImageTifList` / `ImageTifFolder` — the
    modality-agnostic uint8 image folder/list bodies (a `load_file` codec +
    lazy `frame_shape`) behind the `Image/*.tif` previews, the hologram `.tif`
    folder, and -- via a `numpy.load` `load_file` -- `HologramNpyFolder`.
  - `read_npy_shape` — read a 2-D `.npy` array's `(height, width)` without
    loading its data (memory-mapped, `allow_pickle=False`); used to validate the
    `*NpyFolder`s cheaply.
  - `write_npy` — the shared atomic `.npy` writer behind every modality's
    `save_*_npy`.
  - `ensure_file_extension` — the `*List` and single-file `*File` sequences
    (`HologramRawFile`, `TimestampsTxtFile`) now validate each given path's
    `.<FILE_EXT>` at construction, so a wrong-format file fails up front rather
    than on decode. `FILE_EXT` lives on the concrete `*List` (the
    auto-discovering `*Folder` inherits it).
- `iivs.dhm.data.intensity`: read and write Koala float32 `.bin` intensity
  images (the amplitude/intensity reconstruction Koala exports alongside
  phase).
  - `IntensityBinHeader` — the typed header; intensity carries no height scale
    or unit, so Koala's `hconv = -1` / `unit = 0` sentinel is written on save
    and ignored on load.
  - `load_intensity_bin(path, *, return_header=False)` /
    `save_intensity_bin(...)` / `read_intensity_bin_header(path)`.
  - `load_intensity_txt` / `read_intensity_txt_header`, with
    `IntensityTxtFolder` / `IntensityTxtList` — the text twins over Koala's
    `Float/Txt` export (a 2-line header -- no unit/height-conversion -- + grid).
  - `IntensitySequence` modality base, split into `IntensityFloatSequence`
    (quantitative float32) and `IntensityImageSequence` (the uint8
    `Image/*.tif` preview); same-shape sources add `FrameShapedMixin`. Concrete
    `IntensityBinFolder` (a `{index:05d}_intensity.bin` folder)
    and `IntensityBinList` (an arbitrary list of `.bin` files). The `.txt`
    folders/lists are their text twins.
  - `IntensityTifFolder` / `IntensityTifList` — the uint8 `Image/*.tif`
    display-preview sources (an `IntensityImageSequence`); like the phase
    previews, decoding needs the `iivs-lib[image]` extra.
  - `IntensityNpyFolder` — a header-less `{index:05d}_intensity.npy` float32
    folder; `pixel_size` is passed to the constructor (intensity has no unit or
    height scale). Arrays load via `numpy.load(allow_pickle=False)`.
  - `save_intensity_txt` / `save_intensity_npy` single-image writers, and
    `convert_intensity_folder` / `convert_intensity_list` to re-encode intensity
    between the lossless `bin` / `txt` / `npy` formats (`folder` writes a new
    numbered folder under `root`; `list` rewrites each file in place, keeping
    per-file metadata).
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
  - `load_hologram_tif(path)` /
    `save_hologram_tif(path, data, *, overwrite=False)` (uint8 validated via
    `common.validate_uint8_image`).
  - `HologramSequence` — read-only base type for any hologram sequence;
    same-shape sources also mix in `data.common.FrameShapedMixin` for
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
  - `HologramNpyFolder` — a header-less `{index:05d}_holo.npy` uint8 folder (no
    metadata needed); arrays load via `numpy.load(allow_pickle=False)`.
  - `save_hologram_raw` (a multi-frame `.raw` stack writer, with
    `HologramRawHeader.to_dtype`; accepts an array or a `HologramSequence` and
    streams frame by frame, so a large source is never held in memory at once)
    / `save_hologram_npy`, and `convert_hologram_sequence(dest, seq, *,
    ext=...)` to re-encode a hologram sequence between `raw` (one stack file),
    `tif`, and `npy` (per-frame folders).
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
