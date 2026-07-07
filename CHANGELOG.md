# Changelog

All notable changes to this project will be documented in this file.

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `iivs.common.data`: the first technique-agnostic data primitives, hoisted out
  of `iivs.dhm.data.koala` — `read_npy_shape` / `write_npy` (the `.npy` reader /
  writer) and `FrameShapedMixin` (the same-shape marker mixin).
- `iivs.common.data.timestamp`: the technique-agnostic timing types, hoisted out
  of `iivs.dhm.data.timestamp` — the `Timestamp` record, the abstract
  `TimestampSequence` interface (`mean_interval_ms` / `mean_frame_rate`), and the
  synthetic `TimestampsFixedFPS`. Any time-lapse acquisition has per-frame timing,
  so a future technique (`epi` / `rcm`) implements the same `TimestampSequence`
  from OME-TIFF / Micro-Manager metadata without importing `dhm`. The Koala
  `timestamps.txt` reader `TimestampsTxtFile` stays in `iivs.dhm.data.timestamp`.
- `iivs.dhm.data.koala`: shared vocabulary for the repeated dispatch literals —
  the `OnNonFinite` (`"ignore"`/`"warn"`/`"raise"`) and `ValidationLevel`
  (`"names"`/`"headers"`/`"data"`) type aliases, the `FloatFormat` alias with its
  `FLOAT_FORMATS` tuple, and `detect_numbered_format(root, *, stem, formats,
  prefer)` — which discovers a numbered folder's format with `kaparoo`'s
  `search_files` + a `Regex` (no `Path.glob`) and resolves a multi-format
  conflict via `prefer` (the per-modality factories now share these instead of
  each defining their own). `iivs.dhm.data.hologram` adds the matching
  `HologramFormat` / `HOLOGRAM_FORMATS`.
- `iivs.dhm.data.phase`: extension-dispatch entry points that pick the format
  by a path's suffix, so callers need not hand-pick the per-format symbol.
  - `load_phase(path, *, return_header=False)` — dispatch `.bin` / `.txt` /
    `.npy` to `load_phase_bin` / `load_phase_txt` / `load_phase_npy`. Returns
    the image, or `(image, header)` when `return_header` — with `header` `None`
    for the header-less `.npy` (`read_phase_header` instead *raises* on `.npy`,
    since returning the header is its sole job).
  - `read_phase_header(path)` — dispatch `.bin` / `.txt` to their header
    readers. `.npy` is excluded (it is header-less).
  - `save_phase(path, data, ...)` — dispatch the writers; for `.npy` the
    header metadata (`pixel_size` / `unit` / scale) does not apply and is
    dropped with a warning if given.
  - `phase_list(files)` / `phase_folder(root)` — build the matching
    `Phase*List` / `Phase*Folder` by the files' shared extension / the folder's
    contents. `phase_folder` requires `pixel_size` + `unit` for an `.npy`
    folder (header-less) and rejects those args for `.bin` / `.txt`; a `prefer`
    argument resolves a folder that holds more than one format (`None` raises,
    a format or priority sequence picks the first present one).
  - `load_phase_npy(path)` — the previously missing standalone `.npy` loader
    (image only), filling out the `load_phase_{bin,txt,npy}` set.
- `iivs.dhm.data.phase.save_phase_folder(root, images, *, ext, ...)` — write any
  phase image sequence to a numbered folder. Unlike `convert_phase_folder` (which
  reads `pixel_size` / `height_scale` / `unit` off a file folder's header), it
  takes that metadata explicitly, so it accepts header-less sources — `kaparoo`
  composers (`ConcatSequence`, sliced/windowed views), `to_float` / `to_image`
  reconstructions, or a plain list of arrays. `convert_phase_folder` now delegates
  to it.
- `iivs.dhm.data.intensity`: the same suffix-dispatch entry points as `phase`
  (`load_intensity`, `read_intensity_header`, `save_intensity`, `intensity_list`,
  `intensity_folder`, the standalone `load_intensity_npy`), plus
  `save_intensity_folder(root, images, *, ext, pixel_size=..., ...)` for writing
  any intensity image sequence (composer outputs, …) to a numbered folder.
  Intensity's only header field is `pixel_size`. `convert_intensity_folder` now
  delegates to `save_intensity_folder`, and `intensity_folder` takes the same
  `prefer` conflict-resolution argument as `phase_folder`.
- `iivs.dhm.data.hologram.load_hologram_npy(path)` — the standalone `.npy` uint8
  reader, the twin of `load_hologram_tif`.

### Changed

- Lower the minimum Python to **3.13** (from 3.14). The code uses only PEP 695
  generics (3.12+) and `kaparoo-python` supports 3.13, so 3.13 runs the full
  suite unchanged; CI now covers 3.13 and 3.14. The `[torch]` extra keeps its
  `torch>=2.9` floor (wheels for both cp313 and cp314).
- `iivs.common.data`: hoist the dtype-generic image bases out of the (now)
  `iivs.dhm.data.koala` layer — `load_tif` (read any single-page `.tif`, keeping
  its stored dtype) and `ArrayFileList[U]` (the header-less array-list template,
  generic in the item dtype). `koala` keeps the uint8 bindings: `load_uint8_tif`
  (`= validate_uint8_array(load_tif(...))`), the Koala-numbered `ImageFileFolder`,
  and the `.tif` concretes `ImageTifList` / `ImageTifFolder`. Same behavior; the
  uint8 assumption (Koala previews are 8-bit) stays in `koala`, so a future 16-bit
  source binds `ArrayFileList[np.uint16]` on the shared `load_tif`.
- Rename `iivs.dhm.data.common` to `iivs.dhm.data.koala`. The dhm-internal
  cross-modality layer holds Lyncée Tec Koala's proprietary `.bin` / `Float/Txt`
  codecs and its `{index:05d}_<stem>.<ext>` export convention, so the vendor name
  disambiguates it from the technique-agnostic `iivs.common.data`. Update imports
  (`from iivs.dhm.data.koala import ...`); no symbols changed.
- `imagecodecs` (LZW `Image/*.tif` preview decode) ships as a **core
  dependency**, not an extra: it moves into the base dependencies and the
  `[image]` extra is removed. Handling image-like microscope data is this
  library's primary job, so it is always present — `[torch]`
  (`analysis.pytorch`) remains the only extra. `load_uint8_tif` no longer
  raises an "install the `[image]` extra" `ImportError`.
- `iivs.dhm.data.hologram`: `convert_hologram_sequence` and `save_hologram_raw`
  now accept any uint8 `DataSequence`, not just a `HologramSequence`, so a
  `kaparoo` composer (e.g. a `ConcatSequence` of acquisitions) can be re-encoded
  or written to `.raw` directly.
- `phase.save_phase` / `phase.save_phase_folder` / `phase.phase_folder` gained
  `@overload` signatures expressing the `height_scale` XOR `wavelength` +
  `refractive_delta` scale forms (and, for `phase_folder`, the `.npy`-only
  metadata), matching the per-format `save_phase_bin` / `save_phase_txt`. Runtime
  behavior is unchanged.

- Require `kaparoo-python>=0.10.0`. Its filter classes moved to a top-level
  `kaparoo.filters` package; the numbered-folder discovery in
  `koala.sequence` imports `Regex` from there. The per-modality
  factories dispatch on `kaparoo.filesystem`'s `file_extension` and raise its
  `UnsupportedExtensionError` (a `ValueError` subclass) for an unknown suffix,
  in place of this package's own short-lived `file_extension` /
  `unsupported_extension` helpers.
- Membership-validation guards now use `kaparoo.utils.ensure_one_of` (the `ext`
  checks in the `convert` modules, the folder `validate` level, and
  `HologramRawHeader.bit_depth`). The rejection message wording changes
  slightly (e.g. `ext must be one of [...]`).
- Raise the `iivs-lib[torch]` extra's floor to `torch>=2.9` — the first release
  with CPython 3.14 (cp314) wheels, which the project's `requires-python >=3.14`
  needs (2.6–2.8 ship none); resolved installs on 3.14 are unaffected.
- `PhaseFloatSequence.bounds_nm` is now a cached **property**, not a method:
  access it as `seq.bounds_nm` (no call). It still reads every frame on first
  access, then caches the global `(min, max)` for the sequence's lifetime.
- `iivs.dhm.data.koala`: the `.bin` and `Float/Txt` readers are consolidated
  into two header-parameterized engines — `load_bin(path, header_cls)` and
  `load_txt(path, header_codec)`, each returning `(image, header)`. They absorb
  and replace the lower-level `read_bin_pixels` and `parse_txt_grid` (removed
  from the public surface), and `write_txt_grid` is renamed to `write_txt`. Each
  phase / intensity per-format loader is now a thin wrapper over these, so the
  duplicated per-modality read bodies are gone.

### Fixed

- `iivs.dhm.data.koala.write_bin` now rejects a pixel block whose shape
  disagrees with the header, instead of writing a malformed `.bin`.

## [0.1.0] - 2026-06-05

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
    unit conversion; `get_header(index)` reads one file's header (the per-file
    twin of a folder's shared `header`), and `load_with_header(index)` returns
    the decoded image and its header in a single read.
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
    `read_phbounds` validates the `.txt` extension and `write_phbounds` appends
    it when absent, matching the other single-file readers / writers. The I/O
    lives on the value object as `PhaseBounds.from_file` / `to_file` (the free
    functions are thin wrappers); the unit tag is `PhaseBounds.UNIT_TAG`.
    `PhaseBounds.decode_preview` maps a uint8 `Image/*.tif` preview back toward
    phase in nm (the inverse of Koala's `[min, max]`→`0–255` render — lossy,
    8-bit quantized), and `encode_preview` is the forward render (rounded,
    clamped); a degenerate `min == max` is handled without dividing by a zero
    span. Whole-sequence twins of that map run lazily (per frame, on access):
    `PhaseFileList.to_image(bounds=None)` renders a `PhaseFloatSequence` into a
    uint8 `PhaseImageSequence` (each frame put in nm via its header first, so
    `target_unit` is irrelevant; `None` derives the bounds from `bounds_nm`), and
    `PhaseImageSequence.to_float(bounds, *, target_unit=NANOMETERS, height_scale=…)`
    reconstructs a `PhaseFloatSequence` from previews in the requested unit
    (8-bit-quantized — a reconstruction, *not* the quantitative `Float` source;
    NANOMETERS / METERS need no scale, RADIANS needs `height_scale` or
    `wavelength` + `refractive_delta`). The reconstruction view is built on
    `kaparoo`'s `TransformedSequence`; each view exposes `.source` / `.bounds`.
    Verified against a real Koala acquisition: phase previews are *globally*
    normalized via `phbounds.txt` (a single `PhaseBounds` spans the whole
    acquisition, and decode/encode match Koala to within one 8-bit code), while
    intensity previews are normalized *per frame* -- which is why intensity
    carries no bounds record and no `to_float`.
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
  - `KoalaFloatFileList` / `KoalaFloatFileFolder` — the float32 list/folder
    machinery (the `.<FILE_EXT>` check, `get_meta` / `get_header` /
    `load_with_header`, and a folder's shared `header` / `frame_shape` /
    `_validate_content`), generic in the header type over a ``(read_header,
    decode)`` codec whose `_decode` returns ``(image, header)``. Phase and
    intensity bind it and add only modality-specific bits (phase's unit
    conversion / `target_unit`), so the per-modality list/folder bodies are no
    longer duplicated.
  - `validate_float32_image` / `validate_uint8_image` — the shared,
    modality-agnostic image/stack validators (dtype, dimensionality, and -- for
    float32 -- the `on_nonfinite` policy). Both take `allow_stack` (default
    True; pass False to require a single 2-D image), used by the `save_*`
    writers. phase/intensity validate float32, holograms uint8.
  - `parse_txt_grid` and `write_txt_grid` — parse / atomically write a Koala
    `Float/Txt` body (whitespace-separated floats ↔ a float32 `(H, W)` array);
    shared by the `.txt` readers and `save_*_txt` writers. `parse_txt_grid` is
    layout-agnostic: it reshapes the values in row-major order, so it reads a
    grid Koala wrote as `height` rows *or* as a single long line.
  - `KoalaTxtHeaderCodec` — the stateless `Float/Txt` header (de)serializer (the
    text twin of `KoalaBinHeader`'s own `to_dtype` / `from_dtype`), with
    `from_file` / `from_lines` / `to_lines`. `phase` and `intensity` subclass it
    to bridge their text header to/from the matching `*BinHeader`
    (`_from_geometry` / `_extra_lines`), sharing the line-count check and the
    `h/w` + `pixel size` regex.
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
  - `ensure_file_extension` (imported directly from `kaparoo.filesystem`, now
    `>=0.7.0`) — the `*List` and single-file `*File` sequences (`HologramRawFile`,
    `TimestampsTxtFile`) validate each given path's `.<FILE_EXT>` at construction,
    so a wrong-format file fails up front rather than on decode (`FILE_EXT` lives
    on the concrete `*List`; the auto-discovering `*Folder` inherits it). Every
    `save_*` writer passes `add=True` for `np.save`-style behavior: a path with no
    suffix gets its extension appended (``out/00000_phase`` ->
    ``out/00000_phase.bin``), a mismatched one is rejected.
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
  `DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT` (2.0e-4 m³/kg, for dry mass), and the
  measured per-objective pixel sizes `PIXEL_SIZE_10X` / `PIXEL_SIZE_20X` /
  `PIXEL_SIZE_40X` (in m) with their `*_UM` twins (~580 / 285 / 144 nm; Koala
  10X / 20X / 40X) as a header-less fallback for `PhaseBinHeader.pixel_size`.
- `iivs.dhm.analysis`: quantitative analysis derived from phase. The NumPy
  engines and helpers are re-exported from the package root (the Torch twins are
  not, so importing the package never requires PyTorch).
  - `opd.OPDConverter` — convert between phase and optical path difference
    (`OPD = phase * wavelength / (2*pi)`) at a fixed wavelength via
    `convert_to_opd` / `convert_to_phase`, with OPD in nm (the QPI
    convention) and the cached scale exposed as `opd_scale` (nm/rad); construct
    with m or `OPDConverter.from_wavelength_nm`. `opd.phase_to_opd` /
    `opd.opd_to_phase` are one-shot conveniences over it.
  - `drymass.DryMassCalculator` — integrate a background-corrected, optionally
    masked OPD (`calc_from_opd`, in nm) or phase (`calc_from_phase`) into a dry
    mass in pg (Barer), binding pixel size, specific refractive
    increment, and an injected `opd_converter` (for the phase path) once,
    precomputing the per-pixel factor (exposed as `drymass_scale`; sum in
    float64, returned as float32). Inputs are batched `(..., H, W)` — the sum is
    over the last two axes, giving one mass per image `(...)`; a `(N, H, W)` mask
    (validated to `(H, W)` / `(N, H, W)`, matching `opd`'s `(H, W)`) adds a
    trailing axis `(..., N)`, and `reduce=False` returns the per-pixel
    mass-density map instead of the sum. Build it from a wavelength with
    `DryMassCalculator.from_wavelength`; `wavelength` / `wavelength_nm`
    shortcuts read the converter's. `drymass.calc_drymass` /
    `drymass.calc_drymass_from_phase` are one-shot conveniences over it.
    Segmentation and background estimation stay the caller's job.
  - `analysis.pytorch` (the `iivs-lib[torch]` extra) — torch-native twins that
    take and return `torch.Tensor`s, preserving the input tensor's device,
    dtype, and autograd graph (the dry-mass sum still accumulates in float64 for
    precision, then casts back -- so f16 / bf16 (AMP) and f64 are kept, where the
    NumPy twin forces float32). Mirrors the NumPy layout: an `nn.Module` per quantity, named
    for the quantity per the `nn.Module` convention
    (`pytorch.opd.OpticalPathDifference`, `pytorch.drymass.DryMass`, the latter
    holding the former as a submodule) with one-shot free functions wrapping it
    (`phase_to_opd` / `opd_to_phase`, `calc_drymass` / `calc_drymass_from_phase`).
    The calibration scalars are reused from the NumPy engines, so only the
    elementwise ops are torch-native; `calc_*` returns a tensor (never a Python
    `float`). Importing the subpackage without PyTorch raises a pointer to the
    `[torch]` extra.
