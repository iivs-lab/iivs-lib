# iivs-lib

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg?logo=opensourceinitiative&logoColor=white)](./LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/iivs-lib.svg?logo=pypi&logoColor=white)](https://pypi.org/project/iivs-lib/)
[![Downloads](https://pepy.tech/badge/iivs-lib)](https://pypi.org/project/iivs-lib/)
[![Python](https://img.shields.io/badge/python-3.13%20%7C%203.14-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/copier-org/copier)

*A Python toolkit for multi-modal holographic systems and cellular analysis.*

## 📦 Installation

Requires Python 3.13 or newer.

```bash
# With uv (recommended)
uv add iivs-lib

# With pip
pip install iivs-lib
```

Image I/O ships by default: `imagecodecs` (to decode the LZW-compressed Koala
`Image/*.tif` uint8 previews) is a core dependency — handling image-like
microscope data is this library's primary job. The only extra is `[torch]`,
which adds PyTorch for `iivs.dhm.analysis.pytorch` — tensor-in / tensor-out
OPD / dry-mass twins with autograd (`uv add "iivs-lib[torch]"`).

## 🚀 Quick start

```python
from iivs.dhm.data.phase import PhaseBinFolder, PhaseUnit
from iivs.dhm.analysis import DryMassCalculator

# Lazily open a phase acquisition (a folder of numbered .bin frames).
phase = PhaseBinFolder("scan/Phase/Float/Bin", target_unit=PhaseUnit.RADIANS)
phase.frame_shape          # (H, W), shared across frames
img = phase[0]             # first frame as a float32 array (decoded on access)

# Per-frame dry mass over a segmented cell:
calc = DryMassCalculator(pixel_size=phase.header.pixel_size)
mass_pg = calc.calc_from_phase(img, mask=cell_mask)
```

## 🧩 Modules

Each package ships a detailed README in the source tree — the endpoints,
examples, and the inherited sequence interface:
[`hologram`](https://github.com/iivs-lab/iivs-lib/tree/main/iivs/dhm/data/hologram),
[`phase`](https://github.com/iivs-lab/iivs-lib/tree/main/iivs/dhm/data/phase),
[`intensity`](https://github.com/iivs-lab/iivs-lib/tree/main/iivs/dhm/data/intensity),
[`data`](https://github.com/iivs-lab/iivs-lib/tree/main/iivs/dhm/data) (overview
and `timestamp`),
[`analysis`](https://github.com/iivs-lab/iivs-lib/tree/main/iivs/dhm/analysis).

### `iivs.dhm.data`

Readers, writers, and lazy sequences for Lyncée Tec Koala acquisition data
(validated end-to-end against a real Koala acquisition — every format, the
sequences, and the round trips between them).

- **`hologram`** — uint8 holograms: `.tif` via `load_hologram_tif` /
  `save_hologram_tif` with `HologramTifFolder` / `HologramTifList`; a single
  multi-frame `.raw` via `HologramRawFile` (a lazy `np.memmap`),
  `read_hologram_raw_header` / `save_hologram_raw`; header-less `.npy` frames via
  `HologramNpyFolder`.
- **`phase`** — float32 `.bin` phase images: `load_phase_bin` /
  `save_phase_bin` / `read_phase_bin_header`, the typed `PhaseBinHeader` and
  `PhaseUnit`, and `convert_phase_unit`; folder/list sequences
  `PhaseBinFolder` / `PhaseBinList`. The same quantitative phase from Koala's
  `Float/Txt` export via `load_phase_txt`, `PhaseTxtFolder` / `PhaseTxtList`.
  The uint8 `Image/*.tif` display previews (not quantitative) via
  `PhaseTifFolder` / `PhaseTifList`. Header-less `.npy` frames via
  `PhaseNpyFolder` (`pixel_size` / `unit` / `height_scale` passed to the
  constructor; `numpy.load`, pickle disabled).
- **`intensity`** — float32 `.bin` intensity reconstructions (exported
  alongside phase): `load_intensity_bin` / `save_intensity_bin` /
  `read_intensity_bin_header`, the typed `IntensityBinHeader`, and folder/list
  sequences `IntensityBinFolder` / `IntensityBinList`; plus the `Float/Txt`
  twins `load_intensity_txt`, `IntensityTxtFolder` / `IntensityTxtList`, the
  uint8 `Image/*.tif` previews `IntensityTifFolder` / `IntensityTifList`, and
  header-less `.npy` frames via `IntensityNpyFolder` (`pixel_size` passed in).
  The phase and intensity `.bin` formats share the `common.KoalaBinHeader` base.
- **`timestamp`** — per-frame acquisition timing: the `Timestamp` record,
  `TimestampsTxtFile` (Koala `timestamps.txt`), and `TimestampsFixedFPS`
  (synthesized from a frame rate).

`phase` and `intensity` also expose **suffix-dispatch entry points** that pick
the format by a path's extension — `load_phase` / `read_phase_header` /
`save_phase`, `phase_list` / `phase_folder` (and the `intensity` twins) — plus
`save_phase_folder` / `save_intensity_folder` to write any (e.g. a `kaparoo`-
composed) image sequence to a numbered folder. `load_phase` / `load_intensity`
take `return_header` (the header is `None` for the header-less `.npy`).

A **`timelapse`** module opens a whole Koala acquisition from its root folder:
`KoalaTimelapse` composes the per-modality `PhaseGroup` / `IntensityGroup` (each
picking its `Float/{Bin,Txt}` format), the holograms, `timestamps.txt`, and
`phbounds.txt` display bounds, with lazy access, frame-count consistency checks
(`is_consistent`), and content `validate`. `search_timelapses` finds every
acquisition under a root; `KOALA_TIMELAPSE_TREE` describes the layout for
`hierarchy.validate`.

Every sequence is a `kaparoo.data.sequences.DataSequence`, so it indexes,
slices, and iterates lazily; same-shape sources also expose `frame_shape` by
mixing in `iivs.common.data.FrameShapedMixin` (so a uniform source is its
`<Modality>FloatSequence` / `<Modality>ImageSequence` plus that mixin). For
phase and intensity the quantitative float32 sources are
`<Modality>FloatSequence` and the uint8 `Image/*.tif` previews are
`<Modality>ImageSequence`, both under the `<Modality>Sequence` base.
Numbered-folder sequences share the `KoalaFrameFolder` discovery/validation
base in `iivs.dhm.data.koala`, and validate their arrays via `iivs.common.data`'s
`validate_float32_array` / `validate_uint8_array`.

### `iivs.dhm.analysis`

Physical quantities derived from phase, each via an engine object that
precomputes its conversion factor (with one-shot function conveniences):

- **`opd`** — optical path difference (`OPD = phase * wavelength / (2*pi)`, in
  nm). `OPDConverter` (`convert_to_opd` / `convert_to_phase`, scale
  `opd_scale`); `phase_to_opd` / `opd_to_phase`.
- **`drymass`** — dry mass (pg) via the Barer relation. `DryMassCalculator`
  (`calc_from_opd` / `calc_from_phase` over a background-corrected, optionally
  masked map; scale `drymass_scale`); `calc_drymass` / `calc_drymass_from_phase`.

The `wavelength` and `alpha` defaults come from
[`iivs.dhm.constants`](./iivs/dhm/constants.py), the lab's microscope settings —
shared by `analysis` and `data`, and belonging to neither. Override per experiment
when the setup differs.

#### Using with PyTorch (autograd)

The `convert_*` / `calc_*` methods operate on NumPy arrays. Install the
`iivs-lib[torch]` extra for `iivs.dhm.analysis.pytorch` — tensor-in / tensor-out
twins that keep the input tensor's device, dtype, and autograd graph (the
calibration scalars are shared with the NumPy engines):

```python
from iivs.dhm.analysis.pytorch.opd import phase_to_opd
from iivs.dhm.analysis.pytorch.drymass import calc_drymass_from_phase

opd = phase_to_opd(phase, wavelength=666e-9)                    # Tensor (CPU/GPU), grad kept
mass = calc_drymass_from_phase(phase, pixel_size=px, mask=cell) # 0-dim Tensor, grad kept
```

Or, without the dependency, multiply by the cached scale factors (plain floats)
with native ops yourself:

```python
opd = phase * conv.opd_scale                  # phase: Tensor -> OPD (nm), grad kept
mass = opd[mask].sum() * calc.drymass_scale   # OPD -> dry mass (pg), grad kept
```

### `iivs.common.data`

Technique-agnostic data primitives — nothing here knows what a hologram is, so a
future technique can build on them without importing `dhm`.

- **`ArrayFileList[U]`** — the dtype-generic base every file-backed sequence
  extends, with **`FrameShapedMixin`** marking the same-shape sources and
  **`ValueRangeMixin`** adding a cached `value_range()` over all frames.
- **`load_float32_npy`** / **`save_float32_npy`** and the **`uint8`** pair, plus
  `read_npy_shape` / `write_npy` — header-less `.npy` I/O, keyed by dtype (which
  is all that varies) rather than by modality. Pickle is disabled, so an object
  array cannot be written and one written elsewhere is refused rather than
  unpickled.
- **`validate_float32_array`** / **`validate_uint8_array`** and their wider
  `float` / `uint` forms — dtype + shape + non-finite checks, with
  `on_nonfinite` (`"ignore"` / `"warn"` / `"raise"`) the policy every loader and
  saver in the library threads through.
- **`Timestamp`** / **`TimestampSequence`** / **`TimestampsFixedFPS`** — per-frame
  timing, which any time-lapse acquisition has; the Koala `timestamps.txt` reader
  implements this interface from `dhm`.

### `iivs.common.visualization`

Technique-agnostic display helpers, shared across every modality.

- **`auto_rescale`** — ImageJ-style auto-contrast (Enhance Contrast + Normalize):
  clips `saturated`% of pixels via NaN-safe percentile bounds, stretches onto
  `out_range` (or the dtype's full span), and casts back to the input dtype.
  Works on any numeric image or stack, so phase / intensity / hologram share it.

## 📋 TODO

See [TODO.md](./TODO.md) for tracked open items.

## 📜 Changelog

See [CHANGELOG.md](./CHANGELOG.md) for the version history.

## 🙏 Acknowledgements

The file formats read and written by `iivs.dhm.data` are
[Lyncée Tec](https://www.lynceetec.com/)'s proprietary Koala formats. The `.bin`
format — Koala's float32 binary container, shared by the phase and intensity
reconstructions — was cross-checked against their reference implementation,
[`pyKoalaUtils`](https://github.com/lynceetec/pyKoalaUtils) (MIT). iivs-lib is
an independent reimplementation and contains no code from it.

## ⚖️ License

This project is distributed under the terms of the [MIT](./LICENSE) license.
