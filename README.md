# iivs-lib

[![PyPI version](https://img.shields.io/pypi/v/iivs-lib.svg)](https://pypi.org/project/iivs-lib/)
[![Downloads](https://pepy.tech/badge/iivs-lib)](https://pypi.org/project/iivs-lib/)
[![Python](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/copier-org/copier)

*A Python toolkit for multi-modal holographic systems and cellular analysis.*

## 📦 Installation

Requires Python 3.14+.

```bash
# With uv (recommended)
uv add iivs-lib

# With pip
pip install iivs-lib
```

The `[image]` extra adds `imagecodecs`, needed only to decode the
LZW-compressed Koala `Image/*.tif` uint8 previews (`uv add "iivs-lib[image]"`).
The `[torch]` extra adds PyTorch for `iivs.dhm.analysis.pytorch` — tensor-in /
tensor-out OPD / dry-mass twins with autograd (`uv add "iivs-lib[torch]"`).

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
  multi-frame `.raw` via `HologramRawFile` (a lazy `np.memmap`) and
  `read_hologram_raw_header`; header-less `.npy` frames via `HologramNpyFolder`.
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

Every sequence is a `kaparoo.data.sequences.DataSequence`, so it indexes,
slices, and iterates lazily; same-shape sources also expose `frame_shape` by
mixing in `common.FrameShapedMixin` (so a uniform source is its
`<Modality>FloatSequence` / `<Modality>ImageSequence` plus that mixin). For
phase and intensity the quantitative float32 sources are
`<Modality>FloatSequence` and the uint8 `Image/*.tif` previews are
`<Modality>ImageSequence`, both under the `<Modality>Sequence` base.
Numbered-folder sequences share the `common.SequentialFileFolder`
discovery/validation base, and validate their arrays via
`common.validate_float32_image` / `validate_uint8_image`. These cross-modality
building blocks live in `iivs.dhm.data.common`.

### `iivs.dhm.analysis`

Physical quantities derived from phase, each via an engine object that
precomputes its conversion factor (with one-shot function conveniences):

- **`opd`** — optical path difference (`OPD = phase * wavelength / (2*pi)`, in
  nm). `OPDConverter` (`convert_to_opd` / `convert_to_phase`, scale
  `opd_scale`); `phase_to_opd` / `opd_to_phase`.
- **`drymass`** — dry mass (pg) via the Barer relation. `DryMassCalculator`
  (`calc_from_opd` / `calc_from_phase` over a background-corrected, optionally
  masked map; scale `drymass_scale`); `calc_drymass` / `calc_drymass_from_phase`.

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

## 📋 TODO

See [TODO.md](./TODO.md) for tracked open items.

## 📜 Changelog

See [CHANGELOG.md](./CHANGELOG.md) for the version history.

## 🙏 Acknowledgements

The Koala file formats read and written by `iivs.dhm.data` originate from
[Lyncée Tec](https://www.lynceetec.com/)'s Koala software. The phase `.bin`
format was cross-checked against their reference implementation,
[`pyKoalaUtils`](https://github.com/lynceetec/pyKoalaUtils) (MIT). iivs-lib is
an independent reimplementation and contains no code from it.

## ⚖️ License

This project is distributed under the terms of the [MIT](./LICENSE) license.
