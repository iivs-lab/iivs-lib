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

## 🧩 Modules

### `iivs.dhm.data`

Readers, writers, and lazy sequences for Lyncée Tec Koala acquisition data.

- **`phase`** — float32 `.bin` phase images: `load_phase_bin` /
  `save_phase_bin` / `read_phase_bin_header`, the typed `PhaseBinHeader` and
  `PhaseUnit`, `validate_phase`, and `convert_phase_unit`; folder/list
  sequences `PhaseBinFolder` / `PhaseBinList`.
- **`intensity`** — float32 `.bin` intensity reconstructions (exported
  alongside phase): `load_intensity_bin` / `save_intensity_bin` /
  `read_intensity_bin_header`, the typed `IntensityBinHeader`,
  `validate_intensity`, and folder/list sequences `IntensityBinFolder` /
  `IntensityBinList`. The phase and intensity `.bin` formats share the
  `binfile.KoalaBinHeader` base.
- **`hologram`** — uint8 holograms: `.tif` via `load_hologram_tif` /
  `save_hologram_tif` with `HologramTifFolder` / `HologramTifList`; a single
  multi-frame `.raw` via `HologramRawFile` (a lazy `np.memmap`) and
  `read_hologram_raw_header`; plus `validate_hologram`.
- **`timestamp`** — per-frame acquisition timing: the `Timestamp` record,
  `TimestampsTxtFile` (Koala `timestamps.txt`), and `TimestampsFixedFPS`
  (synthesized from a frame rate).

Every sequence is a `kaparoo.data.sequences.DataSequence`, so it indexes,
slices, and iterates lazily; same-shape sources also expose `frame_shape` by
mixing in `common.FrameShapedMixin` (so a uniform source is its
`<Modality>Sequence` plus that mixin). Numbered-folder sequences share the
`common.SequentialFileFolder` discovery/validation base. These cross-modality
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

The `convert_*` / `calc_*` methods operate on NumPy arrays. Inside a model,
keep gradients by multiplying tensors with the cached scale factors (plain
floats) using native ops instead:

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
