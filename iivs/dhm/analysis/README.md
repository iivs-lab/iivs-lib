# `iivs.dhm.analysis`

Physical quantities derived from reconstructed phase. Each quantity has an
**engine object** that binds its parameters once and precomputes a single scalar
conversion factor, plus **one-shot free functions** for convenience (as
`json.dumps` is to `json.JSONEncoder`).

These operate on plain NumPy arrays; they are not sequences. For PyTorch, see
[Using with PyTorch](#using-with-pytorch-autograd).

## `opd` — optical path difference

`OPD = phase * wavelength / (2 * pi)`, in **nm**, independent of any refractive
index (and distinct from the physical height `PhaseUnit.METERS` represents,
which additionally divides by the refractive-index difference).

- `OPDConverter(wavelength=...)` — bind a wavelength (SI, m) once.
  - `from_wavelength_nm(nm)` — construct from a wavelength in nm.
  - `convert_to_opd(phase)` / `convert_to_phase(opd)` — rad ↔ nm.
  - `opd_scale` — the cached nm-of-OPD-per-rad factor (a plain `float`).
  - `wavelength` / `wavelength_nm`.
- `phase_to_opd(phase, *, wavelength=...)` / `opd_to_phase(opd, *, wavelength=...)`
  — the one-shot forms.

## `drymass` — dry mass

Dry mass `= (1 / alpha) * sum(OPD * pixel_area)` (the Barer relation), in **pg**.
The OPD must already be background-corrected (≈ 0 outside the object); pass a
boolean `mask` to restrict the sum to one segmented object — segmentation and
background estimation stay the caller's responsibility.

- `DryMassCalculator(pixel_size, alpha=..., opd_converter=...)` — bind the pixel
  size (m), specific refractive increment (m³/kg), and an `OPDConverter` (for the
  phase path) once.
  - `from_wavelength(pixel_size=..., wavelength=...)` — build the inner converter
    from a wavelength.
  - `calc_from_opd(opd, *, mask=None)` — dry mass from an OPD map (nm).
  - `calc_from_phase(phase, *, mask=None)` — dry mass from a phase map (rad).
  - `drymass_scale` — the cached pg-per-summed-nm factor (a plain `float`).
  - `wavelength` / `wavelength_nm`.
- `calc_drymass(opd, *, pixel_size, alpha=..., mask=None)` /
  `calc_drymass_from_phase(phase, *, pixel_size, wavelength=..., alpha=..., mask=None)`
  — the one-shot forms.

Defaults for `wavelength` and `alpha` come from
[`iivs.dhm.data.constants`](../data/constants.py).

## Using with PyTorch (autograd)

The `convert_*` / `calc_*` methods run on NumPy arrays. Install the
`iivs-lib[torch]` extra for the `pytorch` subpackage — tensor-in / tensor-out
twins (`pytorch.opd`, `pytorch.drymass`) that keep the input tensor's device and
autograd graph. The physical calibration (the scale factors) is reused from the
NumPy engines, so only the elementwise ops are torch-native:

```python
from iivs.dhm.analysis.pytorch.opd import phase_to_opd, opd_to_phase
from iivs.dhm.analysis.pytorch.drymass import calc_drymass, calc_drymass_from_phase

opd = phase_to_opd(phase, wavelength=666e-9)                    # Tensor (CPU/GPU), grad kept
mass = calc_drymass_from_phase(phase, pixel_size=px, mask=cell) # 0-dim Tensor, grad kept
```

`calc_*` returns a 0-dim tensor (never a Python `float`), so it stays on-device
and differentiable. Without the dependency, you can instead multiply by the
cached scale factors (plain floats) with native ops yourself:

```python
opd = phase * conv.opd_scale                  # phase: Tensor -> OPD (nm), grad kept
mass = opd[mask].sum() * calc.drymass_scale   # OPD -> dry mass (pg), grad kept
```

## Example

```python
from iivs.dhm.analysis import OPDConverter, DryMassCalculator

conv = OPDConverter.from_wavelength_nm(666)
opd_nm = conv.convert_to_opd(phase_rad)        # rad -> nm

calc = DryMassCalculator(pixel_size=2.85e-7)   # alpha, wavelength default
mass_pg = calc.calc_from_phase(phase_rad, mask=cell_mask)
```
