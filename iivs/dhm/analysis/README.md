# `iivs.dhm.analysis`

Physical quantities derived from reconstructed phase. Each quantity has an
**engine object** that binds its parameters once and precomputes a single scalar
conversion factor, plus **one-shot free functions** for convenience.

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

Dry mass `= (1 / alpha) * sum(OPD * pixel_area)` (the Barer relation), in **pg**,
summed over the last two axes (H, W). Inputs are **batched** — `opd` / `phase`
have shape `(..., H, W)`, giving one mass per image (`(...)`). The OPD must
already be background-corrected (≈ 0 outside the object); a boolean `mask` of
shape `(H, W)` or `(N, H, W)` (for `N` objects, giving a trailing axis
`(..., N)`) restricts the sum — segmentation and background estimation stay the
caller's responsibility.

- `DryMassCalculator(pixel_size, alpha=..., opd_converter=...)` — bind the pixel
  size (m), specific refractive increment (m³/kg), and an `OPDConverter` (for the
  phase path) once.
  - `from_wavelength(pixel_size=..., wavelength=...)` — build the inner converter
    from a wavelength.
  - `calc_from_opd(opd, *, mask=None, reduce=True)` — dry mass from an OPD map
    (nm). `reduce=False` returns the per-pixel mass-density map (`opd * scale`,
    masked) instead of the sum.
  - `calc_from_phase(phase, *, mask=None, reduce=True)` — dry mass from a phase
    map (rad).
  - `drymass_scale` — the cached pg-per-summed-nm factor (a plain `float`).
  - `wavelength` / `wavelength_nm`.
- `calc_drymass(opd, *, pixel_size, alpha=..., mask=None, reduce=True)` /
  `calc_drymass_from_phase(phase, *, pixel_size, wavelength=..., alpha=..., mask=None, reduce=True)`
  — the one-shot forms.

Defaults for `wavelength` and `alpha` come from
[`iivs.dhm.data.constants`](../data/constants.py).

## Using with PyTorch (autograd)

The `convert_*` / `calc_*` methods run on NumPy arrays. Install the
`iivs-lib[torch]` extra for the `pytorch` subpackage — tensor-in / tensor-out
twins that keep the input tensor's device and autograd graph. The physical
calibration (the scale factors) is reused from the NumPy engines, so only the
elementwise ops are torch-native. It mirrors the NumPy layout: an `nn.Module`
per quantity (named for the quantity, per the `nn.Module` convention —
`OpticalPathDifference` / `DryMass`, not the NumPy engines'
`OPDConverter` / `DryMassCalculator`), with one-shot free functions wrapping it.

```python
from iivs.dhm.analysis.pytorch.opd import OpticalPathDifference, phase_to_opd
from iivs.dhm.analysis.pytorch.drymass import DryMass, calc_drymass_from_phase

# nn.Module layers (compose in a model; the inner OpticalPathDifference is a submodule):
to_opd = OpticalPathDifference(wavelength=666e-9)
mass_head = DryMass.from_wavelength(pixel_size=px, wavelength=666e-9)
opd = to_opd(phase)                          # forward == convert_to_opd
mass = mass_head.calc_from_phase(phase, mask=cell)

# Or one-shot functions:
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
