# `iivs.dhm.analysis`

Physical quantities derived from reconstructed phase. Each quantity has an
**engine object** that binds its parameters once and precomputes a single scalar
conversion factor, plus **one-shot free functions** for convenience.

The quantities form a chain — each is the previous one rescaled or integrated:

```
phase (rad)
  × λ/2π          → OPD (nm)             opd
  ÷ Δn            → optical height (nm)  height
  Σ × pixel_area  → volume (µm³ ≡ fL)    volume   (projected area × mean height)
  × Δn/α          → dry mass (pg)        drymass
```

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

## `height` — optical height

`height = OPD / refractive_delta`, in **nm**: the physical thickness producing
the measured path difference. Transmission QPI literature usually calls this
quantity the sample **thickness** (`phase = 2π·Δn·t/λ`); "height" here follows
Koala's and this library's data-layer naming (the `.bin` header's
`height_scale`, `PhaseUnit.METERS`). It is the same height the data layer's
`PhaseUnit.NANOMETERS` represents, so `convert_from_phase` agrees numerically
with `convert_phase_unit(..., target=NANOMETERS)` for a file whose stored
`height_scale` was built from the same wavelength and delta.

- `OpticalHeightConverter(refractive_delta=..., opd_converter=...)` — bind the
  refractive-index difference (and, for the phase path, an `OPDConverter`) once.
  - `from_wavelength(wavelength=..., refractive_delta=...)`.
  - `convert_to_height(opd)` / `convert_to_opd(height)` — nm ↔ nm.
  - `convert_from_phase(phase)` — rad → nm of height.
  - `height_scale` — the cached nm-of-height-per-rad factor (the nm twin of the
    `.bin` header's m-per-rad `height_scale`).
  - `refractive_delta`, `wavelength` / `wavelength_nm`.
- `opd_to_height` / `height_to_opd` / `phase_to_height` — the one-shot forms.

## `area` — projected area

`area = pixel_count * pixel_size²`, in **µm²**: the footprint the selected
region(s) cover in the image plane; it enters the volume relation as `volume =
area * mean(height)`. The call shape matches the other engines (`image`, `mask`,
`reduce`), but the image's *values* never enter the area — the image fixes the
pixel grid (and any leading batch axes), and each selected pixel contributes the
constant `area_scale`.

- `ProjectedAreaCalculator(pixel_size)` — bind the pixel size (m) once.
  - `calc(image, *, mask=None, reduce=True)` — µm² of each region: `(...)` for a
    boolean `(H, W)` mask (or None, the whole frame), a trailing `(..., R)` axis
    for a `(N, H, W)` stack or a label image. `reduce=False` returns the
    per-pixel area-density map instead (`area_scale` inside a region, 0 outside;
    summing back to the area).
  - `area_scale` — the cached µm²-per-pixel factor.
  - `pixel_size` / `pixel_size_um`.
- `calc_projected_area(image, *, pixel_size, mask=None, reduce=True)` — the
  one-shot form.

## `volume` — optical volume

`volume = sum(height * pixel_area)`, in **µm³** (1 µm³ = 1 fL); equivalently
`projected_area * mean(height)`. The calculator holds both sides of that
relation — a `ProjectedAreaCalculator` (built from `pixel_size`) and an
`OpticalHeightConverter` — so `volume_scale` is their product
(`area_scale * 1e-3 / refractive_delta`). Batching, `mask`, `reduce`, and the
background-correction requirement all match `drymass` below; an empty region
integrates to 0 µm³. Dry mass is the `refractive_delta / alpha` multiple of this
volume (`DryMassCalculator.calc_from_volume`).

- `OpticalVolumeCalculator(pixel_size, height_converter=...)` — bind the pixel
  size (m) and an `OpticalHeightConverter` once.
  - `from_wavelength(pixel_size=..., wavelength=..., refractive_delta=...)`.
  - `calc_from_opd(opd, *, mask=None, reduce=True)` /
    `calc_from_height(height, ...)` / `calc_from_phase(phase, ...)`.
  - `volume_scale` — the cached µm³-per-summed-nm-OPD factor.
  - `area_calculator` / `height_converter` — the two bound engines.
  - `pixel_size` / `pixel_size_um`, `refractive_delta`, `wavelength` /
    `wavelength_nm`.
- `calc_volume(opd, *, pixel_size, refractive_delta=..., mask=None, reduce=True)` /
  `calc_volume_from_phase(phase, *, pixel_size, wavelength=..., refractive_delta=..., ...)`
  — the one-shot forms.

## `drymass` — dry mass

Dry mass `= (1 / alpha) * sum(OPD * pixel_area)` (the Barer relation), in **pg**,
summed over the last two axes (H, W). Inputs are **batched** — `opd` / `phase`
have shape `(..., H, W)`, giving one mass per image (`(...)`). The OPD must
already be background-corrected (≈ 0 outside the object); a `mask` restricts the
sum to region(s): a boolean `(H, W)` (one region) or `(N, H, W)` (`N` regions,
which may overlap, giving a trailing `(..., R)` axis), or an integer label image
`(H, W)` (0 = background, one region per positive label). An empty region (no
pixels) integrates to 0 pg. Segmentation and background estimation stay the
caller's responsibility; the masking + reduction is the `iivs.common.data`
`Sum` reduction, which `DryMassCalculator` holds internally.

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
  - `calc_from_volume(volume, *, refractive_delta=...)` — the Barer bridge from an
    already-integrated optical volume (µm³): `mass = volume * delta / alpha`.
  - `drymass_scale` — the cached pg-per-summed-nm factor (a plain `float`).
  - `wavelength` / `wavelength_nm`, `pixel_size` / `pixel_size_um`.
- `calc_drymass(opd, *, pixel_size, alpha=..., mask=None, reduce=True)` /
  `calc_drymass_from_phase(phase, *, pixel_size, wavelength=..., alpha=..., mask=None, reduce=True)`
  — the one-shot forms.

Defaults for `wavelength` and `alpha` come from
[`iivs.dhm.constants`](../constants.py).

## Using with PyTorch (autograd)

The `convert_*` / `calc_*` methods run on NumPy arrays. Install the
`iivs-lib[torch]` extra for the `pytorch` subpackage — tensor-in / tensor-out
twins that keep the input tensor's device, dtype, and autograd graph. The
physical calibration (the scale factors) is reused from the NumPy engines, so
only the elementwise ops are torch-native.

Unlike the NumPy engines (which fold in the masked reduction), the Torch layers
are **pure pointwise** `nn.Module`s — one op each, so they drop cleanly into
`nn.Sequential`, hooks, `torch.compile`, and fx / export tracing:

- `OpticalPathDifference(wavelength=...)` — `forward(phase) = phase * opd_scale`.
- `OpticalHeight(refractive_delta, wavelength=...)` — `forward(opd) = opd / delta`,
  plus `convert_from_phase` for the rad → nm path.
- `OpticalVolume(pixel_size=..., refractive_delta=...)` — `forward(opd) =
  opd * volume_scale`, the per-pixel volume density (µm³). No `mask` / `reduce`.
- `DryMass(pixel_size=..., alpha=...)` — `forward(opd) = opd * drymass_scale`, the
  per-pixel dry-mass density (pg). No `mask` / `reduce`.

`calc_volume` / `calc_volume_from_phase` are the volume one-shots, and
`calc_projected_area` is the mask-only area twin (no `nn.Module` — counting mask
pixels is not a pointwise map).

Masking into regions and reducing to a total are a **separate** step — the
reductions in [`iivs.common.data.pytorch`](../../common/data) (`Sum`, `Mean`,
`Norm`, ...). Compose them, or let the one-shot free functions do it:

```python
from iivs.dhm.analysis.pytorch.opd import OpticalPathDifference, phase_to_opd
from iivs.dhm.analysis.pytorch.drymass import DryMass, calc_drymass_from_phase
from iivs.common.data.pytorch import Sum

# Compose the pointwise layers with a reduction:
density = DryMass(pixel_size=px)(OpticalPathDifference(666e-9)(phase))  # pg/pixel
mass = Sum(mask=cell)(density)                                          # pg per region

# Or one-shot (composes the same pieces; an empty region -> 0 pg):
opd = phase_to_opd(phase, wavelength=666e-9)                    # Tensor, grad kept
mass = calc_drymass_from_phase(phase, pixel_size=px, mask=cell) # Tensor, grad kept
```

`calc_*` keeps the input's device, dtype, and autograd graph, accumulating each
region's sum in float64 (so f16 / bf16 / f64 all survive, where the NumPy engine
returns float32). Without the dependency, multiply by the cached scale factors
(plain floats) yourself:

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
