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

Every engine is default-constructible from the lab constants in
[`iivs.dhm.constants`](../constants.py) (including the 20X pixel size), receives
its inner engines at construction, and offers a fully-explicit `from_args`
builder taking plain parameters instead.

These operate on plain NumPy arrays; they are not sequences. For PyTorch, see
[Using with PyTorch](#using-with-pytorch-autograd).

## `opd` — optical path difference

`OPD = phase * wavelength / (2 * pi)`, in **nm**, independent of any refractive
index (and distinct from the optical height, which additionally divides by the
refractive-index difference).

- `OPDConverter(wavelength=...)` — bind a wavelength (SI, m) once.
  - `from_wavelength_nm(nm)` — construct from a wavelength in nm.
  - `convert_from_phase(phase)` / `convert_to_phase(opd)` — rad ↔ nm.
  - `opd_scale` — the cached nm-of-OPD-per-rad factor (a plain `float`).
  - `wavelength` / `wavelength_nm`.
- `phase_to_opd(phase, *, wavelength=...)` / `opd_to_phase(opd, *, wavelength=...)`
  — the one-shot forms.

## `height` — optical height

`height = phase · λ/(2π·Δn)`, in **nm**: the physical thickness producing the
measured phase. Transmission QPI literature usually calls this quantity the
sample **thickness** (`phase = 2π·Δn·t/λ`); "height" keeps this library's
established name for it. Phase is the preferred representation: an OPD input
enters through phase via the bound `OPDConverter`, whose wavelength then
cancels (`height == OPD / Δn`).

- `OpticalHeightConverter(refractive_delta=..., opd_converter=...)` — bind the
  refractive-index difference and an `OPDConverter` once.
  - `from_args(wavelength=..., refractive_delta=...)`.
  - `convert_from_phase(phase)` / `convert_to_phase(height)` — rad ↔ nm.
  - `convert_from_opd(opd)` / `convert_to_opd(height)` — the OPD entry / exit,
    routed through phase.
  - `height_scale` — the cached nm-of-height-per-rad factor
    (`wavelength / (2π·Δn)` in nm).
  - `refractive_delta`, `wavelength` / `wavelength_nm`.
- `phase_to_height` / `height_to_phase` / `opd_to_height` / `height_to_opd` —
  the one-shot forms.

## `area` — projected area

`area = pixel_count * pixel_size²`, in **µm²**: the footprint the selected
region(s) cover in the image plane; it enters the volume relation as `volume =
area * mean(height)`. The call shape matches the other engines (`image`, `mask`,
`reduce`), but the image's *values* never enter the area — the image fixes the
pixel grid (and any leading batch axes), and each selected pixel contributes the
constant `area_scale`.

- `ProjectedAreaCalculator(pixel_size=PIXEL_SIZE_20X)` — bind the pixel size (m)
  once; `from_pixel_size_um(um)` is the µm builder twin.
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
`projected_area * mean(height)`. The calculator receives both sides of that
relation at construction — a `ProjectedAreaCalculator` and an
`OpticalHeightConverter` — so `volume_scale` is their product
(`area_scale * height_scale * 1e-3`, µm³ per rad of phase). Phase is the
canonical input: an OPD or height map converts back to phase first. Batching,
`mask`, `reduce`, and the background-correction requirement all match `drymass`
below; an empty region integrates to 0 µm³. Dry mass is this volume's
`refractive_delta / alpha` multiple (see `drymass`).

- `OpticalVolumeCalculator(area_calculator=..., height_converter=...)` — bind
  the two engines once.
  - `from_args(pixel_size=..., wavelength=..., refractive_delta=...)` — build
    both engines from plain parameters.
  - `calc(phase, *, mask=None, reduce=True)` (canonical) /
    `calc_from_opd(opd, ...)` / `calc_from_height(height, ...)` — the latter two
    convert back to phase first.
  - `volume_scale` — the cached µm³-per-summed-rad-phase factor.
  - `pixel_size` / `pixel_size_um`, `refractive_delta`, `wavelength` /
    `wavelength_nm` — re-surfaced from the bound engines.
- `calc_optical_volume(phase, *, pixel_size, wavelength=..., refractive_delta=..., mask=None, reduce=True)`
  (canonical) / `calc_optical_volume_from_opd(opd, ...)` / `calc_optical_volume_from_height(height, ...)`
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

- `DryMassCalculator(volume_converter, alpha=...)` — bind an
  `OpticalVolumeCalculator` (which carries the pixel size, wavelength, and delta)
  and the specific refractive increment (m³/kg) once; the last link of the
  engine chain.
  - `from_args(pixel_size=..., wavelength=..., refractive_delta=..., alpha=...)`
    — build the whole engine chain from plain parameters (all explicit).
  - `calc(phase, *, mask=None, reduce=True)` (canonical) — dry mass from a phase
    map (rad). `reduce=False` returns the per-pixel mass-density map instead.
  - `calc_from_opd(opd, ...)` / `calc_from_height(height, ...)` — the OPD / height
    entry points, converted back to phase first.
  - `drymass_scale` — the cached pg-per-summed-rad-phase factor (a plain `float`).
  - `wavelength` / `wavelength_nm`, `pixel_size` / `pixel_size_um`,
    `refractive_delta` — re-surfaced from the bound volume engine.
- `calc_drymass(phase, *, pixel_size, wavelength=..., refractive_delta=..., alpha=..., mask=None, reduce=True)`
  (canonical) / `calc_drymass_from_opd(opd, ...)` / `calc_drymass_from_height(height, ...)`
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
- `OpticalHeight(refractive_delta=..., opd_converter=...)` — `forward(phase) =
  phase * height_scale`, owning an `OpticalPathDifference` submodule for the
  `convert_from_opd` / `convert_to_opd` entry / exit (`from_args` builds it from
  a wavelength).
- `OpticalVolume(area_calculator=..., height_converter=...)` — `forward(phase) =
  phase * volume_scale`, the per-pixel volume density (µm³); owns a `ProjectedArea`
  and an `OpticalHeight` submodule (`from_args(pixel_size=..., wavelength=...,
  refractive_delta=...)` builds both), and an OPD / height map enters through phase
  via `calc_from_opd` / `calc_from_height`. No `mask` / `reduce`.
- `DryMass(volume_converter=..., alpha=...)` — `forward(phase) =
  phase * drymass_scale`, the per-pixel dry-mass density (pg); owns an
  `OpticalVolume` submodule (`from_args(pixel_size=..., wavelength=...,
  refractive_delta=..., alpha=...)` builds the chain), an OPD / height map entering
  via `calc_from_opd` / `calc_from_height`. No `mask` / `reduce`.

`calc_optical_volume` (phase) / `calc_optical_volume_from_opd` /
`calc_optical_volume_from_height` and `calc_drymass` (phase) /
`calc_drymass_from_opd` / `calc_drymass_from_height` are the one-shots; the shared
`mask` / `reduce` dispatch is `iivs.common.data.pytorch.reduce_regions`.
`ProjectedArea(pixel_size=...)` — `forward(image)` is the constant per-pixel area
density (µm²) over the image's grid (its values never enter, so no gradient flows
from the image); `calc_projected_area` is its mask / reduce one-shot.

Masking into regions and reducing to a total are a **separate** step — the
reductions in [`iivs.common.data.pytorch`](../../common/data) (`Sum`, `Mean`,
`Norm`, ...). Compose them, or let the one-shot free functions do it:

```python
from iivs.dhm.analysis.pytorch.drymass import DryMass, calc_drymass
from iivs.common.data.pytorch import Sum

# Compose the pointwise layer with a reduction (DryMass takes phase directly):
dm = DryMass.from_args(pixel_size=px, wavelength=666e-9, refractive_delta=dn, alpha=a)
mass = Sum(mask=cell)(dm(phase))                               # pg per region

# Or one-shot (composes the same pieces; an empty region -> 0 pg):
mass = calc_drymass(phase, pixel_size=px, mask=cell)          # Tensor, grad kept
```

`calc_*` keeps the input's device, dtype, and autograd graph, accumulating each
region's sum in float64 (so f16 / bf16 / f64 all survive, where the NumPy engine
returns float32). Without the dependency, multiply by the cached scale factors
(plain floats) yourself:

```python
opd = phase * conv.opd_scale                    # phase: Tensor -> OPD (nm), grad kept
mass = phase[mask].sum() * calc.drymass_scale   # phase -> dry mass (pg), grad kept
```

## Example

```python
from iivs.dhm.analysis import OPDConverter, DryMassCalculator

conv = OPDConverter.from_wavelength_nm(666)
opd_nm = conv.convert_from_phase(phase_rad)    # rad -> nm

calc = DryMassCalculator()  # lab defaults: 20X pixel size, 666 nm, alpha 2e-4
mass_pg = calc.calc(phase_rad, mask=cell_mask)
```
