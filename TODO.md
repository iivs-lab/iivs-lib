# TODO

Actionable work on the data layer and beyond, in rough priority order. Not
formal milestones.

## `value_range` — a data-computed value range

Koala's `phbounds.txt` drifts run-to-run even on identical data, so it is a
display bound, not a data-truth range. Give phase a range read from the actual
`Float` data, kept distinct from `PhaseBounds` (which stays the `phbounds.txt`
/ preview-rendering object and a last-resort range source):

- Add `PhaseFileList.value_range(unit: PhaseUnit = NANOMETERS) -> tuple[float,
  float]`: decode each frame, convert via its own `height_scale`, reduce to one
  global `(min, max)`. Generalizes today's `bounds_nm` to any unit.
- Remove `bounds_nm`; `to_image(bounds=None)` builds `PhaseBounds(*value_range(
  NANOMETERS))` where display bounds are needed. `bounds_nm` is `[Unreleased]`,
  so only `CHANGELOG.md` needs updating.
- `intensity` is the same `KoalaFloatFileList`, so a unit-less `value_range()`
  fits it too; add it once a consumer actually needs a range.
- Per-frame is the natural primitive: a `value_ranges(unit) -> NDArray` of shape
  `(N, 2)` reduces to the global `value_range` in one pass, so computing it costs
  nothing extra. It would enable a per-frame `intensity.to_image` (intensity
  previews are normalized per frame, so they carry no global bound) and
  frame-to-frame drift / QC. Expose it only once a concrete consumer lands; keep
  it internal otherwise. RADIANS conversion is per-frame (`height_scale`), so
  decode in the target unit rather than converting a cached nm range.
- Open: cache the nm reduction (as `bounds_nm` did) or recompute per call.

## npy metadata — warn vs. silently ignore

`.npy` is header-less, so `save_phase` / `save_phase_folder` today warn and
drop any `pixel_size` / `unit` / scale passed for it. Decide: keep the warning
(a signal that calibration won't persist) or drop it (smoother format-agnostic
use), then apply the same choice to the `intensity` twins.

## `iivs.dhm.analysis` — split `MaskedReduction` out of dry mass

Make `DryMass` / `DryMassCalculator` compute only the per-pixel mass-density
map (`opd * scale`) and move masking + reduction (sum / norm / mean) into a
separate `MaskedReduction`, which also accepts label images (scikit-image
style, 0 = background) alongside the current one-hot `(N, H, W)` masks. Two
payoffs beyond label support:

- **Clean `nn.Module` composition.** A per-pixel `forward(self, opd) ->
  opd * scale` drops the keyword-only `mask` / `reduce` and the Python
  validation / `raise`, so `DryMass` becomes a shape-preserving pointwise
  layer (fits `nn.Sequential`, hooks, `jit` / `compile`) rather than a
  reduction head, matching `OpticalPathDifference`'s already-clean form.
  Gradients still flow; the reduction stays differentiable when called
  explicitly.
- **Drops the NumPy/Torch duplication.** The mask shape-validation and
  branching leave both engines for the one shared reduction.

Sequence: ① shrink `DryMass` / `DryMassCalculator` to per-pixel; ② add the
shared `MaskedReduction` (NumPy + Torch, label + one-hot).

## Release

`CHANGELOG.md`'s `[Unreleased]` has accumulated substantial public-API changes
since `0.1.0` (the suffix-dispatch factories, the `load_bin` / `load_txt`
engine consolidation, `bounds_nm` as a property, the `imagecodecs` core
dependency). Cut the next version once the data layer settles.
