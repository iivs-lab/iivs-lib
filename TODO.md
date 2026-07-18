# TODO

Actionable work on the data layer and beyond, in rough priority order. Not
formal milestones.

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
  branching leave both engines for the one shared reduction. This is measured,
  not estimated: `calc_from_opd`'s validation block is currently **16 lines
  identical down to the whitespace** between `drymass.py` and
  `pytorch/drymass.py` — it touches only `.ndim` / `.shape`, which is why it
  reads the same in both, and why it belongs to neither. (If this split slips,
  a `_validate_opd_mask(opd, mask) -> use_mask` helper in `drymass.py` is the
  stopgap: `pytorch/drymass.py` already imports `DryMassCalculator` for
  `drymass_scale`, so it adds no coupling. The split then deletes it.)

Sequence: ① shrink `DryMass` / `DryMassCalculator` to per-pixel; ② add the
shared `MaskedReduction` (NumPy + Torch, label + one-hot).

Whatever `MaskedReduction`'s sum ends up looking like, it must keep the
float64-without-copy pattern `calc_from_opd` now uses (sum with `dtype=`, not a
cast of the whole stack): it is the reduction, so the float64 accumulation is its
job.

## Release

`CHANGELOG.md`'s `[Unreleased]` has accumulated substantial public-API changes
since `0.1.0`: the suffix-dispatch factories, the `load_bin` / `load_txt` engine
consolidation, `bounds_nm` replaced by `value_range`, the `imagecodecs` core
dependency, and the whole `KoalaTimelapse` / `ReconstructionGroup` time-lapse
layout (per-modality groups, the `search_*` finders, content `validate`, and the
status flags). The data layer has settled; cut `0.2.0`.

This is a breaking release for a `0.1.0` caller: beyond `iivs.common` and the
time-lapse layer being new additions, the cycle renamed or moved existing public
API (the `iivs.dhm.data.common` → `iivs.dhm.data.koala` module, `bounds_nm` →
`value_range`, the `iivs.dhm.data.constants` → `iivs.dhm.constants` move,
`convert_*_folder`'s `root` → `dest`, and more; see the `[Unreleased]`
**Changed** / **Removed** entries). Pre-1.0 SemVer permits the minor bump.
