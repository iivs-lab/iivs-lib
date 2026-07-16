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
  branching leave both engines for the one shared reduction.

Sequence: ① shrink `DryMass` / `DryMassCalculator` to per-pixel; ② add the
shared `MaskedReduction` (NumPy + Torch, label + one-hot).

## Release

`CHANGELOG.md`'s `[Unreleased]` has accumulated substantial public-API changes
since `0.1.0`: the suffix-dispatch factories, the `load_bin` / `load_txt` engine
consolidation, `bounds_nm` replaced by `value_range`, the `imagecodecs` core
dependency, and the whole `KoalaTimelapse` / `ReconstructionGroup` time-lapse
layout (per-modality groups, the `search_*` finders, content `validate`, and the
status flags). The data layer has settled; cut `0.2.0`.
