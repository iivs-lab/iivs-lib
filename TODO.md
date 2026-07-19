# TODO

Actionable work on the data layer and beyond, in rough priority order. Not
formal milestones.

## Release

`CHANGELOG.md`'s `[Unreleased]` has accumulated substantial public-API changes
since `0.1.0`: the suffix-dispatch factories, the `load_bin` / `load_txt` engine
consolidation, `bounds_nm` replaced by `value_range`, the `imagecodecs` core
dependency, the `iivs.common.data` masked reductions (NumPy + Torch, label +
one-hot masks) with a now-pointwise Torch `DryMass`, and the whole
`KoalaTimelapse` / `ReconstructionGroup` time-lapse layout (per-modality groups,
the `search_*` finders, content `validate`, and the status flags). The data layer
has settled; cut `0.2.0`.

This is a breaking release for a `0.1.0` caller: beyond `iivs.common` and the
time-lapse layer being new additions, the cycle renamed or moved existing public
API (the `iivs.dhm.data.common` → `iivs.dhm.data.koala` module, `bounds_nm` →
`value_range`, the `iivs.dhm.data.constants` → `iivs.dhm.constants` move,
`convert_*_folder`'s `root` → `dest`, and more; see the `[Unreleased]`
**Changed** / **Removed** entries). Pre-1.0 SemVer permits the minor bump.
