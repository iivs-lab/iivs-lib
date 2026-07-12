# TODO

Open items, in roughly the suggested order. These track design-quality work on
the data layer and beyond — not formal milestones.

## Decisions pending (need a call)

- **`save_phase` / `save_phase_folder` npy metadata — warn vs. silently ignore.**
  `.npy` is header-less; today passing `pixel_size` / `unit` / a scale emits a
  warning and drops them. Decide whether to keep the warning (a data-integrity
  signal that provided calibration won't persist) or drop it (smoother
  format-agnostic use). Apply the choice to the `intensity` twins too.
- **`bound` / `value_range` (min/max in `target_unit`).** A likely
  cross-modality API (phase / intensity / hologram), so design it deliberately
  rather than committing to it on `phase` alone.
- **`PhaseCalibration` value object.** Whether to bundle the scattered
  `pixel_size` / scale / `unit` metadata — 3 consumers (`save_phase`,
  `save_phase_folder`, `PhaseNpyFolder`) — into one shape-less type. Also
  cross-modality (intensity carries only `pixel_size`), so a separate proposal.

## Beyond the data layer

- **`iivs.dhm.analysis` — split `MaskedReduction` out of dry mass.** Make
  `DryMass` / `DryMassCalculator` compute only the per-pixel mass-density map
  (`opd * scale`) and move masking + reduction (sum / norm / mean) into a
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
- **`iivs.common.data`.** Re-check the technique-agnostic layer once the above
  settle (e.g. where a shared `MaskedReduction` or calibration type would live).
- **Extract `kaparoo-numpy` (deferred).** The numpy-generic, iivs-agnostic parts
  of `iivs.common.data` — the array validators (`validate_*_array`), `.npy` I/O
  (`read_npy_shape` / `write_npy`), and the array-file base (`ArrayFileList`) —
  are kaparoo-ecosystem utilities, not iivs-specific
  (`ArrayFileList` already extends `kaparoo`'s `FileListSequence`). Split them
  into a sibling `kaparoo-numpy` distribution *once a second consumer needs
  them*: premature now (iivs is the only user, and a separate lib adds
  release / CI / versioning overhead). `timestamp` is acquisition timing, not
  numpy — it stays; `FrameShapedMixin` is borderline (same-shape sequences),
  decide at extraction time. The technique-agnostic rule already keeps these
  free of `dhm` / modality coupling, so the split is mostly mechanical when
  triggered.

## Housekeeping

- **Release.** `CHANGELOG.md`'s `[Unreleased]` has accumulated substantial
  public-API changes since `0.1.0` — the suffix-dispatch factories, the
  `load_bin` / `load_txt` engine consolidation, `bounds_nm` as a property, the
  `imagecodecs` core dependency. Cut the next version once the data layer
  settles.
