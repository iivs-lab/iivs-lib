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

## `iivs.dhm.data`

- **`hologram` focused review.** The one modality without a dedicated design
  pass (only incidental shared changes so far). Check docstring/contract
  consistency across `tif` / `raw` / `npy` / `convert`, and whether
  `HologramRawHeader` is coherent with the phase/intensity header patterns.

## Beyond the data layer

- **`iivs.dhm.analysis`.** Revisit the deferred `MaskedReduction` idea —
  separating mask + reduction (sum / norm / mean) from the dry-mass calculation.
- **`iivs.common.data`.** Re-check the technique-agnostic layer once the above
  settle (e.g. where a shared `MaskedReduction` or calibration type would live).

## Housekeeping

- **Release.** `CHANGELOG.md`'s `[Unreleased]` has accumulated substantial
  public-API changes since `0.1.0` — the suffix-dispatch factories, the
  `load_bin` / `load_txt` engine consolidation, `bounds_nm` as a property, the
  `imagecodecs` core dependency. Cut the next version once the data layer
  settles.
