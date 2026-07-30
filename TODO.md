# TODO

Actionable work on the data layer and beyond, in rough priority order. Not
formal milestones.

## Candidate: overridable `subpath` on the `search_*_folders` wrappers

The six format-specific wrappers (`search_phase_bin_folders` and its siblings)
fix both the subpath (`PHASE_FLOAT_BIN` etc.) and the folder class. Exposing the
subpath as a keyword argument that defaults to that constant, with the folder
class still fixed, would let a caller point the same typed reader at a
non-standard location: e.g. a filtered `.bin` re-exported to
`FilteredPhase/Float/Bin` instead of `Phase/Float/Bin`, still parsed as a
`PhaseBinFolder`. It is backward-compatible (the default is unchanged) and cheap
(six signatures plus docstrings). `open_timelapse_subfolders(root, subpath,
folder, ...)` already takes an explicit subpath, so this is convenience over that
call while keeping the wrapper's typed return. The format-agnostic
`search_phase_folders` / `search_intensity_folders` multiplex formats and stay as
they are.

## On hold: threaded `get_items` (batch reads)

Deferred until real-storage numbers justify it (2026-07-29). The idea: overlap
per-file reads when a caller fetches many frames at once (slicing, explicit
`get_items`, a PyTorch `DataLoader` batch). `load_file` is stateless and the
sequences are read-only after `__init__`, so a caller can already get the full
benefit externally, with no library change:

```python
with ThreadPoolExecutor(max_workers=4) as ex:
    frames = list(ex.map(folder.get_item, indices))  # order preserved
```

A warm-cache benchmark on a local disk (synthetic 800x800 float32 `.bin`, 200
frames, 488 MB, `PhaseBinFolder`) showed only 1.2-1.35x over the sequential
loop, saturating at 2-4 threads — the bottleneck there is GIL-held Python
overhead and memory bandwidth, not IO latency. That does not clear the bar for
the agreed design, whose cost is real: `num_workers: int = 0` threaded through
every `__init__`-defining sequence class (plus the `@overload` stubs), a
`get_items(*, num_workers: int | None = None)` override (`None` = instance
default, `0` = force sequential), implemented in `iivs.common.data`'s
`ArrayFileList` so List and Folder variants both benefit, and a
`__getitems__` delegating method as the `DataLoader` batch-fetch hook.

Revisit when a cold-cache run on the storage that actually hosts acquisitions
(NAS / external HDD) shows ≥ 1.5-2x: threading's real payoff — hiding disk and
network latency — is exactly what a warm page cache hides. Until then, use the
external `ThreadPoolExecutor` pattern at call sites that need it.

## Done

- The `analysis` package deeper pass (2026-07-31): every quantity is now
  phase-canonical (calculators funnel through `calc(phase)`; the one-shots lead
  with the phase form, `calc_optical_volume` / `calc_drymass`, plus `_from_opd` /
  `_from_height` twins). The torch twins own their inner engines as submodules
  (`OpticalVolume` owns area + height, `DryMass` owns volume) and derive their
  scales from them. The calculators share a `MaskedRegionCalculator` base
  (`_require_2d` shape guard plus the `_reduce` mask/reduce dispatch); the torch
  one-shots share `iivs.common.data.pytorch.reduce_regions`; converter forwards
  are `convert_from_phase`; and `pixel_size` defaults to `PIXEL_SIZE_20X` on both
  the NumPy and torch surfaces.
