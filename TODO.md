# TODO

Actionable work on the data layer and beyond, in rough priority order. Not
formal milestones.

## In progress: the `analysis` package needs a deeper pass

The engine-injection redesign (volume owns area + height, dry mass owns
volume) landed fast and module-by-module review is finding real work; `opd`
and `height` are done (2026-07-30, including the phase-preferred height axis
and the torch `OpticalHeight` owning its OPD submodule), with `area`,
`volume`, and `drymass` still to review. Known items so far:

- The torch `OpticalVolume` and `DryMass` still copy scalar factors out of the
  NumPy engines instead of owning their engine submodules the way
  `OpticalHeight` now does; bring them to the same composition once their
  reviews reach them.
- The one-shot and `from_args` surfaces expose chain-building parameters that
  cancel out of the result (`wavelength` / `refractive_delta` on the dry-mass
  paths); each is documented, but the surface deserves one deliberate pass
  rather than per-module accretion.
- `pixel_size` defaulting to `PIXEL_SIZE_20X` everywhere is a convenience /
  safety trade-off: a caller who forgets it silently gets 20X-scaled numbers.
  Revisit whether the engines should keep that default once the review is
  through.
- Docs and tests grew with the churn; after the last module review, sweep for
  leftover wording, duplicated anchors, and the analysis-docs rule (no
  data-layer vocabulary: no Koala, file formats, or phase-unit types).

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
