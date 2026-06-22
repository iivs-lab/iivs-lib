# TODO

Tracked items that are not yet captured in code or tests. Promote an
item to a CHANGELOG entry once it lands.

## Open

- **Exercise the real LZW `Image/*.tif` decode path in tests.** `load_uint8_tif`
  is only tested against an *uncompressed* tif; the actual `imagecodecs` LZW
  decode (what Koala writes) is never run. Add a test that round-trips a
  genuinely LZW-compressed uint8 tif — either a small generated fixture
  (`tifffile.imwrite(path, data, compression="lzw")`, no proprietary data to
  ship) or a minimal real Koala `Image/*.tif` sample asset (mind size and the
  Lyncée Tec data-redistribution question; keep it out of the built
  sdist/wheel). `imagecodecs` is now a core dependency, so the codec is always
  present in CI.
- **Add a dataset/acquisition opener.** Koala nests its export as
  `<Modality>/Float/Bin`, `<Modality>/Float/Txt`, `<Modality>/Image`, plus
  `Holograms/holo.raw`, `timestamps.txt`, and `phbounds.txt` at the root (this
  layout is confirmed against a real acquisition sample); today each leaf path is
  opened separately. Add a top-level opener that takes the acquisition root and
  wires phase (`Phase/Float/Bin`), intensity (`Intensity/Float/Bin`), holograms
  (`Holograms/holo.raw`), and timestamps into one object, tolerating absent
  modalities.
- **Consider matching Koala's exact preview quantization.** Verified on real
  data: Koala renders phase previews by *globally* normalizing `phbounds.txt`'s
  `[min, max]` to 0–255, while intensity previews are normalized *per frame*.
  `PhaseBounds.encode_preview` uses a clean, invertible `round(x * 255)` that
  matches Koala to within 1 code; Koala itself looks like a 256-level
  `round(x * 256)` clamped to 255 (a closer ≈74 % exact match on the sample, but
  not cleanly invertible). Decide whether tighter preview fidelity is worth
  giving up the invertible 255-level map — the `Float` data stays the exact
  source regardless.

## Finalize the WIP dispatch / composer features

Landed as WIP (the two `🚧 WIP` commits since `v0.1.0`) and not yet complete.
They are uneven across modalities and under-tested/-documented; finish each to
parity before the next release.

- **Extension-based dispatch — bring `hologram` to parity and surface at the
  data root.** `phase` and `intensity` have full suffix-dispatch factories
  (`load_*`, `read_*_header`, `save_*`, `*_list`, `*_folder`), but `hologram`
  has *none* — it ships only the `HologramFormat` / `HOLOGRAM_FORMATS`
  vocabulary, with no `load_hologram` / `hologram_folder` / `hologram_list` /
  `save_hologram` (`.tif` / `.raw` / `.npy`). Add the `hologram` factory, then
  re-export the per-modality entry points from `iivs.dhm.data` (the package root
  currently surfaces none of them), so a caller reaches dispatch without diving
  into submodules.
- **Composer-friendly folder export — confirm cross-modality coverage.**
  `save_phase_folder` / `save_intensity_folder` accept any image sequence
  (`kaparoo` composers, `to_float` / `to_image` views, plain lists). Verify the
  hologram side reaches the same place: `convert_hologram_sequence` /
  `save_hologram_raw` already take any uint8 `DataSequence`, but there is no
  `save_hologram_folder` twin for the per-frame `.tif` / `.npy` folders. Add it
  (or document why `raw` is the only stack target) and add the missing
  composer-input tests.
- **`common` shared dispatch — stabilize and test now that the local helpers are
  gone.** `file_extension` / `unsupported_extension` were replaced by
  `kaparoo.filesystem.file_extension` / `UnsupportedExtensionError`; make sure
  every factory dispatches through them consistently, and add direct unit tests
  for `detect_numbered_format` (single/multi-format folders, the `prefer`
  policies, the empty/ambiguous error paths) and the `FloatFormat` /
  `HologramFormat` aliases.
- **Hologram composer support — close the gap with `phase` / `intensity`.**
  The composer acceptance on `convert_hologram_sequence` / `save_hologram_raw`
  is the first half; the modality still lacks the dispatch factory and the
  folder-export twin above. Track these together so `hologram` ends up with the
  same composer-in / dispatch-out surface as the float modalities.

## Planned module structure (future `rcm` / `epi` techniques, shared `common`, viz)

Design decisions for growing `iivs` beyond `dhm`. The governing rule: name
namespaces by **technique** (`dhm`, `rcm`, `epi`), not contrast mechanism; keep
field-standard acronyms (`dhm` / `rcm` / `opd`) but spell out colloquial
abbreviations (`visualization`, not `viz`). Shared layers are pure infra and
**never import a technique** (one-directional dependency); a technique never
imports a sibling technique.

Lab techniques: `dhm` (existing, label-free QPI), `rcm` (re-scan confocal
super-resolution), `epi` (epifluorescence / widefield). `rcm` and `epi` are
**parallel top-level peers**, not one `confocal` namespace with `rcm` nested
inside (this revises the earlier confocal-family plan): both are first-class
techniques the lab runs.

- **`iivs.common` shared namespace.** Add `common` as a sibling of the technique
  namespaces, mirroring their layout (`common.data`, `common.visualization`,
  ...), so every namespace has a uniform `<ns>.data` / `<ns>.visualization`
  shape and `common` is "the shared technique". `common.*` is numpy / tifffile /
  kaparoo only; techniques depend on `common`, not vice versa.
- **Shared camera / OME-TIFF substrate for `rcm` + `epi` (contrast-agnostic).**
  `rcm` and `epi` share their acquisition substrate — multi-channel OME-TIFF /
  Micro-Manager data, single-stack files or T/Z/C-labeled folders, possibly
  16-bit — so that I/O must live in **one** shared place, never copied into each.
  The substrate is **camera-scoped, not fluorescence-scoped**: it is contrast-
  agnostic, so a future bright-field instrument rides on the same layer (see the
  bright-field note below). But `common` stays technique-agnostic pure infra, and
  a technique must not import a sibling, so the shared code cannot live in `rcm`,
  `epi`, or (as acquisition *semantics*) in `common`. Split it on the
  infra/semantics line: the generic container I/O — reading OME-TIFF via
  `tifffile`, parsing Micro-Manager metadata, the multichannel-stack / dimension-
  labeled-folder sequence templates — is format infra and belongs in
  `common.data`; the camera-acquisition *semantics* the peers share (named
  channels, LUTs, channel composite, dimension axes) need a home below both.
  Settle that home when the `rcm` / `epi` data code actually lands and the real
  shared surface is known (don't guess): a dedicated shared module is the likely
  answer, folding the thin remainder into `common.data` the alternative. Validate
  the boundary against the code, not by prediction.
- **Bright-field is deferred — a contrast mode, not a technique.** The lab has a
  dedicated bright-field instrument, but its output is essentially a single-
  channel grayscale intensity raster (≈ the existing `intensity` role), so it has
  no distinct surface to justify its own namespace yet. **No `bfm` namespace
  now.** If added later it is a thin grayscale module on the shared camera
  substrate above — not a fluorescence peer, and not a flat top-level. Contrast
  (fluorescence vs bright-field) is **orthogonal** to the instrument/technique
  axis; keep it a sub-concern, never a top-level (this is why the substrate above
  is camera- not fluorescence-scoped).
- **`common.visualization` core + per-technique adapters.** A
  technique-agnostic render core under `common.visualization`, with each
  `<technique>.visualization` a thin adapter adding semantics (`dhm`: phase →
  colormap + colorbar, optionally over a `PhaseBounds` nm range, intensity /
  hologram grayscale; `rcm` / `epi`: fluorescence channel LUTs + composite,
  bright-field grayscale). Matplotlib is a core dependency; viz is functions
  taking sequences / arrays, never `.show()` methods on data classes (the data
  layer stays matplotlib-free). *First slice landed*: `normalize` / `render`
  (single image → `Axes`) and the `dhm` adapters. **Remaining**: a `PhaseBounds`
  path that puts a `PhaseFloatSequence` in nm before rendering, grid / multi-axis
  layout, channel composite (for `rcm` / `epi`), sequence animation, and
  `save` / `show` helpers.
- **Hoist `common.data` (partly done; finish when `rcm` / `epi` land).** The
  zero-coupling leaves are already in `iivs.common.data`; move the remaining
  *format-agnostic* I/O up too, leaving the Koala-proprietary codecs in `dhm`. Do
  this when `rcm` / `epi` are the real second consumers
  (validate the generic/specific boundary against them, not by guessing) — the
  same pass that places the shared camera substrate above. The new requirements
  (16-bit pixels, single-stack OME-TIFF, T/Z/C-labeled folders) show the value is
  mostly in **generalizing two axes `dhm` never exercised**, not in a bulk move:
  - *Pixel dtype* (uint8 → uint8 / uint16 / float32). `validation`'s
    `validate_uint8_image` / `validate_float32_image` hardcode the dtype, but the
    dimension check is already factored into the generic `_validate_image_dims`;
    re-shape as a dtype-parametric validator (integers need no finite policy, so
    uint16 joins the uint8 side trivially). `image`'s `load_uint8_tif` and
    `ImageFileList` are uint8-locked at the type level — make them generic in the
    pixel dtype, exactly as `KoalaFloatFileList[H]` is generic in the header type
    and modalities bind it: `dhm` binds uint8, `epi` / `rcm` bind uint16.
  - *Frame addressing* (single linear `NNNNN` index → multi-axis + single-file
    stacks). `SequentialFileFolder` and `numbered_name` / `detect_numbered_format`
    hardcode Koala's `{index:05d}_{stem}.{ext}`, so they **stay dhm-specific**;
    the T/Z/C folder is a sibling template on `kaparoo`'s `FileFolderSequence`,
    and the OME-TIFF single-stack reader is **new** code (a `SingleFileSequence`
    like `HologramRawFile`, but OME-axis-aware) — neither is a hoist. The *List*
    layer (arbitrary file list, no naming) is shareable once dtype-generalized;
    only the *Folder* discovery is per-convention.
  - *Move as-is* (no coupling): **done** — `FrameShapedMixin` and the `npy`
    reader / writer (`read_npy_shape` / `write_npy`) now live in `iivs.common.data`
    (re-exported from `dhm.data.common`). `_validate_image_dims` is generic too but
    moves with the `validation` dtype generalization above, so its file is not
    split. *Stays in `dhm`* (Koala-proprietary): `bin`, `txt`, `float`, and the
    Koala numbering helpers above.

  Resolve the resulting double-"common" (`iivs.common.data` vs
  `iivs.dhm.data.common`) by folding / renaming the dhm-internal one (e.g.
  `dhm.data.koala`).
- **Timing / `timestamp` is a shared-metadata candidate for `common.data`.**
  `Timestamp`, the abstract `TimestampSequence` interface (`mean_interval_ms` /
  `mean_frame_rate`), and `TimestampsFixedFPS` (synthesized from a frame rate) are
  technique-agnostic — any time-lapse acquisition has per-frame timing — so they
  hoist alongside the image I/O. Only the Koala `timestamps.txt` reader
  (`TimestampsTxtFile`) stays in `dhm`; `epi` / `rcm` read timing from OME-TIFF /
  Micro-Manager metadata instead, implementing the same `TimestampSequence`
  interface.
