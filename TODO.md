# TODO

Tracked items that are not yet captured in code or tests. Promote an
item to a CHANGELOG entry once it lands.

## Open

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

## Planned module structure (future `confocal` / `rcm`, shared `common`, viz)

Design decisions reached for growing `iivs` beyond `dhm`. The governing rule:
name namespaces by **technique** (`dhm`, `confocal`), not contrast mechanism;
keep field-standard acronyms (`dhm` / `rcm` / `opd`) but spell out colloquial
abbreviations (`visualization`, not `viz`). Shared layers are pure infra and
**never import a modality** (one-directional dependency).

- **`iivs.common` shared namespace.** Add `common` as a sibling of the modality
  namespaces, mirroring their layout (`common.data`, `common.visualization`,
  ...), so every namespace has a uniform `<ns>.data` / `<ns>.visualization`
  shape and `common` is "the shared modality". `common.*` is numpy / tifffile /
  kaparoo only; modalities depend on `common`, not vice versa.
- **`common.visualization` core + per-modality adapters.** Build a
  modality-agnostic render core (normalize, colormap / LUT, colorbar, grid,
  channel composite, sequence animation, save/show, matplotlib backend) under
  `common.visualization`; each `<modality>.visualization` is a thin adapter
  adding semantics (`dhm`: phase → nm range + colormap + colorbar via
  `PhaseBounds`, intensity / hologram grayscale; `confocal`: fluorescence
  channel LUTs + composite, bright-field grayscale). Ship behind an optional
  `iivs-lib[visualization]` extra with a guarded matplotlib import (like
  `[torch]` / `[image]`); keep viz as functions taking sequences / arrays, not
  `.show()` methods on data classes (data layer stays matplotlib-free). This is
  greenfield — build it first.
- **Hoist `common.data` when `confocal` lands.** Move the format-agnostic I/O now
  in `dhm.data.common` (npy, tif / image, validation, the numbered-folder
  sequence templates, `numbered_name`) up to `common.data`, leaving the Koala
  `.bin` / `.txt` codecs (`KoalaBinHeader`, `KoalaTxtHeaderCodec`,
  `KoalaFloatFile*`) in `dhm`. Do this when `confocal` is the real second
  consumer (validate the generic/specific boundary against it, not by guessing).
  Resolve the resulting double-"common" (`iivs.common.data` vs
  `iivs.dhm.data.common`) by folding / renaming the dhm-internal one (e.g.
  `dhm.data.koala`).
- **`iivs.confocal` namespace (camera / Micro-Manager confocal + RCM).** A
  technique-named namespace for the camera-based confocal microscope, which also
  does bright-field — so `confocal` (technique), not `fluorescence` (a contrast
  mode that would exclude bright-field). Axes: contrast (fluorescence /
  bright-field) × resolution (standard / `rcm`). RCM is a **variant within**
  `confocal` (re-scan super-res), not a parallel top-level — it shares the
  multi-channel OME-TIFF / Micro-Manager data and differs mainly in re-scan pixel
  size / resolution. Shared OME-TIFF / Micro-Manager / channel I/O lives in
  `confocal.data.common` (over `common.data`); fluorescence / bright-field are
  contrast sub-modules; the `rcm` resolution variant is thin.
- **(Optional) Upstream the most-generic file machinery to `kaparoo`.** The
  numbered-folder naming + open-index discovery (a `Numbered` filter unifying
  today's `numbered_name` + discovery regex) and the acquisition-tree spec
  (`kaparoo.filesystem.hierarchy`, usable as a `search` predicate via
  `conforms`) are general enough to live in `kaparoo`; `common.data` would then
  build on them. Pending a `kaparoo` release — these are in its post-0.7.0
  unreleased set, which also moves the filter DSL to `kaparoo.filters` (update
  the `Regex` import then).
