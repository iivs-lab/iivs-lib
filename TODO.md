# TODO

Tracked items that are not yet captured in code or tests. Promote an
item to a CHANGELOG entry once it lands.

## Open

- **Exercise the real LZW `Image/*.tif` decode path in tests.** Today the
  `load_uint8_tif` LZW branch is only covered via a monkeypatched
  `tifffile.imread` (`tests/dhm/data/test_common.py`); the actual `imagecodecs`
  decode is never run. Add a test that round-trips a genuinely LZW-compressed
  uint8 tif — either a small generated fixture
  (`tifffile.imwrite(path, data, compression="lzw")`, no proprietary data to
  ship) or a minimal real Koala `Image/*.tif` sample asset (mind size and the
  Lyncée Tec data-redistribution question; keep it out of the built
  sdist/wheel). The `dev` group already pulls `iivs-lib[image]`, so the codec
  is present in CI.
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
