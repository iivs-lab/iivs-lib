# TODO

Tracked items that are not yet captured in code or tests. Promote an
item to a CHANGELOG entry once it lands.

## Open

- **Share the format codec across `.bin` / `.txt` within a modality.** Two
  axes of reuse are already in place: numbered-folder mechanics via
  `data.common.SequentialFileFolder`, and folder/list via each `*Folder`
  subclassing its `*List` (so the folder inherits `load_file`). What remains is
  the *per-format* duplication within a modality: `PhaseBinList` vs
  `PhaseTxtList` differ only by `load_file`'s decoder (`load_phase_bin` vs
  `load_phase_txt`), and `PhaseBinFolder` vs `PhaseTxtFolder` only by the header
  reader in `__init__` / `_validate_content` (`FILE_EXT` aside); the intensity
  pairs likewise. Express the codec (header reader + decode fn) once per
  (modality, format) and inject it, so the list/folder bodies stop repeating
  per format. A maintenance concern, not a correctness one.
- **Add a dataset/acquisition opener.** Koala nests its export as
  `<Modality>/Float/Bin`, `<Modality>/Float/Txt`, `<Modality>/Image`, plus
  `Holograms/holo.raw`, `timestamps.txt`, and `phbounds.txt` at the root; today
  each leaf path is opened separately. Add a top-level opener that takes the
  acquisition root and wires phase (`Phase/Float/Bin`), intensity
  (`Intensity/Float/Bin`), holograms (`Holograms/holo.raw`), and timestamps
  into one object, tolerating absent modalities.
- **Read the `Image/*.tif` previews and `phbounds.txt`.** The `Image/` tifs are
  rendered **uint8 LZW-compressed** grayscale previews (display-only — distinct
  from the quantitative `Float` data). Two blockers deferred them: reading LZW
  needs the **`imagecodecs`** package (not a current dependency, so a core or
  optional-extra decision), and being uint8 they do not fit the float32
  `PhaseSequence` / `IntensitySequence` contract (so they would be a separate
  uint8 preview type, not a phase/intensity sequence). `phbounds.txt` holds the
  phase display bounds in nm (`min max`) used to map the float data into those
  previews (and back toward nm). Plan once the above is resolved.
