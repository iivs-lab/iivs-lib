# TODO

Tracked items that are not yet captured in code or tests. Promote an
item to a CHANGELOG entry once it lands.

## Open

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
