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
- **Use `phbounds.txt` to map previews to/from nm.** The `phbounds.txt` record
  itself now lands: `PhaseBounds.from_file` / `to_file` (the `read_phbounds` /
  `write_phbounds` wrappers; a `[nm]` tag then `min max`, e.g.
  `-403.4911 635.9849`), and `PhaseFloatSequence.bounds_nm()` derives those
  bounds straight from the float source (global min/max in nm), so the previews
  are never authoritative. What
  remains is the *mapping*: the uint8 `Image/*.tif` previews are read as raw
  8-bit values; use a `PhaseBounds` (read from disk, or `bounds_nm()` from the
  `Float` twin) to map a preview back toward nm (lossy, 8-bit quantized — the
  `Float` data stays the exact source). Decide whether to expose it as a
  `PhaseImageSequence` helper or a standalone converter, and whether intensity
  has an analogous bounds file.
