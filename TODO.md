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
- **Use `phbounds.txt` to map previews to/from nm.** The uint8 `Image/*.tif`
  previews are now read (`PhaseTifFolder` / `IntensityTifFolder`, uint8
  `*ImageSequence`; LZW decoding via the `[image]` extra) but as raw 8-bit
  values. `phbounds.txt` holds the phase display bounds in nm — a `[nm]` line
  then `min max` (e.g. `-403.4911 635.9849`) — used to render the float phase
  into 0–255. A reader could map a preview back toward nm (lossy, 8-bit
  quantized — the `Float` data stays the exact source). Decide whether to
  expose it as a `PhaseImageSequence` helper or a standalone converter, and
  whether intensity has an analogous bounds file.
