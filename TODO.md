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
