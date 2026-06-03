# TODO

Tracked items that are not yet captured in code or tests. Promote an
item to a CHANGELOG entry once it lands.

## Open

- **Factor a per-format codec for the list variant + per-file decode.** The
  numbered-folder mechanics (`list_files` / `get_meta` / `validate` /
  `validate_file`) are now shared via `data.common.SequentialFileFolder`
  (a `FileFolderSequence` subclass with a `_validate_content` hook). What
  remains is the per-format `load_file`, still redeclared by each folder *and*
  its list variant (`FileListSequence`), and the list variant's `get_meta`. As
  more formats land, extract a `_<Modality><Format>Codec` holding `load_file`
  (+ `get_meta` for lists) and share it across the folder and list variants.
  Stateless formats (hologram `.tif`) DRY fully; stateful ones (phase/intensity
  `.bin`, shared header) carry per-file conversion. A duplication/maintenance
  concern, not a correctness one.
- **Add a dataset/acquisition opener.** Koala nests its export as
  `<Modality>/Float/Bin`, `<Modality>/Float/Txt`, `<Modality>/Image`, plus
  `Holograms/holo.raw`, `timestamps.txt`, and `phbounds.txt` at the root; today
  each leaf path is opened separately. Add a top-level opener that takes the
  acquisition root and wires phase (`Phase/Float/Bin`), intensity
  (`Intensity/Float/Bin`), holograms (`Holograms/holo.raw`), and timestamps
  into one object, tolerating absent modalities.
- **Read the `Float/Txt` exports for phase and intensity.** Alongside `.bin`,
  Koala writes a text float form (`<idx>_phase.txt` / `<idx>_intensity.txt`)
  with a small header (`h=900 w=900`, `pixel size=...`) followed by the float
  grid. Redundant with `.bin` but planned: add `.txt` loaders and matching
  folder/list sequences.
- **Read the `Image/*.tif` previews and `phbounds.txt`.** The `Image/` tifs are
  rendered 8/16-bit previews (display-only — distinct from the quantitative
  `Float` data and from the raw hologram `.tif`). `phbounds.txt` holds the
  phase display bounds in nm (`min max`) used to map the float data into those
  previews. Planned: tif preview loaders/sequences plus a `phbounds.txt` reader
  (the bounds also let one map a preview back toward nm).
