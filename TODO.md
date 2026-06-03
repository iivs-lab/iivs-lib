# TODO

Tracked items that are not yet captured in code or tests. Promote an
item to a CHANGELOG entry once it lands.

## Open

- **Factor a per-format codec mixin for sequences.** Each file modality
  currently ships a folder variant (`FileFolderSequence`) and a list variant
  (`FileListSequence`) that each redeclare `load_file` / `get_meta`. As more
  formats land this duplicates the per-format adapter across both variants.
  When a third format appears, extract a `_<Modality><Format>Codec` mixin
  holding `load_file` + `get_meta` (with `get_file` declared `TYPE_CHECKING`
  only, mixin placed first in the bases) and combine it into both variants.
  Stateless formats (e.g. hologram `.tif`) DRY fully; stateful ones (phase
  `.bin`, shared header) need the folder variant unified onto per-file decode
  first. Revisit together with the duplicated numbered-folder validation
  (`list_files` / `validate` / `validate_file`). The current diamond
  (`FileFolderSequence` + modality base via `DataSequence`) is benign — this
  is a duplication/maintenance concern, not a correctness one.
