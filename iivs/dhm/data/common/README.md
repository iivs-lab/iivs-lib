# `iivs.dhm.data.common`

Cross-modality building blocks the per-modality packages
([`phase`](../phase/README.md), [`intensity`](../intensity/README.md),
[`hologram`](../hologram/README.md)) compose. **Internal**: these are
infrastructure for the modality code, not the user-facing API. Everything is
re-exported from `common`, so `from iivs.dhm.data.common import X` reaches any
submodule.

This README is a map for contributors; per-format usage lives in the modality
READMEs.

## Submodules

| Submodule | Holds |
| --- | --- |
| `bin` | `KoalaBinHeader` (the 23-byte `.bin` header value object) + `read_bin_pixels` / `write_bin`. |
| `txt` | `KoalaTxtHeaderCodec` (a stateless `Float/Txt` header (de)serializer producing a `KoalaBinHeader`) + `parse_txt_grid` / `write_txt_grid`. |
| `float` | `KoalaFloatFileList[H]` / `KoalaFloatFileFolder[H]` — the float32 list/folder machinery, generic in the header type, over a `(read_header, decode)` codec. |
| `image` | `ImageFileList` / `ImageFileFolder` + `ImageTifList` / `ImageTifFolder` (uint8 image bases) + `load_uint8_tif`. |
| `npy` | `read_npy_shape` (shape from a `.npy` header) + `write_npy`. |
| `sequence` | `SequentialFileFolder` (numbered `{index:05d}_<stem>.<ext>` discovery + validation) + the `FrameShapedMixin` (forces `frame_shape`). |
| `validation` | `validate_float32_image` / `validate_uint8_image`. |
| `extension` | `ensure_file_extension` / `with_file_extension` / `numbered_name`. |

## Design notes

- **Codec over template.** A modality's float list/folder is
  `KoalaFloatFileList[H]` / `KoalaFloatFileFolder[H]` bound to its header type
  `H`, supplying only `FILE_EXT`, `FILE_STEM`, and the `_read_header` / `_decode`
  codec (where `_decode` returns `(image, header)`). Phase adds its unit layer by
  overriding `_postprocess` (per-frame conversion) and `_after_header`
  (`target_unit` resolution); intensity adds nothing. So a new format is a couple
  of codec methods, not a copied list+folder.
- **Folder ⊂ list.** A `*Folder` subclasses its `*List` (mirroring kaparoo's
  `FileFolderSequence ⊂ FileListSequence`) and reuses its `load_file`;
  `SequentialFileFolder` adds numbered discovery, the contiguity-checked
  `validate`, and `FrameShapedMixin`.
- **Header is the value object; the codec is separate.** `KoalaBinHeader`
  (de)serializes itself (`to_dtype` / `from_dtype`); the `Float/Txt` text form is
  handled by the stateless `KoalaTxtHeaderCodec`, kept off the value object so it
  stays free of text-format knowledge.
- **`H` is bound with a string subscript** (`KoalaFloatFileList["PhaseBinHeader"]`)
  so the header import stays under `TYPE_CHECKING` and the `base` ↔ `bin` import
  cycle is avoided.

See the [AGENTS.md](../../../../AGENTS.md) conventions section for the full
naming / structure rules.
