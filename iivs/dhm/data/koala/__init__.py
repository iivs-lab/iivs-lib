"""Building blocks shared across the data modalities (phase, intensity, ...).

The cross-modality primitives the per-modality packages compose, split by concern into
submodules and re-exported here so ``from iivs.dhm.data.koala import X`` reaches any of
them:

- `bin` — the Koala `.bin` header value object and I/O (`KoalaBinHeader`,
  `load_bin`, `write_bin`).
- `txt` — the `Float/Txt` header codec and image I/O (`KoalaTxtHeaderCodec`,
  `load_txt`, `write_txt`).
- `image` — the uint8 `.tif` reader (`load_uint8_tif`) and the numbered folder /
  list concretes (`ImageFileFolder`, `ImageTifFolder` / `ImageTifList`) that bind
  uint8 on `iivs.common.data`'s generic `ImageFileList` and `load_tif`.
- `float` — the float32 file list/folder bases over a ``(read_header, decode)``
  codec, generic in the header type (`KoalaFloatFileList`,
  `KoalaFloatFileFolder`; shared by phase and intensity).
- `sequence` — the numbered-folder base (`SequentialFileFolder`) plus its naming
  / discovery helpers (`numbered_name`, `detect_numbered_format`). The extension
  helpers `file_extension`, `ensure_file_extension` (with its `add=True` mode),
  and the `UnsupportedExtensionError` raised on a bad extension are imported
  directly from `kaparoo.filesystem` where needed, not re-exported here.

The technique-agnostic primitives live in `iivs.common.data`: the `.npy` reader / writer
(`read_npy_shape`, `write_npy`), the same-shape mixin `FrameShapedMixin`, the array
validators (`validate_float_array` / `validate_uint_array` and their float32 / uint8
bindings), and the dtype-generic image bases (`load_tif`, `ImageFileList`); import them
from there.
"""

from __future__ import annotations

__all__ = (
    "FLOAT_FORMATS",
    "FloatFormat",
    "ImageFileFolder",
    "ImageTifFolder",
    "ImageTifList",
    "KoalaBinHeader",
    "KoalaFloatFileFolder",
    "KoalaFloatFileList",
    "KoalaTxtHeaderCodec",
    "SequentialFileFolder",
    "ValidationLevel",
    "detect_numbered_format",
    "load_bin",
    "load_txt",
    "load_uint8_tif",
    "numbered_name",
    "write_bin",
    "write_txt",
)

from iivs.dhm.data.koala.bin import KoalaBinHeader, load_bin, write_bin
from iivs.dhm.data.koala.float import (
    FLOAT_FORMATS,
    FloatFormat,
    KoalaFloatFileFolder,
    KoalaFloatFileList,
)
from iivs.dhm.data.koala.image import (
    ImageFileFolder,
    ImageTifFolder,
    ImageTifList,
    load_uint8_tif,
)
from iivs.dhm.data.koala.sequence import (
    SequentialFileFolder,
    ValidationLevel,
    detect_numbered_format,
    numbered_name,
)
from iivs.dhm.data.koala.txt import KoalaTxtHeaderCodec, load_txt, write_txt
