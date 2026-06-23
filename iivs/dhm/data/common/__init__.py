"""Building blocks shared across the data modalities (phase, intensity, ...).

The cross-modality primitives the per-modality packages compose, split by
concern into submodules and re-exported here so ``from iivs.dhm.data.common
import X`` reaches any of them:

- `bin` — the Koala `.bin` header value object and pixel I/O (`KoalaBinHeader`,
  `read_bin_pixels`, `write_bin`).
- `txt` — the `Float/Txt` header codec and grid I/O (`KoalaTxtHeaderCodec`,
  `parse_txt_grid`, `write_txt_grid`).
- `image` — the uint8 image folder/list bases and `.tif` reader
  (`ImageFileFolder` / `ImageFileList`, `ImageTifFolder` / `ImageTifList`,
  `load_uint8_tif`).
- `float` — the float32 file list/folder bases over a ``(read_header, decode)``
  codec, generic in the header type (`KoalaFloatFileList`,
  `KoalaFloatFileFolder`; shared by phase and intensity).
- `sequence` — the numbered-folder base (`SequentialFileFolder`).
- `validation` — the float32 / uint8 image validators.
- `utils` — the numbered-folder helpers (`numbered_name`,
  `detect_numbered_format`). The extension helpers `file_extension`,
  `ensure_file_extension` (with its `add=True` mode), and the
  `UnsupportedExtensionError` raised on a bad extension are imported directly
  from `kaparoo.filesystem` where needed, not re-exported here.

The technique-agnostic `.npy` reader / writer (`read_npy_shape`, `write_npy`)
and the same-shape mixin `FrameShapedMixin` live in `iivs.common.data`; import
them from there.
"""

from __future__ import annotations

__all__ = (
    "FLOAT_FORMATS",
    "FloatFormat",
    "ImageFileFolder",
    "ImageFileList",
    "ImageTifFolder",
    "ImageTifList",
    "KoalaBinHeader",
    "KoalaFloatFileFolder",
    "KoalaFloatFileList",
    "KoalaTxtHeaderCodec",
    "OnNonFinite",
    "SequentialFileFolder",
    "ValidationLevel",
    "detect_numbered_format",
    "load_uint8_tif",
    "numbered_name",
    "parse_txt_grid",
    "read_bin_pixels",
    "validate_float32_image",
    "validate_uint8_image",
    "write_bin",
    "write_txt_grid",
)

from iivs.dhm.data.common.bin import KoalaBinHeader, read_bin_pixels, write_bin
from iivs.dhm.data.common.float import (
    FLOAT_FORMATS,
    FloatFormat,
    KoalaFloatFileFolder,
    KoalaFloatFileList,
)
from iivs.dhm.data.common.image import (
    ImageFileFolder,
    ImageFileList,
    ImageTifFolder,
    ImageTifList,
    load_uint8_tif,
)
from iivs.dhm.data.common.sequence import SequentialFileFolder, ValidationLevel
from iivs.dhm.data.common.txt import KoalaTxtHeaderCodec, parse_txt_grid, write_txt_grid
from iivs.dhm.data.common.utils import detect_numbered_format, numbered_name
from iivs.dhm.data.common.validation import (
    OnNonFinite,
    validate_float32_image,
    validate_uint8_image,
)
