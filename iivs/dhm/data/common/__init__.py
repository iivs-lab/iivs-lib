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
- `npy` — the header-less `.npy` shape reader and writer (`read_npy_shape`,
  `write_npy`).
- `sequence` — the numbered-folder base and same-shape mixin
  (`SequentialFileFolder`, `FrameShapedMixin`).
- `validation` — the float32 / uint8 image validators.
- `utils` — the numbered-folder filename helper (`numbered_name`). The path
  helper `ensure_file_extension` (with its `add=True` mode) is imported directly
  from `kaparoo.filesystem` where needed, not re-exported here.
"""

from __future__ import annotations

__all__ = (
    "FrameShapedMixin",
    "ImageFileFolder",
    "ImageFileList",
    "ImageTifFolder",
    "ImageTifList",
    "KoalaBinHeader",
    "KoalaFloatFileFolder",
    "KoalaFloatFileList",
    "KoalaTxtHeaderCodec",
    "SequentialFileFolder",
    "load_uint8_tif",
    "numbered_name",
    "parse_txt_grid",
    "read_bin_pixels",
    "read_npy_shape",
    "validate_float32_image",
    "validate_uint8_image",
    "write_bin",
    "write_npy",
    "write_txt_grid",
)

from iivs.dhm.data.common.bin import KoalaBinHeader, read_bin_pixels, write_bin
from iivs.dhm.data.common.float import KoalaFloatFileFolder, KoalaFloatFileList
from iivs.dhm.data.common.image import (
    ImageFileFolder,
    ImageFileList,
    ImageTifFolder,
    ImageTifList,
    load_uint8_tif,
)
from iivs.dhm.data.common.npy import read_npy_shape, write_npy
from iivs.dhm.data.common.sequence import FrameShapedMixin, SequentialFileFolder
from iivs.dhm.data.common.txt import (
    KoalaTxtHeaderCodec,
    parse_txt_grid,
    write_txt_grid,
)
from iivs.dhm.data.common.utils import numbered_name
from iivs.dhm.data.common.validation import (
    validate_float32_image,
    validate_uint8_image,
)
