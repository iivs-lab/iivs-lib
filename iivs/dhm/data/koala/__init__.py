"""Building blocks shared across the data modalities (phase, intensity, ...).

The cross-modality primitives the per-modality packages compose, split by concern into
submodules and re-exported here so ``from iivs.dhm.data.koala import X`` reaches any of
them:

- `bin` — the Koala `.bin` header value object and I/O (`KoalaBinHeader`,
  `load_bin`, `write_bin`).
- `txt` — the `Float/Txt` header codec and image I/O (`KoalaTxtHeaderCodec`,
  `load_txt`, `write_txt`).
- `image` — the uint8 `.tif` reader (`load_uint8_tif`, over `tifffile`) and the
  numbered folder / list concretes (`ImageFileFolder`, `ImageTifFolder` /
  `ImageTifList`) that bind uint8 on `iivs.common.data`'s generic `ArrayFileList`.
- `float` — the float32 file list/folder bases over a ``(read_header, decode)``
  codec, generic in the header type (`KoalaFloatFileList`,
  `KoalaFloatFileFolder`; shared by phase and intensity).
- `frame` — the numbered-folder base (`KoalaFrameFolder`) plus its naming
  / discovery helpers (`koala_frame_name`, `detect_koala_format`). The extension
  helpers `file_extension`, `ensure_file_extension` (with its `add=True` mode),
  and the `UnsupportedExtensionError` raised on a bad extension are imported
  directly from `kaparoo.filesystem` where needed, not re-exported here.
- `constants` — the fixed Koala layout vocabulary: the atomic names (`PHASE`,
  `INTENSITY`, `HOLOGRAMS`, `FLOAT`, `BIN`, `TXT`, `IMAGE`, `TIMESTAMPS`, `PHBOUNDS`)
  and the per-modality path combinations built from them (`PHASE_FLOAT_BIN`,
  `PHASE_IMAGE`, `INTENSITY_FLOAT_BIN`, ...).
- `layout` — the shared time-lapse machinery the per-modality `layout` modules build on:
  the tolerant folder opener (`open_folder`), the time-lapse subfolder walkers
  (`search_timelapse_subdirs`, `search_timelapse_subfolders`),
  the reconstruction group base (`ReconstructionGroup`), and the
  `<Modality>/{Float/{Bin,Txt}, Image}` spec builder (`reconstruction_tree`).

The technique-agnostic primitives live in `iivs.common.data`: the `.npy` reader / writer
(`read_npy_shape`, `write_npy`), the same-shape mixin `FrameShapedMixin`, the array
validators (`validate_float_array` / `validate_uint_array` and their float32 / uint8
bindings), and the dtype-generic array-list base (`ArrayFileList`); import them
from there.
"""

__all__ = (
    "BIN",
    "FLOAT",
    "FLOAT_FORMATS",
    "HOLOGRAMS",
    "IMAGE",
    "INTENSITY",
    "INTENSITY_FLOAT_BIN",
    "INTENSITY_FLOAT_TXT",
    "INTENSITY_IMAGE",
    "PHASE",
    "PHASE_FLOAT_BIN",
    "PHASE_FLOAT_TXT",
    "PHASE_IMAGE",
    "PHBOUNDS",
    "TIMESTAMPS",
    "TXT",
    "FloatFormat",
    "ImageFileFolder",
    "ImageTifFolder",
    "ImageTifList",
    "KoalaBinHeader",
    "KoalaFloatFileFolder",
    "KoalaFloatFileList",
    "KoalaFrameFolder",
    "KoalaTxtHeaderCodec",
    "ReconstructionGroup",
    "ValidationLevel",
    "detect_koala_format",
    "koala_frame_name",
    "load_bin",
    "load_txt",
    "load_uint8_tif",
    "open_folder",
    "reconstruction_tree",
    "search_timelapse_subdirs",
    "search_timelapse_subfolders",
    "write_bin",
    "write_txt",
)

from iivs.dhm.data.koala.bin import KoalaBinHeader, load_bin, write_bin
from iivs.dhm.data.koala.constants import (
    BIN,
    FLOAT,
    HOLOGRAMS,
    IMAGE,
    INTENSITY,
    INTENSITY_FLOAT_BIN,
    INTENSITY_FLOAT_TXT,
    INTENSITY_IMAGE,
    PHASE,
    PHASE_FLOAT_BIN,
    PHASE_FLOAT_TXT,
    PHASE_IMAGE,
    PHBOUNDS,
    TIMESTAMPS,
    TXT,
)
from iivs.dhm.data.koala.float import (
    FLOAT_FORMATS,
    FloatFormat,
    KoalaFloatFileFolder,
    KoalaFloatFileList,
)
from iivs.dhm.data.koala.frame import (
    KoalaFrameFolder,
    ValidationLevel,
    detect_koala_format,
    koala_frame_name,
)
from iivs.dhm.data.koala.image import (
    ImageFileFolder,
    ImageTifFolder,
    ImageTifList,
    load_uint8_tif,
)
from iivs.dhm.data.koala.layout import (
    ReconstructionGroup,
    open_folder,
    reconstruction_tree,
    search_timelapse_subdirs,
    search_timelapse_subfolders,
)
from iivs.dhm.data.koala.txt import KoalaTxtHeaderCodec, load_txt, write_txt
