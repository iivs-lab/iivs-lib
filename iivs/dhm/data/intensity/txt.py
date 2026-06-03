from __future__ import annotations

__all__ = (
    "IntensityTxtFolder",
    "IntensityTxtList",
    "load_intensity_txt",
    "read_intensity_txt_header",
)

import re
from typing import TYPE_CHECKING, ClassVar, overload, override

from kaparoo.filesystem import ensure_file_exists

from iivs.dhm.data.common import parse_txt_grid, validate_float32_image
from iivs.dhm.data.intensity.base import IntensityFileFolder, IntensityFileList
from iivs.dhm.data.intensity.bin import IntensityBinHeader

if TYPE_CHECKING:
    from typing import Literal

    import numpy as np
    from kaparoo.filesystem.types import StrPath
    from numpy.typing import NDArray


# A Koala `Float/Txt` intensity export is a 2-line key=value header (no unit or
# height-conversion line, unlike phase) followed by `height` rows of `width`
# floats:
#     h=900 w=900
#     pixel size=2.84871e-07 m
#     <row 0> ... <row height-1>
_HEADER_LINES = 2
_HW_RE = re.compile(r"h=(\d+)\s+w=(\d+)")
_PIXEL_SIZE_RE = re.compile(r"pixel size=([0-9.eE+-]+)")


def _parse_header(lines: list[str], path: StrPath) -> IntensityBinHeader:
    """Parse the 2-line text header into an `IntensityBinHeader`."""
    if len(lines) < _HEADER_LINES:
        msg = f"intensity txt header needs {_HEADER_LINES} lines (got {len(lines)}): {path}"
        raise ValueError(msg)

    hw = _HW_RE.search(lines[0])
    pixel_size = _PIXEL_SIZE_RE.search(lines[1])
    if hw is None or pixel_size is None:
        msg = f"malformed intensity txt header: {path}"
        raise ValueError(msg)

    return IntensityBinHeader(
        width=int(hw[2]), height=int(hw[1]), pixel_size=float(pixel_size[1])
    )


def read_intensity_txt_header(path: StrPath) -> IntensityBinHeader:
    """Read only the header of a Koala `Float/Txt` intensity file, without the grid.

    Raises:
        FileNotFoundError: If `path` does not exist.
        NotAFileError: If `path` exists but is not a regular file.
        ValueError: If the header is missing or malformed.
    """
    path = ensure_file_exists(path)
    with path.open() as fb:
        lines = [fb.readline() for _ in range(_HEADER_LINES)]
    return _parse_header(lines, path)


@overload
def load_intensity_txt(
    path: StrPath,
    *,
    return_header: Literal[False] = False,
    on_nonfinite: Literal["ignore", "warn", "raise"] = ...,
) -> NDArray[np.float32]: ...


@overload
def load_intensity_txt(
    path: StrPath,
    *,
    return_header: Literal[True],
    on_nonfinite: Literal["ignore", "warn", "raise"] = ...,
) -> tuple[NDArray[np.float32], IntensityBinHeader]: ...


@overload
def load_intensity_txt(
    path: StrPath,
    *,
    return_header: bool,
    on_nonfinite: Literal["ignore", "warn", "raise"] = ...,
) -> NDArray[np.float32] | tuple[NDArray[np.float32], IntensityBinHeader]: ...


def load_intensity_txt(
    path: StrPath,
    *,
    return_header: bool = False,
    on_nonfinite: Literal["ignore", "warn", "raise"] = "ignore",
) -> NDArray[np.float32] | tuple[NDArray[np.float32], IntensityBinHeader]:
    """Load a Koala `Float/Txt` intensity image, and optionally its header.

    The text export holds the same quantitative intensity as the `.bin`, so
    this returns a float32 array (and an `IntensityBinHeader`) like
    `load_intensity_bin`.

    Raises:
        FileNotFoundError: If `path` does not exist.
        NotAFileError: If `path` exists but is not a regular file.
        ValueError: If the header is malformed, the grid does not match it, or
            it holds non-finite values while `on_nonfinite` is "raise".
    """
    path = ensure_file_exists(path)
    lines = path.read_text().splitlines()
    header = _parse_header(lines, path)
    data = parse_txt_grid(lines[_HEADER_LINES:], shape=header.shape)
    data = validate_float32_image(data, on_nonfinite=on_nonfinite)
    return (data, header) if return_header else data


# ========================== #
#          Sequence          #
# ========================== #


class IntensityTxtList(IntensityFileList):
    """An intensity sequence over an explicit, arbitrary list of `Float/Txt` files.

    The text twin of `IntensityBinList` (the `.txt` codec over
    `IntensityFileList`): no naming/contiguity/shared-header constraint; each
    file is read independently. `IntensityTxtFolder` is the auto-discovered,
    same-shape special case of this.

    Args:
        files: The `.txt` files to expose, in the given order.
    """

    @override
    def _read_header(self, path: StrPath) -> IntensityBinHeader:
        """Read the `Float/Txt` header."""
        return read_intensity_txt_header(path)

    @override
    def _decode(
        self,
        path: StrPath,
        *,
        on_nonfinite: Literal["ignore", "warn", "raise"] = "ignore",
    ) -> NDArray[np.float32]:
        """Decode the `Float/Txt` image."""
        return load_intensity_txt(path, on_nonfinite=on_nonfinite)


class IntensityTxtFolder(IntensityFileFolder, IntensityTxtList):
    """An ordered sequence of Koala `Float/Txt` intensity images in a folder.

    The text twin of `IntensityBinFolder`, and the auto-discovered, same-shape
    special case of `IntensityTxtList`: lists `{index:05d}_intensity.txt`,
    sharing one acquisition `header`. Construction and validation are inherited;
    this supplies only the `.txt` extension.

    Args:
        root: The folder to scan.
        validate: Validation level at construction, or None to skip.
    """

    FILE_EXT: ClassVar[str] = "txt"
