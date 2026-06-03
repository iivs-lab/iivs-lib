from __future__ import annotations

__all__ = (
    "PhaseTxtFolder",
    "PhaseTxtList",
    "load_phase_txt",
    "read_phase_txt_header",
)

import re
from typing import TYPE_CHECKING, ClassVar, overload, override

from kaparoo.filesystem import ensure_file_exists

from iivs.dhm.data.common import (
    parse_txt_grid,
    validate_float32_image,
)
from iivs.dhm.data.phase.base import PhaseFileFolder, PhaseFileList
from iivs.dhm.data.phase.bin import PhaseBinHeader
from iivs.dhm.data.phase.core import PhaseUnit

if TYPE_CHECKING:
    from typing import Literal

    import numpy as np
    from kaparoo.filesystem.types import StrPath
    from numpy.typing import NDArray


# A Koala `Float/Txt` phase export is a 4-line key=value header followed by
# `height` rows of `width` floats:
#     h=900 w=900
#     pixel size=2.84871e-07 m
#     data unit=rad
#     height conversion factor (-> m)=2.11994e-07
#     <row 0> ... <row height-1>
_HEADER_LINES = 4
_HW_RE = re.compile(r"h=(\d+)\s+w=(\d+)")
_PIXEL_SIZE_RE = re.compile(r"pixel size=([0-9.eE+-]+)")
_UNIT_RE = re.compile(r"data unit=(\S+)")
_HCONV_RE = re.compile(r"height conversion factor.*=([0-9.eE+-]+)")
_UNIT_BY_NAME = {
    "rad": PhaseUnit.RADIANS,
    "m": PhaseUnit.METERS,
    "none": PhaseUnit.UNKNOWN,
}


def _parse_header(lines: list[str], path: StrPath) -> PhaseBinHeader:
    """Parse the 4-line text header into a `PhaseBinHeader` (same fields as `.bin`)."""
    if len(lines) < _HEADER_LINES:
        msg = f"phase txt header needs {_HEADER_LINES} lines (got {len(lines)}): {path}"
        raise ValueError(msg)

    hw = _HW_RE.search(lines[0])
    pixel_size = _PIXEL_SIZE_RE.search(lines[1])
    hconv = _HCONV_RE.search(lines[3])
    if hw is None or pixel_size is None or hconv is None:
        msg = f"malformed phase txt header: {path}"
        raise ValueError(msg)

    unit_match = _UNIT_RE.search(lines[2])
    unit = (
        _UNIT_BY_NAME.get(unit_match[1].lower(), PhaseUnit.UNKNOWN)
        if unit_match
        else PhaseUnit.UNKNOWN
    )
    return PhaseBinHeader(
        width=int(hw[2]),
        height=int(hw[1]),
        pixel_size=float(pixel_size[1]),
        height_scale=float(hconv[1]),
        unit=unit,
    )


def read_phase_txt_header(path: StrPath) -> PhaseBinHeader:
    """Read only the header of a Koala `Float/Txt` phase file, without the grid.

    Returns the same `PhaseBinHeader` the `.bin` reader uses (width, height,
    pixel size, height scale, unit).

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
def load_phase_txt(
    path: StrPath,
    *,
    return_header: Literal[False] = False,
    on_nonfinite: Literal["ignore", "warn", "raise"] = ...,
) -> NDArray[np.float32]: ...


@overload
def load_phase_txt(
    path: StrPath,
    *,
    return_header: Literal[True],
    on_nonfinite: Literal["ignore", "warn", "raise"] = ...,
) -> tuple[NDArray[np.float32], PhaseBinHeader]: ...


@overload
def load_phase_txt(
    path: StrPath,
    *,
    return_header: bool,
    on_nonfinite: Literal["ignore", "warn", "raise"] = ...,
) -> NDArray[np.float32] | tuple[NDArray[np.float32], PhaseBinHeader]: ...


def load_phase_txt(
    path: StrPath,
    *,
    return_header: bool = False,
    on_nonfinite: Literal["ignore", "warn", "raise"] = "ignore",
) -> NDArray[np.float32] | tuple[NDArray[np.float32], PhaseBinHeader]:
    """Load a Koala `Float/Txt` phase image, and optionally its header.

    The text export holds the same quantitative phase as the `.bin`, so this
    returns a float32 array (and a `PhaseBinHeader`) just like `load_phase_bin`.

    Args:
        path: The `.txt` file to read.
        return_header: Whether to also return the parsed header.
        on_nonfinite: Forwarded to `validate_float32_image`.

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


class PhaseTxtList(PhaseFileList):
    """A phase sequence over an explicit, arbitrary list of `Float/Txt` files.

    The text twin of `PhaseBinList` (the `.txt` codec over `PhaseFileList`):
    no naming/contiguity/shared-header constraint; each file is read
    independently with per-file unit conversion. `PhaseTxtFolder` is the
    auto-discovered, same-shape special case of this.

    Args:
        files: The `.txt` files to expose, in the given order.
        target_unit: Unit to return images in (None keeps each file's stored).
    """

    @override
    def _read_header(self, path: StrPath) -> PhaseBinHeader:
        """Read the `Float/Txt` header."""
        return read_phase_txt_header(path)

    @override
    def _decode(
        self,
        path: StrPath,
        *,
        on_nonfinite: Literal["ignore", "warn", "raise"] = "ignore",
    ) -> tuple[NDArray[np.float32], PhaseBinHeader]:
        """Decode the `Float/Txt` image and its header."""
        return load_phase_txt(path, return_header=True, on_nonfinite=on_nonfinite)


class PhaseTxtFolder(PhaseFileFolder, PhaseTxtList):
    """An ordered sequence of Koala `Float/Txt` phase images in a folder.

    The text twin of `PhaseBinFolder`, and the auto-discovered, same-shape
    special case of `PhaseTxtList`: lists `{index:05d}_phase.txt`, sharing one
    acquisition `header`. Construction and validation are inherited; this
    supplies only the `.txt` extension.

    Args:
        root: The folder to scan.
        target_unit: Unit to return loaded images in (None keeps the stored).
        validate: Validation level at construction, or None to skip.
    """

    FILE_EXT: ClassVar[str] = "txt"
