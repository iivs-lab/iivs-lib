from __future__ import annotations

__all__ = (
    "PhaseTxtFolder",
    "PhaseTxtList",
    "load_phase_txt",
    "read_phase_txt_header",
)

import re
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, overload, override

import numpy as np
from kaparoo.data.sequences import FileListSequence
from kaparoo.filesystem import ensure_file_exists
from kaparoo.utils import replace_if_none
from numpy.typing import NDArray

from iivs.dhm.data.common import (
    FrameShapedMixin,
    SequentialFileFolder,
    parse_txt_grid,
    validate_float32_image,
)
from iivs.dhm.data.phase.base import PhaseSequence
from iivs.dhm.data.phase.bin import PhaseBinHeader
from iivs.dhm.data.phase.core import PhaseUnit, convert_phase_unit

if TYPE_CHECKING:
    from typing import Literal

    from kaparoo.filesystem.types import StrPath, StrPaths


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


class PhaseTxtList(FileListSequence[NDArray[np.float32], Path], PhaseSequence[Path]):
    """A phase sequence over an explicit, arbitrary list of `Float/Txt` files.

    The text twin of `PhaseBinList`: no naming/contiguity/shared-header
    constraint; each file is read independently with per-file unit conversion.
    `PhaseTxtFolder` is the auto-discovered, same-shape special case of this.

    Args:
        files: The `.txt` files to expose, in the given order.
        target_unit: Unit to return images in (None keeps each file's stored).
    """

    def __init__(
        self, files: StrPaths, *, target_unit: PhaseUnit | None = None
    ) -> None:
        super().__init__(files)
        self._target_unit = target_unit

    @property
    def target_unit(self) -> PhaseUnit | None:
        """The unit images are converted to on load, or None to keep each file's."""
        return self._target_unit

    @override
    def get_meta(self, index: int) -> Path:
        """Return the source path of the file at `index`."""
        return self.get_file(index)

    @override
    def load_file(self, path: Path) -> NDArray[np.float32]:
        """Load the image at `path`, converted to `target_unit` if one is set."""
        image, header = load_phase_txt(path, return_header=True)
        target = self._target_unit if self._target_unit is not None else header.unit
        return convert_phase_unit(
            image, source=header.unit, target=target, height_scale=header.height_scale
        )


class PhaseTxtFolder(
    SequentialFileFolder[NDArray[np.float32]],
    PhaseTxtList,
    FrameShapedMixin,
):
    """An ordered sequence of Koala `Float/Txt` phase images in a folder.

    The text twin of `PhaseBinFolder`, and the auto-discovered special case of
    `PhaseTxtList` (it inherits the `load_file`): lists `{index:05d}_phase.txt`,
    shares one acquisition `header` (read from the first file), and converts to
    `target_unit` on load.

    Args:
        root: The folder to scan.
        target_unit: Unit to return loaded images in (None keeps the stored).
        validate: Validation level at construction, or None to skip.
    """

    FILE_STEM: ClassVar[str] = "phase"
    FILE_EXT: ClassVar[str] = "txt"
    LEVELS: ClassVar[tuple[str, ...]] = ("names", "headers", "data")
    DEFAULT_LEVEL: ClassVar[str] = "headers"

    def __init__(
        self,
        root: StrPath,
        *,
        target_unit: PhaseUnit | None = None,
        validate: Literal["names", "headers", "data"] | None = "headers",
    ) -> None:
        super().__init__(root)

        self._header = read_phase_txt_header(self.get_file(0))
        self._target_unit = replace_if_none(target_unit, self._header.unit)

        convert_phase_unit(
            np.empty((0, 0), dtype=np.float32),
            source=self._header.unit,
            target=self._target_unit,
            height_scale=self._header.height_scale,
        )

        if validate is not None:
            self.validate(level=validate)

    @property
    def header(self) -> PhaseBinHeader:
        """The shared acquisition header, read from the first file."""
        return self._header

    @property
    @override
    def frame_shape(self) -> tuple[int, int]:
        """The (height, width) of each image, from the shared header."""
        return self._header.shape

    @override
    def _validate_content(self, path: Path, *, level: str) -> None:
        """Check `path`'s header matches the reference; at "data", decode too."""
        if read_phase_txt_header(path) != self.header:
            msg = f"header of {path.name} differs from the first file"
            raise ValueError(msg)

        if level == "data":
            load_phase_txt(path, on_nonfinite="raise")
