from __future__ import annotations

__all__ = ("KoalaTxtHeaderCodec", "parse_txt_grid", "write_txt_grid")

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from kaparoo.filesystem import StagedFile, ensure_file_exists

from iivs.dhm.data.common.bin import KoalaBinHeader

if TYPE_CHECKING:
    from kaparoo.filesystem.types import StrPath
    from numpy.typing import NDArray


def parse_txt_grid(lines: list[str], *, shape: tuple[int, int]) -> NDArray[np.float32]:
    """Parse whitespace-separated floats into a float32 (H, W) array, layout-agnostic.

    Used by the Koala `Float/Txt` readers: a modality's text export is a small
    key=value header followed by the grid values. Koala may write those values
    as `height` rows of `width` floats *or* as a single long line, so the values
    are read in row-major order and reshaped to `shape` rather than relying on
    the line breaks. Any rectangular layout (including one line) is accepted;
    blank lines are ignored.

    Raises:
        ValueError: If a value is malformed, the lines are raggedly shaped, or
            the value count does not fill `shape`.
    """
    height, width = shape
    try:
        grid = np.loadtxt(lines, dtype=np.float32, ndmin=1)
    except ValueError as exc:
        msg = f"malformed txt grid: {exc}"
        raise ValueError(msg) from exc
    flat = np.ravel(grid)
    if flat.size != height * width:
        msg = f"txt grid must hold {height * width} values (got {flat.size})"
        raise ValueError(msg)
    return flat.reshape(height, width)


def write_txt_grid(
    path: StrPath,
    header: str,
    data: NDArray[np.float32],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically write a Koala `Float/Txt` file: the `header` text then the grid.

    `header` is the already-serialized key=value header (see
    `KoalaTxtHeaderCodec.to_lines`); `data` follows as ``%.8e`` rows. The
    writer twin of `parse_txt_grid`, shared by the per-modality `save_*_txt`.

    Raises:
        FileExistsError: If `path` exists and `overwrite` is False.
        FileNotFoundError: If the parent directory of `path` does not exist.
    """
    with StagedFile(path, binary=True, overwrite=overwrite) as staged:
        staged.write(header.encode("utf-8"))
        np.savetxt(staged.file, data, fmt="%.8e")


class KoalaTxtHeaderCodec[H: KoalaBinHeader](ABC):
    """Stateless (de)serializer between a Koala `Float/Txt` header and a `KoalaBinHeader`.

    Not a header value object -- it is never instantiated and carries no state;
    every method is a classmethod, and `from_lines` / `from_file` return the
    modality's `KoalaBinHeader` (`H`), not a `KoalaTxtHeaderCodec`. It is the
    text twin of the binary (de)serialization that lives on `KoalaBinHeader`
    itself (`to_dtype` / `from_dtype`), kept separate so the header value class
    stays free of text-format knowledge.

    The first two lines are always ``h=<H> w=<W>`` and ``pixel size=<m> m``; a
    modality may add more (phase carries a `data unit` and a `height conversion
    factor` line). A subclass sets `HEADER_LINES` / `MODALITY` and bridges those
    extra lines both ways -- `_from_geometry` parses them into `H`,
    `_extra_lines` serializes them back -- so `phase` and `intensity` share the
    line-count check, the `h/w` + `pixel size` regex, and the file read/write.

    Type Parameters:
        H: The header the subclass produces (e.g. `PhaseBinHeader`).
    """

    HEADER_LINES: ClassVar[int]
    MODALITY: ClassVar[str]
    _HW_RE: ClassVar[re.Pattern[str]] = re.compile(r"h=(\d+)\s+w=(\d+)")
    _PIXEL_SIZE_RE: ClassVar[re.Pattern[str]] = re.compile(r"pixel size=([0-9.eE+-]+)")

    @classmethod
    def from_lines(cls, lines: list[str], path: StrPath) -> H:
        """Parse the header from a file's lines (only the first `HEADER_LINES` matter).

        Accepts the whole file's lines -- any grid that follows is ignored here.

        Raises:
            ValueError: If there are too few lines, or the geometry is malformed.
        """
        if len(lines) < cls.HEADER_LINES:
            msg = f"{cls.MODALITY} txt header needs {cls.HEADER_LINES} lines (got {len(lines)}): {path}"
            raise ValueError(msg)

        hw = cls._HW_RE.search(lines[0])
        pixel_size = cls._PIXEL_SIZE_RE.search(lines[1])
        if hw is None or pixel_size is None:
            msg = f"malformed {cls.MODALITY} txt header: {path}"
            raise ValueError(msg)

        return cls._from_geometry(
            lines,
            height=int(hw[1]),
            width=int(hw[2]),
            pixel_size=float(pixel_size[1]),
            path=path,
        )

    @classmethod
    @abstractmethod
    def _from_geometry(
        cls,
        lines: list[str],
        *,
        height: int,
        width: int,
        pixel_size: float,
        path: StrPath,
    ) -> H:
        """Build the modality header from the shared geometry and any extra lines.

        `lines` carries the full header (at least `HEADER_LINES` long); read any
        modality-specific lines (e.g. phase's unit / height-conversion) here.
        """
        raise NotImplementedError

    @classmethod
    def from_file(cls, path: StrPath) -> H:
        """Open `path` and read just its header.

        Raises:
            FileNotFoundError: If `path` does not exist.
            NotAFileError: If `path` exists but is not a regular file.
            ValueError: If the header is missing or malformed.
        """
        path = ensure_file_exists(path)
        with path.open() as f:
            lines = [f.readline() for _ in range(cls.HEADER_LINES)]
        return cls.from_lines(lines, path)

    @classmethod
    def to_lines(cls, header: H) -> str:
        """Serialize `header` to its `Float/Txt` header text (inverse of `from_lines`).

        Emits the shared ``h/w`` + ``pixel size`` geometry, then any
        modality-specific lines from `_extra_lines`.
        """
        geometry = (
            f"h={header.height} w={header.width}\npixel size={header.pixel_size} m\n"
        )
        return geometry + cls._extra_lines(header)

    @classmethod
    def _extra_lines(cls, header: H) -> str:  # noqa: ARG003
        """Modality-specific header lines after the shared geometry (default: none)."""
        return ""
