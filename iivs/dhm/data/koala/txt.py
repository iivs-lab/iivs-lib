from __future__ import annotations

__all__ = ("KoalaTxtHeaderCodec", "load_txt", "write_txt")

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from kaparoo.filesystem import StagedFile, ensure_file_exists

from iivs.common.data import validate_float32_array
from iivs.dhm.data.koala.bin import KoalaBinHeader

if TYPE_CHECKING:
    from kaparoo.filesystem.types import StrPath
    from numpy.typing import NDArray

    from iivs.common.data import OnNonFinite


class KoalaTxtHeaderCodec[H: KoalaBinHeader](ABC):
    """Stateless (de)serializer between a `Float/Txt` header and a `KoalaBinHeader`.

    Not a header value object; it is never instantiated and carries no state; every
    method is a classmethod, and `from_lines` / `from_file` return the modality's
    `KoalaBinHeader` (`H`), not a `KoalaTxtHeaderCodec`. It is the text twin of the
    binary (de)serialization that lives on `KoalaBinHeader` itself (`to_dtype` /
    `from_dtype`), kept separate so the header value class stays free of text-format
    knowledge.

    The first two lines are always ``h=<H> w=<W>`` and ``pixel size=<m> m``; a modality
    may add more (phase carries a `data unit` and a `height conversion factor` line). A
    subclass sets `HEADER_LINES` / `MODALITY` and bridges those extra lines both ways
    (`_from_geometry` parses them into `H`, `_extra_lines` serializes them back), so
    `phase` and `intensity` share the line-count check, the `h/w` + `pixel size` regex,
    and the file read/write.

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

        Accepts the whole file's lines; any grid that follows is ignored here.

        Raises:
            ValueError: If there are too few lines, or the geometry is malformed.
        """
        actual = len(lines)
        expected = cls.HEADER_LINES
        modality = cls.MODALITY

        if actual < expected:
            msg = f"{modality} txt header needs {expected} lines (got {actual}): {path}"
            raise ValueError(msg)

        hw = cls._HW_RE.search(lines[0])
        pixel_size = cls._PIXEL_SIZE_RE.search(lines[1])
        if hw is None or pixel_size is None:
            msg = f"malformed {modality} txt header: {path}"
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
            ValueError: If `path` is not `.txt`, or the header is missing or
                malformed.
        """
        path = ensure_file_exists(path, ext="txt")
        with path.open() as f:
            lines = [f.readline() for _ in range(cls.HEADER_LINES)]
        return cls.from_lines(lines, path)

    @classmethod
    def to_lines(cls, header: H) -> str:
        """Serialize `header` to its `Float/Txt` header text (inverse of `from_lines`).

        Emits the shared ``h/w`` + ``pixel size`` geometry, then any modality-specific
        lines from `_extra_lines`.
        """
        geometry = (
            f"h={header.height} w={header.width}\npixel size={header.pixel_size} m\n"
        )
        return geometry + cls._extra_lines(header)

    @classmethod
    def _extra_lines(cls, header: H) -> str:  # noqa: ARG003
        """Modality-specific header lines after the shared geometry (default: none)."""
        return ""


def load_txt[H: KoalaBinHeader](
    path: StrPath,
    codec: type[KoalaTxtHeaderCodec[H]],
    *,
    on_nonfinite: OnNonFinite = "ignore",
) -> tuple[NDArray[np.float32], H]:
    """Read a Koala `Float/Txt` file's float32 image and header (the shared engine).

    Reads the header via `codec`, then the float grid that follows. Koala may
    write the grid as `height` rows *or* as a single long line, so the read is
    layout-agnostic. The per-modality `load_*_txt` wrappers bind their codec and add the
    `return_header` ergonomics.

    Args:
        path: The `.txt` file to read.
        codec: The `KoalaTxtHeaderCodec` subclass to parse the header with.
        on_nonfinite: How to handle non-finite values (NaN, +inf, -inf) in the decoded
            data: "ignore" (default) accepts them silently, "warn" emits a
            RuntimeWarning, "raise" raises a ValueError.

    Returns:
        An ``(image, header)`` tuple: the float32 image of shape
        `header.shape` and the header the codec produced.

    Raises:
        FileNotFoundError: If `path` does not exist.
        NotAFileError: If `path` exists but is not a regular file.
        ValueError: If `path` is not `.txt`, the header or grid is malformed, the
            grid does not fill the header shape, or the data holds non-finite values
            while `on_nonfinite` is "raise".
    """
    path = ensure_file_exists(path, ext="txt")
    lines = path.read_text().splitlines()
    header = codec.from_lines(lines, path)

    try:
        grid = np.loadtxt(lines[codec.HEADER_LINES :], dtype=np.float32, ndmin=1)
    except ValueError as exc:
        msg = f"malformed txt grid: {exc}"
        raise ValueError(msg) from exc

    flat = np.ravel(grid)
    if flat.size != header.pixel_count:
        msg = f"txt grid must hold {header.pixel_count} values (got {flat.size})"
        raise ValueError(msg)

    data = flat.reshape(header.shape)
    return validate_float32_array(data, on_nonfinite=on_nonfinite), header


def write_txt[H: KoalaBinHeader](
    path: StrPath,
    codec: type[KoalaTxtHeaderCodec[H]],
    header: H,
    data: NDArray[np.float32],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically write a Koala `Float/Txt` file: `header`'s text then the grid.

    `codec` serializes `header` to its key=value lines; `data` follows as
    ``%.8e`` rows. The text twin of `write_bin`, shared by the per-modality
    `save_*_txt`.

    Raises:
        FileExistsError: If `path` exists and `overwrite` is False.
        FileNotFoundError: If the parent directory of `path` does not exist.
    """
    text = codec.to_lines(header)
    with StagedFile(path, binary=True, overwrite=overwrite) as staged:
        staged.write(text.encode("utf-8"))
        np.savetxt(staged.file, data, fmt="%.8e")
