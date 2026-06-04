from __future__ import annotations

__all__ = (
    "PhaseTxtFolder",
    "PhaseTxtList",
    "load_phase_txt",
    "read_phase_txt_header",
    "save_phase_txt",
)

import re
import warnings
from typing import TYPE_CHECKING, ClassVar, overload, override

from kaparoo.filesystem import ensure_file_exists

from iivs.dhm.data.common import (
    KoalaTxtHeaderCodec,
    ensure_file_extension,
    parse_txt_grid,
    validate_float32_image,
    write_txt_grid,
)
from iivs.dhm.data.phase.base import PhaseFileFolder, PhaseFileList
from iivs.dhm.data.phase.bin import PhaseBinHeader, _to_storable_unit
from iivs.dhm.data.phase.core import PhaseUnit, resolve_height_scale

if TYPE_CHECKING:
    from typing import Literal

    import numpy as np
    from kaparoo.filesystem.types import StrPath
    from numpy.typing import NDArray


class PhaseTxtHeaderCodec(KoalaTxtHeaderCodec[PhaseBinHeader]):
    """Reads a Koala `Float/Txt` phase header into a `PhaseBinHeader`.

    The 4-line header adds a `data unit` and a `height conversion factor` line
    to the shared `h/w` + `pixel size` pair::

        h=900 w=900
        pixel size=2.84871e-07 m
        data unit=rad
        height conversion factor (-> m)=2.11994e-07
    """

    HEADER_LINES: ClassVar[int] = 4
    MODALITY: ClassVar[str] = "phase"
    _UNIT_RE: ClassVar[re.Pattern[str]] = re.compile(r"data unit=(\S+)")
    _HCONV_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"height conversion factor.*=([0-9.eE+-]+)"
    )
    _UNIT_BY_NAME: ClassVar[dict[str, PhaseUnit]] = {
        "rad": PhaseUnit.RADIANS,
        "m": PhaseUnit.METERS,
        "none": PhaseUnit.UNKNOWN,
    }
    _NAME_BY_UNIT: ClassVar[dict[PhaseUnit, str]] = {
        PhaseUnit.RADIANS: "rad",
        PhaseUnit.METERS: "m",
        PhaseUnit.UNKNOWN: "none",
    }

    @classmethod
    @override
    def _from_geometry(
        cls,
        lines: list[str],
        *,
        height: int,
        width: int,
        pixel_size: float,
        path: StrPath,
    ) -> PhaseBinHeader:
        """Add phase's unit and height-conversion lines to the shared geometry."""
        hconv = cls._HCONV_RE.search(lines[3])
        if hconv is None:
            msg = f"malformed {cls.MODALITY} txt header: {path}"
            raise ValueError(msg)

        unit_match = cls._UNIT_RE.search(lines[2])
        unit = (
            cls._UNIT_BY_NAME.get(unit_match[1].lower(), PhaseUnit.UNKNOWN)
            if unit_match
            else PhaseUnit.UNKNOWN
        )
        return PhaseBinHeader(
            width=width,
            height=height,
            pixel_size=pixel_size,
            height_scale=float(hconv[1]),
            unit=unit,
        )

    @classmethod
    @override
    def _extra_lines(cls, header: PhaseBinHeader) -> str:
        """Serialize phase's `data unit` and `height conversion factor` lines."""
        return (
            f"data unit={cls._NAME_BY_UNIT[header.unit]}\n"
            f"height conversion factor (-> m)={header.height_scale}\n"
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
    return PhaseTxtHeaderCodec.from_file(path)


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
    header = PhaseTxtHeaderCodec.from_lines(lines, path)
    data = parse_txt_grid(lines[PhaseTxtHeaderCodec.HEADER_LINES :], shape=header.shape)
    data = validate_float32_image(data, on_nonfinite=on_nonfinite)
    return (data, header) if return_header else data


# ========================== #
#          Writing           #
# ========================== #


@overload
def save_phase_txt(
    path: StrPath,
    data: NDArray[np.float32],
    *,
    pixel_size: float,
    height_scale: float,
    unit: PhaseUnit = ...,
    overwrite: bool = ...,
    on_nonfinite: Literal["ignore", "warn", "raise"] = ...,
) -> None: ...


@overload
def save_phase_txt(
    path: StrPath,
    data: NDArray[np.float32],
    *,
    pixel_size: float,
    wavelength: float,
    refractive_delta: float,
    unit: PhaseUnit = ...,
    overwrite: bool = ...,
    on_nonfinite: Literal["ignore", "warn", "raise"] = ...,
) -> None: ...


def save_phase_txt(
    path: StrPath,
    data: NDArray[np.float32],
    *,
    pixel_size: float,
    height_scale: float | None = None,
    wavelength: float | None = None,
    refractive_delta: float | None = None,
    unit: PhaseUnit = PhaseUnit.RADIANS,
    overwrite: bool = False,
    on_nonfinite: Literal["ignore", "warn", "raise"] = "warn",
) -> None:
    """Save a 2D float32 phase image as a Koala `Float/Txt` file.

    The text twin of `save_phase_bin`: a 4-line ``h/w`` + ``pixel size`` +
    ``data unit`` + ``height conversion factor`` header, then the float grid.
    The phase-to-height scale is given as `height_scale`, or as `wavelength` +
    `refractive_delta` (exactly one form). Written atomically.

    Args:
        path: The `.txt` file to write.
        data: The phase image to save, of shape (H, W).
        pixel_size: Physical size of one (square) pixel, in m.
        height_scale: Height represented by one rad of phase, in m. Mutually
            exclusive with `wavelength` / `refractive_delta`.
        wavelength: Illumination wavelength, in m. Requires `refractive_delta`.
        refractive_delta: Refractive-index difference. Requires `wavelength`.
        unit: Physical unit of `data`. Defaults to RADIANS. NANOMETERS is stored
            as METERS (the file cannot store it); UNKNOWN is stored but warns.
        overwrite: Whether to replace `path` if it already exists. Defaults to
            False.
        on_nonfinite: How to handle non-finite values, forwarded to
            `validate_float32_image`: "ignore" accepts silently, "warn"
            (default) emits a RuntimeWarning, "raise" rejects with a ValueError.

    Raises:
        ValueError: If `path` has a non-`.txt` extension, neither or both scale
            forms are given, `data` is not a single 2D float32 image, or it holds
            non-finite values while `on_nonfinite` is "raise".
        FileExistsError: If `path` exists and `overwrite` is False.
        FileNotFoundError: If the parent directory of `path` does not exist.
    """
    path = ensure_file_extension(path, "txt", add=True)
    height_scale = resolve_height_scale(height_scale, wavelength, refractive_delta)
    data = validate_float32_image(data, on_nonfinite=on_nonfinite, allow_stack=False)
    data, unit = _to_storable_unit(data, unit, height_scale)

    if unit is PhaseUnit.UNKNOWN:
        msg = "saving with unit=UNKNOWN; physical interpretation is undefined"
        warnings.warn(msg, stacklevel=2)

    header = PhaseBinHeader(
        width=int(data.shape[1]),
        height=int(data.shape[0]),
        pixel_size=pixel_size,
        height_scale=height_scale,
        unit=unit,
    )
    write_txt_grid(
        path, PhaseTxtHeaderCodec.to_lines(header), data, overwrite=overwrite
    )


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

    FILE_EXT: ClassVar[str] = "txt"

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
