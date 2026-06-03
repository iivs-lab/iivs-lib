from __future__ import annotations

__all__ = (
    "HologramRawFile",
    "HologramRawHeader",
    "read_hologram_raw_header",
)

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, cast, override

import numpy as np
from kaparoo.data.sequences.templates import SingleFileSequence
from kaparoo.filesystem import ensure_file_exists
from numpy.typing import NDArray

from iivs.dhm.data.common import FrameShapedMixin
from iivs.dhm.data.hologram.base import HologramSequence

if TYPE_CHECKING:
    from typing import IO, Self

    from kaparoo.filesystem.types import StrPath


@dataclass(frozen=True, slots=True)
class HologramRawHeader:
    """The fixed-size header of a Lyncée Tec Koala hologram `.raw` file.

    Four little-endian int32 fields -- width, height, bit depth (bits per
    pixel), and frame count -- followed immediately by `frame_count`
    row-major frames of `height x width` pixels.

    Attributes:
        width: Frame width in pixels.
        height: Frame height in pixels.
        bit_depth: Bits per pixel. Only 8 (uint8) is currently supported.
        frame_count: Number of frames stored after the header.
    """

    # Packed (no alignment padding) -> exactly 16 bytes.
    DTYPE: ClassVar[np.dtype[np.void]] = cast(
        "np.dtype[np.void]",
        np.dtype(
            [
                ("width", "<i4"),
                ("height", "<i4"),
                ("bit_depth", "<i4"),
                ("frame_count", "<i4"),
            ],
        ),
    )
    HEADER_SIZE: ClassVar[int] = DTYPE.itemsize
    # Only 8-bit is verified against a real file; extend once a sample of
    # another depth is available.
    SUPPORTED_BIT_DEPTHS: ClassVar[tuple[int, ...]] = (8,)

    width: int
    height: int
    bit_depth: int
    frame_count: int

    def __post_init__(self) -> None:
        """Validate the fields."""
        if self.width <= 0 or self.height <= 0:
            msg = f"width and height must be positive (got {self.width}x{self.height})"
            raise ValueError(msg)

        if self.frame_count < 0:
            msg = f"frame_count must be non-negative (got {self.frame_count})"
            raise ValueError(msg)

        if self.bit_depth not in self.SUPPORTED_BIT_DEPTHS:
            allowed = list(self.SUPPORTED_BIT_DEPTHS)
            msg = f"bit_depth must be one of {allowed} (got {self.bit_depth})"
            raise ValueError(msg)

    @property
    def shape(self) -> tuple[int, int]:
        """Frame dimensions as (height, width)."""
        return (self.height, self.width)

    @property
    def pixel_count(self) -> int:
        """Pixels per frame (height * width)."""
        return self.height * self.width

    @property
    def pixel_dtype(self) -> np.dtype[np.uint8]:
        """On-disk pixel dtype for this `bit_depth`."""
        return cast("np.dtype[np.uint8]", np.dtype(np.uint8))

    @property
    def frame_nbytes(self) -> int:
        """Bytes per frame."""
        return self.pixel_count * self.pixel_dtype.itemsize

    @property
    def data_nbytes(self) -> int:
        """Total bytes of pixel data across all frames."""
        return self.frame_count * self.frame_nbytes

    @classmethod
    def from_stream(cls, fb: IO[bytes]) -> Self:
        """Read and validate a header from an open binary stream.

        Raises:
            ValueError: If the stream is too small for a header, or holds
                invalid header fields.
        """
        raw = fb.read(cls.HEADER_SIZE)
        if len(raw) < cls.HEADER_SIZE:
            msg = f"file must be at least {cls.HEADER_SIZE} bytes for a header (got {len(raw)})"
            raise ValueError(msg)

        record = np.frombuffer(raw, dtype=cls.DTYPE, count=1)[0]
        return cls(
            width=int(record["width"]),
            height=int(record["height"]),
            bit_depth=int(record["bit_depth"]),
            frame_count=int(record["frame_count"]),
        )

    @classmethod
    def from_file(cls, path: StrPath) -> Self:
        """Open `path` and read its header; a thin wrapper over `from_stream`.

        Raises:
            FileNotFoundError: If `path` does not exist.
            NotAFileError: If `path` exists but is not a regular file.
            ValueError: If the file is too small or holds invalid header fields.
        """
        path = ensure_file_exists(path)
        with path.open("rb") as fb:
            return cls.from_stream(fb)


def read_hologram_raw_header(path: StrPath) -> HologramRawHeader:
    """Read only the header of a Lyncée Tec Koala hologram `.raw` file, without the pixels."""
    return HologramRawHeader.from_file(path)


class HologramRawFile(
    SingleFileSequence[NDArray[np.uint8], int],
    HologramSequence[int],
    FrameShapedMixin,
):
    """An ordered sequence of holograms in a single Lyncée Tec Koala `.raw` file.

    The file is a `HologramRawHeader` followed by its frames, held internally
    as a lazy, read-only `np.memmap` so a large multi-frame file is never
    loaded whole. Each item is a fresh, writable copy of one frame (metadata
    is the frame index) -- matching `PhaseBinFolder` and
    `HologramTifFolder` -- so it can go straight to `torch.from_numpy`. For
    zero-copy bulk access, use the read-only `frames` memmap directly.

    The sequence pickles to just its path -- the memmap re-opens in each
    process rather than copying every frame -- so it is cheap to hand to
    worker processes (e.g. a PyTorch `DataLoader`).

    Args:
        path: The `.raw` file to read.

    Raises:
        FileNotFoundError: If `path` does not exist.
        NotAFileError: If `path` exists but is not a regular file.
        ValueError: If the header is invalid, or the file size does not match
            the header (`HEADER_SIZE + frame_count * frame_nbytes`).
    """

    def __init__(self, path: StrPath) -> None:
        super().__init__(path)

        self._header = HologramRawHeader.from_file(self.path)

        expected = HologramRawHeader.HEADER_SIZE + self._header.data_nbytes
        actual = self.path.stat().st_size
        if actual != expected:
            msg = f"file size must be {expected} bytes (got {actual})"
            raise ValueError(msg)

        self._frames: NDArray[np.uint8] = np.memmap(
            self.path,
            dtype=self._header.pixel_dtype,
            mode="r",
            offset=HologramRawHeader.HEADER_SIZE,
            shape=(self._header.frame_count, *self._header.shape),
        )

    @property
    def header(self) -> HologramRawHeader:
        """The file's header."""
        return self._header

    @property
    def frames(self) -> NDArray[np.uint8]:
        """All frames as a lazy, read-only `(frame_count, H, W)` memmap."""
        return self._frames

    @property
    @override
    def frame_shape(self) -> tuple[int, int]:
        """The (height, width) of each frame, from the header."""
        return self._header.shape

    @override
    def __len__(self) -> int:
        return self._header.frame_count

    @override
    def get_item(self, index: int) -> NDArray[np.uint8]:
        """Return a writable, owned copy of the frame at `index`.

        Use the read-only `frames` memmap directly for zero-copy access.
        """
        return self._frames[index].copy()

    @override
    def get_meta(self, index: int) -> int:
        return index

    @override
    def __reduce__(self) -> tuple[type[HologramRawFile], tuple[StrPath]]:
        # Pickle only the source path; the memmap re-opens per process on load
        # instead of copying every frame into the pickle (multiprocessing-safe).
        return (type(self), (self.path,))
