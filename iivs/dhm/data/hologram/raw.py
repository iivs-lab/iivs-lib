from __future__ import annotations

__all__ = (
    "HologramRawFile",
    "HologramRawHeader",
    "read_hologram_raw_header",
    "save_hologram_raw",
)

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, cast, override

import numpy as np
from kaparoo.data.sequences.templates import SingleFileSequence
from kaparoo.filesystem import StagedFile, ensure_file_exists
from numpy.typing import NDArray

from iivs.dhm.data.common import (
    FrameShapedMixin,
    ensure_file_extension,
    validate_uint8_image,
    with_file_extension,
)
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
            msg = f"bit_depth must be one of {self.SUPPORTED_BIT_DEPTHS} (got {self.bit_depth})"
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
    def from_stream(cls, f: IO[bytes]) -> Self:
        """Read and validate a header from an open binary stream.

        Raises:
            ValueError: If the stream is too small for a header, or holds
                invalid header fields.
        """
        raw = f.read(cls.HEADER_SIZE)
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
        with path.open("rb") as f:
            return cls.from_stream(f)

    def to_dtype(self) -> NDArray[np.void]:
        """Serialize to a 1-element `HologramRawHeader.DTYPE` record array."""
        record = np.zeros(1, dtype=self.DTYPE)
        record["width"] = self.width
        record["height"] = self.height
        record["bit_depth"] = self.bit_depth
        record["frame_count"] = self.frame_count
        return record


def read_hologram_raw_header(path: StrPath) -> HologramRawHeader:
    """Read only the header of a Lyncée Tec Koala hologram `.raw` file, without the pixels."""
    return HologramRawHeader.from_file(path)


def save_hologram_raw(
    path: StrPath,
    frames: NDArray[np.uint8] | HologramSequence[object],
    *,
    overwrite: bool = False,
) -> None:
    """Save uint8 holograms as a Lyncée Tec Koala `.raw` file.

    The file is a 16-byte `HologramRawHeader` (8-bit) followed by the row-major
    frames. Frames are written one at a time, so a large source (a memmapped
    `HologramRawFile` or a big folder) is never held in memory as a whole stack.
    Written atomically.

    Args:
        path: The `.raw` file to write.
        frames: The holograms to save -- a single 2D image, an ``(N, H, W)``
            stack array, or a `HologramSequence`. All frames must share one
            shape.
        overwrite: Whether to replace `path` if it already exists. Defaults to
            False.

    Raises:
        ValueError: If `path` has a non-`.raw` extension, an array is not a 2D
            image or an ``(N, H, W)`` stack, the sequence or stack is empty, or
            its frames are not same-shaped uint8.
        FileExistsError: If `path` exists and `overwrite` is False.
        FileNotFoundError: If the parent directory of `path` does not exist.
    """
    path = with_file_extension(path, "raw")

    if isinstance(frames, HologramSequence):
        count = len(frames)
        if count == 0:
            msg = "cannot save an empty hologram sequence to .raw"
            raise ValueError(msg)

        shape = validate_uint8_image(frames[0], allow_stack=False).shape
        height, width = shape[0], shape[1]
        frame_iter = frames
    else:
        stack = validate_uint8_image(frames, allow_stack=True)
        if stack.ndim == 2:
            stack = stack[np.newaxis]
        if stack.ndim != 3:
            msg = f"frames array must be a 2D image or an (N, H, W) stack (got {stack.ndim}D)"
            raise ValueError(msg)

        count, height, width = stack.shape[0], stack.shape[1], stack.shape[2]
        if count == 0:
            msg = "cannot save an empty hologram stack to .raw"
            raise ValueError(msg)
        frame_iter = stack

    header = HologramRawHeader(
        width=width, height=height, bit_depth=8, frame_count=count
    )

    with StagedFile(path, binary=True, overwrite=overwrite) as staged:
        header.to_dtype().tofile(staged.file)

        for frame in frame_iter:
            frame = validate_uint8_image(frame, allow_stack=False)
            if frame.shape != (height, width):
                msg = (
                    f"all frames must have shape {(height, width)} (got {frame.shape})"
                )
                raise ValueError(msg)

            np.ascontiguousarray(frame, dtype=np.uint8).tofile(staged.file)


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
        ValueError: If `path` does not have a `.raw` extension, the header is
            invalid, or the file size does not match the header
            (`HEADER_SIZE + frame_count * frame_nbytes`).
    """

    def __init__(self, path: StrPath) -> None:
        ensure_file_extension(path, "raw")
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

    def __reduce__(self) -> tuple[type[HologramRawFile], tuple[StrPath]]:
        # Pickle only the source path; the memmap re-opens per process on load
        # instead of copying every frame into the pickle (multiprocessing-safe).
        return (type(self), (self.path,))
