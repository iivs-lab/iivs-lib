from __future__ import annotations

__all__ = ("KoalaFloatFileFolder", "KoalaFloatFileList")

from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, override

import numpy as np
from kaparoo.data.sequences import FileListSequence
from numpy.typing import NDArray

from iivs.dhm.data.common.bin import KoalaBinHeader
from iivs.dhm.data.common.extension import ensure_file_extension
from iivs.dhm.data.common.sequence import SequentialFileFolder

if TYPE_CHECKING:
    from typing import Literal

    from kaparoo.filesystem.types import StrPath, StrPaths


class KoalaFloatFileList[H: KoalaBinHeader](
    FileListSequence[NDArray[np.float32], Path]
):
    """Format-agnostic float32 file list over a ``(read_header, decode)`` codec.

    The machinery shared by the phase and intensity `Float/{Bin,Txt}` (and
    header-less `.npy`) sources: the `.<FILE_EXT>` check, `get_meta` (= source
    path), the cheap header-only `get_header`, and `load_with_header` (image +
    header in one read). A concrete subclass supplies `FILE_EXT` and the
    `_read_header` / `_decode` codec, where `_decode` returns ``(image,
    header)``. `load_file` returns the decoded image after `_postprocess`,
    which a subclass overrides to transform it (e.g. phase's unit conversion);
    the default is identity.

    Type Parameters:
        H: The header the codec produces (e.g. `PhaseBinHeader`).
    """

    FILE_EXT: ClassVar[str]

    def __init__(self, files: StrPaths) -> None:
        super().__init__([ensure_file_extension(f, self.FILE_EXT) for f in files])

    @override
    def get_meta(self, index: int) -> Path:
        """Return the source path of the file at `index`."""
        return self.get_file(index)

    def get_header(self, index: int) -> H:
        """Read just the header of the file at `index`, without the pixels.

        A header accessor named to sit beside `get_item` / `get_meta` (not part
        of kaparoo's protocol). Use `load_with_header` to get the image and its
        header together in one read.
        """
        return self._read_header(self.get_file(index))

    def load_with_header(self, index: int) -> tuple[NDArray[np.float32], H]:
        """Decode the image at `index` and its header in a single read.

        The image is post-processed exactly as `get_item` (e.g. phase's unit
        conversion); the header is the file's own. The single-read twin of a
        `get_item` + `get_header` pair, for callers (e.g. the converters) that
        need both.
        """
        image, header = self._decode(self.get_file(index))
        return self._postprocess(image, header), header

    @override
    def load_file(self, path: Path) -> NDArray[np.float32]:
        """Decode and post-process the image at `path`."""
        image, header = self._decode(path)
        return self._postprocess(image, header)

    def _postprocess(
        self,
        image: NDArray[np.float32],
        header: H,  # noqa: ARG002  # the identity default ignores it; phase uses it
    ) -> NDArray[np.float32]:
        """Transform a freshly decoded image (default: identity).

        The hook `load_file` / `load_with_header` apply on top of `_decode`;
        phase overrides it for per-file unit conversion, intensity keeps the
        identity default.
        """
        return image

    @abstractmethod
    def _read_header(self, path: StrPath) -> H:
        """Read the format's header (subclass codec)."""
        raise NotImplementedError

    @abstractmethod
    def _decode(
        self,
        path: StrPath,
        *,
        on_nonfinite: Literal["ignore", "warn", "raise"] = "ignore",
    ) -> tuple[NDArray[np.float32], H]:
        """Decode the format's image and its header (subclass codec)."""
        raise NotImplementedError


class KoalaFloatFileFolder[H: KoalaBinHeader](
    SequentialFileFolder[NDArray[np.float32]], KoalaFloatFileList[H]
):
    """Format-agnostic float32 folder: numbered discovery + one shared header.

    The auto-discovered, same-shape specialization of `KoalaFloatFileList`:
    reads one shared acquisition `header` from the first file, exposes
    `frame_shape` from it, and checks every other file's header against it.
    Concrete folders supply `FILE_STEM` (and inherit `FILE_EXT` and the codec
    from the matching `*List`). A subclass needing extra construction state
    (e.g. phase's `target_unit`) overrides `_after_header`.
    """

    LEVELS: ClassVar[tuple[str, ...]] = ("names", "headers", "data")
    DEFAULT_LEVEL: ClassVar[str] = "headers"

    def __init__(
        self,
        root: StrPath,
        *,
        validate: Literal["names", "headers", "data"] | None = "headers",
    ) -> None:
        super().__init__(root)  # discovers the files; rejects an empty folder
        self._header = self._read_header(self.get_file(0))
        self._after_header()
        if validate is not None:
            self.validate(level=validate)

    def _after_header(self) -> None:
        """Hook run once the shared `header` is read, before validation.

        Default no-op; a subclass (e.g. phase) resolves extra construction
        state here, after `self._header` is available.
        """

    @property
    def header(self) -> H:
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
        if self._read_header(path) != self.header:
            msg = f"header of {path.name} differs from the first file"
            raise ValueError(msg)

        if level == "data":
            self._decode(path, on_nonfinite="raise")
