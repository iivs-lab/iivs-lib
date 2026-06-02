from __future__ import annotations

__all__ = ("PhaseBinSequence",)

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from kaparoo.data.sequences import FileFolderSequence
from kaparoo.filesystem.search import search_files
from kaparoo.filesystem.search.filters import Regex
from kaparoo.utils import replace_if_none
from natsort import natsorted
from numpy.typing import NDArray

from iivs.dhm.koala.phase.file import convert_phase_unit, load_bin, read_header

if TYPE_CHECKING:
    from typing import Literal

    from kaparoo.filesystem.types import StrPath

    from iivs.dhm.koala.phase.header import PhaseBinHeader, PhaseUnit


class PhaseBinSequence(FileFolderSequence[NDArray[np.float32], Path]):
    """An ordered sequence of Lyncée Tec Koala `.bin` phase images in a folder.

    Lists the direct children matching `{index:05d}_phase.bin` (exactly five
    digits, case-sensitive), in index order. All images are assumed to share
    one acquisition `header`, read once from the first file; that header drives
    the optional unit conversion and `validate`. Each item is the decoded
    float32 image (optionally converted to `target_unit`) and its metadata is
    the source path.

    Args:
        root: The folder to scan. Must exist, be a directory, and contain at
            least one matching file.
        target_unit: Unit to return loaded images in. When it differs from the
            stored unit, images are converted on load via the header's
            `height_scale`. Defaults to None, which keeps the stored unit.
        validate: Run `validate` to this level ("names", "headers", or
            "data") at construction, or None to skip. Defaults to "headers".

    Raises:
        DirectoryNotFoundError: If `root` does not exist.
        NotADirectoryError: If `root` exists but is not a directory.
        FileNotFoundError: If no `NNNNN_phase.bin` files are found in `root`.
        ValueError: If `validate` is set and the sequence fails validation.
    """

    def __init__(
        self,
        root: StrPath,
        *,
        target_unit: PhaseUnit | None = None,
        validate: Literal["names", "headers", "data"] | None = "headers",
    ) -> None:
        super().__init__(root)  # list_files rejects an empty folder

        self._header = read_header(self.get_file(0))
        self._target_unit = replace_if_none(target_unit, self._header.unit)

        if validate is not None:
            self.validate(level=validate)

    @property
    def header(self) -> PhaseBinHeader:
        """The shared acquisition header, read from the first file."""
        return self._header

    @property
    def target_unit(self) -> PhaseUnit:
        """The unit that loaded images are returned in."""
        return self._target_unit

    def validate(
        self, *, level: Literal["names", "headers", "data"] = "headers"
    ) -> None:
        """Validate the sequence to the given `level`.

        Args:
            level: How deep to validate (cumulative): "names" checks only
                that files are numbered contiguously from 0 (index `i` is
                `{i:05d}_phase.bin`); "headers" (default) also that every
                file shares the first file's header; "data" also that every
                image decodes without error (expensive -- it reads every
                pixel).

        Raises:
            ValueError: If the numbering has gaps, a header differs from the
                first, or (at "data") an image fails to load.
        """
        for index in range(len(self)):
            self.validate_file(index, level=level)

    def validate_file(
        self, index: int, *, level: Literal["names", "headers", "data"] = "headers"
    ) -> None:
        """Validate the file at `index` to `level` (see `validate`)."""
        path = self.get_file(index)
        expected = f"{index:05d}_phase.bin"
        if path.name != expected:
            msg = f"non-contiguous numbering: expected {expected} at index {index}, got {path.name}"
            raise ValueError(msg)
        # The first file is the reference header, so it is never compared.
        if level != "names" and index != 0 and read_header(path) != self.header:
            msg = f"header of {path.name} differs from the first file"
            raise ValueError(msg)
        if level == "data":
            load_bin(path, on_nonfinite="raise")

    def list_files(self, root: Path) -> list[Path]:
        files = search_files(root, name_filter=Regex(r"\d{5}_phase\.bin"), max_depth=1)
        if not files:
            msg = f"no NNNNN_phase.bin files found in {root}"
            raise FileNotFoundError(msg)
        return natsorted(files)

    def get_meta(self, index: int) -> Path:
        return self.get_file(index)

    def load_file(self, path: Path) -> NDArray[np.float32]:
        return convert_phase_unit(
            load_bin(path),
            source=self._header.unit,
            target=self._target_unit,
            height_scale=self._header.height_scale,
        )
