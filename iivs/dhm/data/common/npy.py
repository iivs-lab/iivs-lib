from __future__ import annotations

__all__ = ("read_npy_shape", "write_npy")

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from kaparoo.filesystem import StagedFile

if TYPE_CHECKING:
    from typing import Any

    from kaparoo.filesystem.types import StrPath
    from numpy.typing import NDArray


def read_npy_shape(path: StrPath) -> tuple[int, int]:
    """Read a 2D `.npy` array's (height, width) without loading its data.

    Memory-maps the file to read just the shape from the `.npy` header, so a
    header-less `.npy` folder can be validated by shape cheaply. Pickled object
    arrays are rejected (`allow_pickle=False`).

    Raises:
        ValueError: If the array is not 2-dimensional.
    """
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    shape = array.shape
    if len(shape) != 2:
        msg = f"{Path(path).name} must be a 2D array (got {len(shape)}D)"
        raise ValueError(msg)
    return (shape[0], shape[1])


def write_npy(path: StrPath, data: NDArray[Any], *, overwrite: bool = False) -> None:
    """Atomically write `data` as an uncompressed `.npy` file.

    Content is staged to a temp file in the destination's directory and moved
    into place on success. The shared writer behind the per-modality `.npy`
    savers; `.npy` carries no Koala header, so only the raw array is stored.

    Raises:
        FileExistsError: If `path` exists and `overwrite` is False.
        FileNotFoundError: If the parent directory of `path` does not exist.
    """
    with StagedFile(path, binary=True, overwrite=overwrite) as staged:
        np.save(staged.file, data)
