from __future__ import annotations

__all__ = ("read_npy_shape", "write_npy")

from typing import TYPE_CHECKING

import numpy as np
from kaparoo.filesystem import StagedFile

if TYPE_CHECKING:
    from typing import Any

    from kaparoo.filesystem.types import StrPath
    from numpy.typing import NDArray


def read_npy_shape(path: StrPath, expected: int = 2) -> tuple[int, ...]:
    """Read an `.npy` array's shape without loading its data.

    Memory-maps the file to read just the shape from the `.npy` header, so a
    header-less `.npy` folder can be validated by shape cheaply. Pickled object
    arrays are rejected (`allow_pickle=False`).

    Args:
        path: The `.npy` file to inspect.
        expected: The required number of dimensions (positive). Defaults to 2.

    Raises:
        ValueError: If `expected` is not positive, or the array does not have
            `expected` dimensions.
    """
    if expected < 1:
        msg = f"expected must be positive (got {expected})"
        raise ValueError(msg)

    array: NDArray[Any] = np.load(path, mmap_mode="r", allow_pickle=False)
    shape: tuple[int, ...] = array.shape
    if len(shape) != expected:
        msg = f"expected {expected}D but got {len(shape)}D: {path}"
        raise ValueError(msg)
    return shape


def write_npy(path: StrPath, data: NDArray[Any], *, overwrite: bool = False) -> None:
    """Atomically write `data` as an uncompressed `.npy` file.

    Content is staged to a temp file in the destination's directory and moved
    into place on success. The shared writer behind the per-modality `.npy`
    savers; `.npy` carries no header, so only the raw array is stored. Object
    arrays are rejected (`allow_pickle=False`), matching the readers, so every
    file this writes is a clean numeric `.npy` they can load back.

    Raises:
        FileExistsError: If `path` exists and `overwrite` is False.
        FileNotFoundError: If the parent directory of `path` does not exist.
        ValueError: If `data` is an object array (pickle is disabled).
    """
    with StagedFile(path, binary=True, overwrite=overwrite) as staged:
        np.save(staged.file, data, allow_pickle=False)
