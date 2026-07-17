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

    Reads the shape alone, never the data, so a header-less `.npy` folder can be
    validated by shape cheaply.

    Pickle is disabled, so an object array written elsewhere is refused rather than
    unpickled: inspecting an untrusted `.npy` cannot execute code.

    Args:
        path: The `.npy` file to inspect.
        expected: The required number of dimensions (positive). Defaults to 2.

    Raises:
        ValueError: If `expected` is not positive, the array does not have `expected`
            dimensions, or `path` holds an object array (pickle is disabled).
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

    A failed write leaves no partial file, and no existing one clobbered. `.npy` carries
    no header, so only the raw array is stored.

    Pickle is disabled, so an object array cannot be written at all: what lands on disk
    is always plain numeric, loadable by any reader without unpickling.

    Raises:
        FileExistsError: If `path` exists and `overwrite` is False.
        FileNotFoundError: If the parent directory of `path` does not exist.
        ValueError: If `data` is an object array (pickle is disabled).
    """
    with StagedFile(path, binary=True, overwrite=overwrite) as staged:
        np.save(staged.file, data, allow_pickle=False)
