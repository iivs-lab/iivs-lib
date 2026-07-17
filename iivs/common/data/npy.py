from __future__ import annotations

__all__ = (
    "load_float32_npy",
    "load_uint8_npy",
    "read_npy_shape",
    "save_float32_npy",
    "save_uint8_npy",
    "write_npy",
)

from typing import TYPE_CHECKING

import numpy as np
from kaparoo.filesystem import StagedFile, ensure_file_exists, ensure_file_extension

from iivs.common.data.validation import validate_float32_array, validate_uint8_array

if TYPE_CHECKING:
    from typing import Any

    from kaparoo.filesystem.types import StrPath
    from numpy.typing import NDArray

    from iivs.common.data.validation import OnNonFinite


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


def load_float32_npy(
    path: StrPath, *, on_nonfinite: OnNonFinite = "ignore"
) -> NDArray[np.float32]:
    """Load a header-less `.npy` file as a single 2D float32 image.

    Pickle is disabled, so an object array is refused rather than unpickled.

    Args:
        path: The `.npy` file to read.
        on_nonfinite: What to do about NaN / inf in the loaded array: `"ignore"`
            (default), `"warn"`, or `"raise"`.

    Raises:
        FileNotFoundError: If `path` does not exist.
        NotAFileError: If `path` exists but is not a regular file.
        ValueError: If the array is pickled, is not a 2D float32 image, or holds
            non-finite values while `on_nonfinite` is `"raise"`.
    """
    path = ensure_file_exists(path)
    data = np.load(path, allow_pickle=False)
    return validate_float32_array(data, on_nonfinite=on_nonfinite, allow_stack=False)


def save_float32_npy(
    path: StrPath,
    data: NDArray[np.float32],
    *,
    overwrite: bool = False,
    on_nonfinite: OnNonFinite = "warn",
) -> None:
    """Atomically save a 2D float32 image as an uncompressed `.npy` file.

    Args:
        path: The `.npy` file to write; the extension is added when absent.
        data: The image to save, of shape (H, W).
        overwrite: Whether to replace `path` if it already exists. Defaults to False.
        on_nonfinite: What to do about NaN / inf in `data`: `"ignore"`, `"warn"`
            (default), or `"raise"`.

    Raises:
        FileExistsError: If `path` exists and `overwrite` is False.
        ValueError: If `path` has a non-`.npy` extension, `data` is not a 2D float32
            image, or it holds non-finite values while `on_nonfinite` is `"raise"`.
    """
    path = ensure_file_extension(path, "npy", add=True)
    data = validate_float32_array(data, on_nonfinite=on_nonfinite, allow_stack=False)
    write_npy(path, data, overwrite=overwrite)


def load_uint8_npy(path: StrPath) -> NDArray[np.uint8]:
    """Load a header-less `.npy` file as a single 2D uint8 image.

    Pickle is disabled, so an object array is refused rather than unpickled.

    Raises:
        FileNotFoundError: If `path` does not exist.
        NotAFileError: If `path` exists but is not a regular file.
        ValueError: If the array is pickled or is not a 2D uint8 image.
    """
    path = ensure_file_exists(path)
    return validate_uint8_array(np.load(path, allow_pickle=False), allow_stack=False)


def save_uint8_npy(
    path: StrPath, data: NDArray[np.uint8], *, overwrite: bool = False
) -> None:
    """Atomically save a 2D uint8 image as an uncompressed `.npy` file.

    Args:
        path: The `.npy` file to write; the extension is added when absent.
        data: The image to save, of shape (H, W).
        overwrite: Whether to replace `path` if it already exists. Defaults to False.

    Raises:
        FileExistsError: If `path` exists and `overwrite` is False.
        ValueError: If `path` has a non-`.npy` extension, or `data` is not a 2D uint8
            image.
    """
    path = ensure_file_extension(path, "npy", add=True)
    data = validate_uint8_array(data, allow_stack=False)
    write_npy(path, data, overwrite=overwrite)
