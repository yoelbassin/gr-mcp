from __future__ import annotations

from pathlib import Path
from typing import Literal, overload

import numpy as np
import numpy.typing as npt

from marconi.errors import register_error

# The bits layer holds the whole capture unpacked (1 byte/bit) and each op
# allocates fresh full-size arrays. MEASURED peak over input size on a windowless
# whole-stream decode: unpack/pack ~1x, codebook ~3x, single-error block-code
# correction ~7x (the syndrome/match arrays). Budget it: 2^30 bits = ~1 GiB
# unpacked, so a blind correcting decode peaks ~7 GiB. A 4 GB cf32 @ sps 4, QPSK
# capture is ~250M bits (well under); an hours-long multi-GB capture trips this
# with an actionable error instead of OOM-ing. Raise the budget or slice.
_MAX_BITS = 1 << 30


class CaptureTooLarge(Exception):
    pass


register_error(CaptureTooLarge, "invalid_argument")


def _guard(n_items: int, path: Path, item_bytes: int = 1) -> None:
    if n_items > _MAX_BITS:
        raise CaptureTooLarge(
            f"{path.name}: {n_items} items (~{(n_items * item_bytes) >> 20} MiB in "
            f"memory) exceeds the bits-layer budget of {_MAX_BITS} items; the "
            f"layer holds the whole capture in memory (~2-3x transient). Decode a "
            f"bounded slice of the capture instead."
        )


def write_bits(path: Path, bits: npt.ArrayLike) -> None:
    np.asarray(bits, dtype=np.uint8).tofile(path)


def read_bits(path: Path) -> npt.NDArray[np.uint8]:
    _guard(path.stat().st_size, path)
    return np.fromfile(path, dtype=np.uint8)


def write_symbols(path: Path, symbols: npt.ArrayLike) -> None:
    np.asarray(symbols, dtype=np.int16).tofile(path)


@overload
def read_symbols(
    path: Path, item_type: Literal["s"] = ...
) -> npt.NDArray[np.int16]: ...


@overload
def read_symbols(path: Path, item_type: Literal["f"]) -> npt.NDArray[np.float32]: ...


def read_symbols(
    path: Path, item_type: Literal["s", "f"] = "s"
) -> npt.NDArray[np.int16] | npt.NDArray[np.float32]:
    if item_type == "f":
        _guard(path.stat().st_size // 4, path, item_bytes=4)
        return np.fromfile(path, dtype=np.float32)
    _guard(path.stat().st_size // 2, path, item_bytes=2)
    return np.fromfile(path, dtype=np.int16)


def write_llrs(path: Path, llrs: npt.ArrayLike) -> None:
    np.asarray(llrs, dtype=np.float32).tofile(path)


def read_llrs(path: Path) -> npt.NDArray[np.float32]:
    _guard(path.stat().st_size // 4, path)
    return np.fromfile(path, dtype=np.float32)


def harden(llrs: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    return (np.asarray(llrs, dtype=np.float32) < 0).astype(np.uint8)
