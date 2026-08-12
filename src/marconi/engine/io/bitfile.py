from __future__ import annotations

from pathlib import Path
from typing import Literal, overload

import numpy as np
import numpy.typing as npt

from marconi.engine.types.enums import ItemType
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


def _guard_items(path: Path, item_type: ItemType) -> None:
    item_bytes = item_type.item_bytes
    n_items = path.stat().st_size // item_bytes
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
    _guard_items(path, ItemType.B)
    return np.fromfile(path, dtype=np.uint8)


def write_symbols(path: Path, symbols: npt.ArrayLike) -> None:
    np.asarray(symbols, dtype=np.int16).tofile(path)


@overload
def read_symbols(
    path: Path, item_type: Literal[ItemType.S] = ...
) -> npt.NDArray[np.int16]: ...


@overload
def read_symbols(
    path: Path, item_type: Literal[ItemType.F]
) -> npt.NDArray[np.float32]: ...


def read_symbols(
    path: Path, item_type: ItemType = ItemType.S
) -> npt.NDArray[np.int16] | npt.NDArray[np.float32]:
    if item_type is ItemType.F:
        return read_llrs(path)
    _guard_items(path, ItemType.S)
    return np.fromfile(path, dtype=np.int16)


def write_llrs(path: Path, llrs: npt.ArrayLike) -> None:
    np.asarray(llrs, dtype=np.float32).tofile(path)


def read_llrs(path: Path) -> npt.NDArray[np.float32]:
    _guard_items(path, ItemType.F)
    return np.fromfile(path, dtype=np.float32)


def harden(llrs: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    return (np.asarray(llrs, dtype=np.float32) < 0).astype(np.uint8)
