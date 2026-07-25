from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from marconi.errors import register_error

# The bits layer holds the whole capture unpacked (1 byte/bit) and every
# BITS-level op allocates a fresh full-size array (~2-3x transient peak). Budget
# it: 2^30 bits = ~1 GiB unpacked, ~2-3 GiB peak. A 4 GB cf32 @ sps 4, QPSK
# capture is ~250M bits (well under); an hours-long multi-GB capture trips this
# with an actionable error instead of OOM-ing. Raise the budget or slice.
_MAX_BITS = 1 << 30


class CaptureTooLarge(Exception):
    pass


register_error(CaptureTooLarge, "invalid_argument")


def _guard(n_bits: int, path: Path) -> None:
    if n_bits > _MAX_BITS:
        raise CaptureTooLarge(
            f"{path.name}: {n_bits} bits (~{n_bits >> 20} MiB unpacked) exceeds the "
            f"bits-layer budget of {_MAX_BITS} bits (~{_MAX_BITS >> 20} MiB); the "
            f"layer holds the whole capture in memory (~2-3x transient). Decode a "
            f"bounded slice of the capture instead."
        )


def write_bits(path: Path, bits: np.ndarray) -> None:
    np.asarray(bits, dtype=np.uint8).tofile(path)


def read_bits(path: Path) -> np.ndarray:
    _guard(path.stat().st_size, path)
    return np.fromfile(path, dtype=np.uint8)


def write_symbols(path: Path, symbols: np.ndarray) -> None:
    np.asarray(symbols, dtype=np.int16).tofile(path)


def read_symbols(path: Path, item_type: Literal["s", "f"] = "s") -> np.ndarray:
    if item_type == "f":
        _guard(path.stat().st_size // 4, path)
        return np.fromfile(path, dtype=np.float32)
    _guard(path.stat().st_size // 2, path)
    return np.fromfile(path, dtype=np.int16)


def write_llrs(path: Path, llrs: np.ndarray) -> None:
    np.asarray(llrs, dtype=np.float32).tofile(path)


def read_llrs(path: Path) -> np.ndarray:
    _guard(path.stat().st_size // 4, path)
    return np.fromfile(path, dtype=np.float32)


def harden(llrs: np.ndarray) -> np.ndarray:
    return (np.asarray(llrs, dtype=np.float32) < 0).astype(np.uint8)
