from __future__ import annotations

from typing import Literal

import numpy as np


def _bitorder(bit_order: str) -> Literal["little", "big"]:
    return "little" if bit_order == "lsb" else "big"


def bytes_to_bits(data: bytes, bit_order: str = "msb") -> np.ndarray:
    return np.unpackbits(
        np.frombuffer(data, dtype=np.uint8), bitorder=_bitorder(bit_order)
    )


def bits_to_bytes(bits: np.ndarray, bit_order: str = "msb") -> bytes:
    return np.packbits(
        np.asarray(bits, dtype=np.uint8), bitorder=_bitorder(bit_order)
    ).tobytes()


def read_uint(bits: np.ndarray, start: int, width: int, bit_order: str = "msb") -> int:
    field = bits[start : start + width]
    if bit_order == "lsb":
        field = field[::-1]
    return int(field.dot(1 << np.arange(width - 1, -1, -1, dtype=np.int64)))


def translate_bytes(data: bytes, table: list[int]) -> bytes:
    return bytes(table[b] for b in data)
