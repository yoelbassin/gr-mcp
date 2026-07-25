from __future__ import annotations

from functools import lru_cache
from typing import Literal

import numpy as np
from crc import Calculator, Configuration


def _bitorder(bit_order: str) -> Literal["little", "big"]:
    return "little" if bit_order == "lsb" else "big"


@lru_cache(maxsize=None)
def _calculator(
    poly: int, bits: int, init: int, reflected: bool, xorout: int
) -> Calculator:
    return Calculator(
        Configuration(
            width=bits,
            polynomial=poly,
            init_value=init,
            final_xor_value=xorout,
            reverse_input=reflected,
            reverse_output=reflected,
        ),
        optimized=True,  # table engine (~2.1 MB/s) not the bit-by-bit one (~0.30)
    )


def crc_value(
    data: bytes,
    *,
    poly: int,
    bits: int,
    init: int,
    reflected: bool = False,
    xorout: int = 0,
) -> int:
    return _calculator(poly, bits, init, reflected, xorout).checksum(bytes(data))


def _fold(
    body: bytes,
    *,
    poly: int,
    bits: int,
    init: int,
    reflected: bool,
    xorout: int,
    fold_tail: int,
) -> int:
    if fold_tail <= 0:
        return crc_value(
            body, poly=poly, bits=bits, init=init, reflected=reflected, xorout=xorout
        )
    val = crc_value(
        body[:-fold_tail],
        poly=poly,
        bits=bits,
        init=init,
        reflected=reflected,
        xorout=xorout,
    )
    return (val ^ int.from_bytes(body[-fold_tail:], "big")) & ((1 << bits) - 1)


def _crc_bits(
    bits: np.ndarray, *, poly: int, width: int, init: int, xorout: int
) -> int:
    mask = (1 << width) - 1
    top = 1 << (width - 1)
    reg = init & mask
    for b in bits:
        reg ^= (int(b) & 1) << (width - 1)
        reg = ((reg << 1) ^ poly) & mask if reg & top else (reg << 1) & mask
    return reg ^ xorout


def crc_check(
    payload: bytes,
    *,
    poly: int,
    bits: int,
    init: int = 0,
    reflected: bool = False,
    xorout: int = 0,
    bit_order: str = "msb",
    fold_tail: int = 0,
    checksum_le: bool = False,
) -> tuple[bool, bytes]:
    endian = "little" if checksum_le else _bitorder(bit_order)
    n = bits // 8
    if len(payload) < n:
        return False, payload
    body, checksum = payload[:-n], payload[-n:]
    rx = int.from_bytes(checksum, endian)
    ok = (
        _fold(
            body,
            poly=poly,
            bits=bits,
            init=init,
            reflected=reflected,
            xorout=xorout,
            fold_tail=fold_tail,
        )
        == rx
    )
    return ok, body


def crc_check_bits(
    fbits: np.ndarray, *, poly: int, width: int, init: int = 0, xorout: int = 0
) -> tuple[bool, np.ndarray]:
    if fbits.size < width:
        return False, fbits
    body_bits = fbits[: fbits.size - width]
    check_bits = fbits[fbits.size - width :]
    weights = 1 << np.arange(width - 1, -1, -1, dtype=np.int64)
    rx = int(check_bits.astype(np.int64).dot(weights))
    value = _crc_bits(body_bits, poly=poly, width=width, init=init, xorout=xorout)
    return value == rx, body_bits


def crc_append(
    body: bytes,
    *,
    poly: int,
    bits: int,
    init: int = 0,
    reflected: bool = False,
    xorout: int = 0,
    fold_tail: int = 0,
    checksum_le: bool = False,
) -> bytes:
    endian: Literal["little", "big"] = "little" if checksum_le else "big"
    n = bits // 8
    checksum = _fold(
        body,
        poly=poly,
        bits=bits,
        init=init,
        reflected=reflected,
        xorout=xorout,
        fold_tail=fold_tail,
    )
    return body + checksum.to_bytes(n, endian)
