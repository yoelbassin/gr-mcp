"""DMR BPTC(196,96) as caller data for the composition proof (Task 4). Every
constant here is protocol data living in tests/ — the generic vocabulary
(permute_rx + block_code_rx codewords + crc_rx) carries the FEC/CRC steps, so no
DMR-specific code enters src/. Tables verified vs OK-DMRlib + dsd-neo."""

from __future__ import annotations

from typing import cast

import numpy as np
from helpers.bitops import bits_to_bytes
from helpers.crc import crc_value

from marconi.engine.coding.carrier import CodingCarrier
from marconi.engine.coding.ops_bits import block_code_rx, permute_rx
from marconi.engine.models import ModemStep

# fmt: off
DEINTERLEAVE = [
    0, 13, 26, 39, 52, 65, 78, 91, 104, 117, 130, 143, 156, 169, 182, 195, 12, 25,
    38, 51, 64, 77, 90, 103, 116, 129, 142, 155, 168, 181, 194, 11, 24, 37, 50, 63,
    76, 89, 102, 115, 128, 141, 154, 167, 180, 193, 10, 23, 36, 49, 62, 75, 88, 101,
    114, 127, 140, 153, 166, 179, 192, 9, 22, 35, 48, 61, 74, 87, 100, 113, 126,
    139, 152, 165, 178, 191, 8, 21, 34, 47, 60, 73, 86, 99, 112, 125, 138, 151, 164,
    177, 190, 7, 20, 33, 46, 59, 72, 85, 98, 111, 124, 137, 150, 163, 176, 189, 6,
    19, 32, 45, 58, 71, 84, 97, 110, 123, 136, 149, 162, 175, 188, 5, 18, 31, 44,
    57, 70, 83, 96, 109, 122, 135, 148, 161, 174, 187, 4, 17, 30, 43, 56, 69, 82,
    95, 108, 121, 134, 147, 160, 173, 186, 3, 16, 29, 42, 55, 68, 81, 94, 107, 120,
    133, 146, 159, 172, 185, 2, 15, 28, 41, 54, 67, 80, 93, 106, 119, 132, 145, 158,
    171, 184, 1, 14, 27, 40, 53, 66, 79, 92, 105, 118, 131, 144, 157, 170, 183,
]
G_15_11 = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
]
G_13_9 = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
]
# fmt: on
BS_DATA_SYNC = "313333111331131131331131"
CSBK_XOROUT = 0xFFFF ^ 0xA5A5
HEADER_XOROUT = 0xFFFF ^ 0xCCCC

CSBK_CSBKO = (2, 6)
CSBK_TARGET = (32, 24)
CSBK_SOURCE = (56, 24)
HEADER_TARGET = (16, 24)
HEADER_SOURCE = (40, 24)


def _parity_masks_from_G(G: list[list[int]], k: int, cr: int) -> list[int]:
    return [sum(G[i][k + p] << i for i in range(k)) for p in range(cr)]


ROW_MASKS = _parity_masks_from_G(G_15_11, 11, 4)
COL_MASKS = _parity_masks_from_G(G_13_9, 9, 4)

SCATTER_INV = [DEINTERLEAVE.index(j) for j in range(196)]
TRANSPOSE = [(k % 13) * 15 + (k // 13) for k in range(195)]
TRANSPOSE_INV = [(k % 15) * 13 + (k // 15) for k in range(195)]
EXTRACT = list(range(3, 11)) + [r * 15 + c for r in range(1, 9) for c in range(11)]

# DMR's info bits wrap around the 48-bit SYNC/embedded-signalling field at the
# burst centre: a 264-bit burst's payload is bits [0,98) then [166,264). This
# non-contiguous gather is the interior splice the product expresses windowed.
INFO_SPLICE = list(range(0, 98)) + list(range(166, 264))

_SIGN = {"0": 1, "1": 1, "2": -1, "3": -1}
SYNC_SIGNS = np.array([_SIGN[c] for c in BS_DATA_SYNC])


def _block_code(
    bits: np.ndarray, code_bits: int, data_bits: int, masks: list[int]
) -> np.ndarray:
    return block_code_rx(
        CodingCarrier(bits=bits),
        code_bits=code_bits,
        data_bits=data_bits,
        parity_masks=masks,
        correct=True,
        emit="codeword",
    ).bits


def _perm_step(perm: list[int]) -> ModemStep:
    return ModemStep(
        conv="permute",
        params={"perm": cast("list[float | int]", [int(x) for x in perm])},
    )


def _block_step(code_bits: int, data_bits: int, masks: list[int]) -> ModemStep:
    return ModemStep(
        conv="block_code",
        params={
            "code_bits": code_bits,
            "data_bits": data_bits,
            "parity_masks": cast("list[float | int]", masks),
            "correct": True,
            "emit": "codeword",
        },
    )


def splice_step() -> ModemStep:
    """The interior splice as a windowed gather: info bits around the sync."""
    return _perm_step(INFO_SPLICE)


def bptc_steps() -> list[ModemStep]:
    """info196 -> 96-bit payload through the generic vocabulary: scatter
    deinterleave, drop the reserved bit 0, two row/column Hamming passes with a
    transpose between, then extract the message bits. Runs blind (one 196-bit
    block) or windowed (one per seeded burst) — the ops handle both scopes."""
    round_trip = [
        _block_step(15, 11, ROW_MASKS),
        _perm_step(TRANSPOSE),
        _block_step(13, 9, COL_MASKS),
        _perm_step(TRANSPOSE_INV),
    ]
    return [
        _perm_step(SCATTER_INV),
        _perm_step(list(range(1, 196))),  # drop reserved bit 0 -> 13x15 matrix
        *round_trip,
        *round_trip,
        _perm_step(EXTRACT),
    ]


def bptc_generic(info196: np.ndarray) -> np.ndarray:
    x = permute_rx(CodingCarrier(bits=info196), perm=SCATTER_INV).bits
    x = x[1:196]  # drop the reserved bit0 -> 195 = 13x15 row-major matrix
    # Row [data0..10 | parity0..3] / column [data0..8 | parity0..3] stream order
    # already matches block_code_rx's LSB-first codeword basis, so no per-stride
    # reversal is needed (unlike the diagonal-interleaved CSS composition).
    for _ in range(2):
        x = _block_code(x, 15, 11, ROW_MASKS)
        x = permute_rx(CodingCarrier(bits=x), perm=TRANSPOSE).bits
        x = _block_code(x, 13, 9, COL_MASKS)
        x = permute_rx(CodingCarrier(bits=x), perm=TRANSPOSE_INV).bits
    return x[np.asarray(EXTRACT, np.int64)]


def _u(bits: np.ndarray, start: int, length: int) -> int:
    weights = 1 << np.arange(length - 1, -1, -1, dtype=np.int64)
    return int(bits[start : start + length].astype(np.int64).dot(weights))


def _crc_ok(payload: np.ndarray, xorout: int) -> bool:
    data = bits_to_bytes(payload, "msb")
    body, checksum = data[:-2], data[-2:]
    rx = int.from_bytes(checksum, "big")
    return (
        crc_value(body, poly=0x1021, bits=16, init=0, reflected=False, xorout=xorout)
        == rx
    )


def _dibits_to_bits(dibits: np.ndarray) -> np.ndarray:
    pair = np.stack([(dibits >> 1) & 1, dibits & 1], axis=1)
    return pair.reshape(-1).astype(np.uint8)


def parse_payload(payload: np.ndarray) -> dict[str, int | str] | None:
    """A CRC-valid 96-bit BPTC payload -> its source/target IDs. Pure datasheet
    work over FEC-corrected bits: the CRC check and field offsets, not DSP."""
    if payload.size != 96:
        return None
    for kind, xorout in (("csbk", CSBK_XOROUT), ("data_header", HEADER_XOROUT)):
        if not _crc_ok(payload, xorout):
            continue
        if kind == "csbk":
            return {
                "kind": kind,
                "csbko": _u(payload, *CSBK_CSBKO),
                "target": _u(payload, *CSBK_TARGET),
                "source": _u(payload, *CSBK_SOURCE),
            }
        return {
            "kind": kind,
            "target": _u(payload, *HEADER_TARGET),
            "source": _u(payload, *HEADER_SOURCE),
        }
    return None
