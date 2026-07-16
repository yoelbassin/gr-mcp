"""DMR BPTC(196,96) as caller data for the composition proof (Task 4). Every
constant here is protocol data living in tests/ — the generic vocabulary
(permute_rx + block_code_rx codewords + crc_rx) carries the FEC/CRC steps, so no
DMR-specific code enters src/. Tables verified vs OK-DMRlib + dsd-neo."""

from __future__ import annotations

import numpy as np

from marconi.bits.carriers import RxCarrier, _Frame
from marconi.bits.framing import bits_to_bytes, block_code_rx, crc_rx, permute_rx

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

_SIGN = {"0": 1, "1": 1, "2": -1, "3": -1}
SYNC_SIGNS = np.array([_SIGN[c] for c in BS_DATA_SYNC])


def _block_code(
    bits: np.ndarray, code_bits: int, data_bits: int, masks: list[int]
) -> np.ndarray:
    return block_code_rx(
        RxCarrier(bits=bits),
        code_bits=code_bits,
        data_bits=data_bits,
        parity_masks=masks,
        correct=True,
        emit="codeword",
    ).bits


def bptc_generic(info196: np.ndarray) -> np.ndarray:
    x = permute_rx(RxCarrier(bits=info196), perm=SCATTER_INV).bits
    x = x[1:196]  # drop the reserved bit0 -> 195 = 13x15 row-major matrix
    # Row [data0..10 | parity0..3] / column [data0..8 | parity0..3] stream order
    # already matches block_code_rx's LSB-first codeword basis, so no per-stride
    # reversal is needed (unlike the diagonal-interleaved CSS composition).
    for _ in range(2):
        x = _block_code(x, 15, 11, ROW_MASKS)
        x = permute_rx(RxCarrier(bits=x), perm=TRANSPOSE).bits
        x = _block_code(x, 13, 9, COL_MASKS)
        x = permute_rx(RxCarrier(bits=x), perm=TRANSPOSE_INV).bits
    return x[np.asarray(EXTRACT, np.int64)]


def _u(bits: np.ndarray, start: int, length: int) -> int:
    weights = 1 << np.arange(length - 1, -1, -1, dtype=np.int64)
    return int(bits[start : start + length].astype(np.int64).dot(weights))


def _crc_ok(payload: np.ndarray, xorout: int) -> bool:
    frame = _Frame(
        start=0, cursor=0, payload=bits_to_bytes(payload, "msb"), crc_ok=True
    )
    result = crc_rx(
        RxCarrier(bits=np.zeros(0, np.uint8), frames=[frame]),
        poly=0x1021,
        bits=16,
        init=0,
        reflected=False,
        xorout=xorout,
    )
    return bool(result.frames[0].crc_ok)


def _dibits_to_bits(dibits: np.ndarray) -> np.ndarray:
    pair = np.stack([(dibits >> 1) & 1, dibits & 1], axis=1)
    return pair.reshape(-1).astype(np.uint8)


def decode_burst_from_bits(burst264: np.ndarray) -> dict[str, int | str] | None:
    if burst264.size != 264:
        return None
    info = np.concatenate([burst264[0:98], burst264[166:264]])
    payload = bptc_generic(info)
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


def decode_burst(dibits132: np.ndarray) -> dict[str, int | str] | None:
    return decode_burst_from_bits(_dibits_to_bits(dibits132))
