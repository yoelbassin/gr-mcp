from __future__ import annotations

import numpy as np

from marconi.bits.carriers import RxCarrier
from marconi.bits.framing import block_code_rx
from marconi.core.coding import correct_codeword

# Self-contained systematic codes (caller data, NOT DMR-specific — this task proves the
# generic emit mode; DMR's own codes are exercised in Task 4). masks are over the data
# bits, LSB-first — the basis block_code_rx documents.
_CODES = [
    (7, 4, [0b1011, 0b1101, 0b1110]),  # Hamming(7,4)
    (6, 3, [0b110, 0b101, 0b011]),  # a (6,3) parity code
]


def _roundtrip(code_bits: int, data_bits: int, masks: list[int]) -> None:
    flat = np.array(
        [(x >> i) & 1 for x in range(1 << code_bits) for i in range(code_bits)],
        np.uint8,
    )
    got = block_code_rx(
        RxCarrier(bits=flat),
        code_bits=code_bits,
        data_bits=data_bits,
        parity_masks=masks,
        correct=True,
        emit="codeword",
    ).bits.reshape(-1, code_bits)
    for x in range(1 << code_bits):
        expect = correct_codeword(x, masks, data_bits)
        got_int = sum(int(got[x][i]) << i for i in range(code_bits))
        assert got_int == expect, (code_bits, x, got_int, expect)


def test_codeword_mode_matches_correct_codeword() -> None:
    for code_bits, data_bits, masks in _CODES:
        _roundtrip(code_bits, data_bits, masks)


def test_data_mode_unchanged() -> None:
    flat = np.array([1, 0, 1, 1, 0, 1, 0], np.uint8)  # arbitrary 7-bit stride
    a = block_code_rx(
        RxCarrier(bits=flat),
        code_bits=7,
        data_bits=4,
        parity_masks=[0b1011, 0b1101, 0b1110],
        correct=True,
    ).bits
    b = block_code_rx(
        RxCarrier(bits=flat),
        code_bits=7,
        data_bits=4,
        parity_masks=[0b1011, 0b1101, 0b1110],
        correct=True,
        emit="data",
    ).bits
    assert np.array_equal(a, b)
