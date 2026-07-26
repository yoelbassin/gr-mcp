from __future__ import annotations

import numpy as np
import pytest

from marconi.engine.coding import ops_bits
from marconi.engine.coding.carrier import CodingCarrier
from marconi.engine.coding.stages_bits import _CodebookParams

THREE_OF_SIX = [
    0x16, 0x0D, 0x0E, 0x0B, 0x1C, 0x19, 0x1A, 0x13,
    0x2C, 0x25, 0x26, 0x23, 0x34, 0x31, 0x32, 0x29,
]  # fmt: skip


def _bits_of(values: list[int], width: int) -> np.ndarray:
    out: list[int] = []
    for v in values:
        out.extend((v >> (width - 1 - j)) & 1 for j in range(width))
    return np.asarray(out, np.uint8)


def test_nearest_equals_exact_on_clean_input() -> None:
    wire = _bits_of([THREE_OF_SIX[i] for i in (0, 5, 15, 9)], 6)
    exact = ops_bits.codebook_rx(
        CodingCarrier(bits=wire), code_bits=6, data_bits=4, table=THREE_OF_SIX
    )
    near = ops_bits.codebook_rx(
        CodingCarrier(bits=wire),
        code_bits=6,
        data_bits=4,
        table=THREE_OF_SIX,
        decode="nearest",
    )
    assert near.bits.tolist() == exact.bits.tolist()


def test_nearest_corrects_single_chip_error() -> None:
    wire = _bits_of([THREE_OF_SIX[i] for i in (3, 12, 7)], 6)
    wire[2] ^= 1  # one chip flip inside the first codeword
    near = ops_bits.codebook_rx(
        CodingCarrier(bits=wire),
        code_bits=6,
        data_bits=4,
        table=THREE_OF_SIX,
        decode="nearest",
    )
    assert near.bits.tolist() == _bits_of([3, 12, 7], 4).tolist()


def test_nearest_never_allocates_inverse_table() -> None:
    # 32-bit codewords: the exact path would need a 2^32-entry table.
    table = [0x00000000, 0xFFFFFFFF, 0x0F0F0F0F, 0xF0F0F0F0]
    wire = _bits_of([0xFFFFFFFF, 0x0F0F0F0F], 32)
    wire[[0, 7, 40]] ^= 1
    near = ops_bits.codebook_rx(
        CodingCarrier(bits=wire),
        code_bits=32,
        data_bits=2,
        table=table,
        decode="nearest",
    )
    assert near.bits.tolist() == _bits_of([1, 2], 2).tolist()


def test_nearest_tie_breaks_to_lowest_index() -> None:
    table = [0b0000, 0b0011, 0b1100, 0b1111]
    wire = _bits_of([0b0001], 4)  # distance 1 to both index 0 and index 1
    near = ops_bits.codebook_rx(
        CodingCarrier(bits=wire),
        code_bits=4,
        data_bits=2,
        table=table,
        decode="nearest",
    )
    assert near.bits.tolist() == [0, 0]


def test_exact_mode_rejects_wide_codes() -> None:
    with pytest.raises(Exception, match="nearest"):
        _CodebookParams.model_validate(
            {"code_bits": 32, "data_bits": 2, "table": [0, 1, 2, 3]}
        )


def test_decode_param_is_validated() -> None:
    with pytest.raises(Exception, match="exact|nearest"):
        _CodebookParams.model_validate(
            {"code_bits": 6, "data_bits": 4, "table": THREE_OF_SIX, "decode": "fuzzy"}
        )
