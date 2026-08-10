from __future__ import annotations

import numpy as np
import pytest

from marconi.engine.coding import ops_bits
from marconi.engine.coding.builder import CodingBuilder
from marconi.engine.coding.carrier import CodingCarrier, Window
from marconi.engine.coding.stages_bits import CodebookStep
from marconi.engine.coding.stages_symbols import SymbolMap, SymbolMapStep
from marconi.engine.types.enums import DecodeMode

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
        CodebookStep.model_validate(
            {"code_bits": 32, "data_bits": 2, "table": [0, 1, 2, 3]}
        )


def test_nearest_mode_rejects_codes_past_int64_ceiling() -> None:
    with pytest.raises(Exception, match="63"):
        CodebookStep.model_validate(
            {
                "code_bits": 64,
                "data_bits": 2,
                "table": [0, 1, 2, 3],
                "decode": "nearest",
            }
        )


def test_nearest_mode_allows_int64_ceiling() -> None:
    CodebookStep.model_validate(
        {
            "code_bits": 63,
            "data_bits": 2,
            "table": [0, 1, 2, 3],
            "decode": "nearest",
        }
    )


def test_decode_param_is_validated() -> None:
    with pytest.raises(Exception, match="exact|nearest"):
        CodebookStep.model_validate(
            {"code_bits": 6, "data_bits": 4, "table": THREE_OF_SIX, "decode": "fuzzy"}
        )


def test_nearest_rejects_over_width_table_entries() -> None:
    wire = _bits_of([0], 4)
    with pytest.raises(ValueError, match=r"\[0, 2\*\*code_bits\)"):
        ops_bits.codebook_rx(
            CodingCarrier(bits=wire),
            code_bits=4,
            data_bits=2,
            table=[0, 1, 2, 20],
            decode="nearest",
        )


def test_nearest_windowed_scope_corrects_and_preserves_window_boundaries() -> None:
    values = [3, 12, 7]
    wire = _bits_of([THREE_OF_SIX[i] for i in values], 6)
    wire[2] ^= 1  # chip flip inside the codeword covered by the first window
    windows = [Window(start=0, cursor=0), Window(start=12, cursor=12)]
    out = ops_bits.codebook_rx(
        CodingCarrier(bits=wire, windows=windows),
        code_bits=6,
        data_bits=4,
        table=THREE_OF_SIX,
        decode="nearest",
    )
    assert out.bits.tolist() == _bits_of(values, 4).tolist()
    assert [w.start for w in out.windows or []] == [0, 8]


def test_symbol_input_nearest_corrects_off_table_value() -> None:
    # a repetition-style code, far enough apart that a single off-table
    # symbol has exactly one nearest entry
    table = [0b000000, 0b000111, 0b111000, 0b111111]
    off_table = 0b000110  # one bit off table[1] (index 1 -> data value 1)
    carrier = CodingCarrier(
        bits=np.zeros(0, np.uint8), symbols=np.array([off_table], np.int64)
    )
    exact = ops_bits.codebook_rx(
        carrier, code_bits=6, data_bits=2, table=table, symbol_input=True
    )
    near = ops_bits.codebook_rx(
        carrier,
        code_bits=6,
        data_bits=2,
        table=table,
        symbol_input=True,
        decode="nearest",
    )
    assert exact.bits.tolist() == _bits_of([0], 2).tolist()  # unknown -> degrades to 0
    assert near.bits.tolist() == _bits_of([1], 2).tolist()  # nearest -> table[1]


def test_symbol_map_stage_forwards_decode_to_nearest() -> None:
    table = [0b000000, 0b000111, 0b111000, 0b111111]
    off_table = 0b000110
    b = CodingBuilder()
    SymbolMap().emit_rx(
        b,
        SymbolMapStep(
            code_bits=6,
            data_bits=2,
            table=table,
            decode=DecodeMode.NEAREST,
        ),
    )
    src = CodingCarrier(
        bits=np.zeros(0, np.uint8), symbols=np.array([off_table], np.int64)
    )
    out = b.steps[0].call(src)
    assert out.bits.tolist() == _bits_of([1], 2).tolist()
