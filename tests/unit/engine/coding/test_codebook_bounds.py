from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from marconi.engine.coding import ops_bits
from marconi.engine.coding.stages_bits import (
    MAX_CODEBOOK_DATA_BITS,
    CodebookStep,
)
from marconi.engine.coding.stages_symbols import SymbolMapStep


@pytest.mark.parametrize("model", [CodebookStep, SymbolMapStep])
def test_data_bits_is_bounded_before_the_shift(model: type) -> None:
    # unbounded, this materialized a 1 GB int inside the validator and then
    # raised again while rendering it into the error message
    with pytest.raises(ValidationError, match="ceiling"):
        model(code_bits=2, data_bits=1 << 33, table=[1, 2])


@pytest.mark.parametrize("model", [CodebookStep, SymbolMapStep])
def test_largest_writable_codebook_still_validates(model: type) -> None:
    n = 1 << MAX_CODEBOOK_DATA_BITS
    model(
        code_bits=MAX_CODEBOOK_DATA_BITS + 4,
        data_bits=MAX_CODEBOOK_DATA_BITS,
        table=list(range(n)),
        decode="nearest",
    )


def test_nearest_chunk_is_a_byte_budget_not_a_word_count() -> None:
    # (24, 12) is a spec the validator accepts; at a flat 65536-word chunk its
    # distance cube was a single 6.44 GB numpy temporary
    codewords, code_bits = 1 << 12, 24
    chunk = ops_bits._nearest_chunk(codewords, code_bits)
    cube = chunk * codewords * code_bits
    assert cube <= ops_bits._NEAREST_CUBE_BYTES
    assert chunk >= 1
    # a small codebook must not be throttled by the same budget
    assert ops_bits._nearest_chunk(4, 2) > chunk


def test_nearest_decode_is_correct_across_a_chunk_boundary() -> None:
    code_bits = 6
    table = [0b000111, 0b011001, 0b101010, 0b110100]
    chunk = ops_bits._nearest_chunk(len(table), code_bits)
    rng = np.random.default_rng(0)
    values = rng.integers(0, len(table), chunk * 2 + 7)
    rows = ops_bits._pack_symbols(
        np.asarray([table[v] for v in values], np.int64), code_bits
    ).reshape(-1, code_bits)
    got = ops_bits._nearest_values(rows, table, code_bits)
    assert np.array_equal(got, values)
