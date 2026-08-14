"""The free-word tally: a valid word whose corrected codeword is constant is
counted in words_constant, because the all-zero word is a codeword of every
linear code and its validity is evidence of nothing. The quality layer
excludes these from both verdict arms (see test_extractors)."""

from __future__ import annotations

import numpy as np
from helpers.golay import golay_codeword, golay_masks

from marconi.engine.coding.carrier import CodingCarrier
from marconi.engine.coding.ops_bits import block_code_rx, rs_code_rx
from marconi.engine.types.enums import EmitMode


def test_block_code_tallies_all_zero_words_as_constant() -> None:
    out = block_code_rx(
        CodingCarrier(bits=np.zeros(23 * 40, np.uint8)),
        code_bits=23,
        data_bits=12,
        parity_masks=golay_masks(),
        correct=0,
        emit=EmitMode.DATA,
    )
    assert out.stats is not None
    assert out.stats.words_valid == 40
    assert out.stats.words_constant == 40


def test_block_code_varied_valid_words_are_not_constant() -> None:
    words = [golay_codeword(d) for d in (1, 2, 3, 4)]
    bits = np.concatenate([np.asarray(w, np.uint8) for w in words])
    out = block_code_rx(
        CodingCarrier(bits=bits),
        code_bits=23,
        data_bits=12,
        parity_masks=golay_masks(),
        correct=0,
        emit=EmitMode.DATA,
    )
    assert out.stats is not None
    assert out.stats.words_valid == 4
    assert out.stats.words_constant == 0


def test_rs_code_tallies_the_dragged_to_zero_repair_as_constant() -> None:
    # 2 wrong symbols in 15 zeros is inside t=3: the corrector lands on the
    # all-zero codeword and calls it a repair - the free-word tally must see
    # the CORRECTED word, not the received one
    syms = np.zeros(15, np.int64)
    syms[3], syms[7] = 5, 9
    bits = np.unpackbits(
        syms.astype(np.uint8)[:, None], axis=1, count=4, bitorder="little"
    ).reshape(-1)[: 15 * 4]
    out = rs_code_rx(
        CodingCarrier(bits=np.asarray(bits, np.uint8)),
        symbol_bits=4,
        n=15,
        k=9,
        prim_poly=0x13,
        fcr=1,
    )
    assert out.stats is not None
    assert out.stats.words_valid == 1
    assert out.stats.words_constant == 1
