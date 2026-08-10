from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest
from helpers.golay import golay_codeword, golay_masks

from marconi.engine.coding import ops_bits
from marconi.engine.coding.carrier import CodingCarrier
from marconi.engine.coding.primitives import syndrome_table
from marconi.engine.coding.stages_bits import BlockCodeStep
from marconi.engine.types.enums import EmitMode


def _decode(bits: np.ndarray, correct: int) -> np.ndarray:
    return ops_bits.block_code_rx(
        CodingCarrier(bits=bits),
        code_bits=23,
        data_bits=12,
        parity_masks=golay_masks(),
        correct=correct,
        emit=EmitMode.DATA,
    ).bits


def test_golay_syndrome_table_is_perfect() -> None:
    table = syndrome_table(golay_masks(), 23, 12, 3)
    assert len(table) == 23 + 253 + 1771  # 2^11 - 1: every syndrome used


def test_syndrome_table_is_not_the_shared_cache_object() -> None:
    masks = golay_masks()
    first = syndrome_table(masks, 23, 12, 3)
    second = syndrome_table(masks, 23, 12, 3)
    assert first == second
    assert first is not second
    first[next(iter(first))] = -1
    third = syndrome_table(masks, 23, 12, 3)
    assert third == second


def test_golay_corrects_every_triple_error() -> None:
    data = 0b101100111010
    clean = golay_codeword(data)
    expect = clean[:12].tolist()
    for pattern in combinations(range(23), 3):
        word = clean.copy()
        word[list(pattern)] ^= 1
        assert _decode(word, correct=3).tolist() == expect, pattern


def test_golay_corrects_singles_and_doubles() -> None:
    data = 0b010011001101
    clean = golay_codeword(data)
    expect = clean[:12].tolist()
    for w in (1, 2):
        for pattern in combinations(range(23), w):
            word = clean.copy()
            word[list(pattern)] ^= 1
            assert _decode(word, correct=3).tolist() == expect, pattern


def test_correct_beyond_capability_is_rejected() -> None:
    # Hamming(7,4) corrects 1; t=2 syndromes must collide.
    hamming_masks = [0b1011, 0b1101, 0b1110]
    with pytest.raises(Exception, match="collide|correct"):
        BlockCodeStep.model_validate(
            {
                "code_bits": 7,
                "data_bits": 4,
                "parity_masks": hamming_masks,
                "correct": 2,
            }
        )


def test_correct_and_correct_single_are_exclusive() -> None:
    with pytest.raises(Exception, match="correct"):
        BlockCodeStep.model_validate(
            {
                "code_bits": 7,
                "data_bits": 4,
                "parity_masks": [0b1011, 0b1101, 0b1110],
                "correct": 1,
                "correct_single": True,
            }
        )


def test_t1_via_correct_matches_correct_single() -> None:
    masks = [0b1011, 0b1101, 0b1110]
    clean = np.array([1, 0, 1, 1, 0, 1, 0], np.uint8)
    word = clean.copy()
    word[5] ^= 1
    a = ops_bits.block_code_rx(
        CodingCarrier(bits=word),
        code_bits=7,
        data_bits=4,
        parity_masks=masks,
        correct=1,
        emit=EmitMode.DATA,
    ).bits
    b = ops_bits.block_code_rx(
        CodingCarrier(bits=word),
        code_bits=7,
        data_bits=4,
        parity_masks=masks,
        correct_single=True,
        emit=EmitMode.DATA,
    ).bits
    assert a.tolist() == b.tolist()
