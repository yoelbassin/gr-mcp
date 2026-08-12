from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest
from helpers.golay import golay_codeword, golay_masks
from pydantic import ValidationError

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


def _identity_code(k: int) -> tuple[int, int, list[int]]:
    """A systematic (2k, k) code where parity bit p checks data bit p alone, so
    breaking data bit 0 is visible to EXACTLY ONE equation — the last one to
    survive any width-limited syndrome register."""
    return 2 * k, k, [1 << p for p in range(k)]


@pytest.mark.parametrize("n_parity", [8, 63, 64, 65, 100, 200])
def test_a_broken_equation_is_caught_at_every_code_width(n_parity: int) -> None:
    """The syndrome used to be shifted into one fixed-width integer, so every
    equation past bit 63 of it was dropped: a (300,100) code called a wholly
    corrupt stream 400/400 valid and the quality layer read "decoded". A
    syndrome is as wide as the code, and this pins that at the boundary."""
    code_bits, data_bits, masks = _identity_code(n_parity)
    word = np.zeros(code_bits, np.uint8)
    word[0] ^= 1  # only equation 0 sees this

    out = ops_bits.block_code_rx(
        CodingCarrier(bits=word),
        code_bits=code_bits,
        data_bits=data_bits,
        parity_masks=masks,
        correct_single=False,
    )

    assert out.stats is not None
    assert out.stats.words_valid == 0, "a broken parity equation must not pass"
    assert out.stats.words_total == 1


@pytest.mark.parametrize("n_parity", [8, 65, 130])
def test_an_intact_word_still_passes_at_every_code_width(n_parity: int) -> None:
    code_bits, data_bits, masks = _identity_code(n_parity)
    rng = np.random.default_rng(n_parity)
    data = rng.integers(0, 2, data_bits, dtype=np.uint8)
    word = np.concatenate([data, data])  # parity p == data p

    out = ops_bits.block_code_rx(
        CodingCarrier(bits=word),
        code_bits=code_bits,
        data_bits=data_bits,
        parity_masks=masks,
        correct_single=False,
    )

    assert out.stats is not None and out.stats.words_valid == 1
    assert np.array_equal(np.asarray(out.bits), data)


def test_single_error_correction_works_past_the_old_register_width() -> None:
    # a wide Hamming-style code: parity p checks data bits p and p+1, so every
    # column has a distinct nonzero signature
    data_bits = 80
    code_bits = data_bits + data_bits
    masks = [(1 << p) | (1 << ((p + 1) % data_bits)) for p in range(data_bits)]
    rng = np.random.default_rng(7)
    data = rng.integers(0, 2, data_bits, dtype=np.uint8)
    parity = np.array(
        [int(data[p]) ^ int(data[(p + 1) % data_bits]) for p in range(data_bits)],
        np.uint8,
    )
    word = np.concatenate([data, parity])
    word[3] ^= 1  # a single, correctable error

    out = ops_bits.block_code_rx(
        CodingCarrier(bits=word),
        code_bits=code_bits,
        data_bits=data_bits,
        parity_masks=masks,
        correct=1,
    )

    assert out.stats is not None and out.stats.words_valid == 1
    assert np.array_equal(np.asarray(out.bits), data), "the error must be repaired"


def test_an_oversized_code_is_refused_at_the_spec_boundary() -> None:
    from marconi.engine.coding.stages_bits import MAX_CODE_BITS, BlockCodeStep

    with pytest.raises(ValidationError):
        BlockCodeStep(
            code_bits=MAX_CODE_BITS + 1,
            data_bits=2,
            parity_masks=[1] * (MAX_CODE_BITS - 1),
            correct_single=False,
        )


def test_an_unaffordable_syndrome_table_is_refused_not_enumerated() -> None:
    from marconi.engine.coding.primitives import MAX_SYNDROME_PATTERNS, syndrome_table

    with pytest.raises(ValueError, match="syndrome table"):
        syndrome_table([1] * 100, 2000, 1900, 3)
    assert MAX_SYNDROME_PATTERNS > 0
