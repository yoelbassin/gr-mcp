from __future__ import annotations

import numpy as np

from marconi.bits import framing
from marconi.bits.carriers import RxCarrier, _Frame

# POCSAG (ITU-R M.584) BCH(31,21), generator x^10+x^9+x^8+x^6+x^5+x^3+1 (0x769
# with the leading term implicit -> low terms 0b1101101001 = 0x369; full check
# below fixes the exact constant against the published idle codeword).
_GEN = 0b11101101001  # x^10..x^0 coefficients of the published generator
_IDLE = 0x7A89C197  # published POCSAG idle codeword (32nd bit = even parity)


def _bch_check(data21: int) -> int:
    reg = data21 << 10
    for bit in range(30, 9, -1):
        if reg & (1 << bit):
            reg ^= _GEN << (bit - 10)
    return reg & 0x3FF


def _masks_stream_basis() -> list[int]:
    return [
        sum((((_bch_check(1 << (20 - j)) >> (9 - p)) & 1) << j) for j in range(21))
        for p in range(10)
    ]


def _word_bits(word31: int) -> np.ndarray:
    return np.array([(word31 >> (30 - i)) & 1 for i in range(31)], np.uint8)


def test_idle_codeword_is_bch_consistent() -> None:
    # THE external anchor of this whole vector: pins _GEN (the published POCSAG
    # generator) against the published idle codeword. Every parity mask the decode
    # tests use is derived from _GEN via _masks_stream_basis(), so this one check
    # is what makes them trustworthy -- a wrong hand-typed _GEN cannot reproduce
    # the published idle word.
    idle31 = _IDLE >> 1  # strip the trailing even-parity bit
    data, check = idle31 >> 10, idle31 & 0x3FF
    assert _bch_check(data) == check


def test_block_code_decodes_published_idle_codeword() -> None:
    # Exercises production's clean-codeword decode given the generator-derived
    # masks. It does NOT independently validate the mask values: block_fec_decode
    # returns only the 21 data bits, so a wrong parity mask whose effect lands on
    # a discarded parity bit is invisible here. Mask correctness is anchored at
    # _GEN (see test_idle_codeword_is_bch_consistent), not by this test.
    idle31 = _IDLE >> 1
    out = framing.block_code_rx(
        RxCarrier(bits=_word_bits(idle31)),
        code_bits=31,
        data_bits=21,
        parity_masks=_masks_stream_basis(),
    )
    expect = [(idle31 >> (30 - i)) & 1 for i in range(21)]
    assert out.bits.tolist() == expect


def _assert_corrects_every_single_error(word31: int) -> None:
    clean = [(word31 >> (30 - i)) & 1 for i in range(21)]
    for pos in range(31):
        bits = _word_bits(word31)
        bits[pos] ^= 1
        out = framing.block_code_rx(
            RxCarrier(bits=bits),
            code_bits=31,
            data_bits=21,
            parity_masks=_masks_stream_basis(),
        )
        assert out.bits.tolist() == clean, f"uncorrected error at bit {pos}"


def test_block_code_corrects_single_error_without_2n_tables() -> None:
    # Sweeps all 31 codeword columns for two independent valid codewords, so
    # production's syndrome-scan is exercised at every error position (each drives
    # a distinct syndrome), not just one. Like the decode test, this exercises the
    # decoder given the generator-derived masks; it does NOT independently anchor
    # the mask values (a mask error on a discarded parity bit stays invisible --
    # that lives at _GEN). The "without_2n_tables" claim is the acceptance line:
    # BCH(31,21) corrects via an O(31)-column scan, never a 2^31 lookup table.
    _assert_corrects_every_single_error(_IDLE >> 1)
    data = 0b101010101010101010101
    _assert_corrects_every_single_error((data << 10) | _bch_check(data))


def _scalar_reference(
    bits: np.ndarray,
    code_bits: int,
    data_bits: int,
    masks: list[int],
    correct: bool | None,
) -> list[int]:
    from marconi.core.coding import block_fec_decode, can_correct

    do = can_correct(code_bits - data_bits, data_bits) if correct is None else correct
    out: list[int] = []
    for w in range(bits.size // code_bits):
        word = 0
        for j in range(code_bits):
            word |= int(bits[w * code_bits + j]) << j
        val = block_fec_decode(word, masks, data_bits, do)
        out.extend((val >> j) & 1 for j in range(data_bits))
    return out


def test_block_code_rx_matches_scalar_reference() -> None:
    """Random streams pin the whole per-word semantics — no-match syndromes
    (no flip), data-column flips, parity-column flips (data untouched),
    detect-only pass-through, and trailing partial-word truncation."""
    rng = np.random.default_rng(3)
    for code_bits, data_bits, masks, correct in (
        (31, 21, _masks_stream_basis(), None),
        (31, 21, _masks_stream_basis(), False),
        (8, 4, [7, 14, 11, 13], None),
    ):
        bits = rng.integers(0, 2, code_bits * 200 + 5, dtype=np.uint8)
        out = framing.block_code_rx(
            RxCarrier(bits=bits),
            code_bits=code_bits,
            data_bits=data_bits,
            parity_masks=masks,
            correct=correct,
        )
        assert out.bits.tolist() == _scalar_reference(
            bits, code_bits, data_bits, masks, correct
        )


def test_block_code_seeded_decodes_within_frame_window() -> None:
    # two frames, each one (3,2) codeword [d0 d1 p]; parity mask p = d0^d1 => 0b11
    # frame A bits: 1 0 1 (valid), frame B bits: 0 1 1 (valid)
    bits = np.array([1, 0, 1, 0, 1, 1], np.uint8)
    c = RxCarrier(bits=bits, frames=[_Frame(0, 0), _Frame(3, 3)])
    out = framing.block_code_rx(
        c,
        code_bits=3,
        data_bits=2,
        parity_masks=[0b11],
        correct=False,
        emit="data",
    )
    assert out.bits.tolist() == [1, 0, 0, 1]
    assert [f.cursor for f in out.frames] == [0, 2]  # remapped after shrink
