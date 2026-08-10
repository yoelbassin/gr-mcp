from __future__ import annotations

import numpy as np
import pytest
from helpers.blockmath import block_fec_decode, correct_codeword
from pydantic import ValidationError

from marconi.engine.coding import ops_bits
from marconi.engine.coding.builder import CodingBuilder
from marconi.engine.coding.carrier import CodingCarrier, Window
from marconi.engine.coding.primitives import can_correct
from marconi.engine.coding.stages_bits import (
    CODING_BITS_STAGES,
    BlockCode,
    BlockCodeStep,
    Codebook,
    MarkFrame,
    Permute,
    PermuteStep,
    Realign,
    RealignStep,
    SyncWord,
    SyncWordStep,
)
from marconi.engine.stages.base import CodingStage, StageDirectionError
from marconi.engine.types.enums import EmitMode


def test_sync_word_seeds_window_past_each_match() -> None:
    pat = np.array([0, 1, 1, 1, 1, 1, 1, 0], np.uint8)  # 0x7e
    bits = np.concatenate([np.zeros(5, np.uint8), pat, np.ones(16, np.uint8)])
    out = ops_bits.sync_word_rx(CodingCarrier(bits=bits), sync="7e", max_errors=0)
    assert [w.start for w in _wins(out)] == [13]


def test_block_code_window_scope_decodes_per_window_stride() -> None:
    # two windows, each one (7,4) Hamming codeword; single-bit error corrected
    masks = [0b1011, 0b1101, 0b1110]
    word = np.array([1, 0, 1, 1, 0, 1, 0], np.uint8)
    bad = word.copy()
    bad[2] ^= 1
    bits = np.concatenate([word, bad])
    c = CodingCarrier(bits=bits, windows=[Window(0, 0), Window(7, 7)])
    out = ops_bits.block_code_rx(
        c,
        code_bits=7,
        data_bits=4,
        parity_masks=masks,
        correct_single=True,
        emit=EmitMode.DATA,
    )
    assert out.bits.tolist() == [1, 0, 1, 1, 1, 0, 1, 1]
    assert [w.start for w in _wins(out)] == [0, 4]


def test_all_bit_stages_are_coding_stages() -> None:
    for cls in CODING_BITS_STAGES:
        assert issubclass(cls, CodingStage)
        assert cls().directions == frozenset({"rx"})


def test_descramble_restarts_sequence_per_window() -> None:
    seq_bits = np.unpackbits(np.frombuffer(bytes.fromhex("aa"), np.uint8))
    bits = np.zeros(16, np.uint8)
    c = CodingCarrier(bits=bits, windows=[Window(0, 0), Window(8, 8)])
    out = ops_bits.descramble_rx(c, sequence="aa")
    assert out.bits.tolist() == np.tile(seq_bits, 2).tolist()


def _wins(c: CodingCarrier) -> list[Window]:
    assert c.windows is not None
    return c.windows


def _sync_bits(*byte_vals: int) -> np.ndarray:
    return np.unpackbits(np.array(byte_vals, dtype=np.uint8))


def test_seeds_a_window_after_each_sync() -> None:
    stream = np.concatenate(
        [_sync_bits(0x7A, 0x96, 0x11), _sync_bits(0x7A, 0x96, 0x22)]
    )
    c = ops_bits.sync_word_rx(CodingCarrier(bits=stream), sync="7a96")
    assert [w.cursor for w in _wins(c)] == [16, 40]


def test_exact_match_rejects_corrupted_sync() -> None:
    stream = _sync_bits(0x7B, 0x96, 0x11)  # one bit flipped in the sync
    c = ops_bits.sync_word_rx(CodingCarrier(bits=stream), sync="7a96", max_errors=0)
    assert c.windows == []


def test_correlator_tolerates_errors_within_threshold() -> None:
    stream = _sync_bits(0x7B, 0x96, 0x11)  # Hamming-1 from 0x7a96
    c = ops_bits.sync_word_rx(CodingCarrier(bits=stream), sync="7a96", max_errors=1)
    assert [w.cursor for w in _wins(c)] == [16]


def test_sync_word_stage_declares_seeder_and_validates_hex() -> None:
    stage = SyncWord()
    assert stage.seeds_windows is True
    with pytest.raises(Exception):
        stage.step_model.model_validate({"sync": "zz"})  # not hex


def test_bit_string_sync_seeds_at_non_byte_length() -> None:
    # a 5-bit sync word is inexpressible as hex; the bits form covers it
    pat = np.array([1, 0, 1, 1, 0], np.uint8)
    stream = np.concatenate([np.zeros(3, np.uint8), pat, np.ones(6, np.uint8)])
    c = ops_bits.sync_word_rx(CodingCarrier(bits=stream), bits="10110")
    assert [w.cursor for w in _wins(c)] == [8]


def test_sync_word_step_takes_exactly_one_pattern_form() -> None:
    stage = SyncWord()
    assert stage.step_model.model_validate({"bits": "10110"}).bits == "10110"
    for spec in ({}, {"sync": "7e", "bits": "1"}, {"bits": "012"}):
        with pytest.raises(Exception):
            stage.step_model.model_validate(spec)


def test_sync_word_schema_documents_pattern_fields_for_the_agent() -> None:
    # describe_stages surfaces model_json_schema to the driving agent, so the
    # format of each pattern field (hex bytes vs '0'/'1' bits) and the dibit/
    # symbol-string footgun — a digit run that is silently valid hex and matches
    # nothing — must live in the field descriptions, not only in code comments.
    props = SyncWordStep.model_json_schema()["properties"]
    assert "hex" in props["sync"]["description"].lower()
    bits_doc = props["bits"]["description"].lower()
    assert "bit" in bits_doc and "symbol" in bits_doc


def test_mark_frame_seeds_one_window_per_mark() -> None:
    out = ops_bits.mark_frame_rx(
        CodingCarrier(bits=np.zeros(20, np.uint8), marks=(0, 8)), offset_bits=0
    )
    assert [(w.start, w.cursor) for w in _wins(out)] == [(0, 0), (8, 8)]


def test_mark_frame_drops_out_of_bounds() -> None:
    out = ops_bits.mark_frame_rx(
        CodingCarrier(bits=np.zeros(5, np.uint8), marks=(4,)), offset_bits=8
    )
    assert out.windows == []


def test_mark_frame_preserves_marks_and_bits() -> None:
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], np.uint8)
    out = ops_bits.mark_frame_rx(CodingCarrier(bits=bits, marks=(2,)), offset_bits=1)
    assert out.marks == (2,)
    assert out.bits.tolist() == bits.tolist()
    assert [(w.start, w.cursor) for w in _wins(out)] == [(3, 3)]


def test_mark_frame_stage_declares_seeder() -> None:
    assert MarkFrame().seeds_windows is True


def test_segment_seeds_fixed_stride_windows() -> None:
    bits = np.unpackbits(np.array([0xAB, 0xCD, 0x12, 0x34], dtype=np.uint8))
    out = ops_bits.segment_rx(CodingCarrier(bits=bits), frame_body_len=16)
    assert [w.start for w in _wins(out)] == [0, 16]


def test_segment_drops_short_tail() -> None:
    bits = np.zeros(33, dtype=np.uint8)
    out = ops_bits.segment_rx(CodingCarrier(bits=bits), frame_body_len=16)
    assert len(_wins(out)) == 2


_3OF6 = [
    0x16, 0x0D, 0x0E, 0x0B, 0x1C, 0x19, 0x1A, 0x13,
    0x2C, 0x25, 0x26, 0x23, 0x34, 0x31, 0x32, 0x29,
]  # fmt: skip
_MANCHESTER = [0b01, 0b10]


def _codebook_encode(
    bits: np.ndarray, code_bits: int, data_bits: int, table: list[int]
) -> np.ndarray:
    fwd = np.asarray(table, np.int64)
    values = ops_bits._unpack_symbols(bits, data_bits)
    return ops_bits._pack_symbols(fwd[values], code_bits)


def test_manchester_round_trips() -> None:
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    chips = _codebook_encode(bits, code_bits=2, data_bits=1, table=_MANCHESTER)
    # 1->10, 0->01, ...
    assert chips.tolist() == [1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
    back = ops_bits.codebook_rx(
        CodingCarrier(bits=chips), code_bits=2, data_bits=1, table=_MANCHESTER
    )
    assert back.bits.tolist() == bits.tolist()


def test_three_of_six_round_trips_bytes() -> None:
    data = bytes([0x44, 0x2D, 0x2C, 0x11])
    bits = np.unpackbits(np.frombuffer(data, np.uint8))
    chips = _codebook_encode(bits, code_bits=6, data_bits=4, table=_3OF6)
    assert chips.size == len(data) * 2 * 6  # two 6-chip codewords per byte
    # every emitted 6-chip group carries exactly three ones (the code's invariant)
    assert set(int(g.sum()) for g in chips.reshape(-1, 6)) == {3}
    back = ops_bits.codebook_rx(
        CodingCarrier(bits=chips), code_bits=6, data_bits=4, table=_3OF6
    )
    assert np.packbits(back.bits).tobytes() == data


def test_unknown_code_symbol_maps_to_zero() -> None:
    # 111111 is not a valid 3-of-6 codeword; decoder degrades to 0, no crash
    chips = np.ones(6, dtype=np.uint8)
    back = ops_bits.codebook_rx(
        CodingCarrier(bits=chips), code_bits=6, data_bits=4, table=_3OF6
    )
    assert back.bits.tolist() == [0, 0, 0, 0]


def test_codebook_stage_validates_table_size() -> None:
    stage = Codebook()
    stage.step_model.model_validate({"code_bits": 6, "data_bits": 4, "table": _3OF6})
    with pytest.raises(Exception):  # 3 entries != 2^4
        stage.step_model.model_validate(
            {"code_bits": 6, "data_bits": 4, "table": [1, 2, 3]}
        )


def test_codebook_symbol_input_maps_and_scales_marks() -> None:
    out = ops_bits.codebook_rx(
        CodingCarrier(
            bits=np.zeros(0, np.uint8),
            symbols=np.array([3, 2, 0, 1], np.int16),
            marks=(0, 2),
        ),
        code_bits=2,
        data_bits=2,
        table=[0, 1, 2, 3],
        symbol_input=True,
    )
    assert out.bits.tolist() == [1, 1, 1, 0, 0, 0, 0, 1]
    assert out.marks == (0, 4)
    assert out.symbols is None


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
    idle31 = _IDLE >> 1  # strip the trailing even-parity bit
    data, check = idle31 >> 10, idle31 & 0x3FF
    assert _bch_check(data) == check


def test_block_code_decodes_published_idle_codeword() -> None:
    idle31 = _IDLE >> 1
    out = ops_bits.block_code_rx(
        CodingCarrier(bits=_word_bits(idle31)),
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
        out = ops_bits.block_code_rx(
            CodingCarrier(bits=bits),
            code_bits=31,
            data_bits=21,
            parity_masks=_masks_stream_basis(),
        )
        assert out.bits.tolist() == clean, f"uncorrected error at bit {pos}"


def test_block_code_corrects_single_error_without_2n_tables() -> None:
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
        out = ops_bits.block_code_rx(
            CodingCarrier(bits=bits),
            code_bits=code_bits,
            data_bits=data_bits,
            parity_masks=masks,
            correct_single=correct,
        )
        assert out.bits.tolist() == _scalar_reference(
            bits, code_bits, data_bits, masks, correct
        )


def test_block_code_seeded_decodes_within_window() -> None:
    # two windows, each one (3,2) codeword [d0 d1 p]; parity mask p = d0^d1 => 0b11
    # window A bits: 1 0 1 (valid), window B bits: 0 1 1 (valid)
    bits = np.array([1, 0, 1, 0, 1, 1], np.uint8)
    c = CodingCarrier(bits=bits, windows=[Window(0, 0), Window(3, 3)])
    out = ops_bits.block_code_rx(
        c,
        code_bits=3,
        data_bits=2,
        parity_masks=[0b11],
        correct_single=False,
        emit=EmitMode.DATA,
    )
    assert out.bits.tolist() == [1, 0, 0, 1]
    assert [w.cursor for w in _wins(out)] == [0, 2]  # remapped after shrink


_CODES = [
    (7, 4, [0b1011, 0b1101, 0b1110]),  # Hamming(7,4)
    (6, 3, [0b110, 0b101, 0b011]),  # a (6,3) parity code
]


def _roundtrip(code_bits: int, data_bits: int, masks: list[int]) -> None:
    flat = np.array(
        [(x >> i) & 1 for x in range(1 << code_bits) for i in range(code_bits)],
        np.uint8,
    )
    got = ops_bits.block_code_rx(
        CodingCarrier(bits=flat),
        code_bits=code_bits,
        data_bits=data_bits,
        parity_masks=masks,
        correct_single=True,
        emit=EmitMode.CODEWORD,
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
    a = ops_bits.block_code_rx(
        CodingCarrier(bits=flat),
        code_bits=7,
        data_bits=4,
        parity_masks=[0b1011, 0b1101, 0b1110],
        correct_single=True,
    ).bits
    b = ops_bits.block_code_rx(
        CodingCarrier(bits=flat),
        code_bits=7,
        data_bits=4,
        parity_masks=[0b1011, 0b1101, 0b1110],
        correct_single=True,
        emit=EmitMode.DATA,
    ).bits
    assert np.array_equal(a, b)


def test_block_code_stage_emits_full_codeword() -> None:
    b = CodingBuilder()
    b.begin_step("block_code", "block_code")
    BlockCode().emit_rx(
        b,
        BlockCodeStep(
            code_bits=7,
            data_bits=4,
            parity_masks=[0b1011, 0b1101, 0b1110],
            correct_single=True,
            emit=EmitMode.CODEWORD,
        ),
    )
    src = CodingCarrier(bits=np.array([1, 0, 1, 1, 0, 1, 0], np.uint8))
    out = b.steps[0].call(src).bits
    assert out.size == 7  # full codeword, not masked to the 4 data bits


def test_block_code_emit_validator_rejects_unknown() -> None:
    with pytest.raises(ValidationError):
        BlockCode().step_model.model_validate(
            {
                "code_bits": 7,
                "data_bits": 4,
                "parity_masks": [0b1011, 0b1101, 0b1110],
                "emit": "bogus",
            }
        )


def test_permute_gathers_per_block() -> None:
    perm = [2, 0, 3, 1]
    src = np.array([1, 0, 0, 1, 0, 1, 1, 0], np.uint8)
    out = ops_bits.permute_rx(CodingCarrier(bits=src), perm=perm).bits
    assert list(out) == [
        src[2],
        src[0],
        src[3],
        src[1],
        src[4 + 2],
        src[4 + 0],
        src[4 + 3],
        src[4 + 1],
    ]


def test_permute_is_invertible_with_inverse_perm() -> None:
    perm = [3, 1, 0, 2]
    inv = [perm.index(i) for i in range(len(perm))]
    src = np.random.default_rng(0).integers(0, 2, 4 * 5, dtype=np.uint8)
    once = ops_bits.permute_rx(CodingCarrier(bits=src), perm=perm).bits
    back = ops_bits.permute_rx(CodingCarrier(bits=once), perm=inv).bits
    assert np.array_equal(back, src)


def test_permute_seeded_gathers_relative_to_cursor_and_can_drop_bits() -> None:
    # window has a trailing bit to drop: gather indices 0,1,3 of a 4-bit slot
    bits = np.array([1, 0, 1, 1, 0, 1, 0, 1], np.uint8)  # two 4-bit slots
    c = CodingCarrier(bits=bits, windows=[Window(0, 0)])
    out = ops_bits.permute_rx(c, perm=[0, 1, 3, 4, 5, 7])  # drop indices 2 and 6
    assert out.bits.tolist() == [1, 0, 1, 0, 1, 1]
    assert _wins(out)[0].cursor == 0


def test_permute_window_scope_permutes_every_stride_not_just_the_first() -> None:
    # a window spanning multiple perm-strides must permute EVERY stride, matching
    # block_code / descramble / codebook whole-window scope. Regression: the
    # windowed path used to emit only the first stride and drop the rest.
    perm = [2, 0, 1]
    src = np.random.default_rng(0).integers(0, 2, 12, dtype=np.uint8)  # 4 strides
    c = CodingCarrier(bits=src, windows=[Window(0, 0)])
    out = ops_bits.permute_rx(c, perm=perm).bits
    expected = src.reshape(4, 3)[:, perm].reshape(-1)
    assert out.tolist() == expected.tolist()


def test_permute_stage_wired() -> None:
    b = CodingBuilder()
    b.begin_step("permute", "permute")
    Permute().emit_rx(b, PermuteStep(perm=[2, 0, 3, 1]))
    out = b.steps[0].call(CodingCarrier(bits=np.array([1, 0, 0, 1], np.uint8))).bits
    assert list(out) == [0, 1, 1, 0]  # in[2], in[0], in[3], in[1]


def test_realign_drops_leading_bits() -> None:
    c = CodingCarrier(bits=np.array([1, 1, 0, 1, 0], np.uint8))
    out = ops_bits.realign_rx(c, bit_offset=2)
    assert out.bits.tolist() == [0, 1, 0]
    assert out.windows is None


def test_realign_seeded_advances_cursor() -> None:
    c = CodingCarrier(bits=np.zeros(10, np.uint8), windows=[Window(2, 2)])
    out = ops_bits.realign_rx(c, bit_offset=3)
    assert [(w.start, w.cursor) for w in _wins(out)] == [(2, 5)]


def test_realign_emit_tx_raises() -> None:
    with pytest.raises(StageDirectionError):
        Realign().emit_tx(CodingBuilder(), RealignStep(bit_offset=3))


def _differential_encode(bits: np.ndarray, invert: bool) -> np.ndarray:
    inv = np.uint8(1 if invert else 0)
    return np.bitwise_xor.accumulate(np.bitwise_xor(bits, inv).astype(np.uint8))


def test_differential_roundtrip() -> None:
    bits = np.array([1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0], dtype=np.uint8)
    for invert in (False, True):
        enc = _differential_encode(bits, invert)
        dec = ops_bits.differential_rx(CodingCarrier(bits=enc), invert=invert).bits
        assert np.array_equal(dec, bits)


def test_nibble_swap_swaps_per_byte_and_is_involutive() -> None:
    bits = np.array([1, 0, 1, 1, 0, 0, 0, 1], dtype=np.uint8)  # 1011 0001
    out = ops_bits.nibble_swap_rx(CodingCarrier(bits=bits)).bits
    assert list(out) == [0, 0, 0, 1, 1, 0, 1, 1]  # 0001 1011
    back = ops_bits.nibble_swap_rx(CodingCarrier(bits=out)).bits
    assert list(back) == list(bits)


def test_descramble_xor_and_tile() -> None:
    bits = np.ones(12, np.uint8)
    out = ops_bits.descramble_rx(CodingCarrier(bits=bits), sequence="f0")  # 11110000
    assert out.bits.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]


def test_descramble_round_trip() -> None:
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 800).astype(np.uint8)
    seq = "a5c3"  # 16-bit repeating sequence
    enc = ops_bits.descramble_rx(CodingCarrier(bits=bits), sequence=seq).bits
    back = ops_bits.descramble_rx(CodingCarrier(bits=enc), sequence=seq).bits
    assert np.array_equal(back, bits)


def test_descramble_round_trip_multi_chunk_non_period_aligned() -> None:
    seq = "a5c3"  # 16-bit repeating keystream
    rng = np.random.default_rng(1)
    chunks = [rng.integers(0, 2, 30).astype(np.uint8) for _ in range(4)]  # 30%16!=0
    wire = np.concatenate(chunks)
    enc = ops_bits.descramble_rx(CodingCarrier(bits=wire), sequence=seq).bits
    back = ops_bits.descramble_rx(CodingCarrier(bits=enc), sequence=seq).bits
    assert np.array_equal(back, wire)


def test_descramble_restarts_sequence_at_each_window_cursor() -> None:
    # two windows at cursors 0 and 5 over a 13-bit all-zero stream, sequence
    # "f0" (11110000) restarts at each cursor rather than tiling once over the
    # whole stream.
    bits = np.zeros(13, dtype=np.uint8)
    c = CodingCarrier(bits=bits, windows=[Window(0, 0), Window(5, 5)])
    out = ops_bits.descramble_rx(c, sequence="f0")
    seq = ops_bits._seq_bits("f0")
    expected = np.concatenate([seq[:5], seq[:8]])
    assert np.array_equal(out.bits, expected)
    whole_stream = np.bitwise_xor(bits, np.resize(seq, bits.size))
    assert not np.array_equal(out.bits, whole_stream)


def test_correct_single_rejects_ambiguous_syndromes() -> None:
    with pytest.raises(ValidationError, match="single-error"):
        BlockCodeStep(
            code_bits=4,
            data_bits=2,
            parity_masks=[0b0011, 0b0011],
            correct_single=True,
        )


def test_correct_single_accepts_bch_31_21() -> None:
    masks = [
        0x157929,
        0x1F8B7B,
        0x0A6FDF,
        0x14DFBE,
        0x1CC655,
        0x0CF583,
        0x19EB06,
        0x06AF25,
        0x0D5E4A,
        0x1ABC94,
    ]
    p = BlockCodeStep(
        code_bits=31, data_bits=21, parity_masks=masks, correct_single=True
    )
    assert p.correct_single is True


def test_correct_single_rejects_weight_one_parity_column() -> None:
    with pytest.raises(ValidationError, match="weight"):
        BlockCodeStep(
            code_bits=4,
            data_bits=2,
            parity_masks=[0b01, 0b11],
            correct_single=True,
        )


def test_correct_single_rejects_zero_parity_column() -> None:
    with pytest.raises(ValidationError, match="weight"):
        BlockCodeStep(
            code_bits=6,
            data_bits=3,
            parity_masks=[0b110, 0b100, 0b010],
            correct_single=True,
        )


def test_block_code_auto_correct_rejects_degenerate_columns() -> None:
    with pytest.raises(ValidationError, match="pairwise-distinct"):
        BlockCodeStep.model_validate(
            {
                "code_bits": 7,
                "data_bits": 4,
                "parity_masks": [0b0011, 0b0011, 0b1100],
            }
        )


def test_block_code_detect_only_still_accepts_degenerate_columns() -> None:
    p = BlockCodeStep.model_validate(
        {
            "code_bits": 7,
            "data_bits": 4,
            "parity_masks": [0b0011, 0b0011, 0b1100],
            "correct_single": False,
        }
    )
    assert p.correct_single is False
