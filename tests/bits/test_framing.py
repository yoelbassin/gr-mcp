from __future__ import annotations

import numpy as np
import pytest

from marconi.bits.carriers import RxCarrier, TxCarrier, _Frame
from marconi.bits.framing import (
    _decode_charset,
    _destuff,
    _find_flags,
    _seq_bits,
    block_code_rx,
    build_struct,
    crc_value,
    descramble_bits_rx,
    differential_rx,
    differential_tx,
    hdlc_deframe_rx,
    hdlc_deframe_tx,
    parse_fields,
    parse_rx,
    parse_tx,
    permute_rx,
    realign_rx,
)


def test_hdlc_frame_start_is_a_raw_stream_offset() -> None:
    """FrameResult.bit_offset maps a frame back to its position (and time) in
    the capture — start must index the raw bit stream, not the synthetic
    destuffed concatenation the pre-fix accumulator produced."""
    payload = bytes([0x02, 0xAA, 0xBB])
    framed = hdlc_deframe_tx(TxCarrier([payload]), bit_order="msb").items[0]
    stream = np.concatenate([np.zeros(40, np.uint8), framed])
    out = hdlc_deframe_rx(RxCarrier(bits=stream), bit_order="msb")
    assert [f.payload for f in out.frames] == [payload]
    assert out.frames[0].start == 48  # 40 noise bits + the 8-bit opening flag


def test_differential_roundtrip() -> None:
    bits = np.array([1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0], dtype=np.uint8)
    for invert in (False, True):
        enc = differential_tx(TxCarrier([bits]), invert=invert).items[0]
        dec = differential_rx(RxCarrier(bits=enc), invert=invert).bits
        assert np.array_equal(dec, bits)


def test_hdlc_frame_deframe_roundtrip() -> None:
    payload = bytes([0xFF, 0x3E, 0x01])  # 0xFF/0x3E exercise 5-ones bit stuffing
    framed = hdlc_deframe_tx(TxCarrier([payload]), bit_order="msb").items[0]
    out = hdlc_deframe_rx(RxCarrier(bits=framed), bit_order="msb")
    assert len(out.frames) == 1
    assert out.frames[0].payload == payload


def test_descramble_bits_restarts_sequence_at_each_frame_cursor() -> None:
    # seeded scope: two frames at cursors 0 and 5 over a 13-bit all-zero stream,
    # sequence "f0" (11110000) restarts at each cursor rather than tiling once
    # over the whole stream.
    bits = np.zeros(13, dtype=np.uint8)
    frames = [_Frame(start=0, cursor=0), _Frame(start=5, cursor=5)]
    out = descramble_bits_rx(RxCarrier(bits=bits, frames=frames), sequence="f0")
    seq = _seq_bits("f0")
    expected = np.concatenate([seq[:5], seq[:8]])
    assert np.array_equal(out.bits, expected)
    whole_stream = np.bitwise_xor(bits, np.resize(seq, bits.size))
    assert not np.array_equal(out.bits, whole_stream)


def _carved_carrier() -> RxCarrier:
    bits = np.zeros(16, dtype=np.uint8)
    frames = [_Frame(start=0, cursor=0, payload=b"\x01")]
    return RxCarrier(bits=bits, frames=frames)


def test_descramble_bits_rejects_carved_carrier() -> None:
    with pytest.raises(ValueError, match="no carved scope"):
        descramble_bits_rx(_carved_carrier(), sequence="f0")


def test_realign_rejects_carved_carrier() -> None:
    with pytest.raises(ValueError, match="no carved scope"):
        realign_rx(_carved_carrier(), bit_offset=1)


def test_permute_rejects_carved_carrier() -> None:
    with pytest.raises(ValueError, match="no carved scope"):
        permute_rx(_carved_carrier(), perm=[0, 1, 2, 3])


def test_block_code_rejects_carved_carrier() -> None:
    with pytest.raises(ValueError, match="no carved scope"):
        block_code_rx(
            _carved_carrier(),
            code_bits=3,
            data_bits=2,
            parity_masks=[0b11],
            correct=False,
            emit="data",
        )


def test_destuff_rejects_sixth_consecutive_one() -> None:
    assert _destuff(np.array([1, 1, 1, 1, 1, 1, 0, 1], dtype=np.uint8)) is None
    assert _destuff(np.array([0, 1, 1, 1, 1, 1, 1, 1], dtype=np.uint8)) is None


def test_destuff_removes_valid_stuffed_zero() -> None:
    out = _destuff(np.array([1, 1, 1, 1, 1, 0, 1], dtype=np.uint8))
    assert out is not None and np.array_equal(
        out, np.array([1, 1, 1, 1, 1, 1], dtype=np.uint8)
    )


def test_find_flags_catches_zero_shared_flag() -> None:
    seq = np.array([0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0], dtype=np.uint8)
    assert _find_flags(seq) == [0, 7]


def test_crc_x25_and_ccitt_vectors() -> None:
    x25 = crc_value(
        b"123456789", poly=0x1021, bits=16, init=0xFFFF, reflected=True, xorout=0xFFFF
    )
    assert x25 == 0x906E
    ccitt = crc_value(
        b"123456789", poly=0x1021, bits=16, init=0xFFFF, reflected=False, xorout=0
    )
    assert ccitt == 0x29B1


def test_parse_roundtrip_signed() -> None:
    fields = parse_fields(
        [
            {"name": "a", "bits": 6},
            {"name": "b", "bits": 28, "signed": True},
            {"name": "c", "bits": 2},
        ]
    )
    st = build_struct(fields)
    msg = {"a": 5, "b": -1234, "c": 1}
    built = parse_tx(TxCarrier([msg]), struct=st, fields=fields, bit_order="msb").items[
        0
    ]
    car = RxCarrier(
        bits=np.zeros(0, np.uint8),
        frames=[_Frame(start=0, cursor=0, payload=built, crc_ok=True)],
    )
    out = parse_rx(car, struct=st, fields=fields, bit_order="msb")
    assert out.frames[0].message == msg


def test_decode_charset_rejects_short_table() -> None:
    with pytest.raises(ValueError):
        _decode_charset(0, 6, "ABC", 6)
