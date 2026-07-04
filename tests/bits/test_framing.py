from __future__ import annotations

import numpy as np
import pytest

from marconi.bits.carriers import RxCarrier, TxCarrier, _Frame
from marconi.bits.framing import (
    _decode_charset,
    _destuff,
    _find_flags,
    build_struct,
    crc_value,
    differential_rx,
    differential_tx,
    hdlc_deframe_rx,
    hdlc_deframe_tx,
    parse_fields,
    parse_rx,
    parse_tx,
)


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
