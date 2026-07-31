from __future__ import annotations

import numpy as np
from helpers import bitops, framing


def _hdlc_wrap(payload: bytes) -> np.ndarray:
    flag = bitops.bytes_to_bits(bytes([0x7E]))
    body = framing.bitstuff(bitops.bytes_to_bits(payload))
    return np.concatenate([flag, body, flag]).astype(np.uint8)


def test_hdlc_frame_start_is_a_raw_stream_offset() -> None:
    """A frame's start must index the raw bit stream, not the destuffed
    concatenation — offset (and therefore time) has to survive deframing."""
    payload = bytes([0x02, 0xAA, 0xBB])
    framed = _hdlc_wrap(payload)
    stream = np.concatenate([np.zeros(40, np.uint8), framed]).astype(np.uint8)
    out = framing.hdlc_frames(stream)
    assert [p for _, p in out] == [payload]
    assert out[0][0] == 48  # 40 noise bits + the 8-bit opening flag


def test_hdlc_frame_deframe_roundtrip() -> None:
    payload = bytes([0xFF, 0x3E, 0x01])  # 0xFF/0x3E exercise 5-ones bit stuffing
    framed = _hdlc_wrap(payload)
    out = framing.hdlc_frames(framed)
    assert len(out) == 1
    assert out[0][1] == payload


def test_destuff_rejects_sixth_consecutive_one() -> None:
    assert framing._destuff(np.array([1, 1, 1, 1, 1, 1, 0, 1], dtype=np.uint8)) is None
    assert framing._destuff(np.array([0, 1, 1, 1, 1, 1, 1, 1], dtype=np.uint8)) is None


def test_destuff_removes_valid_stuffed_zero() -> None:
    out = framing._destuff(np.array([1, 1, 1, 1, 1, 0, 1], dtype=np.uint8))
    assert out is not None and np.array_equal(
        out, np.array([1, 1, 1, 1, 1, 1], dtype=np.uint8)
    )


def test_find_flags_catches_zero_shared_flag() -> None:
    seq = np.array([0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0], dtype=np.uint8)
    assert framing._find_flags(seq) == [0, 7]


def test_read_uint_lsb_differs_from_msb_and_matches_hand_computed_value() -> None:
    # earliest physical bit becomes the LSB of the result under "lsb" (BLE-style):
    # bits=[0,0,0,0,1,1] read msb-first is 0b000011=3; read lsb-first, bits[0] is
    # weight 2**0 up to bits[5] weight 2**5, so 1*2**4 + 1*2**5 = 48.
    bits = np.array([0, 0, 0, 0, 1, 1], dtype=np.uint8)
    msb = bitops.read_uint(bits, 0, 6, "msb")
    lsb = bitops.read_uint(bits, 0, 6, "lsb")
    assert msb == 3
    assert lsb == 48
    assert msb != lsb


def test_length_frame_carved_recarves_payload_by_embedded_length() -> None:
    # payload: [hdr, len=2, d0, d1, crc0, crc1, junk]; base=4 (hdr+crc), len units=2
    payload = bytes([0x00, 0x02, 0xAA, 0xBB, 0xC0, 0xDE, 0xFF, 0xFF])
    out = framing.carve_length(
        payload, length_bits=8, base_bytes=4, unit_bytes=1, offset_bits=8
    )
    # need = base(4) + len(2) = 6 bytes
    assert out is not None
    assert len(out) == 6
    assert out == payload[:6]


def test_length_frame_carved_lsb_bit_order_reads_length_field_lsb_first() -> None:
    # BLE-style length byte: the length lives in the low 6 bits, LSB-first. The
    # header byte's second byte is 0x0F (0b00001111); an MSB-first read of its
    # first 6 physical bits gives 0b000011=3 (wrong), an LSB-first read gives the
    # true length 0b001111=15 (verified via a scratch read_uint call above).
    length_byte = 0x0F
    payload = bytes([0xAA, length_byte]) + bytes(range(15)) + bytes([0xFF, 0xFF, 0xFF])
    out = framing.carve_length(
        payload,
        length_bits=6,
        base_bytes=2,
        unit_bytes=1,
        offset_bits=8,
        bit_order="lsb",
    )
    assert out is not None
    assert len(out) == 2 + 1 * 15  # base_bytes + unit_bytes*L(=15)
    assert out == payload[:17]

    # same bits, msb order: a different (wrong) length, proving the two diverge
    out_msb = framing.carve_length(
        payload,
        length_bits=6,
        base_bytes=2,
        unit_bytes=1,
        offset_bits=8,
        bit_order="msb",
    )
    assert out_msb is not None
    assert len(out_msb) == 2 + 1 * 3


def test_carve_length_out_of_bounds_returns_none() -> None:
    region = bytes([200]) + bytes([1, 2, 3])  # claims 203 bytes, only 4 present
    out = framing.carve_length(region, length_bits=8, base_bytes=3, unit_bytes=1)
    assert out is None


def test_delimiter_slices_variable_body_plus_tail() -> None:
    wire = bitops.bytes_to_bits(bytes([0xAA, 0xBB, 0x03, 0xDE, 0xAD]))
    out = framing.carve_delimiter(wire, 0, delimiter_hex="03", tail_bits=16)
    assert out is not None
    assert bitops.bits_to_bytes(out) == bytes([0xAA, 0xBB, 0x03, 0xDE, 0xAD])
    assert out.size == 40


def test_frame_without_delimiter_is_dropped() -> None:
    wire = bitops.bytes_to_bits(bytes([0xAA]))
    out = framing.carve_delimiter(wire, 0, delimiter_hex="03", tail_bits=0)
    assert out is None


def test_tail_past_stream_end_drops_frame() -> None:
    wire = bitops.bytes_to_bits(bytes([0x03]))
    out = framing.carve_delimiter(wire, 0, delimiter_hex="03", tail_bits=16)
    assert out is None
