import numpy as np

from marconi.bits import framing
from marconi.bits.builder import ProgramBuilder
from marconi.bits.carriers import RxCarrier, TxCarrier, _Frame
from marconi.bits.framing import _read_uint
from marconi.bits.registry import registry


def _frame(bits, cursor=0):
    return RxCarrier(bits=bits, frames=[_Frame(start=cursor, cursor=cursor)])


def test_carves_payload_from_declared_length():
    # region = [L=9][9 body bytes][2 trailer bytes] + junk; L drives the span
    region = bytes([9]) + bytes(range(9)) + bytes([0xAA, 0xBB])
    bits = np.concatenate([framing.bytes_to_bits(region), np.zeros(40, np.uint8)])
    c = framing.length_frame_rx(_frame(bits), length_bits=8, base_bytes=3, unit_bytes=1)
    assert len(c.frames) == 1
    assert c.frames[0].payload == region  # 12 bytes: L-field + body + trailer


def test_two_back_to_back_frames_seed_independently():
    r1 = bytes([2]) + bytes([0x11, 0x22]) + bytes([0x01, 0x02])  # L=2 -> 5 bytes
    r2 = bytes([3]) + bytes([0x33, 0x44, 0x55]) + bytes([0x03, 0x04])  # L=3 -> 6 bytes
    bits = framing.bytes_to_bits(r1 + r2)
    frames = [_Frame(start=0, cursor=0), _Frame(start=len(r1) * 8, cursor=len(r1) * 8)]
    c = framing.length_frame_rx(
        RxCarrier(bits=bits, frames=frames), length_bits=8, base_bytes=3, unit_bytes=1
    )
    assert [f.payload for f in c.frames] == [r1, r2]


def test_drops_frame_when_declared_length_overruns_stream():
    region = bytes([200]) + bytes([1, 2, 3])  # claims 203 bytes, only 4 present
    c = framing.length_frame_rx(
        _frame(framing.bytes_to_bits(region)),
        length_bits=8,
        base_bytes=3,
        unit_bytes=1,
    )
    assert c.frames == []


def test_tx_is_passthrough():
    payload = bytes([9, 1, 2, 3])
    tx = framing.length_frame_tx(
        TxCarrier([payload]), length_bits=8, base_bytes=3, unit_bytes=1
    )
    assert tx.items == [payload]


def test_registered_and_declares_body_slicer():
    stage = registry()["length_frame"]
    assert stage.slices_body is True


def test_length_frame_carved_recarves_payload_by_embedded_length() -> None:
    # payload: [hdr, len=2, d0, d1, crc0, crc1, junk]; base=4 (hdr+crc), len units=2
    payload = bytes([0x00, 0x02, 0xAA, 0xBB, 0xC0, 0xDE, 0xFF, 0xFF])
    c = RxCarrier(bits=np.zeros(0, np.uint8), frames=[_Frame(0, 0, payload=payload)])
    out = framing.length_frame_rx(
        c, length_bits=8, base_bytes=4, unit_bytes=1, offset_bits=8, bit_order="msb"
    )
    # need = base(4) + len(2) = 6 bytes
    assert len(out.frames[0].payload) == 6
    assert out.frames[0].payload == payload[:6]


def test_read_uint_lsb_differs_from_msb_and_matches_hand_computed_value() -> None:
    # earliest physical bit becomes the LSB of the result under "lsb" (BLE-style):
    # bits=[0,0,0,0,1,1] read msb-first is 0b000011=3; read lsb-first, bits[0] is
    # weight 2**0 up to bits[5] weight 2**5, so 1*2**4 + 1*2**5 = 48.
    bits = np.array([0, 0, 0, 0, 1, 1], dtype=np.uint8)
    msb = _read_uint(bits, 0, 6, "msb")
    lsb = _read_uint(bits, 0, 6, "lsb")
    assert msb == 3
    assert lsb == 48
    assert msb != lsb


def test_length_frame_carved_lsb_bit_order_reads_length_field_lsb_first() -> None:
    # BLE-style length byte: the length lives in the low 6 bits, LSB-first. The
    # header byte's second byte is 0x0F (0b00001111); an MSB-first read of its
    # first 6 physical bits gives 0b000011=3 (wrong), an LSB-first read gives the
    # true length 0b001111=15 (verified via a scratch _read_uint call above).
    length_byte = 0x0F
    payload = bytes([0xAA, length_byte]) + bytes(range(15)) + bytes([0xFF, 0xFF, 0xFF])
    c = RxCarrier(bits=np.zeros(0, np.uint8), frames=[_Frame(0, 0, payload=payload)])
    out = framing.length_frame_rx(
        c, length_bits=6, base_bytes=2, unit_bytes=1, offset_bits=8, bit_order="lsb"
    )
    assert len(out.frames) == 1
    assert len(out.frames[0].payload) == 2 + 1 * 15  # base_bytes + unit_bytes*L(=15)
    assert out.frames[0].payload == payload[:17]

    # same bits, msb order: a different (wrong) length, proving the two diverge
    out_msb = framing.length_frame_rx(
        c, length_bits=6, base_bytes=2, unit_bytes=1, offset_bits=8, bit_order="msb"
    )
    assert len(out_msb.frames[0].payload) == 2 + 1 * 3


def test_length_frame_stage_threads_offset_bits_through_emit_rx() -> None:
    reg = registry()
    b = ProgramBuilder()
    reg["length_frame"].emit_rx(
        b,
        {
            "length_bits": 8,
            "base_bytes": 4,
            "unit_bytes": 1,
            "bit_order": "msb",
            "offset_bits": 8,
        },
    )
    # payload: [hdr=0x00, len=2, d0, d1, crc0, crc1, junk, junk]; length field
    # sits one byte in, so the step must honor offset_bits=8 (not default 0,
    # which would read the header byte 0x00 as length and trim to 4 bytes).
    payload = bytes([0x00, 0x02, 0xAA, 0xBB, 0xC0, 0xDE, 0xFF, 0xFF])
    c = RxCarrier(bits=np.zeros(0, np.uint8), frames=[_Frame(0, 0, payload=payload)])
    out = b.steps[0](c)
    assert len(out.frames[0].payload) == 6
    assert out.frames[0].payload == payload[:6]
