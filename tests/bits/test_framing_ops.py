import numpy as np

from marconi.bits import framing
from marconi.bits.carriers import RxCarrier, TxCarrier, _Frame


def test_segment_then_fixed_frame_carves_payloads():
    bits = np.unpackbits(np.array([0xAB, 0xCD, 0x12, 0x34], dtype=np.uint8))  # 32 bits
    c = framing.segment_rx(RxCarrier(bits=bits), frame_body_len=16)
    assert len(c.frames) == 2
    c = framing.fixed_frame_rx(c, payload_bits=16)
    assert c.frames[0].payload == bytes([0xAB, 0xCD])
    assert c.frames[1].payload == bytes([0x12, 0x34])


def test_fixed_frame_tx_packs_bits():
    tx = framing.fixed_frame_tx(TxCarrier([bytes([0xAB, 0xCD])]), payload_bits=16)
    assert list(tx.items[0]) == list(
        np.unpackbits(np.array([0xAB, 0xCD], dtype=np.uint8))
    )


def test_nibble_swap_swaps_per_byte_and_roundtrips():
    bits = np.array([1, 0, 1, 1, 0, 0, 0, 1], dtype=np.uint8)  # 1011 0001
    out = framing.nibble_swap_rx(RxCarrier(bits=bits)).bits
    assert list(out) == [0, 0, 0, 1, 1, 0, 1, 1]  # 0001 1011
    tx = framing.nibble_swap_tx(TxCarrier([out]))
    assert list(tx.items[0]) == list(bits)


def test_descramble_xors_and_roundtrips():
    seq = "0f10"
    f = _Frame(start=0, cursor=0)
    f.payload = bytes([0xAB, 0xCD])
    c = framing.descramble_rx(
        RxCarrier(bits=np.zeros(0, np.uint8), frames=[f]), sequence=seq
    )
    assert c.frames[0].payload == bytes([0xAB ^ 0x0F, 0xCD ^ 0x10])
    tx = framing.descramble_tx(TxCarrier([c.frames[0].payload]), sequence=seq)
    assert tx.items[0] == bytes([0xAB, 0xCD])


def test_segment_rx_drops_short_tail():
    # 33 bits, frame_body_len=16: frames at [0,16) and [16,32), 1-bit tail dropped
    bits = np.zeros(33, dtype=np.uint8)
    c = framing.segment_rx(RxCarrier(bits=bits), frame_body_len=16)
    assert len(c.frames) == 2


def test_fixed_frame_rx_drops_past_end_frame():
    # 20-bit carrier, cursor=16, payload_bits=16: 16+16=32 > 20 → frame dropped
    bits = np.zeros(20, dtype=np.uint8)
    f = _Frame(start=16, cursor=16)
    c = framing.fixed_frame_rx(RxCarrier(bits=bits, frames=[f]), payload_bits=16)
    assert len(c.frames) == 0


def test_descramble_bits_xor_and_tile():
    bits = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], np.uint8)
    out = framing.descramble_bits_rx(RxCarrier(bits=bits), sequence="f0")  # 11110000
    assert out.bits.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]


def test_descramble_bits_round_trip():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 800).astype(np.uint8)
    seq = "a5c3"  # 16-bit repeating sequence
    framing.descramble_bits_tx(TxCarrier([bits]))  # identity carrier shape
    enc = framing.descramble_bits_rx(RxCarrier(bits=bits), sequence=seq).bits
    back = framing.descramble_bits_rx(RxCarrier(bits=enc), sequence=seq).bits
    assert np.array_equal(back, bits)
