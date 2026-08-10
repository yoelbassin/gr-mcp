"""Window end-of-scope: a decode must stay inside the bits its own window
owns — realign's dropped lead bits belong to the next frame, and a tail
shorter than a frame is not a frame."""

import numpy as np

from marconi.engine.coding.carrier import CodingCarrier
from marconi.engine.coding.ops_bits import codebook_rx, realign_rx, segment_rx

_IDENTITY = [0, 1, 2, 3]


def test_windowed_realign_drops_bits_not_shifts_them() -> None:
    # two 8-bit frames; realign drops each window's leading 2 bits — they must
    # not be decoded into the previous frame's tail
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 16).astype(np.uint8)
    c = segment_rx(CodingCarrier(bits=bits), frame_body_len=8)
    c = realign_rx(c, bit_offset=2)
    out = codebook_rx(c, code_bits=2, data_bits=2, table=_IDENTITY)
    expected = np.concatenate([bits[2:8], bits[10:16]])
    assert np.array_equal(out.bits, expected)


def test_segment_scopes_exact_frames_dropping_the_tail() -> None:
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, 3 * 8 + 5).astype(np.uint8)
    c = segment_rx(CodingCarrier(bits=bits), frame_body_len=8)
    out = codebook_rx(c, code_bits=2, data_bits=2, table=_IDENTITY)
    assert out.bits.size == 3 * 8  # the 5-bit tail is not a frame
    assert np.array_equal(out.bits, bits[: 3 * 8])


def test_realign_of_a_window_shorter_than_the_offset_is_empty() -> None:
    bits = np.ones(10, np.uint8)
    c = segment_rx(CodingCarrier(bits=bits), frame_body_len=5)
    c = realign_rx(c, bit_offset=6)
    out = codebook_rx(c, code_bits=1, data_bits=1, table=[0, 1])
    assert out.bits.size == 0
