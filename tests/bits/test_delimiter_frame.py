from __future__ import annotations

import numpy as np

from marconi.bits import framing
from marconi.bits.carriers import RxCarrier, _Frame


def _bits(data: bytes) -> np.ndarray:
    return framing.bytes_to_bits(data)


def test_delimiter_slices_variable_body_plus_tail() -> None:
    wire = _bits(bytes([0xAA, 0xBB, 0x03, 0xDE, 0xAD]))  # body..ETX(0x03)+CRC16
    out = framing.delimiter_frame_rx(
        RxCarrier(bits=wire, frames=[_Frame(start=0, cursor=0)]),
        delimiter="03",
        tail_bits=16,
    )
    assert out.frames[0].payload == bytes([0xAA, 0xBB, 0x03, 0xDE, 0xAD])
    assert out.frames[0].cursor == 40


def test_frame_without_delimiter_is_dropped() -> None:
    out = framing.delimiter_frame_rx(
        RxCarrier(bits=_bits(bytes([0xAA])), frames=[_Frame(start=0, cursor=0)]),
        delimiter="03",
        tail_bits=0,
    )
    assert out.frames == []


def test_tail_past_stream_end_drops_frame() -> None:
    out = framing.delimiter_frame_rx(
        RxCarrier(bits=_bits(bytes([0x03])), frames=[_Frame(start=0, cursor=0)]),
        delimiter="03",
        tail_bits=16,
    )
    assert out.frames == []
