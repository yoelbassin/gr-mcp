import numpy as np

from marconi.bits.carriers import RxCarrier
from marconi.bits.framing import mark_frame_rx
from marconi.bits.registry import registry


def test_mark_frame_seeds_one_frame_per_mark() -> None:
    out = mark_frame_rx(
        RxCarrier(bits=np.zeros(20, np.uint8), marks=(0, 8)), offset_bits=0
    )
    assert [(f.start, f.cursor) for f in out.frames] == [(0, 0), (8, 8)]


def test_mark_frame_drops_out_of_bounds() -> None:
    out = mark_frame_rx(
        RxCarrier(bits=np.zeros(5, np.uint8), marks=(4,)), offset_bits=8
    )
    assert out.frames == []


def test_mark_frame_preserves_marks_and_bits() -> None:
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], np.uint8)
    out = mark_frame_rx(RxCarrier(bits=bits, marks=(2,)), offset_bits=1)
    assert out.marks == (2,)
    assert out.bits.tolist() == bits.tolist()
    assert [(f.start, f.cursor) for f in out.frames] == [(3, 3)]


def test_mark_frame_stage_registered() -> None:
    stage = registry()["mark_frame"]
    assert stage.seeds_frames is True
