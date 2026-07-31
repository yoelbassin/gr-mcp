from __future__ import annotations

import numpy as np
from helpers._fakegr import FAKE_GR, FakeTag, drive

from marconi.engine.backends.gnuradio.embedded.framing import make_tag_gate


def _gate(frame_len: int):
    return make_tag_gate(FAKE_GR, frame_len=frame_len, tag_name="sync")


def _stream(frame_len: int, gaps: list[int], frames: list[np.ndarray]):
    """Interleave junk gaps and frames; return (stream, sync-offsets)."""
    parts: list[np.ndarray] = []
    offsets: list[int] = []
    pos = 0
    for gap, frame in zip(gaps, frames):
        parts.append(np.full(gap, 0.3, np.float32))
        pos += gap
        offsets.append(pos)
        parts.append(frame.astype(np.float32))
        pos += frame.size
    parts.append(np.full(4, 0.3, np.float32))
    return np.concatenate(parts), offsets


def test_gate_keeps_frames_drops_junk() -> None:
    L = 12
    frames = [(np.arange(L) + 100 * (i + 1)).astype(np.float32) for i in range(3)]
    stream, offsets = _stream(L, [5, 9, 7], frames)
    blk = _gate(L)
    blk.in_tags = [FakeTag(o, "sync") for o in offsets]
    out = drive(blk, stream, chunk=1000, out_dtype=np.float32)
    assert out.tolist() == np.concatenate(frames).tolist()


def test_gate_frame_spans_chunk_boundary() -> None:
    L = 16
    frames = [(np.arange(L) + 100 * (i + 1)).astype(np.float32) for i in range(2)]
    stream, offsets = _stream(L, [3, 11], frames)
    blk = _gate(L)
    blk.in_tags = [FakeTag(o, "sync") for o in offsets]
    # chunk=5 forces frames to straddle general_work calls
    out = drive(blk, stream, chunk=5, out_dtype=np.float32)
    assert out.tolist() == np.concatenate(frames).tolist()


def test_gate_ignores_tag_inside_active_frame() -> None:
    L = 10
    frame = (np.arange(L) + 1).astype(np.float32)
    stream = np.concatenate([np.full(4, 0.3, np.float32), frame]).astype(np.float32)
    blk = _gate(L)
    # a spurious second tag lands 3 items into the frame — must be ignored
    blk.in_tags = [FakeTag(4, "sync"), FakeTag(7, "sync")]
    out = drive(blk, stream, chunk=100, out_dtype=np.float32)
    assert out.tolist() == frame.tolist()


def test_gate_no_tags_emits_nothing() -> None:
    blk = _gate(8)
    blk.in_tags = []
    out = drive(blk, np.full(40, 0.3, np.float32), chunk=7, out_dtype=np.float32)
    assert out.size == 0


def test_gate_reports_truncated_final_frame() -> None:
    """A frame cut short by end-of-stream is unrecoverable (its items never
    arrive), so the gate must say so: diagnostics carry how many items of the
    open frame are missing — the honest EOF signal a consumer can act on."""
    L = 12
    frame = np.arange(L).astype(np.float32)
    blk = _gate(L)
    blk.in_tags = [FakeTag(4, "sync")]
    stream = np.concatenate([np.full(4, 0.3, np.float32), frame[:5]])
    out = drive(blk, stream, chunk=100, out_dtype=np.float32)
    assert out.tolist() == frame[:5].tolist()
    assert blk.diagnostics["truncated_frame_items"] == L - 5


def test_gate_truncation_zero_when_frames_complete() -> None:
    L = 12
    frames = [(np.arange(L) + 100 * (i + 1)).astype(np.float32) for i in range(2)]
    stream, offsets = _stream(L, [5, 9], frames)
    blk = _gate(L)
    blk.in_tags = [FakeTag(o, "sync") for o in offsets]
    drive(blk, stream, chunk=1000, out_dtype=np.float32)
    assert blk.diagnostics["truncated_frame_items"] == 0
