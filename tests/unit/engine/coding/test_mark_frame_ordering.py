"""mark_frame owns the precondition its own output has to satisfy.

window_spans requires non-decreasing cursors and raises CarrierInvariantError
otherwise — an error classified internal_error, which tells the agent the
ENGINE is broken. mark_frame built its windows straight from the carrier's
marks in whatever order they arrived, so the invariant held only because two
unrelated upstreams happened to sort: a diagnostic harvest, and a Symbolstream
validator. Neither is reachable from this function.
"""

from __future__ import annotations

import numpy as np
import pytest

from marconi.engine.coding.carrier import CarrierInvariantError, CodingCarrier
from marconi.engine.coding.ops_bits import mark_frame_rx


def _carrier(marks: tuple[int, ...], n: int = 64) -> CodingCarrier:
    return CodingCarrier(bits=np.zeros(n, np.uint8), marks=marks)


def test_unsorted_marks_still_scope_usable_windows() -> None:
    out = mark_frame_rx(_carrier((30, 4, 17)))
    assert [w.cursor for w in out.windows or []] == [4, 17, 30]
    assert out.window_spans() == [(4, 17), (17, 30), (30, 64)]


def test_duplicate_marks_collapse_to_one_window() -> None:
    """Two windows at one cursor scope an empty span between them, which is
    legal but reports a frame that never existed."""
    out = mark_frame_rx(_carrier((8, 8, 8, 20)))
    assert [w.cursor for w in out.windows or []] == [8, 20]


def test_the_offset_is_applied_before_ordering() -> None:
    out = mark_frame_rx(_carrier((30, 4)), offset_bits=5)
    assert [w.cursor for w in out.windows or []] == [9, 35]


def test_marks_outside_the_stream_are_dropped_not_clamped() -> None:
    out = mark_frame_rx(_carrier((10, 900), n=64))
    assert [w.cursor for w in out.windows or []] == [10]


def test_no_marks_seeds_an_empty_scope_not_a_blind_one() -> None:
    # [] means "a seeder ran and found nothing"; None would mean no seeder ran
    # and every downstream op would go whole-stream.
    out = mark_frame_rx(_carrier(()))
    assert out.windows == []


def test_the_invariant_this_protects_is_real() -> None:
    """Hand-built out-of-order windows still raise, so the guard above is doing
    the work rather than the invariant having quietly gone away."""
    from marconi.engine.coding.carrier import Window

    bad = CodingCarrier(
        bits=np.zeros(64, np.uint8),
        windows=[Window(start=30, cursor=30), Window(start=4, cursor=4)],
    )
    with pytest.raises(CarrierInvariantError, match="non-decreasing"):
        bad.window_spans()
