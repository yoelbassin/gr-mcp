"""Shared lifecycle for the embedded Python blocks — the one home of the
GR-scheduler contract these blocks live by:

- forecast is the only EOF channel: once upstream is exhausted, a block is
  called again only if its forecast demands zero input. Pending output (or
  emittable buffered work) must be announced as zero demand — that is what
  flushes the tail at EOF.
- never infer EOF from len(input) == 0: with zero demand announced, the
  scheduler may deliver zero-input calls at any time mid-stream, so every
  decision must be keyed to buffered content — a wakeup emits exactly what
  a later call would.
- a full-symbol/frame input demand can exceed GR's default stream buffer
  (8191 complex items): accumulate internally off forecast [1] instead of
  demanding multiples (the failure class that motivated OutQueue).

Two block classes intentionally stay outside this module: fixed small-ratio
converters (css_demap — bounded demands, no pending state) and pure
streaming pass-throughs (sym_strip — emits within each granted window).
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

# The block-side diagnostics payload the worker harvests into Diagnostic rows:
# int -> count, float -> value, list[int] -> marks. Any other value type is a
# harvest-time TypeError, never a silent coercion.
Diagnostics = dict[str, "int | float | list[int]"]


def bump(diag: Diagnostics, key: str, n: int = 1) -> None:
    # np.integer is deliberately accepted and narrowed: numpy's own counting
    # (np.count_nonzero, a .sum(), an argmax index) hands back np.int64, which
    # is not an isinstance of int. Rejecting it made a counter poison itself —
    # the value stored fine and the NEXT bump raised inside general_work, so a
    # decode that had already run correctly was returned as "embedded block
    # raised". The guard is here for float counters, and np.floating would
    # have slipped through the same test anyway.
    cur = diag.get(key, 0)
    if isinstance(cur, bool) or not isinstance(cur, (int, np.integer)):
        raise TypeError(f"diagnostic {key!r} is not an int counter: {cur!r}")
    diag[key] = int(cur) + int(n)


class ChunkedBlock(Protocol):
    _pending: npt.NDArray[Any]
    _eof_done: bool
    _out: "OutQueue"
    eof_probe: "EofProbe | None"

    def _process_block(self, block: Any) -> None: ...

    def _flush_tail(self) -> None: ...

    def consume(self, port: int, n: int) -> None: ...

    def nitems_read(self, port: int) -> int: ...


def eof_final(block: ChunkedBlock) -> bool:
    """True only once every input sample this block will ever get has been
    read (source-relative count proven by the build-wired probe). Needs no
    straggler grace: nitems_read reaching the propagated total means the
    block's own buffer already holds the whole capture, so irreversible tail
    work (sub-block remainders, committing an open burst) is safe. Without
    expected_items (unknown rate / a live source) it never fires and the
    sim-pad withhold stands."""
    p = block.eof_probe
    return (
        p is not None
        and p.expected_items is not None
        and int(block.nitems_read(0)) >= p.expected_items
    )


def drain_chunked(
    block: ChunkedBlock,
    input_items: Any,
    output_items: Any,
    *,
    floor: int,
    dtype: Any,
) -> int:
    inp = input_items[0]
    if len(inp):
        block._pending = np.concatenate([block._pending, np.asarray(inp, dtype)])
        block.consume(0, len(inp))
        while len(block._pending) >= floor:
            chunk, block._pending = block._pending[:floor], block._pending[floor:]
            block._process_block(chunk)
    if not block._eof_done and eof_final(block):
        block._eof_done = True
        block._flush_tail()
    return block._out.drain(output_items[0])


class OutQueue:
    """Pending-output queue: push whole results as they complete, drain what
    fits into each granted output window. Totals support offset bookkeeping
    (e.g. stream tags addressed by absolute output position)."""

    def __init__(self, dtype: Any) -> None:
        self._q = np.empty(0, dtype=dtype)
        self.pushed_total = 0
        self.drained_total = 0

    @property
    def pending(self) -> bool:
        return bool(self._q.size)

    @property
    def size(self) -> int:
        return int(self._q.size)

    def push(self, items: Any) -> None:
        arr = np.asarray(items, dtype=self._q.dtype)
        self._q = np.concatenate([self._q, arr])
        self.pushed_total += int(arr.size)

    def drain(self, out: Any) -> int:
        k = int(min(self._q.size, len(out)))
        if k:
            out[:k] = self._q[:k]
            self._q = self._q[k:]
            self.drained_total += k
        return k


def forecast_drain(work_pending: bool, ninputs: int) -> list[int]:
    """[0] while pending work can emit without new input — announcing
    drainability is what buys both mid-stream wakeups and the EOF flush —
    and [1] otherwise."""
    return [0 if work_pending else 1] * ninputs


class EofProbe:
    """The missing half of the EOF story (forecast_drain is the other):
    ``exhausted()`` is true once the pipeline's sole file source has written
    its entire file into the graph — no NEW sample will ever enter, though
    already-written ones may still be in flight through upstream buffers, so
    it licenses only actions that stay correct if more input arrives (e.g.
    releasing whole-window margins early). ``expected_items`` (build.py wires it:
    the source's emitted count carried through a single-edged, integer-ratio
    decimation path — ratio 1 included) makes nitems_read reaching it prove
    every sample is already in the block's hands — true finality, safe
    for irreversible steps like padding out a filter tail. nitems_written is
    a monotonic counter safe to read cross-thread; a stale read only delays
    the flush by one wakeup."""

    def __init__(
        self, source: Any, total_items: int, expected_items: int | None = None
    ) -> None:
        self._source = source
        self._total = int(total_items)
        self.expected_items = None if expected_items is None else int(expected_items)

    def exhausted(self) -> bool:
        return int(self._source.nitems_written(0)) >= self._total
