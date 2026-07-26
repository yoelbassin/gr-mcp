"""Shared lifecycle for the embedded Python blocks — the one home of the
GR-scheduler contract these blocks live by (issues 01/05/16):

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
converters (css_map/css_demap — bounded demands, no pending state) and pure
streaming pass-throughs (sym_strip — emits within each granted window).
"""

from __future__ import annotations

from typing import Any

import numpy as np


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
    releasing whole-window margins early). ``expected_items`` is set when the
    source feeds the holding block DIRECTLY: then nitems_read reaching it
    proves every sample is already in the block's hands — true finality, safe
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
