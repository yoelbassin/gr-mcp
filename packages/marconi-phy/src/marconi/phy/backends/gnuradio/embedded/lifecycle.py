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
  demanding multiples (the chirp_demod failure class).

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
