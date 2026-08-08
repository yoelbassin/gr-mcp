"""Open-loop burst sampler: per-burst phase-acquired decimation for bursty
OOK/PPM envelopes.

Stock symbol_sync_ff measured railing on burst environments at every
loop_bw >= 0.045 (chip rate 2.16-2.83 samp/chip vs nominal 2.0 across two
real assets), and frozen (loop_bw=0) it samples one global interpolant
phase - initialization luck (0/12 CRC frames on a live off-air slice whose
block-synchronized oracle decodes 12). pwr_squelch_cc cannot frame
marginal PPM (0-6/400 full-body holds across alpha 1..64); corr_est_cc is
coherent/complex and needs a datasheet preamble. No stock block does
burst-gated decimation with per-burst phase. Variance-max phase selection
picked the CRC-winning phase 12/12 on that slice; the metric is
modulation-blind (correct phase = bimodal samples = max variance), so the
block carries no protocol knowledge. Guarded by the sibling test file and
the off-air acceptance gates.

Determinism: processing is chunk-independent (fixed internal blocks, no
feedback loop, no volk in-block), so identical input yields identical
chips - the adaptive loop this replaces was the suite's nondeterminism
amplifier. The block withholds an unfinished burst tail at EOF (suite
convention: sims pad)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from marconi.engine.backends.gnuradio.embedded.lifecycle import (
    OutQueue,
    forecast_drain,
)

_FLOOR_BLOCK = 1024  # fixed floor-update granularity (chunk-independent)
_FLOOR_UP = 1.0 / 64.0  # EMA rate when the floor estimate rises
_FLOOR_DOWN = 1.0 / 4.0  # faster fall: never let a burst inflate the floor
_RISE_RATIO = 4.0  # burst starts above 4x floor ...
_RISE_CHIPS = 1  # ... sustained this many chips (rejects single-sample spikes)
_FALL_RATIO = 2.0  # and ends below 2x floor ...
_FALL_CHIPS = 4  # ... sustained this many chips (> longest in-burst calm run)
_PAD_CHIPS = 2  # pre/post pad kept around each burst
_MAX_BURST_CHIPS = 4096  # cap: beyond this, commit phase from the first cap


def _sustained_runs(mask: np.ndarray, run: int) -> np.ndarray:
    """Start indices of every maximal run of >= `run` consecutive True
    values in `mask`, via a vectorized sliding-window sum (cumsum
    difference) - the idle-span hunt is bulk, so this must not be a
    per-sample Python loop the way the (rare, short) in-burst scan is."""
    if run <= 1:
        return np.flatnonzero(mask)
    if len(mask) < run:
        return np.empty(0, dtype=np.int64)
    csum = np.concatenate([[0], np.cumsum(mask, dtype=np.int64)])
    window = csum[run:] - csum[:-run]
    return np.flatnonzero(window >= run)


def make_burst_sampler(gr: Any, *, sps: float) -> Any:
    if sps < 2.0:
        raise ValueError(f"burst_sampler needs sps >= 2, got {sps}")
    stride = float(sps)
    phases = int(math.ceil(stride))
    pad = _PAD_CHIPS * phases
    rise_run = _RISE_CHIPS * phases
    fall_run = _FALL_CHIPS * phases
    max_burst = _MAX_BURST_CHIPS * phases

    class _BurstSampler(gr.basic_block):
        def __init__(self) -> None:
            gr.basic_block.__init__(
                self, name="burst_sampler", in_sig=[np.float32], out_sig=[np.float32]
            )
            self._out = OutQueue(np.float32)
            self._pending = np.empty(0, np.float32)  # < _FLOOR_BLOCK carryover
            self._floor = 0.0
            self._idle_tail = np.empty(0, np.float32)  # pre-pad ring
            self._grid = 0.0  # fractional cursor into the idle span
            self._burst: list[np.ndarray] = []
            self._burst_len = 0
            self._in_burst = False
            self._quiet = 0
            self.diagnostics = {"bursts_flushed": 0}

        def forecast(self, noutput_items: int, ninputs: int) -> list[int]:
            return forecast_drain(self._out.pending, ninputs)

        def general_work(self, input_items: Any, output_items: Any) -> int:
            inp = input_items[0]
            if len(inp):
                self._pending = np.concatenate(
                    [self._pending, np.asarray(inp, np.float32)]
                )
                self.consume(0, len(inp))
                while len(self._pending) >= _FLOOR_BLOCK:
                    block, self._pending = (
                        self._pending[:_FLOOR_BLOCK],
                        self._pending[_FLOOR_BLOCK:],
                    )
                    self._process_block(block)
            return self._out.drain(output_items[0])

        def _process_block(self, block: np.ndarray) -> None:
            med = float(np.median(block))
            if self._floor == 0.0:
                self._floor = med if med > 0.0 else 1e-9
            rate = _FLOOR_UP if med > self._floor else _FLOOR_DOWN
            self._floor += rate * (med - self._floor)
            active = block > _RISE_RATIO * self._floor
            calm = block < _FALL_RATIO * self._floor
            i = 0
            while i < len(block):
                if not self._in_burst:
                    rises = _sustained_runs(active[i:], rise_run)
                    if len(rises) == 0:
                        self._emit_idle(block[i:])
                        return
                    start = i + int(rises[0])
                    self._emit_idle(block[i:start])
                    lead = self._idle_tail[-pad:] if pad else self._idle_tail[:0]
                    self._burst = [lead.copy(), np.empty(0, np.float32)]
                    self._burst_len = len(lead)
                    self._in_burst = True
                    self._quiet = 0
                    i = start
                    continue
                # in burst: scan for sustained calm
                j = i
                while j < len(block):
                    self._quiet = self._quiet + 1 if calm[j] else 0
                    j += 1
                    if self._quiet >= fall_run:
                        break
                self._burst.append(block[i:j])
                self._burst_len += j - i
                i = j
                if self._quiet >= fall_run or self._burst_len >= max_burst:
                    self._flush_burst()

        def _emit_idle(self, span: np.ndarray) -> None:
            if len(span):
                self._idle_tail = np.concatenate([self._idle_tail, span])[
                    -max(pad, 1) :
                ]
                n = len(span)
                picks = []
                g = self._grid
                while g < n:
                    picks.append(int(g))
                    g += stride
                self._grid = g - n
                if picks:
                    self._out.push(span[np.asarray(picks, np.int64)])

        def _flush_burst(self) -> None:
            seg = np.concatenate(self._burst)
            self._burst, self._burst_len = [], 0
            self._in_burst = False
            self._quiet = 0
            best_phase, best_var = 0, -1.0
            k = np.arange(int(len(seg) // stride) + 1)
            for phi in range(phases):
                idx = phi + np.round(k * stride).astype(np.int64)
                idx = idx[idx < len(seg)]
                if len(idx) < 4:
                    continue
                v = float(np.var(seg[idx]))
                if v > best_var:
                    best_phase, best_var = phi, v
            idx = best_phase + np.round(k * stride).astype(np.int64)
            idx = idx[idx < len(seg)]
            self._out.push(seg[idx])
            self._grid = 0.0
            self._idle_tail = self._idle_tail[:0]
            self.diagnostics["bursts_flushed"] += 1

    return _BurstSampler()
