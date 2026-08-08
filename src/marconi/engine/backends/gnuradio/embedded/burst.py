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
amplifier. Decimation runs on ONE global chip grid: chip k draws from input
sample round(k*stride) (plus the burst's phase offset), a monotonic index
that never resets at a burst boundary, so the emitted count is exactly
input_len // stride with no per-burst slack. Only two things vary per burst -
the sampling PHASE (which sub-sample within each chip window is taken,
variance-max acquired) and the NORMALIZATION scale; the chip index stays
global, which is what stops a burst boundary from inserting a chip and
slipping the downstream chip-pair grid mid-frame. The block withholds an
unfinished burst tail at EOF (suite convention: sims pad). Every emitted chip
is normalized (a burst's chips against their own 95th-percentile grid level,
an idle chip against the tracked floor) before it leaves the block, so
downstream fixed-threshold slicing is independent of whatever gain an upstream
stage was at - the open-loop path is AGC-free by design and normalizes
internally instead."""

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
_FALL_CHIPS = 32  # ... sustained this many chips: must clear the longest
# quiet run still INTERIOR to one transmission (structured burst-mode
# signals can carry a quiet stretch several symbols wide inside a single
# burst), while staying far under real inter-burst spacing
_FALL_SMOOTH_CHIPS = 4  # trailing moving-average width applied before the
# fall test only: a run this long tested sample-by-sample against raw,
# noisy envelope values is statistically unreachable by ordinary background
# noise (per-sample calm probability < 1 raised to a large power is ~0) even
# though the noise floor itself is genuinely low - smoothing first is what
# makes a long sustained-fall confirmation achievable at all
_MAX_BURST_CHIPS = 4096  # cap: beyond this, commit phase from the first cap
_NORM_FLOOR_RATIO = 16.0  # emitted-value normalization reference (own
# constant, not tied to _RISE_RATIO: measured on a real off-air capture that
# real background noise's own P95 can reach ~7-12x floor and its tail as far
# as ~26-29x, well past _RISE_RATIO's 4x detection bar - a normalization
# floor that close to the detection bar amplifies exactly those noise-
# triggered false detections up past the downstream 0.5 slicing point,
# defeating the point of normalizing at all. A burst's grid values divide by
# their own 95th percentile (floored at this ratio, so a pathological
# near-silent burst - or one dominated by a noise-scale P95 - cannot divide
# down to noise level); an idle span's by this ratio directly. Every
# emitted chip lands independent of whatever gain an upstream stage was at.


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


def _trailing_mean(block: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average, same length as `block` (front-padded with
    `block[0]` so the first few samples of each fixed-size processing block
    do not see a false dip toward zero from implicit zero-padding)."""
    if window <= 1:
        return block
    kernel = np.full(window, 1.0 / window, dtype=np.float64)
    padded = np.concatenate([np.full(window - 1, block[0], dtype=np.float64), block])
    return np.convolve(padded, kernel, mode="valid")


def make_burst_sampler(gr: Any, *, sps: float) -> Any:
    if sps < 1.0:
        raise ValueError(f"burst_sampler needs sps >= 1, got {sps}")
    stride = float(sps)
    phases = int(math.ceil(stride))
    rise_run = _RISE_CHIPS * phases
    fall_run = _FALL_CHIPS * phases
    fall_smooth = _FALL_SMOOTH_CHIPS * phases
    max_burst = _MAX_BURST_CHIPS * phases

    class _BurstSampler(gr.basic_block):
        def __init__(self) -> None:
            gr.basic_block.__init__(
                self, name="burst_sampler", in_sig=[np.float32], out_sig=[np.float32]
            )
            self._out = OutQueue(np.float32)
            self._pending = np.empty(0, np.float32)  # < _FLOOR_BLOCK carryover
            self._floor = 0.0
            self._abs = 0  # absolute index of the next input sample to classify
            self._k = 0  # next global chip index; its base is round(k*stride)
            self._burst: list[np.ndarray] = []
            self._burst_len = 0
            self._burst_start = 0  # absolute sample index this burst began at
            self._burst_floor = 0.0  # floor at burst start (idle-reference term)
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
            calm = _trailing_mean(block, fall_smooth) < _FALL_RATIO * self._floor
            base = self._abs
            n = len(block)
            i = 0
            while i < n:
                if not self._in_burst:
                    rises = _sustained_runs(active[i:], rise_run)
                    if len(rises) == 0:
                        self._emit_idle(block, base, n)
                        break
                    start = i + int(rises[0])
                    self._emit_idle(block, base, start)
                    self._burst = []
                    self._burst_len = 0
                    self._burst_start = base + start
                    self._burst_floor = self._floor
                    self._in_burst = True
                    self._quiet = 0
                    i = start
                    continue
                # in burst: scan for sustained calm
                j = i
                while j < n:
                    self._quiet = self._quiet + 1 if calm[j] else 0
                    j += 1
                    if self._quiet >= fall_run:
                        break
                self._burst.append(block[i:j])
                self._burst_len += j - i
                i = j
                if self._quiet >= fall_run or self._burst_len >= max_burst:
                    self._flush_burst(base + j)
            self._abs = base + n

        def _emit_idle(self, block: np.ndarray, base: int, hi: int) -> None:
            # Emit every global chip whose base position round(k*stride) lands
            # in [current cursor, base+hi) - the idle default scale, phase 0.
            limit = base + hi
            picks: list[int] = []
            while True:
                b = int(round(self._k * stride))
                if b >= limit:
                    break
                picks.append(b - base)
                self._k += 1
            if picks:
                idx = np.asarray(picks, np.int64)
                self._out.push(block[idx] / (_NORM_FLOOR_RATIO * self._floor))

        def _flush_burst(self, end: int) -> None:
            seg = (
                np.concatenate(self._burst) if self._burst else np.empty(0, np.float32)
            )
            s = self._burst_start
            self._burst = []
            self._burst_len = 0
            self._in_burst = False
            self._quiet = 0
            self.diagnostics["bursts_flushed"] += 1
            length = len(seg)
            # Global chips this burst owns: base round(k*stride) in [s, end).
            # The count is fixed by s/end (never by the phase below), so no
            # burst boundary can insert or drop a chip on the global grid.
            locals_: list[int] = []
            while True:
                b = int(round(self._k * stride))
                if b >= end:
                    break
                locals_.append(b - s)
                self._k += 1
            if not locals_:
                return
            local = np.asarray(locals_, np.int64)
            best_phase = 0
            if phases > 1 and len(local) >= 4:
                best_var = -1.0
                for phi in range(phases):
                    idx = np.minimum(local + phi, length - 1)
                    v = float(np.var(seg[idx]))
                    if v > best_var:
                        best_phase, best_var = phi, v
            idx = np.minimum(local + best_phase, length - 1)
            grid_vals = seg[idx]
            p95 = float(np.percentile(grid_vals, 95)) if len(grid_vals) else 0.0
            burst_scale = max(p95, _NORM_FLOOR_RATIO * self._burst_floor)
            self._out.push(grid_vals / burst_scale)

    return _BurstSampler()
