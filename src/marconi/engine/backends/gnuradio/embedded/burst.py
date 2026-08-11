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
import numpy.typing as npt

from marconi.engine.backends.gnuradio.embedded.envelope import (
    sustained_runs,
    trailing_mean,
)
from marconi.engine.backends.gnuradio.embedded.lifecycle import (
    Diagnostics,
    EofProbe,
    OutQueue,
    bump,
    drain_chunked,
    forecast_drain,
)

_FLOOR_BLOCK = 1024  # floor-update granularity, and the processing-block
# minimum. The block is sized up to hold the widest window a single call to
# _process_block evaluates locally: the rise hunt (sustained_runs returns
# nothing when the mask is shorter than its run) and the fall smoothing
# (trailing_mean degenerates toward a block-wide constant). Both scale with
# sps, so a block pinned here silently stopped detecting any burst at all
# once they outgrew it. self._quiet carries the fall RUN across blocks, so
# fall_run is deliberately not part of the bound.
_BLOCK_WINDOW_MARGIN = 2  # ... and holds that many of the widest window, so
# the trailing mean is an average over real history rather than the boundary
# extension mode="nearest" supplies when the window spans the whole block
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


def make_burst_sampler(gr: Any, *, sps: float) -> Any:
    if sps < 1.0:
        raise ValueError(f"burst_sampler needs sps >= 1, got {sps}")
    stride = float(sps)
    phases = int(math.ceil(stride))
    rise_run = _RISE_CHIPS * phases
    fall_run = _FALL_CHIPS * phases
    fall_smooth = _FALL_SMOOTH_CHIPS * phases
    max_burst = _MAX_BURST_CHIPS * phases
    block_floor = max(_FLOOR_BLOCK, _BLOCK_WINDOW_MARGIN * fall_smooth, rise_run)

    class _BurstSampler(gr.basic_block):
        def __init__(self) -> None:
            gr.basic_block.__init__(
                self, name="burst_sampler", in_sig=[np.float32], out_sig=[np.float32]
            )
            self._out = OutQueue(np.float32)
            self._pending = np.empty(0, np.float32)  # < block_floor carryover
            self._floor = 0.0
            self._abs = 0  # absolute index of the next input sample to classify
            self._k = 0  # next global chip index; its base is round(k*stride)
            self._burst: list[npt.NDArray[np.float32]] = []
            self._burst_len = 0
            self._burst_start = 0  # absolute sample index this burst began at
            self._burst_floor = 0.0  # floor at burst start (idle-reference term)
            self._in_burst = False
            self._quiet = 0
            self.eof_probe: EofProbe | None = None  # set by build wiring
            self._eof_done = False
            self.diagnostics: Diagnostics = {
                "bursts_flushed": 0,
                "bursts_truncated_at_eof": 0,
            }

        def forecast(self, noutput_items: int, ninputs: int) -> list[int]:
            return forecast_drain(self._out.pending, ninputs)

        def general_work(self, input_items: Any, output_items: Any) -> int:
            return drain_chunked(
                self, input_items, output_items, floor=block_floor, dtype=np.float32
            )

        def _flush_tail(self) -> None:
            """At true EOF, drain the withheld tail a real (unpadded) capture
            leaves behind: process the sub-block _pending remainder, then commit
            any burst still open. A burst whose calm tail landed in that
            remainder ends normally here (its own fall-detection fires); one the
            capture truncated mid-transmission is committed from what arrived and
            counted, so it surfaces instead of vanishing as noise chips."""
            if len(self._pending):
                self._process_block(self._pending)
                self._pending = np.empty(0, np.float32)
            if self._in_burst:
                bump(self.diagnostics, "bursts_truncated_at_eof")
                self._flush_burst(self._abs)

        def _process_block(self, block: npt.NDArray[np.float32]) -> None:
            med = float(np.median(block))
            if self._floor == 0.0:
                self._floor = med if med > 0.0 else 1e-9
            rate = _FLOOR_UP if med > self._floor else _FLOOR_DOWN
            self._floor += rate * (med - self._floor)
            active = block > _RISE_RATIO * self._floor
            calm = trailing_mean(block, fall_smooth) < _FALL_RATIO * self._floor
            base = self._abs
            n = len(block)
            i = 0
            while i < n:
                if not self._in_burst:
                    rises = sustained_runs(active[i:], rise_run)
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
                j = self._scan_fall(calm, i, n)
                self._burst.append(block[i:j])
                self._burst_len += j - i
                i = j
                if self._quiet >= fall_run or self._burst_len >= max_burst:
                    self._flush_burst(base + j)
            self._abs = base + n

        def _scan_fall(self, calm: npt.NDArray[np.bool_], i: int, n: int) -> int:
            """One past the sample completing a sustained-calm run of fall_run
            (counting the carried self._quiet), else n — the vectorized twin
            of a per-sample `quiet+1 if calm else 0` walk, which ran inside
            GR's scheduler thread for every in-burst sample."""
            c = calm[i:n]
            if c.size == 0:
                return n
            idx = np.arange(c.size, dtype=np.int64)
            marker = np.where(~c, idx, np.int64(-1))
            runlen = np.where(c, idx - np.maximum.accumulate(marker), np.int64(0))
            falses = np.flatnonzero(~c)
            lead = int(falses[0]) if falses.size else c.size
            runlen[:lead] += self._quiet
            hits = np.flatnonzero(runlen >= fall_run)
            if hits.size:
                self._quiet = int(runlen[hits[0]])
                return i + int(hits[0]) + 1
            self._quiet = int(runlen[-1])
            return n

        def _grid_picks(self, limit: int) -> npt.NDArray[np.int64]:
            """Absolute positions round(k*stride) of every global chip from
            the cursor landing before `limit`, advancing the cursor — the
            bulk twin of the scalar int(round(k*stride)) walk (np.round and
            Python round share half-to-even on float64)."""
            k_hi = int((limit + 0.5) / stride) + 2
            if k_hi <= self._k:
                return np.empty(0, np.int64)
            ks = np.arange(self._k, k_hi, dtype=np.float64)
            bases = np.round(ks * stride).astype(np.int64)
            take = int(np.searchsorted(bases, limit, side="left"))
            self._k += take
            return bases[:take]

        def _emit_idle(
            self, block: npt.NDArray[np.float32], base: int, hi: int
        ) -> None:
            # Emit every global chip whose base position round(k*stride) lands
            # in [current cursor, base+hi) - the idle default scale, phase 0.
            bases = self._grid_picks(base + hi)
            if bases.size:
                self._out.push(block[bases - base] / (_NORM_FLOOR_RATIO * self._floor))

        def _flush_burst(self, end: int) -> None:
            seg = (
                np.concatenate(self._burst) if self._burst else np.empty(0, np.float32)
            )
            s = self._burst_start
            self._burst = []
            self._burst_len = 0
            self._in_burst = False
            self._quiet = 0
            bump(self.diagnostics, "bursts_flushed")
            length = len(seg)
            # Global chips this burst owns: base round(k*stride) in [s, end).
            # The count is fixed by s/end (never by the phase below), so no
            # burst boundary can insert or drop a chip on the global grid.
            bases = self._grid_picks(end)
            if bases.size == 0:
                return
            local = bases - s
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
