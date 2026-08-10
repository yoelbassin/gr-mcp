"""Open-loop Oerder & Meyr symbol timing for bursty/short PSK. Stock
symbol_sync_cc/pfb_clock_sync_ccf are second-order tracking loops: they
settle over many symbols and mistrack before a short burst ends. The O&M
estimator is feedforward — one |x|^2 clock-line phase per burst, no
settling — so it fits bursts a loop cannot.

Contract (mirrors burst_sampler): bursts are detected on smoothed |x|^2
against a persistent tracked peak (_update_peak), buffered whole, and
cubic-resampled at one tau estimated over the burst's edge-excluded
interior — the RRC matched filter's ~span-symbol transient at each burst
edge biases any estimate that includes it. Symbols land on a global grid at
m*stride that never resets at a burst boundary, and every grid position is
emitted exactly once — idle spans, pre-burst instants, and the confirmed-
quiet fall tail as literal zeros (an unambiguous "nothing here" that no
downstream statistic, e.g. a differential decode straddling a burst edge,
can mistake for a constellation point) — so the output count is exactly
input_len // stride. Chunk-independent: fixed internal blocks, no feedback
loop, no volk in-block. Unfinished tails are withheld until the finality
probe proves EOF (suite convention: sims pad).

A region that never falls (a continuous transmission; in a ratio-1 chain
nothing announces its end) commits a chunk every _MAX_REGION_SYMS and keeps
accumulating — which also tracks slow SFO. Power alone cannot tell a
constant-envelope signal from noise, so long regions are additionally gated
on the |x|^2 clock line itself (_clock_line_ratio): a region with no line —
noise promoted to "active" by a cold-start peak — is emitted as zeros,
never resampled into fake symbols, and proves the noise floor. That floor
(also learned from gap blocks under a healthy peak) clamps the peak's decay
at _NOISE_GUARD x floor, so a long gap never descends into the noise band
where it would fragment into pseudo-bursts too short to gate; the rare
noise excursion still crossing the guarded threshold is zeroed by contrast
against that proven floor (_BURST_CONTRAST). Residual noise passed through
before the floor is proven (e.g. an EOF-truncated cold start) is the
quality layer's to flag."""

from __future__ import annotations

from typing import Any

import numpy as np

from marconi.engine.backends.gnuradio.embedded.burst import (
    _sustained_runs,
    _trailing_mean,
)
from marconi.engine.backends.gnuradio.embedded.lifecycle import (
    OutQueue,
    forecast_drain,
)


def estimate_tau(block: np.ndarray, sps: int) -> float:
    e = (block.real.astype(np.float64) ** 2) + (block.imag.astype(np.float64) ** 2)
    whole = (e.size // sps) * sps
    if whole:
        # whole symbol periods only: |x|^2's large DC term leaks into the
        # 1/sps bin over any partial period and biases tau
        e = e[:whole]
    k = np.arange(e.size)
    c = complex(np.sum(e * np.exp(-2j * np.pi * k / sps)))
    return -float(np.angle(c)) / (2.0 * np.pi)


def _cubic(samples: np.ndarray, x: np.ndarray) -> np.ndarray:
    x0 = np.floor(x).astype(np.int64)
    mu = x - x0

    def g(off: int) -> np.ndarray:
        return samples[np.clip(x0 + off, 0, samples.size - 1)]

    m1, p0, p1, p2 = g(-1), g(0), g(1), g(2)
    a0 = -0.5 * m1 + 1.5 * p0 - 1.5 * p1 + 0.5 * p2
    a1 = m1 - 2.5 * p0 + 2.0 * p1 - 0.5 * p2
    a2 = -0.5 * m1 + 0.5 * p1
    return ((a0 * mu + a1) * mu + a2) * mu + p0


def _clock_line_ratio(seg: np.ndarray, sps: int) -> float:
    e = seg.real.astype(np.float64) ** 2 + seg.imag.astype(np.float64) ** 2
    n = (e.size // sps) * sps
    if n < 2 * sps:
        return 0.0
    e = e[:n] - e[:n].mean()
    spec = np.abs(np.fft.rfft(e)) ** 2
    b = n // sps
    lo, hi = max(1, b - 16), min(spec.size - 1, b + 17)
    ref = np.concatenate([spec[lo : b - 1], spec[b + 2 : hi]])
    med = float(np.median(ref)) if ref.size else 0.0
    return float(spec[b]) / med if med > 0.0 else 0.0


_FLOOR_BLOCK = 1024  # fixed detection granularity (chunk-independent)
_THRESH_FRAC = 0.25  # active threshold vs the TRACKED peak; a per-block peak
# reads a gap's own noise top-decile as "active" (verified on pure AWGN)
_PEAK_RISE = 1.0 / 8.0  # fast, so a burst's true level is captured in ~1 block
_PEAK_DECAY = 1.0 / 64.0  # slow, so the peak survives ordinary gaps
_NOISE_GUARD = 12.0  # the peak never decays below this multiple of the
# noise floor: a threshold descending into the noise band fragments a gap
# into short "bursts" (noise bumps over noise dips) too brief for the
# clock-line gate to judge. This multiple puts the activity threshold ~5.7
# sigma above smoothed noise (a lower 8.0 measured ~2.9 sigma: hundreds of
# false micro-bursts over a long gap). The detectability cost is honest — a
# burst needs roughly >= 3 dB SNR to clear a sustained smoothed-power test.
_FLOOR_RISE = 1.0 / 8.0  # noise-floor EMA rates; the floor learns only from
_FLOOR_FALL = 1.0 / 4.0  # proven evidence (see _update_floor callers)
_BURST_CONTRAST = 4.0  # a short flushed region below this multiple of the
# proven floor is a noise excursion over the guarded threshold, not a
# burst. NOTE the dead band this creates with _NOISE_GUARD: detection
# triggers at 3x floor (0.25 * 12), so a genuine burst landing in
# (3x, 4x) — roughly 5-6 dB SNR — is detected, buffered, then zeroed;
# bursts_zeroed_low_contrast is its witness
_SMOOTH_SYMS = 1  # activity-envelope smoothing width
_RISE_SYMS = 1  # sustained-active symbols confirming a burst START
_FALL_SYMS = 2  # sustained-INactive symbols confirming a burst END: PSK's
# matched-filtered envelope never holds a legitimate in-burst dip this long
_MAX_REGION_SYMS = 512  # commit cadence for a region that never falls; far
# above real burst lengths so ordinary bursts always flush on FALL
_MIN_INTERIOR_SYMS = 8  # tau-interior floor: edge exclusion shrinks gradedly
# on short bursts instead of cliff-falling to a whole-burst (edge-biased) tau
_CLOCK_GATE_MIN_SYMS = 256  # regions at least this long are clock-line
# gated. Measured (RRC alpha=0.35, matched): noise ratio p99 ~7 / max ~17
# over 500 trials at 256-512 symbols; real QPSK >= 42 even at 5 dB SNR.
# Shorter real bursts cannot clear the bar reliably, so they pass ungated.
_CLOCK_LINE_MIN_RATIO = 25.0  # calibrated at _CLOCK_CAL_ALPHA; the line
# weakens with pulse-shaping rolloff (512-sym minima at 10 dB: 107/66/46/
# 33/27 at alpha .35/.25/.20/.15/.10 — roughly linear in alpha), so the
# threshold derates with alpha but never below the alpha-independent
# noise ceiling
_CLOCK_CAL_ALPHA = 0.35
_CLOCK_LINE_NOISE_CEILING = 20.0


def _block_peak(smoothed: np.ndarray) -> float:
    p90 = float(np.percentile(smoothed, 90))
    top = smoothed[smoothed > p90]
    return float(np.median(top)) if top.size else p90


def _activity_mask(smoothed: np.ndarray, peak: float) -> np.ndarray:
    return smoothed > _THRESH_FRAC * peak


def make_oerder_meyr(
    gr: Any, *, sps: float, span: int = 11, alpha: float = 0.35
) -> Any:
    stride = int(round(sps))
    if abs(sps - stride) > 1e-6:
        raise ValueError(f"oerder_meyr needs integer sps, got {sps}; resample upstream")
    if stride < 4:
        raise ValueError(
            f"oerder_meyr needs sps >= 4 (the |x|^2 clock line collapses "
            f"toward Nyquist below that), got {sps}"
        )
    if span < 1:
        raise ValueError(f"oerder_meyr needs span >= 1, got {span}")
    if not 0.1 <= alpha <= 1.0:
        raise ValueError(
            f"oerder_meyr needs alpha in [0.1, 1]: the |x|^2 clock line "
            f"scales with excess bandwidth and vanishes as alpha -> 0, "
            f"got {alpha}"
        )
    gate_ratio = max(
        _CLOCK_LINE_NOISE_CEILING,
        _CLOCK_LINE_MIN_RATIO * min(1.0, alpha / _CLOCK_CAL_ALPHA),
    )
    smooth_w = _SMOOTH_SYMS * stride
    rise_run = _RISE_SYMS * stride
    fall_run = _FALL_SYMS * stride
    max_region = _MAX_REGION_SYMS * stride
    gate_min = _CLOCK_GATE_MIN_SYMS * stride

    class _OerderMeyr(gr.basic_block):
        def __init__(self) -> None:
            gr.basic_block.__init__(
                self,
                name="oerder_meyr",
                in_sig=[np.complex64],
                out_sig=[np.complex64],
            )
            self._out = OutQueue(np.complex64)
            self._pending = np.empty(0, np.complex64)  # < _FLOOR_BLOCK carryover
            self._abs = 0  # absolute index of the next input sample to classify
            self._m = 0  # next global symbol index; its base is m*stride
            self._burst: list[np.ndarray] = []
            self._burst_len = 0
            self._burst_start = 0  # absolute sample index this burst began at
            self._in_burst = False
            self._quiet = 0
            self._peak = 0.0  # persistent tracked signal-level peak
            self._floor = 0.0  # noise-floor estimate; 0 until proven
            self.eof_probe: Any = None  # set by build wiring; None => no flush
            self._eof_done = False
            self.diagnostics = {
                "bursts_flushed": 0,
                "bursts_truncated_at_eof": 0,
                "bursts_partial_edge_exclusion": 0,
                "bursts_zeroed_low_contrast": 0,
                "regions_capped": 0,
                "regions_zeroed": 0,
            }

        def forecast(self, noutput_items: int, ninputs: int) -> list[int]:
            return forecast_drain(self._out.pending, ninputs)

        def general_work(self, input_items: Any, output_items: Any) -> int:
            inp = input_items[0]
            if len(inp):
                self._pending = np.concatenate(
                    [self._pending, np.asarray(inp, np.complex64)]
                )
                self.consume(0, len(inp))
                while len(self._pending) >= _FLOOR_BLOCK:
                    block, self._pending = (
                        self._pending[:_FLOOR_BLOCK],
                        self._pending[_FLOOR_BLOCK:],
                    )
                    self._process_block(block)
            if not self._eof_done and self._eof_final():
                self._eof_done = True
                self._flush_tail()
            return self._out.drain(output_items[0])

        def _eof_final(self) -> bool:
            p = self.eof_probe
            return (
                p is not None
                and p.expected_items is not None
                and int(self.nitems_read(0)) >= p.expected_items
            )

        def _flush_tail(self) -> None:
            if len(self._pending):
                self._process_block(self._pending)
                self._pending = np.empty(0, np.complex64)
            if self._in_burst:
                if self._flush_burst(self._abs, self._abs - self._quiet):
                    self.diagnostics["bursts_truncated_at_eof"] += 1
                self._in_burst = False

        def _update_floor(self, med: float) -> None:
            if med <= 0.0:
                return
            if self._floor == 0.0:
                self._floor = med
            else:
                rate = _FLOOR_RISE if med > self._floor else _FLOOR_FALL
                self._floor += rate * (med - self._floor)

        def _update_peak(self, smoothed: np.ndarray) -> float:
            blk_hi = _block_peak(smoothed)
            if self._peak == 0.0:
                self._peak = blk_hi if blk_hi > 0.0 else 1e-9
            else:
                rate = _PEAK_RISE if blk_hi > self._peak else _PEAK_DECAY
                self._peak += rate * (blk_hi - self._peak)
            # a block whose high sits far under the trained peak is a proven
            # gap: its level is trustworthy noise, unlike a cold-start block
            # (blk_hi ~ peak), whose level could equally be a continuous
            # signal — that case is settled by the clock-line gate instead
            if blk_hi < 0.5 * self._peak:
                self._update_floor(float(np.median(smoothed)))
            if self._floor > 0.0:
                self._peak = max(self._peak, _NOISE_GUARD * self._floor)
            return self._peak

        def _process_block(self, block: np.ndarray) -> None:
            power = (
                block.real.astype(np.float64) ** 2 + block.imag.astype(np.float64) ** 2
            )
            smoothed = _trailing_mean(power, smooth_w)
            active = _activity_mask(smoothed, self._update_peak(smoothed))
            base = self._abs
            n = len(block)
            i = 0
            while i < n:
                if not self._in_burst:
                    rises = _sustained_runs(active[i:], rise_run)
                    if len(rises) == 0:
                        self._emit_idle(base, n)
                        break
                    start = i + int(rises[0])
                    self._emit_idle(base, start)
                    self._start_burst(base + start)
                    i = start
                    continue
                j = self._scan_fall(active, i)
                self._burst.append(block[i:j])
                self._burst_len += j - i
                i = j
                if self._quiet >= fall_run:
                    self._flush_burst(base + j, base + j - fall_run)
                    self._in_burst = False
                elif self._burst_len >= max_region:
                    # still active, just too long to keep buffering: commit
                    # this chunk and continue the SAME region without
                    # re-detection. active_end == end: a trailing sub-fall
                    # dip here is an envelope dip of a signal that CONTINUES,
                    # not a confirmed-idle tail — zeroing it would drop one
                    # real symbol per cap on a continuous signal
                    if self._flush_burst(base + j, base + j):
                        self.diagnostics["regions_capped"] += 1
                    # re-seed the next chunk with the last stride samples:
                    # the boundary symbol's timed instant can precede the
                    # chunk split (tau < 0 against the grid), and without
                    # this overlap it would be zeroed at every cap
                    carry = np.concatenate(self._burst)[-stride:]
                    self._start_burst(base + j - len(carry))
                    self._burst = [carry]
                    self._burst_len = len(carry)
            self._abs = base + n

        def _scan_fall(self, active: np.ndarray, i: int) -> int:
            # vectorized equivalent of counting consecutive inactive samples
            # (carrying self._quiet across blocks) until fall_run completes
            seg = active[i:]
            idx = np.arange(seg.size)
            run = idx - np.maximum.accumulate(np.where(seg, idx, -1))
            if self._quiet:
                lead = int(seg.argmax()) if seg.any() else seg.size
                run[:lead] += self._quiet
            hits = np.flatnonzero(run >= fall_run)
            if hits.size:
                self._quiet = fall_run
                return i + int(hits[0]) + 1
            self._quiet = int(run[-1])
            return i + seg.size

        def _start_burst(self, start_abs: int) -> None:
            self._burst = []
            self._burst_len = 0
            self._burst_start = start_abs
            self._in_burst = True
            self._quiet = 0

        def _advance_grid(self, limit: int) -> np.ndarray:
            a = limit - self._m * stride
            if a <= 0:
                return np.empty(0, np.int64)
            count = (a + stride - 1) // stride
            bases = (self._m + np.arange(count, dtype=np.int64)) * stride
            self._m += int(count)
            return bases

        def _emit_idle(self, base: int, hi: int) -> None:
            bases = self._advance_grid(base + hi)
            if bases.size:
                self._out.push(np.zeros(bases.size, np.complex64))

        def _burst_tau(self, seg: np.ndarray) -> float:
            nsym = seg.size // stride
            excl = min(span, max(0, (nsym - _MIN_INTERIOR_SYMS) // 2))
            if excl < span:
                self.diagnostics["bursts_partial_edge_exclusion"] += 1
            lo = excl * stride
            return estimate_tau(seg[lo : seg.size - lo], stride)

        def _flush_burst(self, end: int, active_end: int) -> bool:
            """Commit whatever's buffered as one tau-estimated chunk covering
            the grid up to `end`; slots at/after `active_end` (the caller's
            last-known-active sample) emit the idle zero. Returns whether a
            substantive commit happened — the degenerate branches below
            still consume and zero-fill their grid slots but report False so
            callers don't count them as caps/truncations. Purely a commit:
            the caller owns the _in_burst transition, since a flush can mean
            "region ended" (fall) or "region continues, capped"."""
            seg = (
                np.concatenate(self._burst)
                if self._burst
                else np.empty(0, np.complex64)
            )
            s = self._burst_start
            bases = self._advance_grid(end)
            if bases.size == 0:
                return False
            if seg.size == 0:
                self._out.push(np.zeros(bases.size, np.complex64))
                return False
            power = seg.real.astype(np.float64) ** 2 + seg.imag.astype(np.float64) ** 2
            if seg.size >= gate_min:
                if _clock_line_ratio(seg, stride) < gate_ratio:
                    self.diagnostics["regions_zeroed"] += 1
                    # clock-proven noise: its level seeds/updates the floor,
                    # and via the _NOISE_GUARD clamp lifts the peak clear of
                    # the noise band (ends a cold start's noise-reads-active
                    # state)
                    self._update_floor(
                        float(np.median(_trailing_mean(power, smooth_w)))
                    )
                    self._out.push(np.zeros(bases.size, np.complex64))
                    return True
            elif self._floor > 0.0:
                level = _block_peak(_trailing_mean(power, smooth_w))
                if level < _BURST_CONTRAST * self._floor:
                    self.diagnostics["bursts_zeroed_low_contrast"] += 1
                    self._out.push(np.zeros(bases.size, np.complex64))
                    return True
            self.diagnostics["bursts_flushed"] += 1
            tau = self._burst_tau(seg)
            # bases[0]-s is the first owned symbol's nominal local offset; a
            # burst start rarely lands on the global grid, so snap it to the
            # tau comb {(n + tau)*stride} rather than adding tau*stride to it
            n0 = round(float(bases[0] - s) / stride - tau)
            instants = (n0 + tau) * stride + np.arange(
                bases.size, dtype=np.float64
            ) * stride
            sym = _cubic(seg.astype(np.complex128), instants).astype(np.complex64)
            # grid slots before the buffer or past the last active sample
            # hold no burst energy: emit the idle zero, not interpolated
            # edge samples / noise
            sym[(instants < 0.0) | (bases >= active_end)] = 0
            self._out.push(sym)
            return True

    return _OerderMeyr()
