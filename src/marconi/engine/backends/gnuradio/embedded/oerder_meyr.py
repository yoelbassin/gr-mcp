"""Open-loop Oerder & Meyr symbol timing: a non-data-aided feedforward tau
estimate (spectral-line phase of the squared envelope) plus cubic
resampling, for the same class of problem `burst.py` solves for OOK/PPM -
a bursty/short signal that never gives a tracking LOOP time to converge.
Stock `symbol_sync_cc`/`pfb_clock_sync_ccf` are second-order loops: they
settle over many symbols and rail (mistrack) before a short burst ends. The
Oerder & Meyr estimator has no settling time - one FFT-line phase per burst
- so it fits bursts a loop cannot. `estimate_tau`/`_cubic` are the pure-
numpy estimator and interpolator (this module's functions above);
`make_oerder_meyr` below streams them.

A single tau per FIXED WINDOW (the shipped-then-reworked design) under-
recovers short bursts: measured R4 (constellation cleanliness) was 0.65 at
40-symbol bursts and 0.78 at 80-symbol bursts, flat regardless of window
size, because the RRC matched filter's ~`span`-symbol transient at every
burst EDGE biases any burst-spanning tau estimate - no fixed window can
dodge it, since a window boundary is not a burst boundary. The fix - proven
in `.superpowers/sdd/2026-08-09-open-loop-complex-timing/task-2-rework-
prototype.py` - is per-burst framing: detect each burst on |x|^2 (the same
rise/fall-vs-a-tracked-reference control flow `burst_sampler` uses for OOK's
raw envelope, adapted to PSK's power - see `_update_peak` below for why the
reference is a persistent, asymmetric-EMA-tracked peak rather than a per-
block statistic or a floor), buffer it, and estimate ONE tau over the
burst's EDGE-EXCLUDED interior (drop the
first/last `span` symbols, the RRC transient) before cubic-resampling the
whole burst at that tau. Symbols land on a global grid at m*stride (mirrors
`burst_sampler._k`) that never resets at a burst boundary, with idle/gap
spans emitting a literal zero per grid position (unlike burst_sampler's
scaled-noise idle chip - see `_emit_idle` for why a complex symbol stream
wants an unambiguous "nothing here" rather than a small-but-nonzero,
random-phase sample) - so the emitted count stays exactly input_len //
stride and a burst boundary can never insert or drop a symbol. Determinism
follows `burst_sampler`: fixed-size internal blocks, no feedback loop, no
volk in-block, so identical input yields identical output regardless of how
the caller chunks it. The block withholds an unfinished burst tail at EOF
(suite convention: sims pad).

Idle emission is gated RELATIVE TO A TRACKED SIGNAL-LEVEL PEAK (see
`_update_peak`), not an unconditional guarantee: a genuine gap between real
bursts is correctly read as idle and emits zero, but a capture that carries
no real signal at ALL - pure noise, cover to cover - is not detected as
silence, because a constant-envelope signal and constant-envelope noise are
locally identical in power; telling them apart needs clock-line detection,
out of scope here. That input is processed into garbage symbols instead,
which is the downstream quality layer's job to flag as no-signal - this
block never manufactures a false decode by mistaking noise for zero.

A burst-only fall trigger has a blind spot: a CONTINUOUS transmission is
one active region that never falls, and in a ratio-1 engine chain (this
block's rate matches its neighbors, so the graph's file source feeds it
only indirectly) there is no `expected_items` either - nothing ever tells
the block the region is done, so it withholds forever and emits nothing.
`_MAX_REGION_SYMS` closes this the same way `burst_sampler._MAX_BURST_CHIPS`
bounds a pathological OOK burst: once a region has been buffering active
samples this long with no fall, commit its buffered tau now and keep
accumulating a fresh chunk from there - same region, same `_in_burst`, no
re-detection. The cap sits far above real burst lengths so it only ever
fires on long/continuous signals, never splits an actual burst, and each
capped chunk gets its own tau, which incidentally tracks slow SFO too."""

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


def resample_symbols(
    samples: np.ndarray, sps: int, tau: float, start: float = 0.0
) -> np.ndarray:
    nsym = int((samples.size - start) // sps)
    if nsym <= 0:
        return np.empty(0, np.complex64)
    instants = start + np.arange(nsym) * float(sps) + tau * sps
    return _cubic(samples.astype(np.complex128), instants).astype(np.complex64)


_FLOOR_BLOCK = 1024  # fixed detection granularity (chunk-independent)
_THRESH_FRAC = 0.25  # active threshold, as a fraction of the TRACKED PEAK
# (see _update_peak) - matches the validated prototype's per_burst
# threshold (0.25 * the median of the block's top decile), just against a
# peak that persists across blocks instead of being recomputed from each
# block alone. A PER-BLOCK peak was tried first and is wrong: a gap longer
# than one _FLOOR_BLOCK (or any pure-noise stretch) has its own top decile
# too, so a threshold relative to THAT reads the gap's own noise as
# "active" - verified, pure AWGN false-triggers a burst in ~100% of blocks
# under a per-block peak. This matters for the block's actual use case:
# real inter-burst gaps in a bursty PSK capture are routinely longer than
# one block.
_PEAK_RISE = 1.0 / 8.0  # _peak EMA rate when a block's own high-power level
# exceeds the tracked peak - fast, so a burst's true level is captured in
# essentially one block rather than many
_PEAK_DECAY = 1.0 / 64.0  # ... when it's below - slow, so the peak
# SURVIVES a gap (or any pure-noise stretch) instead of collapsing toward
# that block's own level; a continuous signal, whose blocks never read
# below the peak, never decays it at all and stays classified active
# throughout
_SMOOTH_SYMS = 1  # activity-envelope smoothing width, one symbol (matches
# the prototype's uniform_filter1d(p, SPS))
_RISE_SYMS = 1  # sustained-active symbols to confirm a burst START
_FALL_SYMS = 2  # sustained-INactive symbols to confirm a burst END - short,
# because PSK's matched-filtered envelope (unlike OOK's on/off chips) never
# holds a legitimate in-burst dip this long; keeping it short also keeps
# the noise tail folded into a flushed burst well inside the `span`-symbol
# edge exclusion the tau estimate drops, and out of the burst's resampled
# output (see _emit_idle for why that tail matters beyond the tau estimate)
_MAX_REGION_SYMS = 512  # cap: an active region (burst OR a long/continuous
# signal that never falls) this long commits its buffered tau NOW and keeps
# accumulating a fresh chunk from there - mirrors burst_sampler's
# _MAX_BURST_CHIPS, but the driving case here isn't a pathological burst,
# it's an ordinary CONTINUOUS transmission: with no fall and no finality
# probe (a ratio-1 engine chain gives the block no expected_items), a
# region that never caps would buffer - and withhold - forever. Must stay
# well above real burst lengths (tens-low-hundreds of symbols) so normal
# bursts still flush on FALL, never on the cap; each capped chunk gets its
# own tau, which incidentally also tracks slow SFO across a long capture
_MIN_INTERIOR_SYMS = 2  # tau-estimation interior shorter than this falls
# back to the whole burst


def _block_peak(smoothed: np.ndarray) -> float:
    p90 = float(np.percentile(smoothed, 90))
    top = smoothed[smoothed > p90]
    return float(np.median(top)) if top.size else p90


def _activity_mask(smoothed: np.ndarray, peak: float) -> np.ndarray:
    return smoothed > _THRESH_FRAC * peak


def make_oerder_meyr(gr: Any, *, sps: float, window: int = 64, span: int = 11) -> Any:
    if sps < 1.0:
        raise ValueError(f"oerder_meyr needs sps >= 1, got {sps}")
    if window < 1:
        raise ValueError(f"oerder_meyr needs window >= 1, got {window}")
    if span < 1:
        raise ValueError(f"oerder_meyr needs span >= 1, got {span}")
    stride = int(round(sps))
    edge = span * stride
    min_interior = _MIN_INTERIOR_SYMS * stride
    smooth_w = _SMOOTH_SYMS * stride
    rise_run = _RISE_SYMS * stride
    fall_run = _FALL_SYMS * stride
    max_region = _MAX_REGION_SYMS * stride
    # `window` is unused now that one tau is estimated per burst rather than
    # per fixed window; kept as a parameter for call-site/API stability.

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
            self._m = 0  # next global symbol index; its base is round(m*stride)
            self._burst: list[np.ndarray] = []
            self._burst_len = 0
            self._burst_start = 0  # absolute sample index this burst began at
            self._in_burst = False
            self._quiet = 0
            self._peak = 0.0  # persistent tracked signal-level peak
            self.eof_probe: Any = None  # set by build wiring; None => no flush
            self._eof_done = False
            self.diagnostics = {
                "bursts_flushed": 0,
                "bursts_truncated_at_eof": 0,
                "regions_capped": 0,
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
            """Mirrors burst_sampler._eof_final: true only once every input
            sample this block will ever see has been consumed (source-
            relative count proven by the probe), so the withheld tail is
            safe to commit. Without expected_items the sim-pad withhold
            stands - see test_no_flush_without_a_finality_probe."""
            p = self.eof_probe
            return (
                p is not None
                and p.expected_items is not None
                and int(self.nitems_read(0)) >= p.expected_items
            )

        def _flush_tail(self) -> None:
            """At true EOF, drain the withheld tail a real (unpadded)
            capture leaves behind: process the sub-block _pending remainder,
            then commit any burst still open (bursts-style _flush_tail)."""
            if len(self._pending):
                self._process_block(self._pending)
                self._pending = np.empty(0, np.complex64)
            if self._in_burst:
                if self._flush_burst(self._abs):
                    self.diagnostics["bursts_truncated_at_eof"] += 1
                self._in_burst = False

        def _update_peak(self, smoothed: np.ndarray) -> float:
            blk_hi = _block_peak(smoothed)
            if self._peak == 0.0:
                self._peak = blk_hi
            else:
                rate = _PEAK_RISE if blk_hi > self._peak else _PEAK_DECAY
                self._peak += rate * (blk_hi - self._peak)
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
                j = i
                while j < n:
                    self._quiet = self._quiet + 1 if not active[j] else 0
                    j += 1
                    if self._quiet >= fall_run:
                        break
                self._burst.append(block[i:j])
                self._burst_len += j - i
                i = j
                if self._quiet >= fall_run:
                    self._flush_burst(base + j)
                    self._in_burst = False
                elif self._burst_len >= max_region:
                    # Still active - just too long to keep buffering (an
                    # ordinary burst never gets here; only a long/continuous
                    # region does). Commit this chunk and immediately start
                    # the next one from here, WITHOUT going back through
                    # rise detection - a fresh chunk of the same region, not
                    # a new region.
                    if self._flush_burst(base + j):
                        self.diagnostics["regions_capped"] += 1
                    self._start_burst(base + j)
            self._abs = base + n

        def _start_burst(self, start_abs: int) -> None:
            self._burst = []
            self._burst_len = 0
            self._burst_start = start_abs
            self._in_burst = True
            self._quiet = 0

        def _advance_grid(self, limit: int) -> np.ndarray:
            # Absolute base positions round(m*stride) for every global
            # symbol m owned up to (not including) `limit`, advancing self._m
            # past them - the global grid never resets at a burst boundary.
            a = limit - self._m * stride
            if a <= 0:
                return np.empty(0, np.int64)
            count = (a + stride - 1) // stride
            bases = (self._m + np.arange(count, dtype=np.int64)) * stride
            self._m += int(count)
            return bases

        def _emit_idle(self, base: int, hi: int) -> None:
            # Literal zero, not a raw noisy sample: a zero can never be
            # mistaken for a real constellation point downstream, whereas a
            # small-but-nonzero noise sample still carries a random PHASE
            # that pollutes any downstream statistic spanning a burst
            # boundary (e.g. a differential decode straddling idle). Zero is
            # also the cleanest possible "magnitude-gated downstream" value.
            # Whether a span COUNTS as idle in the first place is decided by
            # the caller's persistent-peak-gated activity mask (_update_peak
            # / _activity_mask), not by anything here - this only fires once
            # that gate has already said "nothing real is happening".
            bases = self._advance_grid(base + hi)
            if bases.size:
                self._out.push(np.zeros(bases.size, np.complex64))

        def _burst_tau(self, seg: np.ndarray) -> float:
            lo, hi = edge, seg.size - edge
            interior = seg[lo:hi] if hi - lo >= min_interior else seg
            return estimate_tau(interior, stride)

        def _flush_burst(self, end: int) -> bool:
            """Commit whatever's buffered as one tau-estimated chunk, and
            report whether there was anything to commit. Purely a commit: it
            does NOT touch _in_burst/_quiet - the caller owns that
            transition, since a flush can mean either "this region just
            ended" (exit burst mode) or "this region is still going, just
            capped" (start a fresh chunk immediately, same region). The
            bool return lets a caller avoid counting an empty commit (e.g. a
            cap that landed exactly on the last sample, immediately
            restarted, then found nothing buffered at EOF) as a real flush."""
            seg = (
                np.concatenate(self._burst)
                if self._burst
                else np.empty(0, np.complex64)
            )
            s = self._burst_start
            bases = self._advance_grid(end)
            if bases.size == 0 or seg.size == 0:
                return False
            self.diagnostics["bursts_flushed"] += 1
            tau = self._burst_tau(seg)
            # tau is the offset (in [-0.5, 0.5) symbols) from LOCAL INDEX 0 of
            # seg to the nearest true symbol peak - the whole comb of peaks is
            # {(n + tau)*stride : n integer}. bases[0]-s is this burst's first
            # owned GLOBAL symbol's nominal (untimed) local offset, which
            # generally isn't itself a multiple of stride (a burst start
            # rarely lands on the global grid), so it must be snapped to the
            # nearest comb point rather than just added to tau*stride.
            n0 = round(float(bases[0] - s) / stride - tau)
            instant0 = (n0 + tau) * stride
            instants = instant0 + np.arange(bases.size, dtype=np.float64) * stride
            sym = _cubic(seg.astype(np.complex128), instants).astype(np.complex64)
            self._out.push(sym)
            return True

    return _OerderMeyr()
