"""Open-loop Oerder & Meyr symbol timing: a non-data-aided feedforward tau
estimate (spectral-line phase of the squared envelope) plus cubic
resampling, for the same class of problem `burst.py` solves for OOK/PPM -
a bursty/short signal that never gives a tracking LOOP time to converge.
Stock `symbol_sync_cc`/`pfb_clock_sync_ccf` are second-order loops: they
settle over many symbols and rail (mistrack) before a short burst ends. The
Oerder & Meyr estimator has no settling time - one FFT-line phase per block
- so it fits bursts a loop cannot. `estimate_tau`/`resample_symbols` are the
pure-numpy estimator and interpolator (this module's functions above);
`make_oerder_meyr` below streams them: a global symbol grid at m*sps
(mirrors `burst_sampler._k`) so window boundaries never insert or drop a
symbol, a fresh tau per window of `window` symbols (tracks slow SFO across
windows), and no in-block volk or feedback loop, so identical input yields
identical output regardless of how the caller chunks it."""

from __future__ import annotations

from typing import Any

import numpy as np

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


def make_oerder_meyr(gr: Any, *, sps: float, window: int = 64) -> Any:
    if sps < 1.0:
        raise ValueError(f"oerder_meyr needs sps >= 1, got {sps}")
    if window < 1:
        raise ValueError(f"oerder_meyr needs window >= 1, got {window}")
    stride = int(round(sps))
    win_syms = int(window)
    margin = 2 * stride  # retained trailing/leading history for the cubic

    class _OerderMeyr(gr.basic_block):
        def __init__(self) -> None:
            gr.basic_block.__init__(
                self,
                name="oerder_meyr",
                in_sig=[np.complex64],
                out_sig=[np.complex64],
            )
            self._out = OutQueue(np.complex64)
            self._pending = np.empty(0, np.complex64)
            self._abs = 0  # absolute sample index of self._pending[0]
            self._m = 0  # next global symbol index to emit; always a
            # multiple of win_syms outside of _flush_tail
            self.eof_probe: Any = None  # set by build wiring; None => no flush
            self._eof_done = False

        def forecast(self, noutput_items: int, ninputs: int) -> list[int]:
            return forecast_drain(self._out.pending, ninputs)

        def general_work(self, input_items: Any, output_items: Any) -> int:
            inp = input_items[0]
            if len(inp):
                self._pending = np.concatenate(
                    [self._pending, np.asarray(inp, np.complex64)]
                )
                self.consume(0, len(inp))
                self._process_ready()
            if not self._eof_done and self._eof_final():
                self._eof_done = True
                self._flush_tail()
            return self._out.drain(output_items[0])

        def _eof_final(self) -> bool:
            """Mirrors burst_sampler._eof_final: true only once every input
            sample this block will ever see has been consumed (source-
            relative count proven by the probe), so the withheld tail
            window is safe to commit. Without expected_items the sim-pad
            withhold stands - see test_no_flush_without_a_finality_probe."""
            p = self.eof_probe
            return (
                p is not None
                and p.expected_items is not None
                and int(self.nitems_read(0)) >= p.expected_items
            )

        def _process_ready(self) -> None:
            # Global symbol grid: window w covers symbols [m, m+win_syms) at
            # nominal samples [m*stride, (m+win_syms)*stride) - fixed by m
            # alone, so no chunk boundary can ever insert or drop a symbol.
            available = self._abs + self._pending.size
            while True:
                start_abs = self._m * stride
                end_abs = start_abs + win_syms * stride
                if end_abs + margin > available:
                    break
                self._emit_window(start_abs, end_abs, available)
            self._trim(self._m * stride - margin)

        def _emit_window(self, start_abs: int, end_abs: int, available: int) -> None:
            raw = self._pending[start_abs - self._abs : end_abs - self._abs]
            tau = estimate_tau(raw, stride)
            buf_lo = max(self._abs, start_abs - margin)
            buf_hi = min(available, end_abs + margin)
            buf = self._pending[buf_lo - self._abs : buf_hi - self._abs]
            sym = resample_symbols(buf, stride, tau, start=float(start_abs - buf_lo))
            self._out.push(sym[:win_syms])
            self._m += win_syms

        def _trim(self, keep_from_abs: int) -> None:
            keep_from_abs = max(keep_from_abs, self._abs)
            drop = keep_from_abs - self._abs
            if drop > 0:
                self._pending = self._pending[drop:]
                self._abs += drop

        def _flush_tail(self) -> None:
            """At true EOF, commit whatever symbols the retained tail still
            holds - a window that streaming withheld only for lack of
            trailing margin, plus any final partial window - as one
            tau-estimated block (bursts-style _flush_tail)."""
            total_abs = self._abs + self._pending.size
            start_abs = self._m * stride
            remaining = (total_abs - start_abs) // stride
            if remaining > 0:
                end_abs = start_abs + remaining * stride
                raw = self._pending[start_abs - self._abs : end_abs - self._abs]
                tau = estimate_tau(raw, stride)
                buf_lo = max(self._abs, start_abs - margin)
                buf = self._pending[buf_lo - self._abs :]
                sym = resample_symbols(
                    buf, stride, tau, start=float(start_abs - buf_lo)
                )
                self._out.push(sym[:remaining])
                self._m += remaining
            self._pending = np.empty(0, np.complex64)
            self._abs = total_abs

    return _OerderMeyr()
