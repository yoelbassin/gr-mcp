from __future__ import annotations

from typing import Any

import numpy as np


def _complex_preamble(preamble_i: list[float], preamble_q: list[float]) -> np.ndarray:
    return (
        np.asarray(preamble_i, dtype=np.float64)
        + 1j * np.asarray(preamble_q, dtype=np.float64)
    ).astype(np.complex64)


def _ramp(n: int) -> np.ndarray:
    """Constant-modulus alternating +/-1 settle pattern: transition-rich for the
    Gardner TED and decorrelated from a random preamble, so it never false-peaks."""
    r = np.ones(int(n), dtype=np.complex64)
    r[1::2] = -1.0
    return r


def make_sym_prepend(
    gr: Any, preamble_i: list[float], preamble_q: list[float], pad_symbols: int
) -> Any:
    """TX: emit [ramp ++ preamble] once, then pass the payload through 1:1."""
    prepend = np.concatenate(
        [_ramp(pad_symbols), _complex_preamble(preamble_i, preamble_q)]
    )

    class _SymPrepend(gr.basic_block):
        def __init__(self) -> None:
            gr.basic_block.__init__(
                self,
                name="sym_prepend",
                in_sig=[np.complex64],
                out_sig=[np.complex64],
            )
            self._pre = prepend
            self._done = False

        def forecast(self, noutput_items: int, ninputs: int) -> list[int]:
            return [0] * ninputs if not self._done else [noutput_items] * ninputs

        def general_work(self, input_items: Any, output_items: Any) -> int:
            out = output_items[0]
            o = 0
            if not self._done:
                k = min(len(out), len(self._pre))
                out[:k] = self._pre[:k]
                self._pre = self._pre[k:]
                o += k
                if len(self._pre) == 0:
                    self._done = True
                if o == len(out):
                    return o
            x = input_items[0]
            m = min(len(out) - o, len(x))
            if m:
                out[o : o + m] = x[:m]
                self.consume(0, m)
            return o + m

    return _SymPrepend()


def _acquire_lock(
    seg: np.ndarray, pre: np.ndarray, min_buf: int, threshold: float
) -> tuple[int, float] | None:
    n = len(pre)
    max_lag = min(len(seg) - n, min_buf - n) + 1
    if max_lag < 1:
        return None
    corr = np.array([np.vdot(pre, seg[lag : lag + n]) for lag in range(max_lag)])
    k = int(np.argmax(np.abs(corr)))
    if np.abs(corr[k]) / (float(np.mean(np.abs(corr))) + 1e-9) < threshold:
        return None
    return k, float(np.angle(corr[k]))


def make_sym_acquire(
    gr: Any,
    preamble_i: list[float],
    preamble_q: list[float],
    pad_symbols: int,
    threshold: float,
) -> Any:
    """RX: buffer; correlate vs the preamble; on a sharp peak lock (lag, phase),
    strip the ramp+preamble, then emit payload * exp(-j phase). Flushes the
    buffered tail at EOF (file-based pipeline: tags do not survive file_sink, so
    the strip is physical). The preamble must be drawn from the modulation's
    constellation so the carrier loop tracks it cleanly — an off-constellation
    preamble is mistracked by higher-order PSK and yields a wrong phase lock."""
    pre = _complex_preamble(preamble_i, preamble_q).astype(np.complex128)
    n = len(pre)
    min_buf = max(int(pad_symbols) + 2 * n, 4 * n)

    class _SymAcquire(gr.basic_block):
        def __init__(self) -> None:
            gr.basic_block.__init__(
                self,
                name="sym_acquire",
                in_sig=[np.complex64],
                out_sig=[np.complex64],
            )
            self._buf = np.empty(0, dtype=np.complex64)
            self._locked = False
            self._phase = 0.0

        def forecast(self, noutput_items: int, ninputs: int) -> list[int]:
            # Locked: run without input only while there is buffered payload to
            # flush; once drained, require input again so EOF propagates from
            # upstream instead of self-terminating on a momentarily empty input.
            if not self._locked:
                return [noutput_items] * ninputs
            return [0] * ninputs if self._buf.size else [1] * ninputs

        def general_work(self, input_items: Any, output_items: Any) -> int:
            x = input_items[0]
            if len(x):
                self._buf = np.concatenate([self._buf, x])
                self.consume(0, len(x))
            if not self._locked:
                if len(self._buf) < min_buf:
                    return 0
                lock = _acquire_lock(
                    self._buf.astype(np.complex128), pre, min_buf, threshold
                )
                if lock is None:
                    return 0
                k, self._phase = lock
                self._locked = True
                self._buf = self._buf[k + n :]
            out = output_items[0]
            m = min(len(out), len(self._buf))
            if m:
                out[:m] = self._buf[:m] * np.exp(-1j * self._phase)
                self._buf = self._buf[m:]
            return m

    return _SymAcquire()
