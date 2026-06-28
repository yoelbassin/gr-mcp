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
    the strip is physical)."""
    pre = _complex_preamble(preamble_i, preamble_q)
    length = len(pre)
    min_buf = max(int(pad_symbols) + 2 * length, 4 * length)

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
            # locked: drain the internal buffer with no new input, so the
            # scheduler keeps calling work at EOF to flush the payload tail.
            return [0] * ninputs if self._locked else [noutput_items] * ninputs

        def general_work(self, input_items: Any, output_items: Any) -> int:
            x = input_items[0]
            if len(x):
                self._buf = np.concatenate([self._buf, x])
                self.consume(0, len(x))
            if not self._locked:
                if len(self._buf) < min_buf:
                    return 0
                seg = self._buf
                corr = np.array(
                    [
                        np.vdot(pre, seg[lag : lag + length])
                        for lag in range(len(seg) - length)
                    ]
                )
                k = int(np.argmax(np.abs(corr)))
                if np.abs(corr[k]) / (float(np.mean(np.abs(corr))) + 1e-9) < threshold:
                    return 0
                self._locked = True
                self._phase = float(np.angle(corr[k]))
                self._buf = self._buf[k + length :]
            out = output_items[0]
            m = min(len(out), len(self._buf))
            if m:
                out[:m] = self._buf[:m] * np.exp(-1j * self._phase)
                self._buf = self._buf[m:]
            elif self._locked and len(x) == 0:
                return -1  # WORK_DONE: buffer drained, input exhausted
            return m

    return _SymAcquire()
