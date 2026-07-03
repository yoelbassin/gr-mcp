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


def sym_prefix(
    preamble_i: list[float], preamble_q: list[float], pad_symbols: int
) -> np.ndarray:
    return np.concatenate(
        [_ramp(pad_symbols), _complex_preamble(preamble_i, preamble_q)]
    )


def make_sym_strip(gr: Any, *, n_pre: int) -> Any:
    """RX back half of preamble_sync: consume the phase_est tag emitted by an
    upstream corr_est_cc, latch the FIRST detection (a later stronger payload
    replica cannot steal the lock), strip pad+preamble, and derotate the payload
    by exp(-j*phase_est). corr_est_cc delays its pass-through by the correlator
    length, so the payload begins n_pre+1 samples past the tagged offset."""
    pmt = gr.pmt

    class _SymStrip(gr.basic_block):
        def __init__(self) -> None:
            gr.basic_block.__init__(
                self,
                name="sym_strip",
                in_sig=[np.complex64],
                out_sig=[np.complex64],
            )
            self._locked = False
            self._rot: complex = 1.0 + 0j
            self._payload_start = 0

        def forecast(self, noutput_items: int, ninputs: int) -> list[int]:
            return [noutput_items] * ninputs

        def general_work(self, input_items: Any, output_items: Any) -> int:
            x = input_items[0]
            base = self.nitems_read(0)
            if not self._locked:
                for t in self.get_tags_in_window(0, 0, len(x)):
                    if pmt.symbol_to_string(t.key) == "phase_est":
                        self._rot = complex(np.exp(-1j * pmt.to_double(t.value)))
                        self._payload_start = t.offset + n_pre + 1
                        self._locked = True
                        break
                if not self._locked:
                    self.consume(0, len(x))
                    return 0
            s = max(0, self._payload_start - base)
            if s >= len(x):
                self.consume(0, len(x))
                return 0
            out = output_items[0]
            k = min(len(out), len(x) - s)
            out[:k] = (x[s : s + k] * self._rot).astype(np.complex64)
            self.consume(0, s + k)
            return int(k)

    return _SymStrip()
