from __future__ import annotations

from typing import Any

import numpy as np


def make_sym_strip(gr: Any, *, n_pre: int) -> Any:
    """RX back half of preamble_sync: consume the phase_est tags emitted by an
    upstream corr_est_cc. Every detection (re-)arms the strip — a capture with
    N bursts gets N per-burst phase estimates, each stripping its own
    pad+preamble; false peaks are corr_est's threshold's job. corr_est_cc
    delays its pass-through by the correlator length, so a payload begins
    n_pre+1 samples past its tagged offset."""
    pmt = gr.pmt

    class _SymStrip(gr.basic_block):
        def __init__(self) -> None:
            gr.basic_block.__init__(
                self,
                name="sym_strip",
                in_sig=[np.complex64],
                out_sig=[np.complex64],
            )
            self._armed = False
            self._rot: complex = 1.0 + 0j
            self._skip_to = 0  # absolute input offset where the payload resumes

        def forecast(self, noutput_items: int, ninputs: int) -> list[int]:
            return [noutput_items] * ninputs

        def _emit(self, x: Any, out: Any, r: int, upto: int, w: int) -> tuple[int, int]:
            if not self._armed:
                return upto, w
            k = min(upto - r, len(out) - w)
            out[w : w + k] = (x[r : r + k] * self._rot).astype(np.complex64)
            return r + k, w + k

        def general_work(self, input_items: Any, output_items: Any) -> int:
            x = input_items[0]
            out = output_items[0]
            base = self.nitems_read(0)
            n = len(x)
            r = min(n, max(0, self._skip_to - base))
            w = 0
            tags = sorted(
                (
                    t
                    for t in self.get_tags_in_window(0, 0, n)
                    if pmt.symbol_to_string(t.key) == "phase_est"
                ),
                key=lambda t: t.offset,
            )
            for t in tags:
                det = int(t.offset - base)
                if det < r:  # inside a span already skipped by a prior detection
                    continue
                r, w = self._emit(x, out, r, det, w)
                if r < det:  # out filled mid-segment; the tag re-appears next call
                    self.consume(0, r)
                    return int(w)
                self._rot = complex(np.exp(-1j * pmt.to_double(t.value)))
                self._armed = True
                self._skip_to = int(t.offset) + n_pre + 1
                r = min(n, self._skip_to - base)
            r, w = self._emit(x, out, r, n, w)
            self.consume(0, r if self._armed else n)
            return int(w)

    return _SymStrip()
