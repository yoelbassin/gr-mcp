"""Quantized-argmax decision: float32 spectrum vector -> int16 index. Thin
adapter — stock argmax_fs emits a signed int16 raw index, which wraps for
vector lengths > 32768; the decision must happen at full precision."""

from __future__ import annotations

from typing import Any

import numpy as np

from marconi.engine.backends.base import DiagnosticKey
from marconi.engine.backends.gnuradio.embedded.lifecycle import Diagnostics, bump

# Peak dominance (peak / median of the decision vector) separates a decisive
# argmax from one made over noise; the block tallies it per symbol so the
# quality layer can judge the run's fractions. Measured through this exact
# fold math (pinned in the css quality tests): noise dominance tops out near
# 1.0-1.25x sqrt(log2(vlen)) depending on oversampling, real off-air chirp
# symbols read 40-850, and AWGN chirps hold the floor down to -15 dB while
# symbol errors only begin near -18 dB. At the 1.35 margin the worst measured
# noise tail (a 32-bin vector at critical sampling) keeps 12% of windows
# above the floor — the chance ceiling reported below; every other measured
# configuration keeps 3% or less.
_DOMINANCE_MARGIN = 1.35
_DOMINANCE_CHANCE = 0.15


def make_peak_decision(gr: Any, *, vlen: int, divisor: int, modulo: int) -> Any:
    floor = _DOMINANCE_MARGIN * float(np.sqrt(np.log2(vlen)))

    class _PeakDecision(gr.sync_block):
        def __init__(self) -> None:
            gr.sync_block.__init__(
                self,
                name="peak_decision",
                in_sig=[(np.float32, vlen)],
                out_sig=[np.int16],
            )
            self.diagnostics: Diagnostics = {
                DiagnosticKey.DOMINANT_SYMBOLS: 0,
                DiagnosticKey.SYMBOLS_TOTAL: 0,
                "dominance_floor": floor,
                DiagnosticKey.DOMINANCE_CHANCE: _DOMINANCE_CHANCE,
            }

        def work(self, input_items: Any, output_items: Any) -> int:
            vecs = input_items[0]
            out = output_items[0]
            k = min(len(vecs), len(out))
            v = vecs[:k]
            idx = np.argmax(v, axis=1)  # full precision, no int16 wrap
            out[:k] = np.remainder(np.round(idx / divisor).astype(np.int64), modulo)
            peak = v.max(axis=1, initial=0.0)
            med = np.median(v, axis=1) if k else np.zeros(0)
            dominant = np.where(med > 0.0, peak >= floor * med, peak > 0.0)
            bump(
                self.diagnostics,
                DiagnosticKey.DOMINANT_SYMBOLS,
                int(np.count_nonzero(dominant)),
            )
            bump(self.diagnostics, DiagnosticKey.SYMBOLS_TOTAL, int(k))
            return k

    return _PeakDecision()
