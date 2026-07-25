"""Quantized-argmax decision: float32 spectrum vector -> int16 index. Thin
adapter — stock argmax_fs emits a signed int16 raw index, which wraps for
vector lengths > 32768; the decision must happen at full precision."""

from __future__ import annotations

from typing import Any

import numpy as np


def make_peak_decision(gr: Any, *, vlen: int, divisor: int, modulo: int) -> Any:
    class _PeakDecision(gr.sync_block):
        def __init__(self) -> None:
            gr.sync_block.__init__(
                self,
                name="peak_decision",
                in_sig=[(np.float32, vlen)],
                out_sig=[np.int16],
            )

        def work(self, input_items: Any, output_items: Any) -> int:
            vecs = input_items[0]
            out = output_items[0]
            k = min(len(vecs), len(out))
            for i in range(k):
                out[i] = round(int(np.argmax(vecs[i])) / divisor) % modulo
            return k

    return _PeakDecision()
