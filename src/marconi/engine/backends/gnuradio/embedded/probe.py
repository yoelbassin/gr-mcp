"""Passthrough burst-mark probe: records each "burst" stream tag's absolute
item offset into run diagnostics — the side-channel that carries burst starts
across the file seam to the bits layer's marked decode (tags don't survive a
file sink)."""

from __future__ import annotations

from typing import Any

import numpy as np

from marconi.engine.backends.base import DiagnosticKey


def make_burst_probe(gr: Any) -> Any:
    pmt = gr.pmt

    class _BurstProbe(gr.sync_block):
        def __init__(self) -> None:
            gr.sync_block.__init__(
                self, name="burst_probe", in_sig=[np.int16], out_sig=[np.int16]
            )
            self.diagnostics: dict[str, list[int]] = {DiagnosticKey.BURSTS: []}

        def work(self, input_items: Any, output_items: Any) -> int:
            x = input_items[0]
            out = output_items[0]
            k = min(len(x), len(out))
            out[:k] = x[:k]
            for t in self.get_tags_in_window(0, 0, k):
                if pmt.symbol_to_string(t.key) == "burst":
                    self.diagnostics[DiagnosticKey.BURSTS].append(int(t.offset))
            return k

    return _BurstProbe()
