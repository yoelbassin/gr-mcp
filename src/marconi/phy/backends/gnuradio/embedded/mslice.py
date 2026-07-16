from __future__ import annotations

from typing import Any

import numpy as np

from marconi.phy.backends.gnuradio.embedded.lifecycle import OutQueue, forecast_drain


def make_mslice(gr: Any, *, thresholds: list[float], code: list[int], bits: int) -> Any:
    edges = np.asarray(thresholds, np.float32)
    table = np.asarray(
        [[(c >> (bits - 1 - j)) & 1 for j in range(bits)] for c in code], np.uint8
    )

    class _Mslice(gr.basic_block):
        def __init__(self) -> None:
            gr.basic_block.__init__(
                self, name="mslice", in_sig=[np.float32], out_sig=[np.uint8]
            )
            self._out = OutQueue(np.uint8)

        def forecast(self, noutput_items: int, ninputs: int) -> list[int]:
            return forecast_drain(self._out.pending, ninputs)

        def general_work(self, input_items: Any, output_items: Any) -> int:
            inp = input_items[0]
            if len(inp):
                region = np.searchsorted(edges, inp)
                self._out.push(table[region].reshape(-1))
                self.consume(0, len(inp))
            return self._out.drain(output_items[0])

    return _Mslice()
