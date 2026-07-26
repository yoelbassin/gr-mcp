from __future__ import annotations

from typing import Any

import numpy as np


def make_tag_gate(gr: Any, *, frame_len: int, tag_name: str) -> Any:
    """Keep exactly ``frame_len`` float items after each ``tag_name`` tag, drop
    everything between frames. The stock counterpart to sym_strip's skip: an
    upstream correlate_access_code_tag_ff tags each payload start, and this gates
    the fixed-length coded frame that a trellis decoder then consumes contiguously
    — the bridge with no stock block (header_payload_demux stalls without a header
    message loop). Emission comes from the current input window, never a buffer,
    so no OutQueue and forecast is a plain pass-through demand."""

    pmt = gr.pmt

    class _TagGate(gr.basic_block):
        def __init__(self) -> None:
            gr.basic_block.__init__(
                self,
                name="tag_gate",
                in_sig=[np.float32],
                out_sig=[np.float32],
            )
            self._end = 0  # absolute input offset where the active frame ends
            # post-run value = items of the open frame the stream ended short
            # of; a truncated final frame is unrecoverable but must be visible
            self.diagnostics = {"truncated_frame_items": 0}

        def forecast(self, noutput_items: int, ninputs: int) -> list[int]:
            return [noutput_items] * ninputs

        def general_work(self, input_items: Any, output_items: Any) -> int:
            x = input_items[0]
            out = output_items[0]
            base = self.nitems_read(0)
            n = len(x)
            starts = sorted(
                int(t.offset)
                for t in self.get_tags_in_window(0, 0, n)
                if pmt.symbol_to_string(t.key) == tag_name
            )
            r = 0
            w = 0
            while r < n and w < len(out):
                abs_r = base + r
                if abs_r < self._end:  # inside a frame: emit up to its end
                    hi = min(self._end - base, n)
                    k = min(hi - r, len(out) - w)
                    out[w : w + k] = x[r : r + k]
                    w += k
                    r += k
                    continue
                nxt = next((o for o in starts if o >= abs_r and o >= self._end), None)
                if nxt is None:  # no further frame opens in this window: drop rest
                    r = n
                    break
                r = nxt - base  # drop the inter-frame gap
                self._end = nxt + frame_len
            self.consume(0, r)
            self.diagnostics["truncated_frame_items"] = max(0, self._end - (base + r))
            return int(w)

    return _TagGate()
