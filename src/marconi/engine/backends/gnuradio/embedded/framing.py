from __future__ import annotations

from typing import Any

import numpy as np

from marconi.engine.backends.base import DiagnosticKey
from marconi.engine.backends.gnuradio.embedded.lifecycle import Diagnostics, bump


def make_tag_gate(
    gr: Any, *, frame_len: int, tag_name: str, chance_per_item: float
) -> Any:
    """Keep exactly ``frame_len`` float items after each ``tag_name`` tag, drop
    everything between frames. The stock counterpart to sym_strip's skip: an
    upstream correlate_access_code_tag_ff tags each payload start, and this gates
    the fixed-length coded frame that a trellis decoder then consumes contiguously
    — the bridge with no stock block (header_payload_demux stalls without a header
    message loop). Emission comes from the current input window, never a buffer,
    so no OutQueue and forecast is a plain pass-through demand.

    Also the sync-evidence census point: it counts the correlator's tags in the
    consumed region plus the chance expectation (``chance_per_item`` x items
    scanned) into diagnostics, so GR-native sync paths feed the quality layer.
    ``chance_per_item`` has no default: zero clears every significance bar, so
    a caller that cannot state one must not get a silent 0.0."""
    if chance_per_item <= 0.0:
        raise ValueError(
            f"tag_gate needs a positive chance_per_item, got {chance_per_item!r}; "
            "the tag count it reports is judged against this expectation and "
            "zero would certify a single chance-level hit"
        )

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
            self._scanned = 0
            # truncated_frame_items post-run = items of the open frame the
            # stream ended short of; a truncated final frame is unrecoverable
            # but must be visible
            self.diagnostics: Diagnostics = {
                "truncated_frame_items": 0,
                DiagnosticKey.SYNC_TAGS: 0,
                DiagnosticKey.SYNC_CHANCE: 0.0,
                DiagnosticKey.SYNC_ITEMS_SCANNED: 0,
            }

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
                inside_frame = abs_r < self._end
                if inside_frame:
                    hi = min(self._end - base, n)
                    k = min(hi - r, len(out) - w)
                    out[w : w + k] = x[r : r + k]
                    w += k
                    r += k
                    continue
                next_frame = next(
                    (o for o in starts if o >= abs_r and o >= self._end), None
                )
                if next_frame is None:
                    r = n
                    break
                r = next_frame - base  # skip the inter-frame gap
                self._end = next_frame + frame_len
            self.consume(0, r)
            self._scanned += r
            # only consumed-region tags: the unconsumed remainder reappears
            # in the next window and would double-count
            bump(
                self.diagnostics,
                DiagnosticKey.SYNC_TAGS,
                sum(1 for o in starts if o < base + r),
            )
            self.diagnostics[DiagnosticKey.SYNC_ITEMS_SCANNED] = self._scanned
            self.diagnostics[DiagnosticKey.SYNC_CHANCE] = (
                self._scanned * chance_per_item
            )
            self.diagnostics["truncated_frame_items"] = max(0, self._end - (base + r))
            return int(w)

    return _TagGate()
