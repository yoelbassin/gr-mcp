"""Generic depuncture (erasure insertion) embedded block — the one operation no
stock GR block does (it expands: sum(keep_mask) inputs -> len(keep_mask) outputs,
inputs where the mask is 1, 0.0 erasures where 0). Puncture tables are parameters."""

from __future__ import annotations

from typing import Any

import numpy as np


def make_depuncture(gr: Any, *, keep_mask: list) -> Any:
    mask = np.asarray([int(m) for m in keep_mask], dtype=np.int8)
    block_out = int(mask.size)
    block_in = int(mask.sum())
    keep_idx = np.flatnonzero(mask)

    class _Depuncture(gr.basic_block):
        def __init__(self) -> None:
            gr.basic_block.__init__(
                self, name="depuncture", in_sig=[np.float32], out_sig=[np.float32]
            )
            self._buf = np.empty(0, dtype=np.float32)
            self._out = np.empty(0, dtype=np.float32)

        def forecast(self, noutput_items: int, ninputs: int) -> list:
            return [0] * ninputs if self._out.size else [1] * ninputs

        def general_work(self, input_items: Any, output_items: Any) -> int:
            inp = input_items[0]
            if inp.size:
                self._buf = np.concatenate([self._buf, np.asarray(inp, np.float32)])
                self.consume(0, len(inp))
            while self._buf.size >= block_in:
                blk = np.zeros(block_out, dtype=np.float32)
                blk[keep_idx] = self._buf[:block_in]
                self._buf = self._buf[block_in:]
                self._out = np.concatenate([self._out, blk])
            out = output_items[0]
            if self._out.size:
                k = min(self._out.size, len(out))
                out[:k] = self._out[:k]
                self._out = self._out[k:]
                return int(k)
            return 0

    return _Depuncture()
