"""OFDM frame sync + CP-strip (embedded GR block, scalar). Null-syncs and, per
frame, emits the FIC symbols' CP-stripped useful parts back-to-back, for a stock
stream_to_vector -> fft_vcc to transform. Generic; geometry is parameters."""

from __future__ import annotations

from typing import Any

import numpy as np

from marconi.phy.modulation.ofdm import primitives as prim


def make_ofdm_frame_sync(
    gr: Any,
    *,
    fft_len: int,
    cp_len: int,
    sym_len: int,
    null_len: int,
    frame_len: int,
    data_syms: int,
) -> Any:
    def _frame_usefuls(buf: np.ndarray, base: int) -> np.ndarray:
        blocks = [
            buf[base + m * sym_len + cp_len : base + m * sym_len + cp_len + fft_len]
            for m in range(data_syms + 1)
        ]
        return np.concatenate(blocks).astype(np.complex64)

    class _OfdmFrameSync(gr.basic_block):
        def __init__(self) -> None:
            gr.basic_block.__init__(
                self,
                name="ofdm_frame_sync",
                in_sig=[np.complex64],
                out_sig=[np.complex64],
            )
            self._buf = np.empty(0, dtype=np.complex64)
            self._base: int | None = None
            self._out = np.empty(0, dtype=np.complex64)

        def forecast(self, noutput_items: int, ninputs: int) -> list:
            return [1] * ninputs

        def general_work(self, input_items: Any, output_items: Any) -> int:
            inp = input_items[0]
            if inp.size:
                self._buf = np.concatenate([self._buf, np.asarray(inp, np.complex64)])
                self.consume(0, len(inp))
            if (
                self._base is None
                and self._buf.size >= null_len + (data_syms + 1) * sym_len
            ):
                self._base = prim.find_null(
                    self._buf, null_len=null_len, frame_len=frame_len, sym_len=sym_len
                )
            while (
                self._base is not None
                and self._base + (data_syms + 1) * sym_len <= self._buf.size
            ):
                self._out = np.concatenate(
                    [self._out, _frame_usefuls(self._buf, self._base)]
                )
                self._base += frame_len
            out = output_items[0]
            if self._out.size:
                k = min(self._out.size, len(out))
                out[:k] = self._out[:k]
                self._out = self._out[k:]
                return int(k)
            return 0

    return _OfdmFrameSync()
