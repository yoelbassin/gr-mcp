"""OFDM frame sync + CP-strip (embedded GR block, scalar). Null-syncs and, per
frame, emits the FIC symbols' CP-stripped useful parts back-to-back, for a stock
stream_to_vector -> fft_vcc to transform. Generic; geometry is parameters."""

from __future__ import annotations

from typing import Any

import numpy as np

from marconi.phy.modulation.ofdm import primitives as prim


def _resync_base(
    buf: np.ndarray,
    predicted: int,
    *,
    null_len: int,
    tol: int,
    max_corr: int,
) -> int:
    """Refine the next frame's base by locating the null nearest ``predicted``.
    Snaps to ``predicted`` within ``tol`` (detection noise / negligible SFO) and
    ignores implausible jumps beyond ``max_corr``, so a drift-free capture strides
    exactly as before; genuine accumulated drift follows the detected null."""
    lo, hi = predicted - null_len - max_corr, predicted + max_corr
    if lo < 0 or hi > buf.size:
        return predicted
    p = np.abs(buf[lo:hi]) ** 2
    cs = np.concatenate([np.zeros(1), np.cumsum(p)])
    energy = cs[null_len:] - cs[: p.size - null_len + 1]
    j = int(np.argmin(energy))
    thr = 0.25 * float(np.mean(p))
    if energy[j] > thr * null_len:
        return predicted
    while j < p.size and p[j] <= thr:
        j += 1
    delta = lo + j - predicted
    return lo + j if tol < abs(delta) <= max_corr else predicted


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
        # Normalize per frame: the raw capture stores large-amplitude ADC values;
        # the downstream stock soft decoder needs unit-ish magnitude input (the
        # de-risk normalized the IQ globally). Per-frame keeps the differential
        # scale-consistent within a frame and is magnitude-invariant for the lock.
        arr = np.concatenate(blocks).astype(np.complex64)
        std = float(np.std(arr))
        if std < 1e-12:
            return np.zeros_like(arr)
        return (arr / std).astype(np.complex64)

    tol, max_corr = max(1, cp_len // 2), sym_len

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
            return [0] * ninputs if self._out.size else [1] * ninputs

        def general_work(self, input_items: Any, output_items: Any) -> int:
            inp = input_items[0]
            if inp.size:
                self._buf = np.concatenate([self._buf, np.asarray(inp, np.complex64)])
                self.consume(0, len(inp))
            if (
                self._base is None
                and self._buf.size >= null_len + (data_syms + 1) * sym_len
            ):
                try:
                    self._base = prim.find_null(
                        self._buf,
                        null_len=null_len,
                        frame_len=frame_len,
                        sym_len=sym_len,
                    )
                except ValueError:
                    pass  # null not yet in buffer; retry on next call
            while (
                self._base is not None
                and self._base + (data_syms + 1) * sym_len <= self._buf.size
            ):
                self._out = np.concatenate(
                    [self._out, _frame_usefuls(self._buf, self._base)]
                )
                self._base = _resync_base(
                    self._buf,
                    self._base + frame_len,
                    null_len=null_len,
                    tol=tol,
                    max_corr=max_corr,
                )
            out = output_items[0]
            if self._out.size:
                k = min(self._out.size, len(out))
                out[:k] = self._out[:k]
                self._out = self._out[k:]
                return int(k)
            return 0

    return _OfdmFrameSync()
