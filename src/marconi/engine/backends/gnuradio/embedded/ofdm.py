from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from marconi.engine.backends.gnuradio.embedded.lifecycle import (
    Diagnostics,
    OutQueue,
    forecast_drain,
)
from marconi.engine.modulation.ofdm import primitives as prim


def _resync_base(
    buf: npt.NDArray[np.complex64],
    predicted: int,
    *,
    null_len: int,
    tol: int,
    max_corr: int,
) -> int:
    """Null nearest ``predicted``; snap within ``tol``, reject jumps past
    ``max_corr`` — drift-free strides unchanged, real drift follows the null."""
    lo, hi = predicted - null_len - max_corr, predicted + max_corr
    if lo < 0 or hi > buf.size:
        return predicted
    p = np.abs(buf[lo:hi]) ** 2
    cs = np.concatenate([np.zeros(1), np.cumsum(p)])
    energy = cs[null_len:] - cs[: p.size - null_len + 1]
    j = int(np.argmin(energy))
    floor = float(energy[j]) / null_len
    rest = (float(np.sum(p)) - float(energy[j])) / (p.size - null_len)
    # a null is present when the quietest window sits well below the rest of
    # the buffer — contrast, not an absolute quarter-mean, so a noise-filled
    # null at low SNR still registers
    if floor > 0.5 * rest:
        return predicted
    edge = prim.step_boundary(p, j, p.size, floor=floor, sig=rest)
    if edge is None:
        return predicted
    delta = lo + edge - predicted
    return lo + edge if tol < abs(delta) <= max_corr else predicted


@dataclass(frozen=True)
class FrameSyncGeometry:
    """The frame anatomy detection and emission are measured in."""

    fft_len: int
    cp_len: int
    sym_len: int
    null_len: int
    frame_len: int
    data_syms: int
    tol: int
    max_corr: int
    usefuls_len: int
    detect_cap: int

    @classmethod
    def build(
        cls,
        *,
        fft_len: int,
        cp_len: int,
        sym_len: int,
        null_len: int,
        frame_len: int,
        data_syms: int,
    ) -> "FrameSyncGeometry":
        usefuls_len = (data_syms + 1) * sym_len
        return cls(
            fft_len=fft_len,
            cp_len=cp_len,
            sym_len=sym_len,
            null_len=null_len,
            frame_len=frame_len,
            data_syms=data_syms,
            tol=max(1, cp_len // 2),
            max_corr=sym_len,
            usefuls_len=usefuls_len,
            detect_cap=null_len + usefuls_len + 4 * frame_len,
        )


class OfdmFrameSyncCore:
    """Null-symbol frame detection and per-frame useful extraction. Plain and
    gr-free: every decision here is keyed to buffered content alone, never to
    call timing, which is precisely the property worth asserting directly."""

    def __init__(self, geom: FrameSyncGeometry) -> None:
        self.geom = geom
        self.out = OutQueue(np.complex64)
        # post-run value = usefuls of the open frame the stream ended short
        # of; the atomic-frame drop at EOF made visible
        self.diagnostics: Diagnostics = {"truncated_frame_items": 0}
        self._buf = np.empty(0, dtype=np.complex64)
        self._base: int | None = None
        self._emitted = False
        self._detect_len = geom.null_len + geom.usefuls_len

    def usefuls_ready(self) -> bool:
        return (
            self._base is not None
            and not self._emitted
            and self._base + self.geom.usefuls_len <= self._buf.size
        )

    def feed(self, samples: npt.NDArray[np.complex64]) -> None:
        self._buf = np.concatenate([self._buf, np.asarray(samples, np.complex64)])

    def _frame_usefuls(self, base: int) -> npt.NDArray[np.complex64]:
        g = self.geom
        blocks = [
            self._buf[
                base
                + m * g.sym_len
                + g.cp_len : base
                + m * g.sym_len
                + g.cp_len
                + g.fft_len
            ]
            for m in range(g.data_syms + 1)
        ]
        # Per-frame std-normalize: the stock soft decoder zeros on large input.
        arr = np.concatenate(blocks).astype(np.complex64)
        std = float(np.std(arr))
        if std < 1e-12:
            return np.zeros_like(arr)
        return (arr / std).astype(np.complex64)

    def _detect(self) -> None:
        """Scan fixed-length prefixes (a ladder growing by frame_len) so the
        found base is a function of stream content, never of how much extra
        data the scheduler happened to deliver (find_null's median threshold
        shifts with buffer length). Past detect_cap the window slides instead —
        the dropped front was already fully scanned — so a null-less wrong-band
        stream holds a bounded buffer, not the whole capture."""
        g = self.geom
        while self._base is None and self._buf.size >= self._detect_len:
            try:
                self._base = prim.find_null(
                    self._buf[: self._detect_len],
                    null_len=g.null_len,
                    frame_len=g.frame_len,
                    sym_len=g.sym_len,
                )
            except ValueError:
                if self._detect_len + g.frame_len <= g.detect_cap:
                    self._detect_len += g.frame_len
                else:
                    self._buf = self._buf[g.frame_len :]

    def _emit_frames(self) -> None:
        """A frame's usefuls emit once buffered — that alone flushes the final
        frame at EOF; the base advances (and the buffer trims) only behind a
        forward null window (frame_len + max_corr). A naive trim that ignored
        the forward window dropped all but the first few frames of a real
        multi-frame capture."""
        g = self.geom
        while self._base is not None:
            if not self._emitted:
                if self._base + g.usefuls_len > self._buf.size:
                    return
                self.out.push(self._frame_usefuls(self._base))
                self._emitted = True
            if self._base + g.frame_len + g.max_corr > self._buf.size:
                return
            nb = _resync_base(
                self._buf,
                self._base + g.frame_len,
                null_len=g.null_len,
                tol=g.tol,
                max_corr=g.max_corr,
            )
            self._buf = self._buf[nb:]
            self._base = 0
            self._emitted = False

    def pump(self) -> None:
        self._detect()
        self._emit_frames()
        self.diagnostics["truncated_frame_items"] = (
            max(0, self._base + self.geom.usefuls_len - self._buf.size)
            if self._base is not None and not self._emitted
            else 0
        )


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
    """The GR shell. Frame detection is OfdmFrameSyncCore."""
    geom = FrameSyncGeometry.build(
        fft_len=fft_len,
        cp_len=cp_len,
        sym_len=sym_len,
        null_len=null_len,
        frame_len=frame_len,
        data_syms=data_syms,
    )

    class _OfdmFrameSync(gr.basic_block):
        def __init__(self) -> None:
            gr.basic_block.__init__(
                self,
                name="ofdm_frame_sync",
                in_sig=[np.complex64],
                out_sig=[np.complex64],
            )
            self._core = OfdmFrameSyncCore(geom)
            self._out = self._core.out
            self.diagnostics = self._core.diagnostics

        def forecast(self, noutput_items: int, ninputs: int) -> list[int]:
            return forecast_drain(
                self._out.pending or self._core.usefuls_ready(), ninputs
            )

        def general_work(self, input_items: Any, output_items: Any) -> int:
            inp = input_items[0]
            if inp.size:
                self._core.feed(inp)
                self.consume(0, len(inp))
            self._core.pump()
            return self._out.drain(output_items[0])

    return _OfdmFrameSync()
