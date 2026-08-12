from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from marconi.engine.backends.base import DiagnosticKey
from marconi.engine.backends.gnuradio.embedded.lifecycle import (
    Diagnostics,
    OutQueue,
    bump,
    forecast_drain,
)
from marconi.engine.modulation.ofdm.primitives import LOCK_MIN_RATIO_DEFAULT

_DROP_FRAC = 0.25
_DROP_SYMS = 8
_EMA = 0.05
_OUT_CAP = 1 << 15


@dataclass(frozen=True)
class CpSyncGeometry:
    """The symbol arithmetic the acquisition and the emit loop are measured in."""

    fft_len: int
    cp_len: int
    sym_len: int
    warmup_syms: int
    lock_min_ratio: float
    need: int
    buf_cap: int

    @classmethod
    def build(
        cls,
        *,
        fft_len: int,
        cp_len: int,
        warmup_syms: int,
        lock_min_ratio: float,
    ) -> "CpSyncGeometry":
        sym_len = fft_len + cp_len
        need = warmup_syms * sym_len + fft_len + cp_len
        return cls(
            fft_len=fft_len,
            cp_len=cp_len,
            sym_len=sym_len,
            warmup_syms=warmup_syms,
            lock_min_ratio=lock_min_ratio,
            need=need,
            buf_cap=need + 8 * sym_len,
        )


class CpSyncCore:
    """Cyclic-prefix correlation acquisition, CFO estimation, and the
    lock-loss watch that re-arms the search. A plain class: the drop detector
    and the re-acquisition path are what a resync bug lives in, and neither
    needs a flowgraph to drive. Sync tags are RECORDED here as absolute output
    offsets and emitted by the shell — the only part that needs GNU Radio."""

    def __init__(self, geom: CpSyncGeometry) -> None:
        self.geom = geom
        self.out = OutQueue(np.complex64)
        self.diagnostics: Diagnostics = {
            "locks": 0,
            DiagnosticKey.LOCK_RATIO_BEST: 0.0,
            DiagnosticKey.LOCK_MIN: float(geom.lock_min_ratio),
        }
        self.tagq: list[int] = []  # absolute out offsets of sync_start
        self._best_ratio = 0.0
        self._buf = np.empty(0, dtype=np.complex64)
        self._pos = 0
        self._unlock()

    def _unlock(self) -> None:
        self._ready = False
        self._cfo = 0.0
        self._abs = 0
        self._metric_ref = 0.0
        self._metric = 0.0
        self._low = 0

    def accept(self, inp: npt.NDArray[np.complex64]) -> int:
        """Buffer what fits under the cap and report how much was taken, so the
        shell consumes exactly that and GR backpressure does the rest."""
        take = int(min(len(inp), max(self.geom.buf_cap - self._buf.size, 0)))
        if take:
            self._buf = np.concatenate(
                [self._buf, np.asarray(inp[:take], np.complex64)]
            )
        return take

    def has_work(self) -> bool:
        avail = self._buf.size - self._pos
        need = self.geom.sym_len if self._ready else self.geom.need
        return bool(self.out.pending or avail >= need)

    def pump(self) -> None:
        if not self._ready:
            self._acquire()
        if self._ready:
            self._emit_symbols()

    def _cp_metric(self, s: int) -> float:
        g = self.geom
        a = self._buf[s : s + g.cp_len]
        b = self._buf[s + g.fft_len : s + g.fft_len + g.cp_len]
        p = float(np.sum(np.abs(a) ** 2) + np.sum(np.abs(b) ** 2)) + 1e-12
        return float(2.0 * np.abs(np.vdot(a, b)) / p)

    def _acquire(self) -> None:
        g = self.geom
        while self._buf.size - self._pos >= g.need:
            seg = self._buf[self._pos : self._pos + g.need]
            length = seg.size - (g.fft_len + g.cp_len)
            prod = seg[:length] * np.conj(seg[g.fft_len : g.fft_len + length])
            metric = np.convolve(prod, np.ones(g.cp_len), "valid")
            n_grid = (metric.size - 1) // g.sym_len
            lattice = np.arange(n_grid) * g.sym_len
            strength = np.array(
                [float(np.abs(metric[o + lattice]).sum()) for o in range(g.sym_len)]
            )
            off = int(np.argmax(strength))
            ratio = float(strength[off] / (np.median(strength) + 1e-12))
            self._best_ratio = max(self._best_ratio, ratio)
            self.diagnostics[DiagnosticKey.LOCK_RATIO_BEST] = self._best_ratio
            if ratio < g.lock_min_ratio:
                self._pos += g.need - (g.fft_len + g.cp_len)
                self._trim()
                continue
            starts = off + lattice
            self._cfo = float(np.angle(metric[starts].sum()) / (2 * np.pi * g.fft_len))
            self._pos += off
            self._trim()
            self._abs = 0
            refs = [
                self._cp_metric(self._pos + m * g.sym_len)
                for m in range(min(g.warmup_syms, n_grid))
            ]
            self._metric_ref = float(np.mean(refs))
            self._metric = self._metric_ref
            self._low = 0
            self._ready = True
            bump(self.diagnostics, "locks")
            return

    def _emit_symbols(self) -> None:
        g = self.geom
        while self._buf.size - self._pos >= g.sym_len and self.out.size < _OUT_CAP:
            m = self._cp_metric(self._pos)
            self._metric = (1 - _EMA) * self._metric + _EMA * m
            if self._metric < _DROP_FRAC * self._metric_ref:
                self._low += 1
            else:
                self._low = 0
            if self._low >= _DROP_SYMS:
                self.tagq.append(self.out.pushed_total)
                self._trim()
                self._unlock()
                return
            s = self._pos + g.cp_len
            n = self._abs + g.cp_len + np.arange(g.fft_len)
            u = self._buf[s : s + g.fft_len] * np.exp(2j * np.pi * self._cfo * n)
            self.out.push(u.astype(np.complex64))
            self._pos += g.sym_len
            self._abs += g.sym_len
        self._trim()

    def _trim(self) -> None:
        if self._pos:
            self._buf = self._buf[self._pos :]
            self._pos = 0


def make_cp_symbol_sync(
    gr: Any,
    *,
    fft_len: int,
    cp_len: int,
    warmup_syms: int,
    lock_min_ratio: float = LOCK_MIN_RATIO_DEFAULT,
) -> Any:
    """The GR shell: buffering, backpressure and stream tags. The synchronizer
    is CpSyncCore, which this delegates to."""
    pmt = gr.pmt
    geom = CpSyncGeometry.build(
        fft_len=fft_len,
        cp_len=cp_len,
        warmup_syms=warmup_syms,
        lock_min_ratio=lock_min_ratio,
    )

    class _CpSymbolSync(gr.basic_block):
        def __init__(self) -> None:
            gr.basic_block.__init__(
                self,
                name="cp_symbol_sync",
                in_sig=[np.complex64],
                out_sig=[np.complex64],
            )
            self._core = CpSyncCore(geom)
            self._out = self._core.out
            self.diagnostics = self._core.diagnostics

        def forecast(self, noutput_items: int, ninputs: int) -> list[int]:
            return forecast_drain(self._core.has_work(), ninputs)

        def general_work(self, input_items: Any, output_items: Any) -> int:
            inp = input_items[0]
            if len(inp):
                self.consume(0, self._core.accept(inp))
            self._core.pump()
            self._emit_tags()
            return self._out.drain(output_items[0])

        def _emit_tags(self) -> None:
            for offset in self._core.tagq:
                self.add_item_tag(0, offset, pmt.intern("sync_start"), pmt.PMT_NIL)
            self._core.tagq.clear()

    return _CpSymbolSync()
