from __future__ import annotations

from typing import Any

import numpy as np

from marconi.engine.backends.gnuradio.embedded.lifecycle import OutQueue, forecast_drain

_LOCK_MIN_RATIO = 2.0  # calibrated: synthetic lattice ~5-8, pure noise ~1.3-1.6
_DROP_FRAC = 0.25
_DROP_SYMS = 8
_EMA = 0.05
_OUT_CAP = 1 << 15


def make_cp_symbol_sync(gr: Any, *, fft_len: int, cp_len: int, warmup_syms: int) -> Any:
    sym_len = fft_len + cp_len
    need = warmup_syms * sym_len + fft_len + cp_len
    buf_cap = need + 8 * sym_len
    pmt = gr.pmt

    class _CpSymbolSync(gr.basic_block):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            gr.basic_block.__init__(
                self,
                name="cp_symbol_sync",
                in_sig=[np.complex64],
                out_sig=[np.complex64],
            )
            self.diagnostics: dict[str, int] = {"locks": 0}
            self._out = OutQueue(np.complex64)
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

        def forecast(self, noutput_items: int, ninputs: int) -> list[int]:
            avail = self._buf.size - self._pos
            work = self._out.pending or avail >= (sym_len if self._ready else need)
            return forecast_drain(bool(work), ninputs)

        def general_work(self, input_items: Any, output_items: Any) -> int:
            inp = input_items[0]
            if len(inp):
                take = int(min(len(inp), max(buf_cap - self._buf.size, 0)))
                if take:
                    self._buf = np.concatenate(
                        [self._buf, np.asarray(inp[:take], np.complex64)]
                    )
                self.consume(0, take)
            if not self._ready:
                self._acquire()
            if self._ready:
                self._emit_symbols()
            return self._out.drain(output_items[0])

        def _cp_metric(self, s: int) -> float:
            a = self._buf[s : s + cp_len]
            b = self._buf[s + fft_len : s + fft_len + cp_len]
            p = float(np.sum(np.abs(a) ** 2) + np.sum(np.abs(b) ** 2)) + 1e-12
            return float(2.0 * np.abs(np.vdot(a, b)) / p)

        def _acquire(self) -> None:
            while self._buf.size - self._pos >= need:
                seg = self._buf[self._pos : self._pos + need]
                length = seg.size - (fft_len + cp_len)
                prod = seg[:length] * np.conj(seg[fft_len : fft_len + length])
                metric = np.convolve(prod, np.ones(cp_len), "valid")
                n_grid = (metric.size - 1) // sym_len
                lattice = np.arange(n_grid) * sym_len
                strength = np.array(
                    [float(np.abs(metric[o + lattice]).sum()) for o in range(sym_len)]
                )
                off = int(np.argmax(strength))
                ratio = float(strength[off] / (np.median(strength) + 1e-12))
                if ratio < _LOCK_MIN_RATIO:
                    self._pos += need - (fft_len + cp_len)
                    self._trim()
                    continue
                starts = off + lattice
                self._cfo = float(
                    np.angle(metric[starts].sum()) / (2 * np.pi * fft_len)
                )
                self._pos += off
                self._trim()
                self._abs = 0
                refs = [
                    self._cp_metric(self._pos + m * sym_len)
                    for m in range(min(warmup_syms, n_grid))
                ]
                self._metric_ref = float(np.mean(refs))
                self._metric = self._metric_ref
                self._low = 0
                self._ready = True
                self.diagnostics["locks"] += 1
                return

        def _emit_symbols(self) -> None:
            while self._buf.size - self._pos >= sym_len and self._out.size < _OUT_CAP:
                m = self._cp_metric(self._pos)
                self._metric = (1 - _EMA) * self._metric + _EMA * m
                if self._metric < _DROP_FRAC * self._metric_ref:
                    self._low += 1
                else:
                    self._low = 0
                if self._low >= _DROP_SYMS:
                    self.add_item_tag(
                        0,
                        self._out.pushed_total,
                        pmt.intern("sync_start"),
                        pmt.PMT_NIL,
                    )
                    self._trim()
                    self._unlock()
                    return
                s = self._pos + cp_len
                n = self._abs + cp_len + np.arange(fft_len)
                u = self._buf[s : s + fft_len] * np.exp(2j * np.pi * self._cfo * n)
                self._out.push(u.astype(np.complex64))
                self._pos += sym_len
                self._abs += sym_len
            self._trim()

        def _trim(self) -> None:
            if self._pos:
                self._buf = self._buf[self._pos :]
                self._pos = 0

    return _CpSymbolSync()
