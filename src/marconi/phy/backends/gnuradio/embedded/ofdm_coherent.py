from __future__ import annotations

from typing import Any

import numpy as np

from marconi.phy.backends.gnuradio.embedded.lifecycle import OutQueue, forecast_drain


def _pilot_period(pilot_sets: list[frozenset[int]], n_frame_syms: int) -> int:
    """Smallest frame-symbol shift over which the scattered-pilot carrier SET
    repeats — the gap the channel-independent frame sync differences across."""
    for gap in range(1, n_frame_syms):
        if all(
            pilot_sets[(fs + gap) % n_frame_syms] == pilot_sets[fs]
            for fs in range(n_frame_syms)
        ):
            return gap
    return n_frame_syms


def make_ofdm_coherent_sync(
    gr: Any,
    *,
    fft_len: int,
    cp_len: int,
    sym_len: int,
    n_frame_syms: int,
    n_carriers: int,
    kmin: int,
    dc_search: int,
    warmup_syms: int,
    pilot_lens: list[int],
    pilot_carriers: list[int],
    pilot_i: list[float],
    pilot_q: list[float],
    fp_carriers: list[int],
    fp_i: list[float],
    fp_q: list[float],
) -> Any:
    dc0 = fft_len // 2
    pilot_sets: list[frozenset[int]] = []
    pilot_vals: list[dict[int, complex]] = []
    off = 0
    for length in pilot_lens:
        ks = [int(k) for k in pilot_carriers[off : off + length]]
        vals = {
            ks[j]: complex(pilot_i[off + j], pilot_q[off + j]) for j in range(length)
        }
        pilot_sets.append(frozenset(ks))
        pilot_vals.append(vals)
        off += length
    fp_vals = [complex(fp_i[j], fp_q[j]) for j in range(len(fp_carriers))]
    gap = _pilot_period(pilot_sets, n_frame_syms)
    union = np.array(sorted(set().union(*pilot_sets)), dtype=np.int64)
    union_index = {int(k): i for i, k in enumerate(union)}
    # active carriers, DC (kmin..kmin+n_carriers spans one null) dropped, ascending
    emit = np.array([k for k in range(kmin, kmin + n_carriers + 1) if k != 0])

    class _OfdmCoherentSync(gr.basic_block):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            gr.basic_block.__init__(
                self,
                name="ofdm_coherent_sync",
                in_sig=[np.complex64],
                out_sig=[np.complex64],
            )
            self._buf = np.empty(0, dtype=np.complex64)
            self._ready = False
            self._off_grid = 0
            self._cfo = 0.0
            self._delta = 0
            self._theta = 0.0
            self._phi = 0
            self._emit_bins = np.empty(0, dtype=np.int64)
            self._x: list[np.ndarray] = []
            self._nsym_fft = 0
            self._node_ms: list[list[int]] = [[] for _ in union]
            self._node_hs: list[list[complex]] = [[] for _ in union]
            self._frames_emitted = 0
            self._out = OutQueue(np.complex64)

        def forecast(self, noutput_items: int, ninputs: int) -> list[int]:
            return forecast_drain(self._out.pending or self._frame_ready(), ninputs)

        def _frame_ready(self) -> bool:
            if not self._ready:
                return False
            avail = (len(self._x) - self._phi) // n_frame_syms
            return avail > self._frames_emitted

        def general_work(self, input_items: Any, output_items: Any) -> int:
            inp = input_items[0]
            if inp.size:
                self._buf = np.concatenate([self._buf, np.asarray(inp, np.complex64)])
                self.consume(0, len(inp))
            if not self._ready:
                self._try_estimate()
            if self._ready:
                self._fft_new_symbols()
                self._emit_ready_frames()
            return self._out.drain(output_items[0])

        def _fft_symbol(self, m: int) -> np.ndarray:
            s = self._off_grid + sym_len * m
            u = self._buf[s + cp_len : s + cp_len + fft_len].astype(np.complex128)
            rot = np.exp(
                -1j * 2 * np.pi * self._cfo * (np.arange(fft_len) + s + cp_len)
            )
            return np.fft.fftshift(np.fft.fft(u * rot))

        def _try_estimate(self) -> None:
            if self._buf.size < (warmup_syms + 2) * sym_len + fft_len:
                return
            seg = self._buf[: warmup_syms * sym_len + fft_len + cp_len]
            length = seg.size - (fft_len + cp_len)
            prod = seg[:length] * np.conj(seg[fft_len : fft_len + length])
            metric = np.convolve(prod, np.ones(cp_len), "valid")
            n_grid = (metric.size - 1) // sym_len
            lattice = np.arange(n_grid) * sym_len
            strength = [
                float(np.abs(metric[o + lattice]).sum()) for o in range(sym_len)
            ]
            self._off_grid = int(np.argmax(strength))
            starts = self._off_grid + lattice
            self._cfo = float(np.angle(metric[starts].sum()) / (2 * np.pi * fft_len))

            xp = np.array([self._fft_symbol(m) for m in range(warmup_syms)])
            dcpow = [
                float(np.mean(np.abs(xp[:, dc0 + d]) ** 2))
                for d in range(-dc_search, dc_search + 1)
            ]
            self._delta = int(np.argmin(dcpow)) - dc_search
            # Fine CFO from the frequency pilots: the CP-based coarse estimate
            # misses a fractional-carrier residual and the channel estimate rings
            # (constant-magnitude, random-phase) without this derotation.
            pil = np.zeros((warmup_syms, len(fp_carriers)), dtype=np.complex128)
            for j, k in enumerate(fp_carriers):
                pil[:, j] = xp[:, k + dc0 + self._delta] * np.conj(fp_vals[j])
            self._theta = float(np.angle(np.sum(pil[1:] * np.conj(pil[:-1]))))
            self._phi = self._estimate_phi(xp)

            self._emit_bins = emit + dc0 + self._delta
            self._x = list(xp)
            self._nsym_fft = warmup_syms
            for m in range(warmup_syms):
                self._gather_nodes(m)
            self._ready = True

        def _estimate_phi(self, xp: np.ndarray) -> int:
            best_phi, best = 0, -1.0
            for phi in range(n_frame_syms):
                fsyms = (np.arange(warmup_syms) - phi) % n_frame_syms
                num, den = 0j, 0.0
                for m in range(warmup_syms - gap):
                    fs = int(fsyms[m])
                    fs2 = (fs + gap) % n_frame_syms
                    ks = [k for k in pilot_sets[fs] if k in pilot_vals[fs2]]
                    bins = np.array([k + dc0 + self._delta for k in ks])
                    ok = (bins >= 0) & (bins < fft_len)
                    ks = [k for k, keep in zip(ks, ok) if keep]
                    bins = bins[ok]
                    if not ks:
                        continue
                    ratio = xp[m + gap, bins] / (xp[m, bins] + 1e-12)
                    ratio = ratio / (np.abs(ratio) + 1e-12)
                    pred = np.array(
                        [pilot_vals[fs2][k] / pilot_vals[fs][k] for k in ks]
                    )
                    pred = pred / np.abs(pred)
                    num += complex(np.sum(ratio * np.conj(pred)))
                    den += len(ks)
                score = abs(num) / max(den, 1.0)
                if score > best:
                    best, best_phi = score, phi
            return best_phi

        def _fft_new_symbols(self) -> None:
            nsy_now = (self._buf.size - self._off_grid) // sym_len
            for m in range(self._nsym_fft, nsy_now):
                self._x.append(self._fft_symbol(m))
                self._gather_nodes(m)
            self._nsym_fft = nsy_now

        def _gather_nodes(self, m: int) -> None:
            fs = (m - self._phi) % n_frame_syms
            xd = self._x[m] * np.exp(-1j * self._theta * m)
            vals = pilot_vals[fs]
            for k in pilot_sets[fs]:
                i = union_index[k]
                self._node_ms[i].append(m)
                self._node_hs[i].append(complex(xd[k + dc0 + self._delta] / vals[k]))

        def _emit_ready_frames(self) -> None:
            avail = (len(self._x) - self._phi) // n_frame_syms
            while self._frames_emitted < avail:
                self._out.push(self._equalize_frame(self._frames_emitted))
                self._frames_emitted += 1

        def _equalize_frame(self, f: int) -> np.ndarray:
            base = self._phi + n_frame_syms * f
            frame_end = base + n_frame_syms - 1
            syms = base + np.arange(n_frame_syms)
            hodd = np.empty((union.size, n_frame_syms), dtype=np.complex128)
            for i in range(union.size):
                ms = np.asarray(self._node_ms[i], dtype=np.float64)
                hs = np.asarray(self._node_hs[i], dtype=np.complex128)
                j = int(np.searchsorted(ms, frame_end, side="right"))
                hodd[i] = np.interp(syms, ms[:j], hs[:j].real) + 1j * np.interp(
                    syms, ms[:j], hs[:j].imag
                )
            out = np.empty(n_frame_syms * emit.size, dtype=np.complex64)
            for si in range(n_frame_syms):
                heq = np.interp(emit, union, hodd[:, si].real) + 1j * np.interp(
                    emit, union, hodd[:, si].imag
                )
                xd = self._x[base + si] * np.exp(-1j * self._theta * (base + si))
                out[si * emit.size : (si + 1) * emit.size] = xd[self._emit_bins] / heq
            return out

    return _OfdmCoherentSync()
