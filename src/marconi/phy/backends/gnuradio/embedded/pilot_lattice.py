from __future__ import annotations

from typing import Any

import numpy as np

from marconi.phy.backends.gnuradio.embedded.lifecycle import OutQueue, forecast_drain

_LOCK_MIN_SCORE = 0.35  # calibrated: synthetic lattice locks ~0.9, noise ~0.1


def _pilot_period(pilot_sets: list[frozenset[int]], n_frame_syms: int) -> int:
    for gap in range(1, n_frame_syms):
        if all(
            pilot_sets[(fs + gap) % n_frame_syms] == pilot_sets[fs]
            for fs in range(n_frame_syms)
        ):
            return gap
    return n_frame_syms


def make_pilot_lattice_equalizer(
    gr: Any,
    *,
    fft_len: int,
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
    emit = np.array([k for k in range(kmin, kmin + n_carriers + 1) if k != 0])
    keep_margin = n_frame_syms + gap
    vec_cap = 4 * n_frame_syms + keep_margin + warmup_syms

    class _PilotLatticeEqualizer(gr.basic_block):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            gr.basic_block.__init__(
                self,
                name="pilot_lattice_equalizer",
                in_sig=[(np.complex64, fft_len)],
                out_sig=[np.complex64],
            )
            self.diagnostics: dict[str, int] = {
                "locks": 0,
                "relocks": 0,
                "lock_score_permille": 0,
                "frames_emitted": 0,
            }
            self._out = OutQueue(np.complex64)
            self._reset_lock()

        def _reset_lock(self) -> None:
            # FIFO: everything pushed before this reset drains before any
            # new-epoch item, so the push watermark is the stale/new boundary
            self._drained_baseline = self._out.pushed_total
            self._ready = False
            self._warm: list[np.ndarray] = []
            self._theta = 0.0
            self._delta = 0
            self._phi = 0
            self._emit_bins = np.empty(0, dtype=np.int64)
            self._m = 0
            self._first = 0
            self._vecs: list[np.ndarray] = []
            self._node_ms: list[list[int]] = [[] for _ in union]
            self._node_hs: list[list[complex]] = [[] for _ in union]
            self._frames_emitted = 0

        def forecast(self, noutput_items: int, ninputs: int) -> list[int]:
            return forecast_drain(self._out.pending or self._frame_ready(), ninputs)

        def _frame_ready(self) -> bool:
            if not self._ready:
                return False
            avail = (self._m - self._phi) // n_frame_syms
            return avail > self._frames_emitted

        def general_work(self, input_items: Any, output_items: Any) -> int:
            vin = input_items[0]
            n = len(vin)
            if n:
                resets = self._reset_offsets(n)
                done = 0
                for i in range(n):
                    if len(self._vecs) >= vec_cap:
                        break
                    if i in resets:
                        if self._ready:
                            self.diagnostics["relocks"] += 1
                        self._reset_lock()
                    self._ingest(np.asarray(vin[i], np.complex128))
                    done += 1
                self.consume(0, done)
            if self._ready:
                self._emit_ready_frames()
                self._prune()
            return self._out.drain(output_items[0])

        def _reset_offsets(self, n: int) -> set[int]:
            base = self.nitems_read(0)
            pmt = gr.pmt
            return {
                int(t.offset - base)
                for t in self.get_tags_in_window(0, 0, n)
                if pmt.symbol_to_string(t.key) == "sync_start"
            }

        def _ingest(self, vec: np.ndarray) -> None:
            if self._ready:
                self._vecs.append(vec)
                self._gather_nodes(self._m, vec)
                self._m += 1
                return
            self._warm.append(vec)
            if len(self._warm) > warmup_syms:
                self._warm.pop(0)
            if len(self._warm) == warmup_syms:
                self._try_lock()

        def _try_lock(self) -> None:
            xp = np.array(self._warm)
            dcpow = [
                float(np.mean(np.abs(xp[:, dc0 + d]) ** 2))
                for d in range(-dc_search, dc_search + 1)
            ]
            delta = int(np.argmin(dcpow)) - dc_search
            pil = np.zeros((warmup_syms, len(fp_carriers)), dtype=np.complex128)
            for j, k in enumerate(fp_carriers):
                pil[:, j] = xp[:, k + dc0 + delta] * np.conj(fp_vals[j])
            theta = float(np.angle(np.sum(pil[1:] * np.conj(pil[:-1]))))
            phi, score = self._estimate_phi(xp, delta)
            self.diagnostics["lock_score_permille"] = int(score * 1000)
            if score < _LOCK_MIN_SCORE:
                return
            self._delta, self._theta, self._phi = delta, theta, phi
            self._emit_bins = emit + dc0 + delta
            self._vecs = list(self._warm)
            self._warm = []
            self._first = 0
            self._m = warmup_syms
            self._ready = True
            self.diagnostics["locks"] += 1
            for m in range(warmup_syms):
                self._gather_nodes(m, self._vecs[m])

        def _estimate_phi(self, xp: np.ndarray, delta: int) -> tuple[int, float]:
            best_phi, best = 0, -1.0
            for phi in range(n_frame_syms):
                fsyms = (np.arange(warmup_syms) - phi) % n_frame_syms
                num, den = 0j, 0.0
                for m in range(warmup_syms - gap):
                    fs = int(fsyms[m])
                    fs2 = (fs + gap) % n_frame_syms
                    ks = [k for k in pilot_sets[fs] if k in pilot_vals[fs2]]
                    bins = np.array([k + dc0 + delta for k in ks])
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
            return best_phi, best

        def _gather_nodes(self, m: int, vec: np.ndarray) -> None:
            fs = (m - self._phi) % n_frame_syms
            xd = vec * np.exp(-1j * self._theta * m)
            vals = pilot_vals[fs]
            for k in pilot_sets[fs]:
                b = k + dc0 + self._delta
                if 0 <= b < fft_len:
                    i = union_index[k]
                    self._node_ms[i].append(m)
                    self._node_hs[i].append(complex(xd[b] / vals[k]))

        def _emit_ready_frames(self) -> None:
            avail = (self._m - self._phi) // n_frame_syms
            while self._frames_emitted < avail:
                self._out.push(self._equalize_frame(self._frames_emitted))
                self._frames_emitted += 1
                self.diagnostics["frames_emitted"] += 1

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
                m = base + si
                xd = self._vecs[m - self._first] * np.exp(-1j * self._theta * m)
                out[si * emit.size : (si + 1) * emit.size] = xd[self._emit_bins] / heq
            return out

        def _prune(self) -> None:
            # retain until DRAINED, not merely pushed — pushed-keyed pruning
            # keeps vecs flat while _out grows unbounded (vec_cap never gates)
            frame_size = n_frame_syms * emit.size
            epoch_drained = max(0, self._out.drained_total - self._drained_baseline)
            drained_frames = epoch_drained // frame_size
            keep_from = self._phi + n_frame_syms * drained_frames - keep_margin
            drop = min(keep_from - self._first, len(self._vecs))
            if drop > 0:
                del self._vecs[:drop]
                self._first += drop
            for i in range(union.size):
                ms = self._node_ms[i]
                j = 0
                while j < len(ms) and ms[j] < keep_from:
                    j += 1
                if j:
                    del ms[:j]
                    del self._node_hs[i][:j]

    return _PilotLatticeEqualizer()
