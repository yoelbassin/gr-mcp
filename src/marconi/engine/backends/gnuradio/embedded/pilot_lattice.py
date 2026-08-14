from __future__ import annotations

import math
from collections.abc import Sequence
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
from marconi.engine.modulation.ofdm.primitives import carrier_bin_problem

# per-ATTEMPT calibration: synthetic lattice ~0.9, noise ~0.1
_LOCK_MIN_SCORE = 0.35
# The lock search is a selection: _try_lock slides once per ingested symbol
# and best-score is a running max, so a fixed per-attempt floor is crossed by
# pure noise given enough attempts (measured, suite geometry, den=48: 1/10
# seeds by 60 symbols, 10/10 by 500, median false lock ~350 — minting a
# decode-grade "decoded" from noise). The effective floor therefore grows
# with the attempt count A: bar(A) = sqrt(ln(n_frame_syms * A^2 / alpha) /
# den), den the attempt's smallest per-phi pilot-pair mass — per phi the
# score is |sum of den unit phasors|/den, tail P(> s) = exp(-s^2 * den) —
# which bounds attempt i's false-cross odds by alpha/i^2, summing to
# <= 1.65*alpha over ANY run length. alpha=0.01 puts bar(1)=0.353 at the
# calibrated per-attempt floor for den=48 and holds the null over 8000 noise
# symbols (10 seeds, worst score-minus-bar -0.100, best selection max 0.488
# at attempt 1978), while a clean lattice's first aligned attempt measures
# 0.999 (0.943 at 10 dB SNR) — headroom for ~1e9 (~9e7) noise attempts. The
# measured cost: a 0-dB lattice (score ~0.45) locks only within its first ~6
# attempts, where the old fixed floor admitted it at any time — and admitted
# the noise with it.
_LOCK_EV_ALPHA = 0.01


@dataclass(frozen=True)
class PilotLattice:
    """Per-frame-symbol scattered-pilot maps plus the frequency pilots.
    from_flat is the one decoder of the flat ParamValue wire form (parallel
    index-coupled lists) — no other module may reassemble it."""

    pilot_sets: tuple[frozenset[int], ...]
    pilot_vals: tuple[dict[int, complex], ...]
    fp_carriers: tuple[int, ...]
    fp_vals: tuple[complex, ...]

    @classmethod
    def from_flat(
        cls,
        *,
        pilot_lens: Sequence[int],
        pilot_carriers: Sequence[int],
        pilot_i: Sequence[float],
        pilot_q: Sequence[float],
        fp_carriers: Sequence[int],
        fp_i: Sequence[float],
        fp_q: Sequence[float],
    ) -> "PilotLattice":
        n = sum(pilot_lens)
        if not len(pilot_carriers) == len(pilot_i) == len(pilot_q) == n:
            raise ValueError("pilot arrays must match sum(pilot_lens)")
        if not len(fp_carriers) == len(fp_i) == len(fp_q):
            raise ValueError("fp arrays must be equal length")
        sets: list[frozenset[int]] = []
        vals: list[dict[int, complex]] = []
        off = 0
        for length in pilot_lens:
            ks = [int(k) for k in pilot_carriers[off : off + length]]
            vals.append(
                {
                    ks[j]: complex(pilot_i[off + j], pilot_q[off + j])
                    for j in range(length)
                }
            )
            sets.append(frozenset(ks))
            off += length
        return cls(
            pilot_sets=tuple(sets),
            pilot_vals=tuple(vals),
            fp_carriers=tuple(int(k) for k in fp_carriers),
            fp_vals=tuple(complex(i, q) for i, q in zip(fp_i, fp_q)),
        )


def _pilot_period(pilot_sets: Sequence[frozenset[int]], n_frame_syms: int) -> int:
    for gap in range(1, n_frame_syms):
        if all(
            pilot_sets[(fs + gap) % n_frame_syms] == pilot_sets[fs]
            for fs in range(n_frame_syms)
        ):
            return gap
    return n_frame_syms


@dataclass(frozen=True)
class LatticeGeometry:
    """The bin arithmetic every step below is expressed in, derived once from
    the frame's carrier plan and validated against the FFT."""

    fft_len: int
    n_frame_syms: int
    dc_search: int
    warmup_syms: int
    lattice: PilotLattice
    lock_min_score: float
    dc0: int
    gap: int
    union: npt.NDArray[np.int64]
    union_index: dict[int, int]
    emit: npt.NDArray[np.int64]
    keep_margin: int
    vec_cap: int

    @classmethod
    def build(
        cls,
        *,
        fft_len: int,
        n_frame_syms: int,
        n_carriers: int,
        kmin: int,
        dc_search: int,
        warmup_syms: int,
        lattice: PilotLattice,
        lock_min_score: float,
    ) -> "LatticeGeometry":
        if not kmin <= 0 <= kmin + n_carriers:
            raise ValueError(
                "carrier span must straddle DC (kmin <= 0 <= kmin + n_carriers): "
                f"kmin={kmin}, n_carriers={n_carriers}"
            )
        if dc_search < 0:
            # range(-dc_search, dc_search+1) is empty for a negative value:
            # the guard below inverts and _try_lock argmins an empty sequence
            raise ValueError(f"dc_search must be >= 0, got {dc_search}")
        problem = carrier_bin_problem(
            fft_len=fft_len,
            n_carriers=n_carriers,
            kmin=kmin,
            dc_search=dc_search,
            pilot_carriers=[k for s in lattice.pilot_sets for k in s],
            fp_carriers=lattice.fp_carriers,
        )
        if problem is not None:
            raise ValueError(problem)
        dc0 = fft_len // 2
        gap = _pilot_period(lattice.pilot_sets, n_frame_syms)
        union = np.array(sorted(set().union(*lattice.pilot_sets)), dtype=np.int64)
        keep_margin = n_frame_syms + gap
        return cls(
            fft_len=fft_len,
            n_frame_syms=n_frame_syms,
            dc_search=dc_search,
            warmup_syms=warmup_syms,
            lattice=lattice,
            lock_min_score=lock_min_score,
            dc0=dc0,
            gap=gap,
            union=union,
            union_index={int(k): i for i, k in enumerate(union)},
            emit=np.array(
                [k for k in range(kmin, kmin + n_carriers + 1) if k != 0],
                dtype=np.int64,
            ),
            keep_margin=keep_margin,
            vec_cap=4 * n_frame_syms + keep_margin + warmup_syms,
        )


class PilotLatticeCore:
    """Lock acquisition, per-carrier channel nodes and frame equalization. A
    plain class rather than a body nested in the factory closure: the lock
    search and the pruning invariant are the parts worth driving directly, and
    neither of them needs a flowgraph."""

    def __init__(self, geom: LatticeGeometry) -> None:
        self.geom = geom
        self.out = OutQueue(np.complex64)
        # LOCK_SCORE_BEST/LOCK_SCORE_MIN appear only once a lock was actually
        # attempted: an initialized 0.0 from an unfed equalizer read as a
        # decode-grade negative measured on nothing
        self.diagnostics: Diagnostics = {
            "locks": 0,
            "relocks": 0,
            "frames_emitted": 0,
        }
        self._best_score = 0.0
        # attempts persist across relocks like best-score does: the quality
        # judgment is over every attempt the run ever made
        self._attempts = 0
        self.reset_lock()

    def reset_lock(self) -> None:
        # FIFO: everything pushed before this reset drains before any
        # new-epoch item, so the push watermark is the stale/new boundary
        self._drained_baseline = self.out.pushed_total
        self._ready = False
        self._warm: list[npt.NDArray[np.complex128]] = []
        self._theta = 0.0
        self._delta = 0
        self._phi = 0
        self._emit_bins = np.empty(0, dtype=np.int64)
        self._m = 0
        self._first = 0
        self._vecs: list[npt.NDArray[np.complex128]] = []
        self._node_ms: list[list[int]] = [[] for _ in self.geom.union]
        self._node_hs: list[list[complex]] = [[] for _ in self.geom.union]
        self._frames_emitted = 0

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def saturated(self) -> bool:
        return len(self._vecs) >= self.geom.vec_cap

    def frame_ready(self) -> bool:
        if not self._ready:
            return False
        avail = (self._m - self._phi) // self.geom.n_frame_syms
        return avail > self._frames_emitted

    def restart(self) -> None:
        if self._ready:
            bump(self.diagnostics, "relocks")
        self.reset_lock()

    def ingest(self, vec: npt.NDArray[np.complex128]) -> None:
        if self._ready:
            self._vecs.append(vec)
            self._gather_nodes(self._m, vec)
            self._m += 1
            return
        self._warm.append(vec)
        if len(self._warm) > self.geom.warmup_syms:
            self._warm.pop(0)
        if len(self._warm) == self.geom.warmup_syms:
            self._try_lock()

    def _try_lock(self) -> None:
        g = self.geom
        xp = np.array(self._warm)
        dcpow = [
            float(np.mean(np.abs(xp[:, g.dc0 + d]) ** 2))
            for d in range(-g.dc_search, g.dc_search + 1)
        ]
        delta = int(np.argmin(dcpow)) - g.dc_search
        fp_carriers, fp_vals = g.lattice.fp_carriers, g.lattice.fp_vals
        pil = np.zeros((g.warmup_syms, len(fp_carriers)), dtype=np.complex128)
        for j, k in enumerate(fp_carriers):
            pil[:, j] = xp[:, k + g.dc0 + delta] * np.conj(fp_vals[j])
        theta = float(np.angle(np.sum(pil[1:] * np.conj(pil[:-1]))))
        phi, score, den = self._estimate_phi(xp, delta)
        self._attempts += 1
        floor = self._lock_floor(den)
        self._best_score = max(self._best_score, score)
        self.diagnostics[DiagnosticKey.LOCK_SCORE_BEST] = self._best_score
        self.diagnostics[DiagnosticKey.LOCK_SCORE_MIN] = floor
        self.diagnostics["lock_attempts"] = self._attempts
        if score < floor:
            return
        self._delta, self._theta, self._phi = delta, theta, phi
        self._emit_bins = g.emit + g.dc0 + delta
        self._vecs = list(self._warm)
        self._warm = []
        self._first = 0
        self._m = g.warmup_syms
        self._ready = True
        bump(self.diagnostics, "locks")
        for m in range(g.warmup_syms):
            self._gather_nodes(m, self._vecs[m])

    def _lock_floor(self, den: float) -> float:
        g = self.geom
        trials = g.n_frame_syms * self._attempts * self._attempts
        bar = math.sqrt(math.log(trials / _LOCK_EV_ALPHA) / max(den, 1.0))
        return max(g.lock_min_score, bar)

    def _estimate_phi(
        self, xp: npt.NDArray[np.complex128], delta: int
    ) -> tuple[int, float, float]:
        g = self.geom
        pilot_sets, pilot_vals = g.lattice.pilot_sets, g.lattice.pilot_vals
        best_phi, best = 0, -1.0
        den_min = math.inf
        for phi in range(g.n_frame_syms):
            fsyms = (np.arange(g.warmup_syms) - phi) % g.n_frame_syms
            num, den = 0j, 0.0
            for m in range(g.warmup_syms - g.gap):
                fs = int(fsyms[m])
                fs2 = (fs + g.gap) % g.n_frame_syms
                ks = [k for k in pilot_sets[fs] if k in pilot_vals[fs2]]
                bins = np.array([k + g.dc0 + delta for k in ks])
                ok = (bins >= 0) & (bins < g.fft_len)
                ks = [k for k, keep in zip(ks, ok) if keep]
                bins = bins[ok]
                if not ks:
                    continue
                ratio = xp[m + g.gap, bins] / (xp[m, bins] + 1e-12)
                ratio = ratio / (np.abs(ratio) + 1e-12)
                pred = np.array([pilot_vals[fs2][k] / pilot_vals[fs][k] for k in ks])
                pred = pred / np.abs(pred)
                num += complex(np.sum(ratio * np.conj(pred)))
                den += len(ks)
            score = abs(num) / max(den, 1.0)
            if den > 0.0:
                den_min = min(den_min, den)
            if score > best:
                best, best_phi = score, phi
        return best_phi, best, den_min if math.isfinite(den_min) else 1.0

    def _gather_nodes(self, m: int, vec: npt.NDArray[np.complex128]) -> None:
        g = self.geom
        fs = (m - self._phi) % g.n_frame_syms
        xd = vec * np.exp(-1j * self._theta * m)
        vals = g.lattice.pilot_vals[fs]
        for k in g.lattice.pilot_sets[fs]:
            b = k + g.dc0 + self._delta
            if 0 <= b < g.fft_len:
                i = g.union_index[k]
                self._node_ms[i].append(m)
                self._node_hs[i].append(complex(xd[b] / vals[k]))

    def emit_ready_frames(self) -> None:
        avail = (self._m - self._phi) // self.geom.n_frame_syms
        while self._frames_emitted < avail:
            self.out.push(self._equalize_frame(self._frames_emitted))
            self._frames_emitted += 1
            bump(self.diagnostics, "frames_emitted")

    def _equalize_frame(self, f: int) -> npt.NDArray[np.complex64]:
        g = self.geom
        union, emit = g.union, g.emit
        base = self._phi + g.n_frame_syms * f
        frame_end = base + g.n_frame_syms - 1
        syms = base + np.arange(g.n_frame_syms)
        hodd = np.empty((union.size, g.n_frame_syms), dtype=np.complex128)
        for i in range(union.size):
            ms = np.asarray(self._node_ms[i], dtype=np.float64)
            hs = np.asarray(self._node_hs[i], dtype=np.complex128)
            j = int(np.searchsorted(ms, frame_end, side="right"))
            hodd[i] = np.interp(syms, ms[:j], hs[:j].real) + 1j * np.interp(
                syms, ms[:j], hs[:j].imag
            )
        out = np.empty(g.n_frame_syms * emit.size, dtype=np.complex64)
        for si in range(g.n_frame_syms):
            heq = np.interp(emit, union, hodd[:, si].real) + 1j * np.interp(
                emit, union, hodd[:, si].imag
            )
            m = base + si
            xd = self._vecs[m - self._first] * np.exp(-1j * self._theta * m)
            out[si * emit.size : (si + 1) * emit.size] = xd[self._emit_bins] / heq
        return out

    def prune(self) -> None:
        # retain until DRAINED, not merely pushed — pushed-keyed pruning
        # keeps vecs flat while _out grows unbounded (vec_cap never gates)
        g = self.geom
        frame_size = g.n_frame_syms * g.emit.size
        epoch_drained = max(0, self.out.drained_total - self._drained_baseline)
        drained_frames = epoch_drained // frame_size
        keep_from = self._phi + g.n_frame_syms * drained_frames - g.keep_margin
        drop = min(keep_from - self._first, len(self._vecs))
        if drop > 0:
            del self._vecs[:drop]
            self._first += drop
        for i in range(g.union.size):
            ms = self._node_ms[i]
            j = 0
            while j < len(ms) and ms[j] < keep_from:
                j += 1
            if j:
                del ms[:j]
                del self._node_hs[i][:j]


def make_pilot_lattice_equalizer(
    gr: Any,
    *,
    fft_len: int,
    n_frame_syms: int,
    n_carriers: int,
    kmin: int,
    dc_search: int,
    warmup_syms: int,
    lattice: PilotLattice,
    lock_min_score: float = _LOCK_MIN_SCORE,
) -> Any:
    """The GR shell: stream tags in, equalized carriers out. The equalizer is
    PilotLatticeCore, which this delegates to."""
    geom = LatticeGeometry.build(
        fft_len=fft_len,
        n_frame_syms=n_frame_syms,
        n_carriers=n_carriers,
        kmin=kmin,
        dc_search=dc_search,
        warmup_syms=warmup_syms,
        lattice=lattice,
        lock_min_score=lock_min_score,
    )

    class _PilotLatticeEqualizer(gr.basic_block):
        def __init__(self) -> None:
            gr.basic_block.__init__(
                self,
                name="pilot_lattice_equalizer",
                in_sig=[(np.complex64, fft_len)],
                out_sig=[np.complex64],
            )
            self._core = PilotLatticeCore(geom)
            self._out = self._core.out
            self.diagnostics = self._core.diagnostics

        def forecast(self, noutput_items: int, ninputs: int) -> list[int]:
            return forecast_drain(
                self._out.pending or self._core.frame_ready(), ninputs
            )

        def general_work(self, input_items: Any, output_items: Any) -> int:
            vin = input_items[0]
            n = len(vin)
            if n:
                resets = self._reset_offsets(n)
                done = 0
                for i in range(n):
                    if self._core.saturated:
                        break
                    if i in resets:
                        self._core.restart()
                    self._core.ingest(np.asarray(vin[i], np.complex128))
                    done += 1
                self.consume(0, done)
            if self._core.ready:
                self._core.emit_ready_frames()
                self._core.prune()
            return self._out.drain(output_items[0])

        def _reset_offsets(self, n: int) -> set[int]:
            base = self.nitems_read(0)
            pmt = gr.pmt
            return {
                int(t.offset - base)
                for t in self.get_tags_in_window(0, 0, n)
                if pmt.symbol_to_string(t.key) == "sync_start"
            }

    return _PilotLatticeEqualizer()
