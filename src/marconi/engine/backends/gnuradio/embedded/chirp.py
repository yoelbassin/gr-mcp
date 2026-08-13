from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from marconi.engine.backends.gnuradio.embedded.lifecycle import (
    Diagnostics,
    EofProbe,
    OutQueue,
    bump,
    forecast_drain,
)
from marconi.engine.coding.primitives import gray_decode


def _base_upchirp(sf: int, oversample: int) -> npt.NDArray[np.complex64]:
    n = 1 << sf
    t = np.arange(oversample * n) / oversample
    return np.exp(1j * 2 * np.pi * (t * t / (2 * n) - t / 2)).astype(np.complex64)


@dataclass(eq=False)
class _Grid:
    sf: int
    oversample: int
    zero_pad: int
    up_ref: npt.NDArray[np.complex64] = field(init=False)
    down_ref: npt.NDArray[np.complex64] = field(init=False)

    def __post_init__(self) -> None:
        up = _base_upchirp(self.sf, self.oversample)
        self.up_ref = np.conj(up)  # dechirp reference for up-chirps
        self.down_ref = up.copy()  # dechirp reference for down-chirps

    @property
    def bins(self) -> int:
        return (1 << self.sf) * self.zero_pad

    @property
    def fft_len(self) -> int:
        return self.oversample * (1 << self.sf) * self.zero_pad

    @property
    def sample_num(self) -> int:
        return self.oversample * (1 << self.sf)


def _spectrum(
    signal: npt.NDArray[np.complex64], x: int, grid: _Grid, up: bool = True
) -> npt.NDArray[np.complexfloating[Any, Any]]:
    seg = signal[x : x + grid.sample_num]
    ref = grid.up_ref if up else grid.down_ref
    return np.fft.fft(seg * ref, grid.fft_len)


def _folded_mag(
    spec: npt.NDArray[np.complexfloating[Any, Any]], grid: _Grid
) -> npt.NDArray[np.floating[Any]]:
    return np.abs(spec[: grid.bins]) + np.abs(spec[grid.fft_len - grid.bins :])


def _folded(
    signal: npt.NDArray[np.complex64], x: int, grid: _Grid, up: bool = True
) -> npt.NDArray[np.floating[Any]]:
    return _folded_mag(_spectrum(signal, x, grid, up), grid)


def _fine_peak(
    signal: npt.NDArray[np.complex64], x: int, grid: _Grid, up: bool = True
) -> tuple[float, int]:
    f = _folded(signal, x, grid, up=up)
    peak = int(np.argmax(f))
    return float(f[peak]), peak


@dataclass
class _DetectScan:
    grid: _Grid
    detect_run: int
    x: int = 0
    run: list[int] = field(default_factory=list)

    def step(self, signal: npt.NDArray[np.complex64]) -> int | None:
        sn = self.grid.sample_num
        while self.x < len(signal) - sn * self.detect_run:
            if len(self.run) == self.detect_run - 1:
                back = int(
                    round(self.run[-1] / self.grid.zero_pad * self.grid.oversample)
                )
                return self.x - back
            _, peak = _fine_peak(signal, self.x, self.grid)
            if self.run:
                delta = (self.run[-1] - peak) % self.grid.bins
                delta = min(delta, self.grid.bins - delta)
                self.run = self.run + [peak] if delta <= self.grid.zero_pad else [peak]
            else:
                self.run = [peak]
            self.x += sn
        return None


def _sfd_sync(
    signal: npt.NDArray[np.complex64],
    x: int,
    grid: _Grid,
    cap: int,
    sfd_symbols: float,
) -> int | None:
    sn = grid.sample_num
    found = False
    while x < len(signal) - sn:
        if x >= cap:
            raise ValueError("no SFD within the preamble span")
        up_h, _ = _fine_peak(signal, x, grid)
        dn_h, _ = _fine_peak(signal, x, grid, up=False)
        x += sn
        if dn_h > up_h:
            found = True
            break
    if not found:
        return None
    if x + 2 * sn > len(signal):
        return None
    _, dn_bin = _fine_peak(signal, x, grid, up=False)
    offset = (dn_bin - grid.bins) if dn_bin > grid.bins / 2 else dn_bin
    x += int(round(offset / grid.zero_pad))
    up_h, _ = _fine_peak(signal, x - sn, grid)
    dn_h, _ = _fine_peak(signal, x - sn, grid, up=False)
    sfd_syms = sfd_symbols if up_h > dn_h else sfd_symbols - 1
    return x + int(round(sfd_syms * sn))


def _preamble_end(
    signal: npt.NDArray[np.complex64], x: int, grid: _Grid, cap: int
) -> int | None:
    sn = grid.sample_num
    while x < len(signal) - sn:
        if x >= cap:
            raise ValueError("preamble run never departs within its declared span")
        if abs(_peak_bin(signal, x, grid, up=True)) > grid.zero_pad:
            return x
        x += sn
    return None


def _parabolic(f: npt.NDArray[np.floating[Any]], p: int) -> float:
    """3-point parabolic sub-bin refinement of an FFT-magnitude peak at index p."""
    if 0 < p < len(f) - 1:
        denom = f[p - 1] - 2.0 * f[p] + f[p + 1]
        if denom != 0.0:
            return float(p + 0.5 * (f[p - 1] - f[p + 1]) / denom)
    return float(p)


def _peak_bin(
    signal: npt.NDArray[np.complex64], x: int, grid: _Grid, up: bool
) -> float:
    """Signed, sub-bin dechirp peak of the window at x (fold-bin units)."""
    f = _folded(signal, x, grid, up=up)
    b = _parabolic(f, int(np.argmax(f)))
    return b - grid.bins if b > grid.bins / 2 else b


def _peak_complex(signal: npt.NDArray[np.complex64], x: int, grid: _Grid) -> complex:
    """Complex dechirp coefficient at the window's folded peak (the stronger
    of the two spectral images)."""
    spec = _spectrum(signal, x, grid)
    p = int(np.argmax(_folded_mag(spec, grid)))
    lo, hi = spec[p], spec[grid.fft_len - grid.bins + p]
    return complex(lo if abs(lo) >= abs(hi) else hi)


def _joint_sync(
    signal: npt.NDArray[np.complex64],
    payload_start: int,
    grid: _Grid,
    preamble_len: int,
    sfd_symbols: float,
    sync_symbols: int,
) -> tuple[float, float]:
    """Joint estimate from the preamble up-chirps + SFD down-chirps. A preamble
    up-chirp dechirps to bin ~ (CFO + STO); an SFD down-chirp to ~ (CFO - STO).
    Returns (cfo_bins, sto_bins) as signed fractional fold-bins."""
    sn = grid.sample_num
    sfd_start = payload_start - int(round(sfd_symbols * sn))
    ups = [sfd_start - (sync_symbols + i) * sn for i in range(1, preamble_len)]
    u = float(np.median([_peak_bin(signal, x, grid, True) for x in ups if x >= 0]))
    down_windows = min(2, int(sfd_symbols))
    d = float(
        np.median(
            [
                _peak_bin(signal, sfd_start + j * sn, grid, False)
                for j in range(down_windows)
            ]
        )
    )
    return (u + d) / 2.0, (u - d) / 2.0


def _preamble_sync(
    signal: npt.NDArray[np.complex64], end: int, grid: _Grid, preamble_len: int
) -> tuple[float, float]:
    """Estimate from the preamble run alone (no SFD). The dechirp peak carries
    (CFO + STO) jointly; the CFO alone advances the dechirped tone's phase by
    2*pi*CFO*T_sym per symbol, so the inter-symbol phase slope recovers CFO
    within +/- half a bin. Any whole-bin CFO remainder lands in the STO term,
    where its timing and carrier residuals cancel in payload dechirp bins.
    Returns (cfo_bins, sto_bins) as signed fractional fold-bins."""
    sn = grid.sample_num
    xs = [end - i * sn for i in range(preamble_len, 0, -1) if end - i * sn >= 0]
    b = float(np.median([_peak_bin(signal, x, grid, True) for x in xs]))
    zs = [_peak_complex(signal, x, grid) for x in xs]
    rot = complex(sum(z2 * np.conj(z1) for z1, z2 in zip(zs, zs[1:])))
    cfo = float(np.angle(rot)) / (2.0 * np.pi) * grid.zero_pad
    return cfo, b - cfo


# Each GR class is defined INSIDE its builder so that `gr` is never a module-
# level name (satisfies the engine ⊥ gnuradio invariant in test_invariants).


def dechirp_ref(sf: int, oversample: int) -> npt.NDArray[np.complex64]:
    return np.conj(_base_upchirp(sf, oversample))


def make_css_demap(gr: Any, sf: int) -> Any:
    class _CssDemap(gr.basic_block):
        def __init__(self) -> None:
            gr.basic_block.__init__(
                self,
                name="css_demap",
                in_sig=[np.int16],
                out_sig=[np.uint8],
            )

        def forecast(self, noutput_items: int, ninputs: int) -> list[int]:
            return [max(1, noutput_items // sf)] * ninputs

        def general_work(self, input_items: Any, output_items: Any) -> int:
            x = input_items[0]
            out = output_items[0]
            nsym = min(len(x), len(out) // sf)
            for i in range(nsym):
                s = gray_decode(int(x[i]) & ((1 << sf) - 1))
                bits = [(s >> (sf - 1 - j)) & 1 for j in range(sf)]
                out[i * sf : (i + 1) * sf] = np.asarray(bits, dtype=np.uint8)
            self.consume(0, nsym)
            return nsym * sf

    return _CssDemap()


_NTAPS = 33
_HALF = (_NTAPS - 1) // 2
# FIR group delay + fractional-timing rounding slack; stays under one symbol
# window (sn >= 32) so pad zeros can only complete the final real window, never
# form a bogus trailing one
_EOF_PAD = _HALF + 8
_OUT_CAP = 1 << 16  # pending-out saturation: stop consuming, let GR backpressure


@dataclass(frozen=True)
class ChirpSyncGeometry:
    """The burst anatomy the detector and the estimators are measured in."""

    grid: _Grid
    sn: int
    bandwidth: float
    oversample: int
    zero_pad: int
    preamble_len: int
    sfd_symbols: float
    sync_symbols: int
    detect_run: int
    hunt_span: int

    @classmethod
    def build(
        cls,
        *,
        sf: int,
        oversample: int,
        zero_pad: int,
        preamble_len: int,
        bandwidth: float,
        sfd_symbols: float,
        sync_symbols: int,
    ) -> "ChirpSyncGeometry":
        grid = _Grid(sf, oversample, zero_pad)
        sn = grid.sample_num
        return cls(
            grid=grid,
            sn=sn,
            bandwidth=bandwidth,
            oversample=oversample,
            zero_pad=zero_pad,
            preamble_len=preamble_len,
            sfd_symbols=sfd_symbols,
            sync_symbols=sync_symbols,
            detect_run=preamble_len - sync_symbols,
            # the whole declared anatomy plus detector slack
            hunt_span=(preamble_len + sync_symbols + int(np.ceil(sfd_symbols)) + 2)
            * sn,
        )


class ChirpSyncCore:
    """Detection, per-burst CFO/STO estimation, and the fractional-timing FIR.
    A plain class rather than a body nested in the factory closure: the EOF
    rules and the tag bookkeeping are the parts that have been wrong before,
    and they are worth driving without a flowgraph. The two EOF facts it cannot
    know — whether the source is exhausted, and whether every sample is already
    in hand — arrive as arguments from the shell that can answer them."""

    def __init__(self, geom: ChirpSyncGeometry) -> None:
        self.geom = geom
        self.out = OutQueue(np.complex64)
        self.diagnostics: Diagnostics = {"locks": 0, "eof_flushed": 0}
        self.tagq: list[int] = []  # out indices (absolute) of burst starts
        self._buf = np.empty(0, dtype=np.complex64)
        self._armed = False
        self._f_cfo = 0.0
        self._n_out = 0  # per-burst payload samples, for CFO phase continuity
        self._taps = np.zeros(_NTAPS, dtype=np.complex64)
        self._hist = np.zeros(_NTAPS - 1, dtype=np.complex64)
        self._drop = 0  # FIR group-delay outputs still to discard
        self._seg = 0  # samples appended to out for the current segment
        self._scan = _DetectScan(geom.grid, geom.detect_run)
        self._det_x: int | None = None
        self._eof_padded = False  # FIR tail zeros injected (once, at flush)

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def saturated(self) -> bool:
        return self.out.size >= _OUT_CAP

    def feed(self, x: npt.NDArray[np.complex64]) -> None:
        self._buf = np.concatenate([self._buf, x])

    def has_work(self, exhausted: bool) -> bool:
        return bool(
            self.out.pending
            or (self._armed and self.cleared() > 0)
            or self.eof_flushable(exhausted) > 0
        )

    def eof_flushable(self, exhausted: bool) -> int:
        """Whole symbol windows of buffered payload releasable once the
        source is exhausted: emitting them early is exactly what normal
        streaming would do, so a straggler still in flight through an
        upstream buffer appends seamlessly (window alignment holds)."""
        if not self._armed or not exhausted:
            return 0
        sn = self.geom.sn
        return max(0, (self._seg + len(self._buf)) // sn * sn - self._seg)

    def cleared(self) -> int:
        """Samples of _buf beyond any claim by an in-progress detection
        (with one symbol of slack for the joint estimator's look-back)."""
        sn = self.geom.sn
        c = self._scan.x - (len(self._scan.run) + 1) * sn
        if self._det_x is not None:
            c = min(c, self._det_x)
        return max(0, min(c - sn, len(self._buf)))

    def hunt(self) -> tuple[int, float, float] | None:
        g = self.geom
        if self._det_x is None:
            self._det_x = self._scan.step(self._buf)
        if self._det_x is None:
            return None
        cap = self._det_x + g.hunt_span
        try:
            if g.sfd_symbols:
                payload_start = _sfd_sync(
                    self._buf, self._det_x, g.grid, cap, g.sfd_symbols
                )
            else:
                end = _preamble_end(self._buf, self._det_x, g.grid, cap)
                payload_start = None if end is None else end + g.sync_symbols * g.sn
        except ValueError:
            self._det_x = None
            self._scan.run = []
            return None
        if payload_start is None or payload_start + g.sn > len(self._buf):
            return None
        if g.sfd_symbols:
            cfo_bins, sto_bins = _joint_sync(
                self._buf,
                payload_start,
                g.grid,
                g.preamble_len,
                g.sfd_symbols,
                g.sync_symbols,
            )
        else:
            cfo_bins, sto_bins = _preamble_sync(
                self._buf,
                payload_start - g.sync_symbols * g.sn,
                g.grid,
                g.preamble_len,
            )
        return payload_start, cfo_bins, sto_bins

    def _fir_to_out(self, upto: int) -> None:
        if upto <= 0:
            return
        g = self.geom
        ext = np.concatenate([self._hist, self._buf[:upto]])
        y = np.convolve(ext, self._taps, "valid")
        self._hist = ext[-(_NTAPS - 1) :]
        self._trim(upto)
        if self._drop:
            d = min(self._drop, len(y))
            y = y[d:]
            self._drop -= d
        k = self._n_out + np.arange(len(y))
        rotated = (
            y * np.exp(-1j * 2 * np.pi * self._f_cfo * k / (g.oversample * g.bandwidth))
        ).astype(np.complex64)
        self._n_out += len(y)
        self.out.push(rotated)
        self._seg += len(rotated)

    def _trim(self, cut: int) -> None:
        if cut <= 0:
            return
        self._buf = self._buf[cut:]
        self._scan.x = max(0, self._scan.x - cut)
        if self._det_x is not None:
            self._det_x = max(0, self._det_x - cut)

    def relock(self, payload_start: int, cfo_bins: float, sto_bins: float) -> None:
        g = self.geom
        if self._armed:
            # flush the old segment up to an emitted-count symbol boundary
            # (so a downstream demod's windows stay aligned across bursts)
            boundary = self._det_x if self._det_x is not None else payload_start
            flush = max(0, ((self._seg + boundary) // g.sn) * g.sn - self._seg)
            self._fir_to_out(flush)
            payload_start -= flush  # _fir_to_out trimmed _buf by `flush`
        self._f_cfo = cfo_bins * g.bandwidth / g.grid.bins
        sto = sto_bins * g.oversample / g.zero_pad  # fractional sample timing
        n_int = int(round(sto))
        mu = sto - n_int
        k = np.arange(_NTAPS) - _HALF
        h = np.sinc(k - mu) * np.blackman(_NTAPS)
        self._taps = (h / h.sum()).astype(np.complex64)
        start = max(_NTAPS - 1, payload_start - n_int)
        self._hist = self._buf[start - (_NTAPS - 1) : start].copy()
        self._trim(start)
        self._drop = _HALF
        self._n_out = 0
        self._seg = 0
        self._scan = _DetectScan(g.grid, g.detect_run)
        self._det_x = None
        self.tagq.append(self.out.pushed_total)
        self._armed = True
        bump(self.diagnostics, "locks")

    def pump(self, *, exhausted: bool, final: bool) -> None:
        """One detect-and-emit pass. `final` licenses the irreversible FIR-tail
        pad: exhausted() alone is NOT finality — for a small capture the source
        finishes almost immediately while the stream is still in flight, and a
        mid-stream pad would splice zeros into real payload."""
        while True:
            lock = self.hunt()
            if lock is None:
                break
            self.relock(*lock)
        if not self._armed:
            self._trim(self.cleared())
            return
        if self.saturated:
            return
        upto = self.cleared()
        if final and not self._eof_padded:
            # the FIR's last outputs need future samples (group delay plus the
            # fractional-timing shift); none can ever come, so zeros expel the
            # real tail — they only feed the interpolation edge past the last
            # sample
            self._buf = np.concatenate([self._buf, np.zeros(_EOF_PAD, np.complex64)])
            self._eof_padded = True
        flush = self.eof_flushable(exhausted)
        if flush > upto:
            bump(self.diagnostics, "eof_flushed", flush - upto)
            upto = flush
        self._fir_to_out(upto)


def make_chirp_sync(
    gr: Any,
    sf: int,
    oversample: int,
    zero_pad: int,
    preamble_len: int,
    bandwidth: float,
    sfd_symbols: float,
    sync_symbols: int,
) -> Any:
    """RX: buffer IQ, detect CSS preamble+SFD, jointly estimate CFO + fractional
    STO per burst, apply the fractional sample timing (streaming windowed-sinc
    FIR) and derotate the CFO, emitting payload IQ. The hunt never stops: each
    further preamble re-arms with fresh estimates; every burst's first payload
    sample carries a "burst" stream tag, and each burst segment is emitted as a
    whole number of symbol windows so a downstream demod stays aligned.
    Emission trails the detector by a bounded margin (samples an in-progress
    detection could still claim are withheld, including at EOF).

    The shell below owns only what needs GNU Radio — the EOF probe, stream tags
    and backpressure; the synchronizer is ChirpSyncCore."""
    pmt = gr.pmt
    geom = ChirpSyncGeometry.build(
        sf=sf,
        oversample=oversample,
        zero_pad=zero_pad,
        preamble_len=preamble_len,
        bandwidth=bandwidth,
        sfd_symbols=sfd_symbols,
        sync_symbols=sync_symbols,
    )

    class _ChirpSync(gr.basic_block):
        def __init__(self) -> None:
            gr.basic_block.__init__(
                self, name="chirp_sync", in_sig=[np.complex64], out_sig=[np.complex64]
            )
            self._core = ChirpSyncCore(geom)
            self._out = self._core.out
            self.diagnostics = self._core.diagnostics
            self.eof_probe: EofProbe | None = None

        def _exhausted(self) -> bool:
            return self.eof_probe is not None and self.eof_probe.exhausted()

        def _final(self) -> bool:
            """Every source sample is provably buffered already (direct source
            feed + read counter at total)."""
            p = self.eof_probe
            return (
                self._core.armed
                and p is not None
                and p.expected_items is not None
                and self.nitems_read(0) >= p.expected_items
            )

        def forecast(self, noutput_items: int, ninputs: int) -> list[int]:
            return forecast_drain(self._core.has_work(self._exhausted()), ninputs)

        def general_work(self, input_items: Any, output_items: Any) -> int:
            x = input_items[0]
            # Saturated pending-out: leave the input unconsumed so GR's own
            # backpressure reaches upstream instead of _out growing unboundedly
            # under a slow consumer.
            if len(x) and not self._core.saturated:
                self._core.feed(x)
                self.consume(0, len(x))
            self._core.pump(exhausted=self._exhausted(), final=self._final())
            before = self._out.drained_total
            k = self._out.drain(output_items[0])
            if k:
                self._emit_tags(before, k)
            return k

        def _emit_tags(self, before: int, k: int) -> None:
            for tag in self._core.tagq:
                if before <= tag < before + k:
                    self.add_item_tag(
                        0,
                        self.nitems_written(0) + (tag - before),
                        pmt.intern("burst"),
                        pmt.PMT_NIL,
                    )
            self._core.tagq = [t for t in self._core.tagq if t >= before + k]

    return _ChirpSync()
