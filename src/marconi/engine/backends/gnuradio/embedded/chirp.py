from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from marconi.engine.backends.gnuradio.embedded.lifecycle import OutQueue, forecast_drain
from marconi.engine.coding.primitives import gray_decode, gray_encode


def _base_upchirp(sf: int, oversample: int) -> np.ndarray:
    n = 1 << sf
    t = np.arange(oversample * n) / oversample
    return np.exp(1j * 2 * np.pi * (t * t / (2 * n) - t / 2)).astype(np.complex64)


@dataclass(eq=False)
class _Grid:
    sf: int
    oversample: int
    zero_pad: int
    up_ref: np.ndarray = field(init=False)
    down_ref: np.ndarray = field(init=False)

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


def _spectrum(signal: np.ndarray, x: int, grid: _Grid, up: bool = True) -> np.ndarray:
    seg = signal[x : x + grid.sample_num]
    ref = grid.up_ref if up else grid.down_ref
    return np.fft.fft(seg * ref, grid.fft_len)


def _folded_mag(spec: np.ndarray, grid: _Grid) -> np.ndarray:
    return np.abs(spec[: grid.bins]) + np.abs(spec[grid.fft_len - grid.bins :])


def _folded(signal: np.ndarray, x: int, grid: _Grid, up: bool = True) -> np.ndarray:
    return _folded_mag(_spectrum(signal, x, grid, up), grid)


def _fine_peak(
    signal: np.ndarray, x: int, grid: _Grid, up: bool = True
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

    def step(self, signal: np.ndarray) -> int | None:
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
    signal: np.ndarray, x: int, grid: _Grid, cap: int, sfd_symbols: float
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


def _preamble_end(signal: np.ndarray, x: int, grid: _Grid, cap: int) -> int | None:
    sn = grid.sample_num
    while x < len(signal) - sn:
        if x >= cap:
            raise ValueError("preamble run never departs within its declared span")
        if abs(_peak_bin(signal, x, grid, up=True)) > grid.zero_pad:
            return x
        x += sn
    return None


def _parabolic(f: np.ndarray, p: int) -> float:
    """3-point parabolic sub-bin refinement of an FFT-magnitude peak at index p."""
    if 0 < p < len(f) - 1:
        denom = f[p - 1] - 2.0 * f[p] + f[p + 1]
        if denom != 0.0:
            return p + 0.5 * (f[p - 1] - f[p + 1]) / denom
    return float(p)


def _peak_bin(signal: np.ndarray, x: int, grid: _Grid, up: bool) -> float:
    """Signed, sub-bin dechirp peak of the window at x (fold-bin units)."""
    f = _folded(signal, x, grid, up=up)
    b = _parabolic(f, int(np.argmax(f)))
    return b - grid.bins if b > grid.bins / 2 else b


def _peak_complex(signal: np.ndarray, x: int, grid: _Grid) -> complex:
    """Complex dechirp coefficient at the window's folded peak (the stronger
    of the two spectral images)."""
    spec = _spectrum(signal, x, grid)
    p = int(np.argmax(_folded_mag(spec, grid)))
    lo, hi = spec[p], spec[grid.fft_len - grid.bins + p]
    return complex(lo if abs(lo) >= abs(hi) else hi)


def _joint_sync(
    signal: np.ndarray,
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
    signal: np.ndarray, end: int, grid: _Grid, preamble_len: int
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


def _modulate_symbol(s: int, sf: int, oversample: int) -> np.ndarray:
    n = 1 << sf
    t = np.arange(oversample * n) / oversample
    return (np.exp(1j * 2 * np.pi * s * t / n) * _base_upchirp(sf, oversample)).astype(
        np.complex64
    )


# Each GR class is defined INSIDE its builder so that `gr` is never a module-
# level name (satisfies the phy ⊥ gnuradio invariant checked by test_invariants).


def chirp_prefix(
    sf: int, oversample: int, preamble_len: int, sfd_symbols: float
) -> np.ndarray:
    sn = oversample * (1 << sf)
    up = _base_upchirp(sf, oversample)
    down = np.conj(up)
    full = int(sfd_symbols)
    frac = int(round((sfd_symbols - full) * sn))
    sfd = np.concatenate([np.tile(down, full), down[:frac]])
    return np.concatenate([np.tile(up, preamble_len), sfd]).astype(np.complex64)


def dechirp_ref(sf: int, oversample: int) -> np.ndarray:
    return np.conj(_base_upchirp(sf, oversample))


def make_chirp_mod(gr: Any, sf: int, oversample: int) -> Any:
    sn = oversample * (1 << sf)

    class _ChirpMod(gr.basic_block):
        def __init__(self) -> None:
            gr.basic_block.__init__(
                self,
                name="chirp_mod",
                in_sig=[np.int16],
                out_sig=[np.complex64],
            )
            self._out = OutQueue(np.complex64)

        def forecast(self, noutput_items: int, ninputs: int) -> list[int]:
            return forecast_drain(self._out.pending, ninputs)

        def general_work(self, input_items: Any, output_items: Any) -> int:
            x = input_items[0]
            out = output_items[0]
            if not self._out.pending and len(x):
                # One symbol can exceed the granted output window (sn > len(out)
                # for large sf); synthesize just enough and drain across calls.
                nsym = min(len(x), -(-len(out) // sn))
                self._out.push(
                    np.concatenate(
                        [
                            _modulate_symbol(int(s) % (1 << sf), sf, oversample)
                            for s in x[:nsym]
                        ]
                    )
                )
                self.consume(0, nsym)
            return self._out.drain(out)

    return _ChirpMod()


def make_css_map(gr: Any, sf: int) -> Any:
    class _CssMap(gr.basic_block):
        def __init__(self) -> None:
            gr.basic_block.__init__(
                self,
                name="css_map",
                in_sig=[np.uint8],
                out_sig=[np.int16],
            )

        def forecast(self, noutput_items: int, ninputs: int) -> list[int]:
            return [noutput_items * sf] * ninputs

        def general_work(self, input_items: Any, output_items: Any) -> int:
            x = input_items[0]
            out = output_items[0]
            nsym = min(len(out), len(x) // sf)
            for i in range(nsym):
                bits = x[i * sf : (i + 1) * sf]
                s = int(np.asarray(bits, dtype=int).dot(1 << np.arange(sf)[::-1]))
                out[i] = gray_encode(s)
            self.consume(0, nsym * sf)
            return nsym

    return _CssMap()


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
    detection could still claim are withheld, including at EOF)."""
    pmt = gr.pmt
    grid = _Grid(sf, oversample, zero_pad)
    sn = grid.sample_num
    detect_run = preamble_len - sync_symbols
    _NTAPS = 33
    _HALF = (_NTAPS - 1) // 2
    _HUNT_SPAN = (
        preamble_len + sync_symbols + int(np.ceil(sfd_symbols)) + 2
    ) * sn  # the whole declared anatomy plus detector slack
    # FIR group delay + fractional-timing rounding slack; stays under one
    # symbol window (sn >= 32) so pad zeros can only complete the final real
    # window, never form a bogus trailing one
    _EOF_PAD = _HALF + 8
    _OUT_CAP = 1 << 16  # pending-out saturation: stop consuming, let GR backpressure

    class _ChirpSync(gr.basic_block):
        def __init__(self) -> None:
            gr.basic_block.__init__(
                self, name="chirp_sync", in_sig=[np.complex64], out_sig=[np.complex64]
            )
            self._buf = np.empty(0, dtype=np.complex64)
            self._armed = False
            self._f_cfo = 0.0
            self._n_out = 0  # per-burst payload samples, for CFO phase continuity
            self._taps = np.zeros(_NTAPS, dtype=np.complex64)
            self._hist = np.zeros(_NTAPS - 1, dtype=np.complex64)
            self._drop = 0  # FIR group-delay outputs still to discard
            self._seg = 0  # samples appended to _out for the current segment
            self._scan = _DetectScan(grid, detect_run)
            self._det_x: int | None = None
            self._out = OutQueue(np.complex64)
            self._tagq: list[int] = []  # _out indices (absolute) of burst starts
            self.eof_probe: Any = None
            self._eof_padded = False  # FIR tail zeros injected (once, at flush)
            self.diagnostics = {"locks": 0, "eof_flushed": 0}

        def forecast(self, noutput_items: int, ninputs: int) -> list[int]:
            work = (
                self._out.pending
                or (self._armed and self._cleared() > 0)
                or self._eof_flushable() > 0
            )
            return forecast_drain(bool(work), ninputs)

        def _eof_flushable(self) -> int:
            """Whole symbol windows of buffered payload releasable once the
            source is exhausted: emitting them early is exactly what normal
            streaming would do, so a straggler still in flight through an
            upstream buffer appends seamlessly (window alignment holds)."""
            if (
                not self._armed
                or self.eof_probe is None
                or not self.eof_probe.exhausted()
            ):
                return 0
            return max(0, (self._seg + len(self._buf)) // sn * sn - self._seg)

        def _eof_final(self) -> bool:
            """True only when every source sample is provably in _buf already
            (direct source feed + read counter at total) — the license for
            the irreversible FIR-tail pad. exhausted() alone is NOT finality:
            for a small capture the source finishes almost immediately while
            the stream is still in flight, and a mid-stream pad would splice
            zeros into real payload."""
            p = self.eof_probe
            return (
                self._armed
                and p is not None
                and p.expected_items is not None
                and self.nitems_read(0) >= p.expected_items
            )

        def _cleared(self) -> int:
            """Samples of _buf beyond any claim by an in-progress detection
            (with one symbol of slack for the joint estimator's look-back)."""
            c = self._scan.x - (len(self._scan.run) + 1) * sn
            if self._det_x is not None:
                c = min(c, self._det_x)
            return max(0, min(c - sn, len(self._buf)))

        def _hunt(self) -> tuple[int, float, float] | None:
            if self._det_x is None:
                self._det_x = self._scan.step(self._buf)
            if self._det_x is None:
                return None
            cap = self._det_x + _HUNT_SPAN
            try:
                if sfd_symbols:
                    payload_start = _sfd_sync(
                        self._buf, self._det_x, grid, cap, sfd_symbols
                    )
                else:
                    end = _preamble_end(self._buf, self._det_x, grid, cap)
                    payload_start = None if end is None else end + sync_symbols * sn
            except ValueError:
                self._det_x = None
                self._scan.run = []
                return None
            if payload_start is None or payload_start + sn > len(self._buf):
                return None
            if sfd_symbols:
                cfo_bins, sto_bins = _joint_sync(
                    self._buf,
                    payload_start,
                    grid,
                    preamble_len,
                    sfd_symbols,
                    sync_symbols,
                )
            else:
                cfo_bins, sto_bins = _preamble_sync(
                    self._buf, payload_start - sync_symbols * sn, grid, preamble_len
                )
            return payload_start, cfo_bins, sto_bins

        def _fir_to_out(self, upto: int) -> None:
            """Move _buf[:upto] through the current FIR+CFO into _out."""
            if upto <= 0:
                return
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
                y * np.exp(-1j * 2 * np.pi * self._f_cfo * k / (oversample * bandwidth))
            ).astype(np.complex64)
            self._n_out += len(y)
            self._out.push(rotated)
            self._seg += len(rotated)

        def _trim(self, cut: int) -> None:
            if cut <= 0:
                return
            self._buf = self._buf[cut:]
            self._scan.x = max(0, self._scan.x - cut)
            if self._det_x is not None:
                self._det_x = max(0, self._det_x - cut)

        def _relock(self, payload_start: int, cfo_bins: float, sto_bins: float) -> None:
            if self._armed:
                # flush the old segment up to an emitted-count symbol boundary
                # (so a downstream demod's windows stay aligned across bursts)
                boundary = self._det_x if self._det_x is not None else payload_start
                flush = max(0, ((self._seg + boundary) // sn) * sn - self._seg)
                self._fir_to_out(flush)
                payload_start -= flush  # _fir_to_out trimmed _buf by `flush`
            self._f_cfo = cfo_bins * bandwidth / grid.bins
            sto = sto_bins * oversample / zero_pad  # fractional sample timing
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
            self._scan = _DetectScan(grid, detect_run)
            self._det_x = None
            self._tagq.append(self._out.pushed_total)
            self._armed = True
            self.diagnostics["locks"] += 1

        def general_work(self, input_items: Any, output_items: Any) -> int:
            x = input_items[0]
            # Saturated pending-out: leave the input unconsumed so GR's own
            # backpressure reaches upstream instead of _out growing unboundedly
            # under a slow consumer.
            if len(x) and self._out.size < _OUT_CAP:
                self._buf = np.concatenate([self._buf, x])
                self.consume(0, len(x))
            while True:
                lock = self._hunt()
                if lock is None:
                    break
                self._relock(*lock)
            if self._armed and self._out.size < _OUT_CAP:
                upto = self._cleared()
                if self._eof_final() and not self._eof_padded:
                    # the FIR's last outputs need future samples (group delay
                    # plus the fractional-timing shift); none can ever come,
                    # so zeros expel the real tail — they only feed the
                    # interpolation edge past the last sample
                    self._buf = np.concatenate(
                        [self._buf, np.zeros(_EOF_PAD, np.complex64)]
                    )
                    self._eof_padded = True
                flush = self._eof_flushable()
                if flush > upto:
                    self.diagnostics["eof_flushed"] += flush - upto
                    upto = flush
                self._fir_to_out(upto)
            elif not self._armed:
                self._trim(self._cleared())
            before = self._out.drained_total
            k = self._out.drain(output_items[0])
            if k:
                for tag in self._tagq:
                    if before <= tag < before + k:
                        self.add_item_tag(
                            0,
                            self.nitems_written(0) + (tag - before),
                            pmt.intern("burst"),
                            pmt.PMT_NIL,
                        )
                self._tagq = [t for t in self._tagq if t >= before + k]
            return k

    return _ChirpSync()
