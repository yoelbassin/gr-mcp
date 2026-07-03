from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from marconi.phy.modulation.css.coding import gray_decode, gray_encode


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


def _folded(signal: np.ndarray, x: int, grid: _Grid, up: bool = True) -> np.ndarray:
    sn = grid.sample_num
    seg = signal[x : x + sn]
    ref = grid.up_ref if up else grid.down_ref
    spec = np.fft.fft(seg * ref, grid.fft_len)
    return np.abs(spec[: grid.bins]) + np.abs(spec[grid.fft_len - grid.bins :])


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


def _sfd_sync(signal: np.ndarray, x: int, grid: _Grid, cap: int) -> int | None:
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
    sfd_syms = 2.25 if up_h > dn_h else 1.25
    return x + int(round(sfd_syms * sn))


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


def _joint_sync(
    signal: np.ndarray, payload_start: int, grid: _Grid, preamble_len: int
) -> tuple[float, float]:
    """Joint estimate from the preamble up-chirps + SFD down-chirps. A preamble
    up-chirp dechirps to bin ~ (CFO + STO); an SFD down-chirp to ~ (CFO - STO).
    Returns (cfo_bins, sto_bins) as signed fractional fold-bins."""
    sn = grid.sample_num
    sfd_start = payload_start - int(round(2.25 * sn))
    u = float(
        np.median(
            [
                _peak_bin(signal, sfd_start - i * sn, grid, True)
                for i in range(1, preamble_len)
            ]
        )
    )
    d = float(
        np.median(
            [_peak_bin(signal, sfd_start + j * sn, grid, False) for j in range(2)]
        )
    )
    return (u + d) / 2.0, (u - d) / 2.0


def _modulate_symbol(s: int, sf: int, oversample: int) -> np.ndarray:
    n = 1 << sf
    t = np.arange(oversample * n) / oversample
    return (np.exp(1j * 2 * np.pi * s * t / n) * _base_upchirp(sf, oversample)).astype(
        np.complex64
    )


def _demod_symbol(chunk: np.ndarray, grid: _Grid) -> int:
    """Dechirp one symbol chunk (len == grid.sample_num) and return symbol index.
    Assumes CFO has already been removed (preamble_bin == 0 after chirp_sync)."""
    _, fine = _fine_peak(chunk, 0, grid, up=True)
    n = 1 << grid.sf
    return int(round(fine % grid.bins / grid.zero_pad)) % n


# Each GR class is defined INSIDE its builder so that `gr` is never a module-
# level name (satisfies the phy ⊥ gnuradio invariant checked by test_invariants).


def chirp_prefix(sf: int, oversample: int, preamble_len: int) -> np.ndarray:
    sn = oversample * (1 << sf)
    up = _base_upchirp(sf, oversample)
    down = np.conj(up)
    sfd = np.concatenate([down, down, down[: sn // 4]])
    return np.concatenate([np.tile(up, preamble_len), sfd]).astype(np.complex64)


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
            self._out = np.empty(0, dtype=np.complex64)

        def forecast(self, noutput_items: int, ninputs: int) -> list[int]:
            return [0 if self._out.size else 1] * ninputs

        def general_work(self, input_items: Any, output_items: Any) -> int:
            x = input_items[0]
            out = output_items[0]
            if not self._out.size and len(x):
                # One symbol can exceed the granted output window (sn > len(out)
                # for large sf); synthesize just enough and drain across calls.
                nsym = min(len(x), -(-len(out) // sn))
                self._out = np.concatenate(
                    [
                        _modulate_symbol(int(s) % (1 << sf), sf, oversample)
                        for s in x[:nsym]
                    ]
                )
                self.consume(0, nsym)
            k = min(self._out.size, len(out))
            out[:k] = self._out[:k]
            self._out = self._out[k:]
            return int(k)

    return _ChirpMod()


def make_chirp_demod(gr: Any, sf: int, oversample: int, zero_pad: int) -> Any:
    """RX: oversample*(1<<sf) complex64 chirp samples → int16 symbol index."""
    grid = _Grid(sf, oversample, zero_pad)
    sn = grid.sample_num

    class _ChirpDemod(gr.basic_block):
        def __init__(self) -> None:
            gr.basic_block.__init__(
                self,
                name="chirp_demod",
                in_sig=[np.complex64],
                out_sig=[np.int16],
            )
            self._buf = np.empty(0, dtype=np.complex64)

        def forecast(self, noutput_items: int, ninputs: int) -> list[int]:
            # A full-symbol input demand (sn) can exceed GR's default stream
            # buffer for large sf; accumulate internally and announce
            # drainability instead (the depuncture/ofdm lifecycle).
            return [0 if self._buf.size >= sn else 1] * ninputs

        def general_work(self, input_items: Any, output_items: Any) -> int:
            x = input_items[0]
            if len(x):
                self._buf = np.concatenate([self._buf, np.asarray(x, np.complex64)])
                self.consume(0, len(x))
            out = output_items[0]
            nsym = min(len(out), self._buf.size // sn)
            for i in range(nsym):
                out[i] = _demod_symbol(self._buf[i * sn : (i + 1) * sn], grid)
            if nsym:
                self._buf = self._buf[nsym * sn :]
            return nsym

    return _ChirpDemod()


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
) -> Any:
    """RX: buffer IQ, detect CSS preamble+SFD, jointly estimate CFO + fractional
    STO, apply the fractional sample timing (streaming windowed-sinc FIR) and
    derotate the CFO, emitting payload IQ."""
    grid = _Grid(sf, oversample, zero_pad)
    sn = grid.sample_num
    detect_run = preamble_len - 2
    min_buf = (preamble_len + 6) * sn
    _NTAPS = 33
    _HALF = (_NTAPS - 1) // 2
    _SFD_SPAN = (preamble_len + 6) * sn
    _LOOKBACK = _SFD_SPAN + _NTAPS - 1

    class _ChirpSync(gr.basic_block):
        def __init__(self) -> None:
            gr.basic_block.__init__(
                self, name="chirp_sync", in_sig=[np.complex64], out_sig=[np.complex64]
            )
            self._buf = np.empty(0, dtype=np.complex64)
            self._locked = False
            self._f_cfo = 0.0
            self._n_out = 0  # payload samples emitted, for CFO phase continuity
            self._taps = np.zeros(_NTAPS, dtype=np.complex64)
            self._hist = np.zeros(_NTAPS - 1, dtype=np.complex64)
            self._drop = 0  # FIR group-delay outputs still to discard
            self._scan = _DetectScan(grid, detect_run)
            self._det_x: int | None = None

        def forecast(self, noutput_items: int, ninputs: int) -> list[int]:
            if not self._locked:
                return [noutput_items] * ninputs
            return [0 if len(self._buf) else 1] * ninputs

        def _trim_prelock(self) -> None:
            anchor = self._scan.x if self._det_x is None else self._det_x
            cut = anchor - _LOOKBACK
            if cut > 0:
                self._buf = self._buf[cut:]
                self._scan.x -= cut
                if self._det_x is not None:
                    self._det_x -= cut

        def _try_lock(self) -> bool:
            if self._det_x is None:
                self._det_x = self._scan.step(self._buf)
            if self._det_x is None:
                self._trim_prelock()
                return False
            try:
                payload_start = _sfd_sync(
                    self._buf, self._det_x, grid, self._det_x + _SFD_SPAN
                )
            except ValueError:
                self._det_x = None
                self._scan.run = []
                self._trim_prelock()
                return False
            if payload_start is None:
                return False
            if payload_start + sn > len(self._buf):
                return False
            cfo_bins, sto_bins = _joint_sync(
                self._buf, payload_start, grid, preamble_len
            )
            self._f_cfo = cfo_bins * bandwidth / grid.bins
            sto = sto_bins * oversample / zero_pad  # fractional sample timing
            n_int = int(round(sto))
            mu = sto - n_int
            k = np.arange(_NTAPS) - _HALF
            h = np.sinc(k - mu) * np.blackman(_NTAPS)
            self._taps = (h / h.sum()).astype(np.complex64)
            start = payload_start - n_int  # +est_sto correction = read earlier by n_int
            if start < _NTAPS - 1:
                return False
            self._hist = self._buf[start - (_NTAPS - 1) : start].copy()
            self._buf = self._buf[start:]
            self._drop = _HALF
            return True

        def general_work(self, input_items: Any, output_items: Any) -> int:
            x = input_items[0]
            if len(x):
                self._buf = np.concatenate([self._buf, x])
                self.consume(0, len(x))
            if not self._locked:
                if len(self._buf) < min_buf and len(x) > 0:
                    return 0
                if not self._try_lock():
                    return 0
                self._locked = True
            out = output_items[0]
            m = min(len(out), len(self._buf))
            if m:
                ext = np.concatenate([self._hist, self._buf[:m]])
                y = np.convolve(ext, self._taps, "valid")  # fractional delay, len m
                self._hist = ext[-(_NTAPS - 1) :]
                self._buf = self._buf[m:]
                if self._drop:
                    d = min(self._drop, len(y))
                    y = y[d:]
                    self._drop -= d
                k = self._n_out + np.arange(len(y))
                out[: len(y)] = (
                    y
                    * np.exp(
                        -1j * 2 * np.pi * self._f_cfo * k / (oversample * bandwidth)
                    )
                ).astype(np.complex64)
                self._n_out += len(y)
                return len(y)
            return 0

    return _ChirpSync()
