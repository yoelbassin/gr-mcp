from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ─── Module-level numpy DSP (zero GR imports) ────────────────────────────────
# Ported verbatim from the de-risk scripts (derisk_css.py / derisk_css2.py)
# that reproduced the Flinders SF11 oracle (conf 0.9961, 20/20 symbols).


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


def _detect(signal: np.ndarray, grid: _Grid, detect_run: int) -> int:
    sn = grid.sample_num
    x = 0
    run: list[int] = []
    while x < len(signal) - sn * detect_run:
        if len(run) == detect_run - 1:
            return x - int(round(run[-1] / grid.zero_pad * grid.oversample))
        _, peak = _fine_peak(signal, x, grid)
        if run:
            delta = (run[-1] - peak) % grid.bins
            delta = min(delta, grid.bins - delta)
            run = run + [peak] if delta <= grid.zero_pad else [peak]
        else:
            run = [peak]
        x += sn
    raise ValueError("no CSS preamble detected")


def _synchronize(signal: np.ndarray, grid: _Grid, detect_run: int) -> tuple[int, int]:
    """Return (payload_start, preamble_start) sample offsets into `signal`."""
    sn = grid.sample_num
    x = _detect(signal, grid, detect_run)
    found = False
    while x < len(signal) - sn:
        up_h, _ = _fine_peak(signal, x, grid)
        dn_h, _ = _fine_peak(signal, x, grid, up=False)
        x += sn
        if dn_h > up_h:
            found = True
            break
    if not found:
        raise ValueError("SFD not found")
    _, dn_bin = _fine_peak(signal, x, grid, up=False)
    offset = (dn_bin - grid.bins) if dn_bin > grid.bins / 2 else dn_bin
    x += int(round(offset / grid.zero_pad))
    preamble_start = x - 4 * sn
    up_h, _ = _fine_peak(signal, x - sn, grid)
    dn_h, _ = _fine_peak(signal, x - sn, grid, up=False)
    sfd_syms = 2.25 if up_h > dn_h else 1.25
    payload_start = x + int(round(sfd_syms * sn))
    return payload_start, preamble_start


def _cfo_hz(preamble_bin: int, grid: _Grid, bandwidth: float) -> float:
    """Convert preamble dechirp bin to a carrier-frequency offset in Hz."""
    n = 1 << grid.sf
    pb_n = preamble_bin / grid.zero_pad
    if pb_n > n / 2:
        pb_n -= n
    return pb_n * bandwidth / n


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
    signal: np.ndarray, payload_start: int, grid: _Grid
) -> tuple[float, float]:
    """Joint estimate from the preamble up-chirps + SFD down-chirps. A preamble
    up-chirp dechirps to bin ~ (CFO + STO); an SFD down-chirp to ~ (CFO - STO).
    Returns (cfo_bins, sto_bins) as signed fractional fold-bins.

    _folded with down_ref=up (not time-reversed up) places SFD peaks at
    (CFO - STO - zero_pad/oversample); adding zero_pad/oversample corrects d."""
    sn = grid.sample_num
    sfd_start = payload_start - int(round(2.25 * sn))
    u = float(
        np.median(
            [_peak_bin(signal, sfd_start - i * sn, grid, True) for i in range(1, 8)]
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


def _gray_encode(n: int) -> int:
    return n ^ (n >> 1)


def _gray_decode(g: int) -> int:
    n = g
    g >>= 1
    while g:
        n ^= g
        g >>= 1
    return n


# ─── Embedded GR-block builders ──────────────────────────────────────────────
# Each GR class is defined INSIDE its builder so that `gr` is never a module-
# level name (satisfies the phy ⊥ gnuradio invariant checked by test_invariants).


def make_chirp_prepend(gr: Any, sf: int, oversample: int, preamble_len: int) -> Any:
    """TX: emit preamble_len up-chirps + 2.25 down-chirps, then pass IQ 1:1."""
    sn = oversample * (1 << sf)
    up = _base_upchirp(sf, oversample)
    down = np.conj(up)
    sfd = np.concatenate([down, down, down[: sn // 4]])
    prepend = np.concatenate([np.tile(up, preamble_len), sfd]).astype(np.complex64)

    class _ChirpPrepend(gr.basic_block):
        def __init__(self) -> None:
            gr.basic_block.__init__(
                self,
                name="chirp_prepend",
                in_sig=[np.complex64],
                out_sig=[np.complex64],
            )
            self._pre = prepend.copy()
            self._done = False

        def forecast(self, noutput_items: int, ninputs: int) -> list[int]:
            return [0] * ninputs if not self._done else [noutput_items] * ninputs

        def general_work(self, input_items: Any, output_items: Any) -> int:
            out = output_items[0]
            o = 0
            if not self._done:
                k = min(len(out), len(self._pre))
                out[:k] = self._pre[:k]
                self._pre = self._pre[k:]
                o += k
                if len(self._pre) == 0:
                    self._done = True
                if o == len(out):
                    return o
            x = input_items[0]
            m = min(len(out) - o, len(x))
            if m:
                out[o : o + m] = x[:m]
                self.consume(0, m)
            return o + m

    return _ChirpPrepend()


def make_chirp_mod(gr: Any, sf: int, oversample: int) -> Any:
    """TX: int16 symbol index → oversample*(1<<sf) complex64 chirp samples."""
    sn = oversample * (1 << sf)
    table = np.stack([_modulate_symbol(s, sf, oversample) for s in range(1 << sf)])

    class _ChirpMod(gr.basic_block):
        def __init__(self) -> None:
            gr.basic_block.__init__(
                self,
                name="chirp_mod",
                in_sig=[np.int16],
                out_sig=[np.complex64],
            )

        def forecast(self, noutput_items: int, ninputs: int) -> list[int]:
            return [max(1, noutput_items // sn)] * ninputs

        def general_work(self, input_items: Any, output_items: Any) -> int:
            x = input_items[0]
            out = output_items[0]
            nsym = min(len(x), len(out) // sn)
            for i in range(nsym):
                out[i * sn : (i + 1) * sn] = table[int(x[i]) % (1 << sf)]
            self.consume(0, nsym)
            return nsym * sn

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

        def forecast(self, noutput_items: int, ninputs: int) -> list[int]:
            return [noutput_items * sn] * ninputs

        def general_work(self, input_items: Any, output_items: Any) -> int:
            x = input_items[0]
            out = output_items[0]
            nsym = min(len(out), len(x) // sn)
            for i in range(nsym):
                out[i] = _demod_symbol(x[i * sn : (i + 1) * sn], grid)
            self.consume(0, nsym * sn)
            return nsym

    return _ChirpDemod()


def make_css_map(gr: Any, sf: int) -> Any:
    """TX: sf uint8 bits (MSB-first) → one Gray-encoded int16 symbol index."""

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
                out[i] = _gray_encode(s)
            self.consume(0, nsym * sf)
            return nsym

    return _CssMap()


def make_css_demap(gr: Any, sf: int) -> Any:
    """RX: one Gray-encoded int16 symbol index → sf uint8 bits (MSB-first)."""

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
                s = _gray_decode(int(x[i]) & ((1 << sf) - 1))
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
    derotate the CFO, emitting payload IQ. Mirrors make_sym_acquire's
    buffer/lock/EOF-flush shape."""
    grid = _Grid(sf, oversample, zero_pad)
    sn = grid.sample_num
    detect_run = preamble_len - 2
    min_buf = (preamble_len + 6) * sn
    _NTAPS = 33
    _HALF = (_NTAPS - 1) // 2

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
            self._tail_flushed = (
                False  # True once the _HALF-zero tail has been injected
            )

        def forecast(self, noutput_items: int, ninputs: int) -> list[int]:
            return [0] * ninputs if self._locked else [noutput_items] * ninputs

        def _try_lock(self) -> bool:
            try:
                payload_start, _ = _synchronize(self._buf, grid, detect_run)
            except ValueError:
                return False
            if payload_start + sn > len(self._buf):
                return False
            cfo_bins, sto_bins = _joint_sync(self._buf, payload_start, grid)
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
            # When all real payload is consumed at EOF, inject _HALF zeros so the
            # streaming FIR outputs the last _HALF held-in-history payload samples.
            # This runs in the SAME general_work call (no separate GR scheduling
            # call needed), because the zero injection and processing happen inline.
            if (
                self._locked
                and len(x) == 0
                and len(self._buf) == 0
                and not self._tail_flushed
            ):
                self._buf = np.zeros(_HALF, dtype=np.complex64)
                self._tail_flushed = True
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
            if self._locked and len(x) == 0 and len(self._buf) == 0:
                return -1  # WORK_DONE: locked + buffer drained + tail flushed
            return 0

    return _ChirpSync()
