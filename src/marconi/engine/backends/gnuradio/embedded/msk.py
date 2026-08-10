"""Coherent MSK demod (h=0.5 CPFSK) to one soft float per bit.

MSK is OQPSK with a half-sine pulse: a carrier PLL derotates the complex
baseband, a half-cosine matched filter integrates over two bit periods, and
each bit is decided on an alternating I/Q rail. The carrier loop is
decision-directed (Costas-style), a first-order leaky integrator producing a
frequency correction. The bit clock free-runs at the nominal baud (the
DECOUPLED path, ratified for baseband input where the subcarrier the reference
passband detector once slaved its clock to has already been removed by
conditioning); a polyphase branch of the oversampled matched filter supplies
sub-sample timing. Coherent detection of a raw (non-precoded) MSK signal
recovers the payload differentially-encoded (NRZI) — the data rides in the
±90°/bit phase step; the per-link differential decode is a bits-layer concern,
so this block stays protocol-agnostic and emits the rail decisions directly.
Constants are pinned from the validated reference detector.
"""

from __future__ import annotations

import cmath
import math
from typing import Any

import numpy as np
import numpy.typing as npt

from marconi.engine.backends.gnuradio.embedded.lifecycle import OutQueue, forecast_drain

# Half-cosine matched-filter oversampling: the number of polyphase branches
# from which sub-sample timing selects one. Default from the reference detector;
# caller-overridable via the mf_oversample param.
_MF_OVERSAMPLE = 12

# The decision-directed carrier loop needs BOTH a loop gain (loop_bw) and a
# leaky-integrator pole (loop_pole); defaults from the validated reference
# detector. freq_corr = pole·freq_corr + (1−pole)·gain·err
_LOOP_POLE = 0.52
_NORM_EPS = 1e-8

# Measured crossover between the two _process media (kbit/s, this block, 4000
# bits): sps 10 → scalar 308 / vectorized 234; sps 16 → 228 / 230; sps 64 →
# 73 / 188; sps 128 → 38 / 159. Per-bit numpy ufunc dispatch dominates below,
# per-sample Python dominates above, so the medium is chosen per sps. Bit
# decisions of the two paths agree exactly across sps 4-128.
_VECTOR_SPS_MIN = 16.0


def _matched_filter(sps: float, flen: int, oversample: int) -> npt.NDArray[np.float64]:
    """Half-cosine MSK matched pulse, oversampled `oversample`×: one positive
    lobe of a cosine at the MSK deviation (baud/4), spanning two bit periods,
    negatives clamped to zero. The deviation-over-rate ratio is 1/(4·sps)."""
    n_taps = flen * oversample + 1
    center = (n_taps - 1) / 2.0
    i = np.arange(n_taps, dtype=np.float64)
    h = np.cos(math.pi / (2.0 * sps * oversample) * (i - center))
    h[h < 0.0] = 0.0
    return h


def _polyphase_taps(sps: float, flen: int, oversample: int) -> npt.NDArray[np.float64]:
    """(oversample+1, flen) branch matrix: branch b is the matched filter
    decimated by `oversample` starting at offset b, so a matched-filter dump
    is a single dot product of a branch against the last `flen` baseband
    samples (chronological, oldest first)."""
    h = _matched_filter(sps, flen, oversample)
    return np.stack(
        [h[b : b + flen * oversample : oversample] for b in range(oversample + 1)]
    )


def _dump_index(bit: int, sps: float) -> int:
    """Absolute input-sample index at which `bit` is dumped. The free-running
    clock fires every `sps` samples; the k-th fire lands where the accumulated
    phase first crosses the (k+1)-th bit boundary."""
    return math.ceil((bit + 1) * sps - 0.5) - 1


def _branch(bit: int, sps: float, oversample: int) -> int:
    """Polyphase branch for `bit` from its residual sub-sample timing error
    (0 for integer sps → the centre branch; cycles for fractional sps)."""
    frac = (bit + 1) * sps
    resid = math.ceil(frac - 0.5) - frac  # in [-0.5, 0.5)
    b = int(oversample * (resid + 0.5))
    return min(max(b, 0), oversample)


def _decide(z: complex, bit: int) -> tuple[float, float]:
    """Alternating I/Q rail decision: (soft value, carrier phase error)."""
    if bit & 1:
        rail = z.imag
        err = -z.real if z.imag >= 0.0 else z.real
    else:
        rail = z.real
        err = z.imag if z.real >= 0.0 else -z.imag
    return (-rail if (bit & 2) else rail), err


def make_msk_demod(
    gr: Any,
    *,
    sps: float,
    loop_bw: float = 0.0038,
    loop_pole: float = _LOOP_POLE,
    mf_oversample: int = _MF_OVERSAMPLE,
) -> Any:
    """Coherent MSK demod: rail de-rotation to OQPSK, matched integration over
    2T, alternating I/Q rail decisions with decision-directed carrier tracking.
    Measured ~2-3 dB more sensitive at operational BER than the best stock
    composition (matched filter + delay-and-multiply differential detection),
    and within ~1-2 dB of the coherent-BPSK bound; stock GR 3.10 has no coherent
    MSK/OQPSK receiver, so this custom block is the only path to that gain
    (guarded by
    tests/integration/engine/modulation/test_msk_snr_margin.py). One soft float
    per symbol, sign = bit."""

    flen = int(2.0 * sps) + 1
    taps_rows = [
        tuple(float(t) for t in row)
        for row in _polyphase_taps(sps, flen, mf_oversample)
    ]
    taps_arr = np.asarray(_polyphase_taps(sps, flen, mf_oversample))
    gain = float(loop_bw)
    pole = float(loop_pole)
    vectorized = sps >= _VECTOR_SPS_MIN
    j_ramp = np.arange(1.0, math.ceil(sps) + 2.0)

    class _MskDemod(gr.basic_block):
        def __init__(self) -> None:
            gr.basic_block.__init__(
                self, name="msk_demod", in_sig=[np.complex64], out_sig=[np.float32]
            )
            self._out = OutQueue(np.float32)
            self._buf = np.empty(0, np.complex64)
            # derotated matched-filter window, stored in the chosen medium
            self._win: list[complex] = [] if vectorized else [0j] * flen
            self._win_arr = np.zeros(flen if vectorized else 0, np.complex128)
            self._carrier_phase = 0.0
            self._freq_corr = 0.0  # PLL frequency correction (rad/sample)
            self._bit = 0  # running bit index: rail parity + π/2 derotation
            self._consumed = 0  # input samples already resolved into bits

        def forecast(self, noutput_items: int, ninputs: int) -> list[int]:
            return forecast_drain(self._out.pending, ninputs)

        def general_work(self, input_items: Any, output_items: Any) -> int:
            inp = input_items[0]
            if len(inp):
                self._buf = np.concatenate([self._buf, inp])
                self.consume(0, len(inp))
                if vectorized:
                    self._process_vectorized()
                else:
                    self._process_scalar()
            return self._out.drain(output_items[0])

        def _process_scalar(self) -> None:
            # The decision-directed feedback (each bit's err retunes the next
            # bit's derotation) makes this loop irreducibly sequential. Below
            # the _VECTOR_SPS_MIN crossover it runs on Python scalars: per-bit
            # numpy ufunc dispatch on the ~sps-sample segments measures slower
            # than the scalar rotor there.
            softs: list[float] = []
            samples: list[complex] = self._buf.tolist()
            pos, n = 0, len(samples)
            win = self._win
            phase = self._carrier_phase
            fc = self._freq_corr  # baseband: nominal VCO is 0
            bit = self._bit
            consumed = self._consumed
            two_pi = 2.0 * math.pi
            while True:
                step = _dump_index(bit, sps) + 1 - consumed
                if n - pos < step:
                    break
                r = cmath.exp(-1j * fc)
                rot = cmath.exp(-1j * phase)
                new = []
                for s in samples[pos : pos + step]:
                    rot *= r
                    new.append(s * rot)
                pos += step
                consumed += step
                phase = (phase + fc * step) % two_pi

                win = win[step:] + new
                v = 0j
                for t, w in zip(taps_rows[_branch(bit, sps, mf_oversample)], win):
                    v += t * w
                z = v / (abs(v) + _NORM_EPS)

                soft, err = _decide(z, bit)
                softs.append(soft)
                fc = pole * fc + (1.0 - pole) * gain * err
                bit += 1
            self._buf = self._buf[pos:]
            self._win = win
            self._carrier_phase = phase
            self._freq_corr = fc
            self._bit = bit
            self._consumed = consumed
            if softs:
                self._out.push(np.asarray(softs, np.float32))

        def _process_vectorized(self) -> None:
            # Same loop, wideband medium: the derotation frequency is constant
            # within a bit, so each bit is a handful of vectorized ops on the
            # sps-sample segment instead of a per-sample Python rotor; only
            # the decision feedback stays sequential.
            softs: list[float] = []
            buf = self._buf.astype(np.complex128)
            pos, n = 0, buf.size
            win = self._win_arr
            phase = self._carrier_phase
            fc = self._freq_corr  # baseband: nominal VCO is 0
            bit = self._bit
            consumed = self._consumed
            two_pi = 2.0 * math.pi
            while True:
                step = _dump_index(bit, sps) + 1 - consumed
                if n - pos < step:
                    break
                rotors = np.exp((-1j * fc) * j_ramp[:step] - 1j * phase)
                new = buf[pos : pos + step] * rotors
                pos += step
                consumed += step
                phase = (phase + fc * step) % two_pi

                keep = flen - step
                win[:keep] = win[step:]
                win[keep:] = new
                v = complex(np.dot(taps_arr[_branch(bit, sps, mf_oversample)], win))
                z = v / (abs(v) + _NORM_EPS)

                soft, err = _decide(z, bit)
                softs.append(soft)
                fc = pole * fc + (1.0 - pole) * gain * err
                bit += 1
            self._buf = self._buf[pos:]
            self._carrier_phase = phase
            self._freq_corr = fc
            self._bit = bit
            self._consumed = consumed
            if softs:
                self._out.push(np.asarray(softs, np.float32))

    return _MskDemod()
