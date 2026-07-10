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
Constants are pinned from the task-1/2 validated reference detector.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from marconi.phy.backends.gnuradio.embedded.lifecycle import OutQueue, forecast_drain

# Half-cosine matched-filter oversampling: the number of polyphase branches
# from which sub-sample timing selects one (reference-detector empirical pin).
_MF_OVERSAMPLE = 12

# The decision-directed carrier loop needs BOTH a loop gain and a leaky-
# integrator pole; the block's single `loop_bw` param carries the gain, so the
# pole is pinned module-internal at its reference-detector value (Gate-B
# validated in task-2). freq_corr = _LOOP_POLE·freq_corr + (1−_LOOP_POLE)·gain·err
_LOOP_POLE = 0.52
_NORM_EPS = 1e-8


def _matched_filter(sps: float, flen: int) -> np.ndarray:
    """Half-cosine MSK matched pulse, oversampled _MF_OVERSAMPLE×: one positive
    lobe of a cosine at the MSK deviation (baud/4), spanning two bit periods,
    negatives clamped to zero. The deviation-over-rate ratio is 1/(4·sps)."""
    n_taps = flen * _MF_OVERSAMPLE + 1
    center = (n_taps - 1) / 2.0
    i = np.arange(n_taps, dtype=np.float64)
    h = np.cos(math.pi / (2.0 * sps * _MF_OVERSAMPLE) * (i - center))
    h[h < 0.0] = 0.0
    return h


def _polyphase_taps(sps: float, flen: int) -> np.ndarray:
    """(_MF_OVERSAMPLE+1, flen) branch matrix: branch b is the matched filter
    decimated by _MF_OVERSAMPLE starting at offset b, so a matched-filter dump
    is a single dot product of a branch against the last `flen` baseband
    samples (chronological, oldest first)."""
    h = _matched_filter(sps, flen)
    return np.stack(
        [
            h[b : b + flen * _MF_OVERSAMPLE : _MF_OVERSAMPLE]
            for b in range(_MF_OVERSAMPLE + 1)
        ]
    )


def _dump_index(bit: int, sps: float) -> int:
    """Absolute input-sample index at which `bit` is dumped. The free-running
    clock fires every `sps` samples; the k-th fire lands where the accumulated
    phase first crosses the (k+1)-th bit boundary."""
    return math.ceil((bit + 1) * sps - 0.5) - 1


def _branch(bit: int, sps: float) -> int:
    """Polyphase branch for `bit` from its residual sub-sample timing error
    (0 for integer sps → the centre branch; cycles for fractional sps)."""
    frac = (bit + 1) * sps
    resid = math.ceil(frac - 0.5) - frac  # in [-0.5, 0.5)
    b = int(_MF_OVERSAMPLE * (resid + 0.5))
    return min(max(b, 0), _MF_OVERSAMPLE)


def make_msk_demod(gr: Any, *, sps: float, loop_bw: float = 0.0038) -> Any:
    """Coherent MSK demod: rail de-rotation to OQPSK, matched integration over
    2T, alternating I/Q rail decisions with decision-directed carrier tracking.
    Measured several dB more sensitive than quadrature_demod+Gardner on weak
    captures (issue 22, guarded by tests/phy/test_msk_snr_margin.py); one soft
    float per symbol, sign = bit."""

    flen = int(2.0 * sps) + 1
    taps = _polyphase_taps(sps, flen)
    gain = float(loop_bw)

    class _MskDemod(gr.basic_block):
        def __init__(self) -> None:
            gr.basic_block.__init__(
                self, name="msk_demod", in_sig=[np.complex64], out_sig=[np.float32]
            )
            self._out = OutQueue(np.float32)
            self._buf = np.empty(0, np.complex64)
            self._history = np.zeros(flen - 1, np.complex128)  # matched-filter tail
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
                self._process()
            return self._out.drain(output_items[0])

        def _process(self) -> None:
            softs: list[float] = []
            while True:
                step = _dump_index(self._bit, sps) + 1 - self._consumed
                if len(self._buf) < step:
                    break
                seg = self._buf[:step].astype(np.complex128)
                self._buf = self._buf[step:]
                self._consumed += step

                increment = self._freq_corr  # baseband: nominal VCO is 0
                phase = self._carrier_phase + increment * np.arange(1, step + 1)
                baseband = seg * np.exp(-1j * phase)
                self._carrier_phase = float((phase[-1]) % (2.0 * math.pi))

                window = np.concatenate([self._history, baseband])[-flen:]
                self._history = window[-(flen - 1) :]
                v = complex(np.dot(taps[_branch(self._bit, sps)], window))
                z = v / (abs(v) + _NORM_EPS)

                if self._bit & 1:
                    rail = z.imag
                    err = -z.real if z.imag >= 0.0 else z.real
                else:
                    rail = z.real
                    err = z.imag if z.real >= 0.0 else -z.imag
                softs.append(-rail if (self._bit & 2) else rail)

                self._freq_corr = (
                    _LOOP_POLE * self._freq_corr + (1.0 - _LOOP_POLE) * gain * err
                )
                self._bit += 1
            if softs:
                self._out.push(np.asarray(softs, np.float32))

    return _MskDemod()
