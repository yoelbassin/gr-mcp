from __future__ import annotations

from pathlib import Path

import numpy as np
from helpers import _synth as synth

_SR, _SYM, _DEV = 4.0, 1.0, 1.0


def make_clean_capture(tmp_path: Path, n_bits: int = 4096) -> Path:
    """A clean CPFSK capture for the quality layer to score. Synthesized rather
    than run through a modulator flowgraph: the quality assertions are about
    what the RX reports, so the signal only has to be a correct FSK waveform."""
    bits = np.random.default_rng(7).integers(0, 2, n_bits).astype(np.uint8)
    return synth.write(
        tmp_path / "clean.cf32",
        synth.cpfsk(bits, sps=int(_SR / _SYM), deviation=_DEV, sample_rate=_SR),
    )


def make_multilevel_capture(tmp_path: Path, n_symbols: int = 8000) -> Path:
    """A clean 4-level FM capture — the M-ary eye a bare discriminator front
    end reads. The outer tone is _DEV, so the same FskStep(deviation=_DEV) that
    demodulates the binary capture demodulates this one."""
    levels = np.random.default_rng(11).choice([-3.0, -1.0, 1.0, 3.0], n_symbols)
    return synth.write(
        tmp_path / "multilevel.cf32",
        synth.fm_levels(
            levels, sps=int(_SR / _SYM), deviation=_DEV / 3.0, sample_rate=_SR
        ),
    )


def make_noise_capture(tmp_path: Path, n_symbols: int = 8000) -> Path:
    rng = np.random.default_rng(12)
    n = n_symbols * int(_SR / _SYM)
    return synth.write(
        tmp_path / "noise.cf32",
        (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64),
    )
