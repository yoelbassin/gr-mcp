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
