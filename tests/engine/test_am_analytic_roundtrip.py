from __future__ import annotations

from pathlib import Path

import numpy as np
from engine._dsp import aligned_ber_best

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.io.bitfile import read_bits
from marconi.engine.modulation.fsk.stages import FskStep
from marconi.engine.stages.conditioning import AmStep, AnalyticStep, ChannelizeStep
from marconi.engine.stages.general import SliceStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

IQ = Descriptor(Level.IQ, "c")
RATE = 48000.0
BAUD = 2400.0
SPS = int(RATE / BAUD)
MARK_HZ, SPACE_HZ = 2400.0, 1200.0  # tone per bit, phase-continuous
# The stage default (1024) has a 2046-sample dc_blocker_ff group delay -- for a
# 20-sps burst that throws Gardner acquisition off by a ~100-symbol fixed offset
# no warmup length fixes (measured: BER stays ~0.5 even with a 512-symbol warmup,
# since it's a shift, not front-loss). A short-burst value settles inside one
# warmup and matches a real ACARS-length transmission better than the default.
DC_BLOCK_LEN = 64


def _am_afsk(bits: np.ndarray) -> np.ndarray:
    freqs = np.where(bits.repeat(SPS) == 1, MARK_HZ, SPACE_HZ)
    phase = 2 * np.pi * np.cumsum(freqs) / RATE
    audio = np.sin(phase)
    return ((0.6 + 0.35 * audio)).astype(np.complex64)


def test_am_afsk_decodes_ber0(tmp_path: Path) -> None:
    ensure_worker_warm()
    rng = np.random.default_rng(7)
    bits = rng.integers(0, 2, 2000).astype(np.uint8)
    warm = np.ones(64, np.uint8)  # settle preamble for the clock loop
    iq = _am_afsk(np.concatenate([warm, bits]))
    src = tmp_path / "am.cf32"
    iq.tofile(src)
    snk = tmp_path / "bits.u8"
    modem = Modem(
        symbol_rate=BAUD,
        path=[
            AmStep(dc_block_len=DC_BLOCK_LEN),
            AnalyticStep(),
            ChannelizeStep(decim=1, bandwidth_hz=2400.0, center_hz=1800.0),
            FskStep(deviation=600.0),
            SliceStep(),
        ],
    )
    pipe = compile_modem(
        modem,
        stage_registry(),
        direction="rx",
        sample_rate=RATE,
        start=IQ,
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )
    r = GnuRadioBackend().run_pipeline(pipe, timeout=60.0)
    assert r.status == "ok", r
    out = read_bits(snk)
    n_sent = warm.size + bits.size
    assert out.size >= n_sent - 96, f"tail loss too large: {out.size}/{n_sent}"
    # A residual few-symbol Gardner acquisition offset survives even at
    # DC_BLOCK_LEN=64, so alignment needs a shift search (issue 05: paired with
    # the length assert above, never aligned_ber alone).
    # global inversion is a legitimate hypothesis: accept either polarity
    assert aligned_ber_best([out, 1 - out], bits) == 0.0
