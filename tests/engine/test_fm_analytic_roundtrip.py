"""FM discriminator to audio opens the demod-then-demod class the AM path
already had: an AFSK subcarrier inside an FM carrier decodes end-to-end
through fm_demod -> analytic -> channelize -> fsk, all stock blocks. The
inner AFSK chain and its tuning are shared with the AM round-trip; only the
outer modulation differs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from engine._dsp import aligned_ber_best

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.io.bitfile import read_bits
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.levels import Level
from marconi.engine.types.models import ModemSpec, ModemStep

IQ = Descriptor(Level.IQ, "c")
RATE = 48000.0
BAUD = 2400.0
SPS = int(RATE / BAUD)
MARK_HZ, SPACE_HZ = 2400.0, 1200.0
DEVIATION = 3000.0
DC_BLOCK_LEN = 64  # short-burst value; see test_am_analytic_roundtrip.py
# The deviation-normalized discriminator hands the clock loop +-1.0 audio
# (vs AM's +-0.35), and Gardner's error gain scales with amplitude: measured,
# the timing transient dies only ~170 bits in. 256 settles it with margin.
WARM_BITS = 256


def _fm_afsk(bits: np.ndarray) -> np.ndarray:
    freqs = np.where(bits.repeat(SPS) == 1, MARK_HZ, SPACE_HZ)
    audio = np.sin(2 * np.pi * np.cumsum(freqs) / RATE)
    phase = 2 * np.pi * DEVIATION * np.cumsum(audio) / RATE
    return np.exp(1j * phase).astype(np.complex64)


def test_fm_carried_afsk_decodes_ber0(tmp_path: Path) -> None:
    ensure_worker_warm()
    rng = np.random.default_rng(7)
    bits = rng.integers(0, 2, 2000).astype(np.uint8)
    warm = np.ones(WARM_BITS, np.uint8)
    iq = _fm_afsk(np.concatenate([warm, bits]))
    src = tmp_path / "fm.cf32"
    iq.tofile(src)
    snk = tmp_path / "bits.u8"
    modem = ModemSpec(
        symbol_rate=BAUD,
        path=[
            ModemStep(
                conv="fm_demod",
                params={"deviation": DEVIATION, "dc_block_len": DC_BLOCK_LEN},
            ),
            ModemStep(conv="analytic", params={}),
            ModemStep(
                conv="channelize",
                params={"decim": 1, "bandwidth_hz": 2400.0, "center_hz": 1800.0},
            ),
            ModemStep(conv="fsk", params={"deviation": 600.0}),
            ModemStep(conv="slice", params={}),
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
    shift = WARM_BITS + 96  # alignment lag = warm preamble + chain group delay
    # global inversion is a legitimate hypothesis: accept either polarity
    assert aligned_ber_best([out, 1 - out], bits, max_shift=shift) == 0.0
