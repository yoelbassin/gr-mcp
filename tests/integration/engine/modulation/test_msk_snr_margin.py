"""First SNR-margin guard in the suite: the coherent msk RX
holds BER-0 at an AWGN SNR where the non-coherent fsk RX measurably fails.

_SNR_DB pinned from a 7x5 sweep over
S in {28,26,24,22,20,18,16} dB x seeds {0..4}, both paths matched at
deviation = symbol_rate/4 (h=0.5). The brief's originally-suggested sweep
range (2-14 dB) was calibrated for a wider FSK deviation elsewhere in the
suite (h=2.0); at this h=0.5 matched-deviation operating point the fsk
symbol-timing loop (Gardner TED, narrow tone separation) collapses to
near-random far higher up the SNR axis, so the sweep was re-centred on the
transition actually observed (probed first: fsk BER-0 by 40 dB, ~0.0015 at
30 dB, waterfall to ~0.19 by 20 dB, ~0.45 by 18 dB; msk stays BER-0 down to
about -2 dB). Measured sweep (BER per seed 0-4):

msk (all cells 0.0 at every S tested, 16-28 dB):
  S=28..16: 0.0000 0.0000 0.0000 0.0000 0.0000  (every row)

fsk:
  S=28: 0.0078 0.0049 0.0059 0.0098 0.0059
  S=26: 0.0244 0.0333 0.0279 0.0289 0.0244
  S=24: 0.0655 0.0724 0.0641 0.0699 0.0562
  S=22: 0.1183 0.1301 0.1134 0.1193 0.1144
  S=20: 0.1902 0.1731 0.1672 0.1721 0.1746
  S=18: 0.4503 0.2352 0.2416 0.3345 0.4147
  S=16: 0.3487 0.2778 0.3993 0.4566 0.4429

Pin rule (amended in review): pin a MID-WATERFALL S where fsk BER >= 5x
the 0.02 gate on all seeds and msk stays BER-0 with >= 2 notches of
headroom -- NOT the highest qualifying S. The original highest-S rule
selected 26 (fsk 0.0244-0.0333), leaving a worst-case margin of only
0.0044 over the gate; with documented GR run-to-run numeric
nondeterminism (which same-seed 10x reruns don't exercise), an
edge-of-gate pin is fragile. S=22 is mid-waterfall: fsk 0.1134-0.1301
(~6-9x the gate) on every seed, msk BER-0 there and at every notch below
through S=16 (and to about -2 dB by probe). _SNR_DB = 22.0. (S=28 fails
the gate outright, max fsk BER 0.0098.)

Non-vacuity, checked once and reverted: pinning _SNR_DB = 28.0 fails the
fsk assertion (min BER 0.0049 !> 0.02) -- the test is not vacuously true."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from helpers import _synth as synth
from helpers._dsp import aligned_ber_best, channel, read_bits

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.modulation.fsk.stages import FskStep, MskStep
from marconi.engine.stages.general import SliceStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem
from marconi.engine.types.step import Step

IQ = Descriptor(Level.IQ, ItemType.C)
_SR, _SYM = 20.0, 1.0
_SNR_DB = 22.0
_SEEDS = (0, 1, 2, 3, 4)


def _modem(rx_conv: str) -> Modem:
    rx_step: Step = FskStep(deviation=_SYM / 4.0) if rx_conv == "fsk" else MskStep()
    return Modem(
        symbol_rate=_SYM,
        path=[rx_step, SliceStep()],
    )


def _score(rx_conv: str, out: np.ndarray, bits: np.ndarray) -> float:
    if rx_conv == "msk":
        h = out.astype(np.uint8)
        nrzi = (h[1:] ^ h[:-1]).astype(np.uint8)
        return aligned_ber_best([nrzi, 1 - nrzi], bits)
    return aligned_ber_best([out, 1 - out], bits)


def _ber(rx_conv: str, seed: int, tmp_path: Path) -> float:
    bits = np.random.default_rng(100 + seed).integers(0, 2, 2048).astype(np.uint8)
    clean = tmp_path / f"c{rx_conv}{seed}.iq"
    imp = tmp_path / f"i{rx_conv}{seed}.iq"
    op = tmp_path / f"o{rx_conv}{seed}.bits"
    # both paths are fed the SAME h=0.5 CPFSK waveform; only the RX differs,
    # which is the whole point of the margin measurement
    synth.write(
        clean,
        synth.cpfsk(bits, sps=int(_SR / _SYM), deviation=_SYM / 4.0, sample_rate=_SR),
    )
    channel(clean, imp, snr_db=_SNR_DB, sample_rate=_SR, seed=seed)
    rx = compile_modem(
        _modem(rx_conv),
        stage_registry(),
        sample_rate=_SR,
        start=IQ,
        source_io={"path": str(imp)},
        sink_io={"path": str(op)},
    )
    assert GnuRadioBackend().run_pipeline(rx, timeout=60.0).status == "ok"
    out = read_bits(op)
    assert out.size >= bits.size - 96
    return _score(rx_conv, out, bits)


def test_msk_ber0_where_fsk_fails(tmp_path: Path) -> None:
    ensure_worker_warm()
    for seed in _SEEDS:
        assert _ber("msk", seed, tmp_path) == 0.0
    assert min(_ber("fsk", s, tmp_path) for s in _SEEDS) > 0.02
