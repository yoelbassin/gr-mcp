from math import pi
from pathlib import Path

import numpy as np
import pytest
from helpers._dsp import aligned_ber, channel

from marconi.engine.backends.gnuradio.runner import ensure_worker_warm
from marconi.engine.io.bitfile import read_bits
from marconi.engine.io.source import SourceSlice
from marconi.engine.modulation.psk.stages import (
    DifferentialDemodStep,
    PskDemapStep,
    SampleSymbolsStep,
    SymbolSyncStep,
)
from marconi.engine.run import run_rx
from marconi.engine.stages.conditioning import AgcStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import AgcMode, ItemType, PskOrder
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

_SPS = 4
_SETTLE_BITS = 384

IQ = Descriptor(Level.IQ, ItemType.C)


def _const_angles() -> np.ndarray:
    from gnuradio import digital

    return np.angle(np.asarray(digital.constellation_qpsk().points()))


def _pi4_dqpsk_iq(bits: np.ndarray, phase0: float) -> np.ndarray:
    from gnuradio import filter as gr_filter

    idx = (bits[0::2] << 1) | bits[1::2]
    dphi = _const_angles()[idx] + pi / 4
    phases = phase0 + np.cumsum(dphi)
    syms = np.exp(1j * phases)
    taps = np.asarray(
        gr_filter.firdes.root_raised_cosine(_SPS, _SPS, 1.0, 0.35, 11 * _SPS + 1)
    )
    up = np.zeros(syms.size * _SPS, np.complex128)
    up[::_SPS] = syms
    return np.convolve(up, taps, mode="same").astype(np.complex64)


def _modem() -> Modem:
    return Modem(
        symbol_rate=1.0,
        path=[
            AgcStep(
                mode=AgcMode("feedback"),
                attack_symbols=16.0,
                decay_symbols=16.0,
            ),
            SymbolSyncStep(sps=_SPS),
            SampleSymbolsStep(),
            DifferentialDemodStep(delay=1, rotate=-pi / 4),
            PskDemapStep(order=PskOrder(4)),
        ],
    )


@pytest.mark.parametrize("phase0", [0.0, pi / 7])
def test_pi4_dqpsk_roundtrip(tmp_path: Path, phase0: float) -> None:
    ensure_worker_warm()
    rng = np.random.default_rng(11)
    bits = rng.integers(0, 2, 4096).astype(np.uint8)
    clean = tmp_path / "clean.cf32"
    _pi4_dqpsk_iq(bits, phase0).tofile(clean)
    imp = channel(clean, tmp_path / "imp.cf32", snr_db=25.0, seed=1)
    r = run_rx(
        _modem(),
        stage_registry(),
        sample_rate=float(_SPS),
        start=IQ,
        workdir=tmp_path,
        source=SourceSlice(path=imp),
    )
    assert r.status == "ok", r
    assert r.bitstream is not None
    rx = read_bits(r.bitstream.path)
    assert aligned_ber(rx[_SETTLE_BITS:], bits[_SETTLE_BITS:], max_shift=512) == 0.0
