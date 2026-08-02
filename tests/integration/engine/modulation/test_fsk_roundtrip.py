from pathlib import Path

import numpy as np
from helpers._dsp import aligned_ber, channel, read_bits, write_bits

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.modulation.fsk.stages import FskStep
from marconi.engine.stages.general import SliceStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

IQ = Descriptor(Level.IQ, ItemType.C)
_SR = 4.0  # sample_rate; OSR = 4 over symbol_rate = 1.0
_SYM = 1.0
_DEV = 1.0


def _modem() -> Modem:
    return Modem(
        symbol_rate=_SYM,
        path=[FskStep(deviation=_DEV), SliceStep()],
    )


def _tx(bits_path: Path, iq_path: Path):
    return compile_modem(
        _modem(),
        stage_registry(),
        direction="tx",
        sample_rate=_SR,
        start=IQ,
        source_io={"path": str(bits_path)},
        sink_io={"path": str(iq_path)},
    )


def _rx(iq_path: Path, bits_path: Path):
    return compile_modem(
        _modem(),
        stage_registry(),
        direction="rx",
        sample_rate=_SR,
        start=IQ,
        source_io={"path": str(iq_path)},
        sink_io={"path": str(bits_path)},
    )


def test_fsk_roundtrip_clean_ber0(tmp_path: Path) -> None:
    ensure_worker_warm()
    be = GnuRadioBackend()
    bits = np.random.default_rng(0).integers(0, 2, 512).astype(np.uint8)
    bp = write_bits(tmp_path / "in.bits", bits)
    ip = tmp_path / "sig.iq"
    op = tmp_path / "out.bits"
    assert be.run_pipeline(_tx(bp, ip)).status == "ok"
    assert be.run_pipeline(_rx(ip, op)).status == "ok"
    assert aligned_ber(read_bits(op), bits) == 0.0


def test_fsk_roundtrip_survives_combined_impairments(tmp_path: Path) -> None:
    ensure_worker_warm()
    be = GnuRadioBackend()
    bits = np.random.default_rng(3).integers(0, 2, 512).astype(np.uint8)
    bp = write_bits(tmp_path / "in.bits", bits)
    clean = tmp_path / "clean.iq"
    imp = tmp_path / "imp.iq"
    op = tmp_path / "out.bits"
    assert be.run_pipeline(_tx(bp, clean)).status == "ok"
    # >=244 ppm actually resamples 2048 samples; 50 ppm rounds to a no-op
    channel(
        clean,
        imp,
        snr_db=20.0,
        cfo_hz=0.25 * _DEV,
        sto=1.5,
        sfo_ppm=500.0,
        sample_rate=_SR,
        seed=7,
    )
    assert be.run_pipeline(_rx(imp, op)).status == "ok"
    assert aligned_ber(read_bits(op), bits) == 0.0
