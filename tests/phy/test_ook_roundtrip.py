from pathlib import Path

import numpy as np
from phy._dsp import aligned_ber, channel, read_bits, write_bits

from marconi.core.descriptor import Descriptor
from marconi.core.levels import Level
from marconi.phy.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.phy.compiler import compile_modem
from marconi.phy.models import ModemSpec, ModemStep
from marconi.phy.stages import stage_registry

IQ = Descriptor(Level.IQ, "c")
_SR, _SYM = 4.0, 1.0


def _modem(direction: str) -> ModemSpec:
    rx = [
        ModemStep(conv="agc"),
        ModemStep(conv="ook_envelope"),
        ModemStep(conv="slice"),
    ]
    tx = [ModemStep(conv="ook_envelope"), ModemStep(conv="slice")]
    return ModemSpec(symbol_rate=_SYM, path=rx if direction == "rx" else tx)


def _compile(direction: str, src: Path, snk: Path):
    return compile_modem(
        _modem(direction),
        stage_registry(),
        direction=direction,
        sample_rate=_SR,
        start=IQ,
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )


def test_ook_roundtrip_clean_ber0(tmp_path: Path) -> None:
    ensure_worker_warm()
    be = GnuRadioBackend()
    bits = np.random.default_rng(0).integers(0, 2, 2048).astype(np.uint8)
    bp, ip, op = (
        write_bits(tmp_path / "in.bits", bits),
        tmp_path / "s.iq",
        tmp_path / "o.bits",
    )
    assert be.run_pipeline(_compile("tx", bp, ip)).status == "ok"
    assert be.run_pipeline(_compile("rx", ip, op)).status == "ok"
    assert aligned_ber(read_bits(op), bits) == 0.0


def test_ook_roundtrip_survives_impairments(tmp_path: Path) -> None:
    # Big CFO (0.05*rate) proves envelope CFO-immunity; STO/SFO/AWGN realistic.
    ensure_worker_warm()
    be = GnuRadioBackend()
    bits = np.random.default_rng(5).integers(0, 2, 2048).astype(np.uint8)
    bp = write_bits(tmp_path / "in.bits", bits)
    clean, imp, op = tmp_path / "c.iq", tmp_path / "i.iq", tmp_path / "o.bits"
    assert be.run_pipeline(_compile("tx", bp, clean)).status == "ok"
    channel(
        clean,
        imp,
        snr_db=20.0,
        cfo_hz=0.05 * _SR,
        sto=1.5,
        sfo_ppm=500.0,
        sample_rate=_SR,
        seed=5,
    )
    assert be.run_pipeline(_compile("rx", imp, op)).status == "ok"
    assert aligned_ber(read_bits(op), bits) == 0.0
