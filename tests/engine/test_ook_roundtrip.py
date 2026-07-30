from pathlib import Path

import numpy as np
from engine._dsp import aligned_ber, channel, read_bits, write_bits

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.modulation.ook.stages import OokEnvelopeStep
from marconi.engine.stages.conditioning import AgcStep
from marconi.engine.stages.general import SliceStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem
from marconi.engine.types.step import Step

IQ = Descriptor(Level.IQ, "c")
_SR, _SYM = 4.0, 1.0


def _modem(direction: str) -> Modem:
    rx: list[Step] = [AgcStep(), OokEnvelopeStep(), SliceStep()]
    tx: list[Step] = [OokEnvelopeStep(), SliceStep()]
    return Modem(symbol_rate=_SYM, path=rx if direction == "rx" else tx)


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
