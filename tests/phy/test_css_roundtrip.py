from pathlib import Path

import numpy as np
import pytest
from phy._dsp import aligned_ber, channel, read_bits, write_bits

from marconi.core.descriptor import Descriptor
from marconi.core.levels import Level
from marconi.core.params import ParamValue
from marconi.phy.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.phy.compiler import compile_modem
from marconi.phy.models import ModemSpec, ModemStep

IQ = Descriptor(Level.IQ, "c")
_OS, _ZP, _SYM = 2, 4, 1.0


def _modem(sf: int) -> ModemSpec:
    p: dict[str, ParamValue] = {
        "sf": sf,
        "oversample": _OS,
        "zero_pad": _ZP,
        "preamble_len": 8,
    }
    return ModemSpec(
        symbol_rate=_SYM,
        path=[
            ModemStep(conv="chirp_sync", params=p),
            ModemStep(conv="dechirp", params=p),
            ModemStep(conv="css_demap", params=p),
        ],
    )


def _compile(modem: ModemSpec, direction: str, rate: float, src: Path, snk: Path):
    from marconi.phy.stages import stage_registry

    return compile_modem(
        modem,
        stage_registry(),
        direction=direction,
        sample_rate=rate,
        start=IQ,
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )


@pytest.mark.parametrize("sf", [7, 11])
def test_css_ber0_impaired(sf: int, tmp_path: Path) -> None:
    ensure_worker_warm()
    be = GnuRadioBackend()
    rate = _OS * (1 << sf) * _SYM  # oversample * bandwidth
    bw = _SYM * (1 << sf)
    bits = np.random.default_rng(sf).integers(0, 2, sf * 40).astype(np.uint8)
    bp = write_bits(tmp_path / "in.bits", bits)
    clean, imp, op = tmp_path / "c.iq", tmp_path / "i.iq", tmp_path / "o.bits"
    m = _modem(sf)
    assert be.run_pipeline(_compile(m, "tx", rate, bp, clean)).status == "ok"
    # The joint up/down-chirp estimator recovers FRACTIONAL sample timing (and a
    # clean CFO), so the block dechirp now decodes through a sub-sample STO that
    # previously smeared marginal symbols (the old integer-only sto=2.0 limit).
    channel(
        clean,
        imp,
        snr_db=15.0,
        cfo_hz=0.03 * bw,
        sto=1.5,
        sfo_ppm=0.0,
        sample_rate=rate,
        seed=sf,
    )
    assert be.run_pipeline(_compile(m, "rx", rate, imp, op)).status == "ok"
    out = read_bits(op)
    assert len(out) >= sf * 30  # preamble stripped, payload recovered
    assert aligned_ber(out, bits, max_shift=8 * sf) == 0.0


@pytest.mark.xfail(
    reason="chirp_sync does not correct SFO; clock_correct owns it "
    "(see test_clock_correct_roundtrip). Pins the coverage boundary.",
    strict=False,
)
@pytest.mark.parametrize("sf", [7])
def test_css_ber0_sfo_documents_gap(sf: int, tmp_path: Path) -> None:
    ensure_worker_warm()
    be = GnuRadioBackend()
    rate = _OS * (1 << sf) * _SYM
    bw = _SYM * (1 << sf)
    bits = np.random.default_rng(sf).integers(0, 2, sf * 40).astype(np.uint8)
    bp = write_bits(tmp_path / "in.bits", bits)
    clean, imp, op = tmp_path / "c.iq", tmp_path / "i.iq", tmp_path / "o.bits"
    m = _modem(sf)
    assert be.run_pipeline(_compile(m, "tx", rate, bp, clean)).status == "ok"
    channel(
        clean,
        imp,
        snr_db=15.0,
        cfo_hz=0.03 * bw,
        sto=1.5,
        sfo_ppm=50.0,
        sample_rate=rate,
        seed=sf,
    )
    assert be.run_pipeline(_compile(m, "rx", rate, imp, op)).status == "ok"
    out = read_bits(op)
    assert aligned_ber(out, bits, max_shift=8 * sf) == 0.0  # xfails: SFO uncorrected
