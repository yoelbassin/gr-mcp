from pathlib import Path

import numpy as np
import pytest
from phy._dsp import aligned_ber, channel, read_bits, write_bits

from marconi.core.descriptor import Descriptor
from marconi.core.levels import Level
from marconi.core.params import ParamValue
from marconi.phy.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.phy.compiler import CompileError, compile_modem
from marconi.phy.models import ModemSpec, ModemStep

IQ = Descriptor(Level.IQ, "c")
_OS, _ZP, _SYM = 2, 4, 1.0


def _modem(sf: int, oversample: int = _OS) -> ModemSpec:
    p: dict[str, ParamValue] = {
        "sf": sf,
        "oversample": oversample,
        "zero_pad": _ZP,
        "preamble_len": 8,
        "sfd_symbols": 2.25,
        "sync_symbols": 2,
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


@pytest.mark.parametrize(
    ("sf", "osr"),
    [(7, 2), (11, 2), (12, 2), (12, 4)],
    # sf=12 osr=2 is the first mode whose symbol (osr*2^sf = 8192 items)
    # exceeds GR's 8191-item default stream buffer; osr=4 doubles it.
)
def test_css_ber0_impaired(sf: int, osr: int, tmp_path: Path) -> None:
    ensure_worker_warm()
    be = GnuRadioBackend()
    rate = osr * (1 << sf) * _SYM  # oversample * bandwidth
    bw = _SYM * (1 << sf)
    bits = np.random.default_rng(sf).integers(0, 2, sf * 40).astype(np.uint8)
    bp = write_bits(tmp_path / "in.bits", bits)
    clean, imp, op = tmp_path / "c.iq", tmp_path / "i.iq", tmp_path / "o.bits"
    m = _modem(sf, osr)
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


def test_dechirp_rejects_wrong_input_rate(tmp_path: Path) -> None:
    # dechirp's window is oversample*2^sf samples/symbol, so the input rate must
    # be ~oversample*2^sf*symbol_rate (here 2*128*1.0 = 256). A grossly wrong rate
    # would dechirp garbage; the compiler must reject it, not emit it (issue 06).
    m = _modem(7, 2)
    with pytest.raises(CompileError) as e:
        _compile(m, "rx", 200.0, tmp_path / "i", tmp_path / "o")
    assert "rate" in str(e.value).lower()


def test_dechirp_accepts_matching_input_rate(tmp_path: Path) -> None:
    m = _modem(7, 2)
    _compile(m, "rx", 256.0, tmp_path / "i", tmp_path / "o")  # exact: compiles


def test_dechirp_rate_check_tolerates_clock_correct_ppm(tmp_path: Path) -> None:
    # clock_correct resamples by 1/(1+ppm) — a legitimate sub-percent rate shift
    # (0.005% at 50 ppm) the tolerance must admit, not reject (issue 06).
    p: dict[str, ParamValue] = {
        "sf": 7,
        "oversample": 2,
        "zero_pad": _ZP,
        "preamble_len": 8,
        "sfd_symbols": 2.25,
        "sync_symbols": 2,
    }
    m = ModemSpec(
        symbol_rate=_SYM,
        path=[
            ModemStep(conv="clock_correct", params={"ppm": 50.0}),
            ModemStep(conv="chirp_sync", params=p),
            ModemStep(conv="dechirp", params=p),
            ModemStep(conv="css_demap", params=p),
        ],
    )
    _compile(m, "rx", 256.0, tmp_path / "i", tmp_path / "o")  # within tolerance


def test_css_two_burst_capture_decodes_both(tmp_path: Path) -> None:
    """A capture with two CSS bursts — each under a DIFFERENT carrier offset —
    decodes both in one run: chirp_sync re-arms per preamble with fresh
    CFO/timing estimates (issue 03; the pre-fix block applied burst 1's
    correction to the whole remainder of the capture). Also pins the
    re-lock diagnostics surfaced through RunResult."""
    from phy._dsp import read_complex

    ensure_worker_warm()
    be = GnuRadioBackend()
    sf, osr = 7, 2
    rate = osr * (1 << sf) * _SYM
    bw = _SYM * (1 << sf)
    sn = osr * (1 << sf)
    rng = np.random.default_rng(21)
    bits1 = rng.integers(0, 2, sf * 30).astype(np.uint8)
    bits2 = rng.integers(0, 2, sf * 30).astype(np.uint8)
    m = _modem(sf, osr)
    f1, f2 = tmp_path / "f1.iq", tmp_path / "f2.iq"
    b1 = write_bits(tmp_path / "b1.bits", bits1)
    b2 = write_bits(tmp_path / "b2.bits", bits2)
    assert be.run_pipeline(_compile(m, "tx", rate, b1, f1)).status == "ok"
    assert be.run_pipeline(_compile(m, "tx", rate, b2, f2)).status == "ok"
    i1, i2 = tmp_path / "i1.iq", tmp_path / "i2.iq"
    channel(f1, i1, cfo_hz=0.03 * bw, sample_rate=rate, seed=1)
    channel(f2, i2, cfo_hz=-0.04 * bw, sample_rate=rate, seed=2)
    gap = 0.02 * (rng.standard_normal(20 * sn) + 1j * rng.standard_normal(20 * sn))
    tail = 0.02 * (rng.standard_normal(14 * sn) + 1j * rng.standard_normal(14 * sn))
    cap = np.concatenate(
        [
            read_complex(i1),
            gap.astype(np.complex64),
            read_complex(i2),
            tail.astype(np.complex64),
        ]
    ).astype(np.complex64)
    cap_p, imp, op = tmp_path / "cap.iq", tmp_path / "imp.iq", tmp_path / "o.bits"
    cap.tofile(cap_p)
    channel(cap_p, imp, snr_db=20.0, sample_rate=rate, seed=3)
    r = be.run_pipeline(_compile(m, "rx", rate, imp, op))
    assert r.status == "ok"
    assert any(d.get("locks") == 2 for d in r.diagnostics.values()), r.diagnostics
    out = read_bits(op)
    assert len(out) > len(bits1) + len(bits2)  # both payloads (plus gap junk)
    assert aligned_ber(out[: len(bits1) + 8 * sf], bits1, max_shift=8 * sf) == 0.0
    assert aligned_ber(out[len(bits1) :], bits2, max_shift=40 * sf) == 0.0


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
