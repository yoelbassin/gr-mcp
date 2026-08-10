from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from helpers._dsp import aligned_ber, channel, read_bits, write_bits

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import CompileError, compile_modem
from marconi.engine.compile.ir import GrPipeline
from marconi.engine.modulation.css.stages import (
    ChirpSyncStep,
    CssDemapStep,
    DechirpStep,
)
from marconi.engine.stages.conditioning import ClockCorrectStep
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

IQ = Descriptor(Level.IQ, ItemType.C)
_OS, _ZP, _SYM = 2, 4, 1.0


def _modem(sf: int, oversample: int = _OS) -> Modem:
    return Modem(
        symbol_rate=_SYM,
        path=[
            ChirpSyncStep(
                sf=sf,
                oversample=oversample,
                zero_pad=_ZP,
                preamble_len=8,
                sfd_symbols=2.25,
                sync_symbols=2,
            ),
            DechirpStep(sf=sf, oversample=oversample, zero_pad=_ZP),
            CssDemapStep(sf=sf),
        ],
    )


def _compile(
    modem: Modem, direction: Literal["rx", "tx"], rate: float, src: Path, snk: Path
) -> GrPipeline:
    from marconi.engine.stages.registry import stage_registry

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
    # full payload, no trailing pad: the capture ends right after the last
    # symbol and the eof flush must still deliver every one of them
    assert len(out) >= sf * 40
    assert aligned_ber(out, bits, max_shift=8 * sf) == 0.0


def test_dechirp_rejects_wrong_input_rate(tmp_path: Path) -> None:
    # dechirp's window is oversample*2^sf samples/symbol, so the input rate must
    # be ~oversample*2^sf*symbol_rate (here 2*128*1.0 = 256). A grossly wrong rate
    # would dechirp garbage; the compiler must reject it, not emit it.
    m = _modem(7, 2)
    with pytest.raises(CompileError) as e:
        _compile(m, "rx", 200.0, tmp_path / "i", tmp_path / "o")
    assert "rate" in str(e.value).lower()


def test_dechirp_accepts_matching_input_rate(tmp_path: Path) -> None:
    m = _modem(7, 2)
    _compile(m, "rx", 256.0, tmp_path / "i", tmp_path / "o")  # exact: compiles


def test_dechirp_rate_check_tolerates_clock_correct_ppm(tmp_path: Path) -> None:
    # clock_correct resamples by 1/(1+ppm) — a legitimate sub-percent rate shift
    # (0.005% at 50 ppm) the tolerance must admit, not reject.
    m = Modem(
        symbol_rate=_SYM,
        path=[
            ClockCorrectStep(ppm=50.0),
            ChirpSyncStep(
                sf=7,
                oversample=2,
                zero_pad=_ZP,
                preamble_len=8,
                sfd_symbols=2.25,
                sync_symbols=2,
            ),
            DechirpStep(sf=7, oversample=2, zero_pad=_ZP),
            CssDemapStep(sf=7),
        ],
    )
    _compile(m, "rx", 256.0, tmp_path / "i", tmp_path / "o")  # within tolerance


def test_css_two_burst_capture_decodes_both(tmp_path: Path) -> None:
    """A capture with two CSS bursts — each under a DIFFERENT carrier offset —
    decodes both in one run: chirp_sync re-arms per preamble with fresh
    CFO/timing estimates (the pre-fix block applied burst 1's
    correction to the whole remainder of the capture). Also pins the
    re-lock diagnostics surfaced through RunResult."""
    from helpers._dsp import read_complex

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
    assert any(d.key == "locks" and d.count == 2 for d in r.diagnostics), r.diagnostics
    out = read_bits(op)
    assert len(out) > len(bits1) + len(bits2)  # both payloads (plus gap junk)
    assert aligned_ber(out[: len(bits1) + 8 * sf], bits1, max_shift=8 * sf) == 0.0
    assert aligned_ber(out[len(bits1) :], bits2, max_shift=40 * sf) == 0.0


def _anatomy_modem(
    sf: int, osr: int, sfd_symbols: float, sync_symbols: int, preamble_len: int = 8
) -> Modem:
    return Modem(
        symbol_rate=_SYM,
        path=[
            ChirpSyncStep(
                sf=sf,
                oversample=osr,
                zero_pad=_ZP,
                preamble_len=preamble_len,
                sfd_symbols=sfd_symbols,
                sync_symbols=sync_symbols,
            ),
            DechirpStep(sf=sf, oversample=osr, zero_pad=_ZP),
            CssDemapStep(sf=sf),
        ],
    )


@pytest.mark.parametrize(
    ("sf", "osr", "cfo_bins"),
    [(7, 2, 0.4), (11, 2, 0.4), (7, 2, 3.84)],
    # Without a down-chirp the CFO/STO split is only resolvable modulo one bin
    # (both project onto the same dechirp peak); the phase-slope estimator
    # recovers the fractional part and any whole-bin remainder self-cancels in
    # the payload dechirp bins (timing and carrier residuals are equal and
    # opposite). 0.4 pins the in-range case, 3.84 the whole-bin cancellation.
)
def test_css_nosfd_ber0_impaired(
    sf: int, osr: int, cfo_bins: float, tmp_path: Path
) -> None:
    """The second CSS anatomy: bare preamble run straight into payload — no
    down-chirp SFD, no sync gap. Pins that chirp_sync's acquisition grammar is
    parametric, not LoRa's frame layout."""
    ensure_worker_warm()
    be = GnuRadioBackend()
    rate = osr * (1 << sf) * _SYM
    bits = np.random.default_rng(sf).integers(0, 2, sf * 40).astype(np.uint8)
    # Departure from the preamble anchors on the first non-base window, so the
    # first payload symbol must sit clear of bin 0 — set its MSB.
    bits[0] = 1
    bp = write_bits(tmp_path / "in.bits", bits)
    clean, imp, op = tmp_path / "c.iq", tmp_path / "i.iq", tmp_path / "o.bits"
    m = _anatomy_modem(sf, osr, sfd_symbols=0.0, sync_symbols=0)
    assert be.run_pipeline(_compile(m, "tx", rate, bp, clean)).status == "ok"
    channel(
        clean,
        imp,
        snr_db=15.0,
        cfo_hz=cfo_bins * _SYM,
        sto=1.5,
        sfo_ppm=0.0,
        sample_rate=rate,
        seed=sf,
    )
    assert be.run_pipeline(_compile(m, "rx", rate, imp, op)).status == "ok"
    out = read_bits(op)
    assert len(out) >= sf * 40
    assert aligned_ber(out, bits, max_shift=8 * sf) == 0.0


def test_css_single_symbol_sfd_ber0(tmp_path: Path) -> None:
    """sfd_symbols=1.0 has always been validator-legal; the joint estimator
    must read exactly the down-chirp windows the params declare, not LoRa's
    two — one payload window mistaken for SFD corrupts the timing estimate."""
    ensure_worker_warm()
    be = GnuRadioBackend()
    sf, osr = 7, 2
    rate = osr * (1 << sf) * _SYM
    bw = _SYM * (1 << sf)
    bits = np.random.default_rng(3).integers(0, 2, sf * 40).astype(np.uint8)
    bp = write_bits(tmp_path / "in.bits", bits)
    clean, imp, op = tmp_path / "c.iq", tmp_path / "i.iq", tmp_path / "o.bits"
    m = _anatomy_modem(sf, osr, sfd_symbols=1.0, sync_symbols=0)
    assert be.run_pipeline(_compile(m, "tx", rate, bp, clean)).status == "ok"
    channel(
        clean,
        imp,
        snr_db=15.0,
        cfo_hz=0.03 * bw,
        sto=1.5,
        sfo_ppm=0.0,
        sample_rate=rate,
        seed=3,
    )
    assert be.run_pipeline(_compile(m, "rx", rate, imp, op)).status == "ok"
    out = read_bits(op)
    assert len(out) >= sf * 40
    assert aligned_ber(out, bits, max_shift=8 * sf) == 0.0


def test_css_long_sync_gap_locks(tmp_path: Path) -> None:
    """A declared 9-symbol gap between preamble run and SFD must be honored:
    the SFD hunt span derives from the declared anatomy (not a constant sized
    to LoRa's 2+2.25), and the preamble look-back skips the gap instead of
    letting gap symbols pollute the estimator's median (low same-sign gap
    bins outnumber the preamble windows in the look-back, so a median over
    the mixture lands on a gap bin, not the preamble's)."""
    from marconi.engine.backends.gnuradio.embedded import chirp

    ensure_worker_warm()
    be = GnuRadioBackend()
    sf, osr, pl, sync = 7, 2, 12, 9
    rate = osr * (1 << sf) * _SYM
    bw = _SYM * (1 << sf)
    sn = osr * (1 << sf)
    sync_syms = [20, 30, 40, 25, 35, 45, 28, 38, 22]
    payload = [37, 101, 5, 64, 90, 12, 77, 55, 3, 118]
    rng = np.random.default_rng(11)
    lead = 0.02 * (
        rng.standard_normal(2 * sn + 37) + 1j * rng.standard_normal(2 * sn + 37)
    )
    tail = 0.02 * (rng.standard_normal(14 * sn) + 1j * rng.standard_normal(14 * sn))
    frame = np.concatenate(
        [
            lead.astype(np.complex64),
            chirp.chirp_prefix(sf, osr, pl, 0.0),
            np.concatenate([chirp._modulate_symbol(s, sf, osr) for s in sync_syms]),
            chirp.chirp_prefix(sf, osr, 0, 2.25),
            np.concatenate([chirp._modulate_symbol(s, sf, osr) for s in payload]),
            tail.astype(np.complex64),
        ]
    ).astype(np.complex64)
    cap_p, imp = tmp_path / "cap.iq", tmp_path / "imp.iq"
    frame.tofile(cap_p)
    channel(cap_p, imp, snr_db=20.0, cfo_hz=0.02 * bw, sample_rate=rate, seed=11)
    m = Modem(
        symbol_rate=_SYM,
        path=[
            ChirpSyncStep(
                sf=sf,
                oversample=osr,
                zero_pad=_ZP,
                preamble_len=pl,
                sfd_symbols=2.25,
                sync_symbols=sync,
            ),
            DechirpStep(sf=sf, oversample=osr, zero_pad=_ZP),
        ],
    )
    snk = tmp_path / "syms.s16"
    r = be.run_pipeline(_compile(m, "rx", rate, imp, snk))
    assert r.status == "ok", r
    syms = np.fromfile(snk, dtype=np.int16)
    assert len(syms) >= len(payload), f"no lock or truncated: {len(syms)} symbols"
    assert syms[: len(payload)].tolist() == payload, (
        f"payload mismatch:\n  got:    {syms[: len(payload)].tolist()}\n"
        f"  expect: {payload}"
    )


@pytest.mark.xfail(
    reason="chirp_sync does not correct SFO; clock_correct owns it "
    "(see test_clock_correct_roundtrip). Pins the coverage boundary.",
    strict=True,
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
