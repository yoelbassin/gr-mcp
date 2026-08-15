from __future__ import annotations

from pathlib import Path

import numpy as np
from integration.engine.quality._capture import (
    make_clean_capture,
    make_multilevel_capture,
    make_noise_capture,
)

from marconi.engine.io.source import SourceSlice
from marconi.engine.modulation.fsk.stages import FskStep, MfskSoftDemapStep
from marconi.engine.quality import _SOFT_MULTILEVEL_SEPARATION, QualityReport
from marconi.engine.run import run_rx
from marconi.engine.stages.general import SliceStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

IQ = Descriptor(Level.IQ, ItemType.C)
_SR, _SYM, _DEV = 4.0, 1.0, 1.0


def _soft_rx(symbol_rate: float) -> Modem:
    return Modem(
        symbol_rate=symbol_rate,
        path=[FskStep(deviation=_DEV), MfskSoftDemapStep(levels=[-1.0, 1.0])],
    )


def test_clean_soft_run_is_decoded(tmp_path: Path) -> None:
    iq = make_clean_capture(tmp_path)
    workdir = tmp_path / "rx"
    workdir.mkdir()
    res = run_rx(
        _soft_rx(_SYM),
        stage_registry(),
        sample_rate=_SR,
        start=IQ,
        workdir=workdir,
        source=SourceSlice(path=iq),
    )
    assert res.status == "ok", res
    assert res.quality is not None
    assert res.quality.verdict == "decoded"
    assert any(e.metric == "soft_confidence" for e in res.quality.evidence)


def test_hard_slice_of_bare_demod_is_signal_present_not_decoded(
    tmp_path: Path,
) -> None:
    # A bits-final path hard-slices the demod's soft SYMBOLS: the run taps that
    # wire, but a per-symbol eye is a decision-quality proxy, not a per-bit
    # confidence -- a clean eye can be a mistimed demod's clean-but-wrong
    # symbols (msk on real GMSK measures a cleaner eye than the correct fsk
    # decode). So the tap rides the detection tier (soft_eye): signal-present,
    # verdict "uncertain", never a self-certified "decoded". A soft-demap stage
    # (bits-level LLRs) is what earns "decoded" -- test_clean_soft_run_is_decoded.
    iq = make_clean_capture(tmp_path)
    rx = Modem(symbol_rate=_SYM, path=[FskStep(deviation=_DEV), SliceStep()])
    workdir = tmp_path / "rx_hard"
    workdir.mkdir()
    res = run_rx(
        rx,
        stage_registry(),
        sample_rate=_SR,
        start=IQ,
        workdir=workdir,
        source=SourceSlice(path=iq),
    )
    assert res.status == "ok", res
    assert res.bitstream is not None  # the delivered stream is still hard bits
    assert res.quality is not None
    assert res.quality.verdict == "uncertain", res.quality
    soft = [e for e in res.quality.evidence if e.source == "soft_stream"]
    assert soft and all(
        e.metric == "soft_eye" and e.assessment == "positive" for e in soft
    ), res.quality
    assert "detection only" in res.quality.rationale
    # No msk polarity hint (the fsk discriminator has a fixed bit sense). The
    # open-loop hint now DOES appear: a bare demod no longer self-certifies
    # "decoded", so a closed-loop fsk landing "uncertain" is nudged toward
    # loop_bw=0 -- the recovery a bursty/short capture needs.
    assert not any("polarity may be inverted" in h for h in res.hints)
    assert any("loop_bw" in h for h in res.hints), res.hints


def _bare_demod_quality(iq: Path, work: Path) -> QualityReport:
    work.mkdir()
    res = run_rx(
        Modem(symbol_rate=_SYM, path=[FskStep(deviation=_DEV)]),
        stage_registry(),
        sample_rate=_SR,
        start=IQ,
        workdir=work,
        source=SourceSlice(path=iq),
    )
    assert res.status == "ok", res
    assert res.quality is not None
    return res.quality


def test_a_multilevel_eye_earns_the_positive_that_noise_cannot(tmp_path: Path) -> None:
    """The reading the off-air DMR gate rides on, in reach of the suite: a bare
    discriminator over a 4-level FM capture must earn a soft_eye POSITIVE at or
    above the multilevel separation bar, and the same front end over noise must
    earn nothing.

    The absence of a negative proves nothing here and is why that gate's old
    assertion could not fail: at symbols grade _emit_soft drops the negative
    (a correct decode of a bursty capture reads below the noise floor), and no
    other producer is reachable from a demod-only path — sync/word-validity
    need stages this path does not have. So both runs read "uncertain", noise
    included, and only the POSITIVE separates them.

    Measured through these two captures: the 4-level eye scores separation
    18.0, while noise forced to a 4-level fit reaches 3.20 (margin 1.56, no
    evidence) — the 4.0 bar sits between them."""
    good = _bare_demod_quality(make_multilevel_capture(tmp_path), tmp_path / "rx_4lvl")
    eye = [e for e in good.evidence if e.metric == "soft_eye"]
    assert eye and all(e.assessment == "positive" for e in eye), good
    assert all(e.value >= _SOFT_MULTILEVEL_SEPARATION for e in eye), good
    assert good.margin is not None and good.margin >= _SOFT_MULTILEVEL_SEPARATION

    noise = _bare_demod_quality(make_noise_capture(tmp_path), tmp_path / "rx_noise")
    assert not [e for e in noise.evidence if e.assessment == "positive"], noise
    # A CHARACTERIZATION of what a demod-only path can say today, not a decision
    # that it should: signal and noise landing on the same verdict is the whole
    # reason the e2e gate had to move to the positive. A stage that gives this
    # path a decode-grade tap SHOULD split these two, and this line is then the
    # thing to change -- do not read it as a rule that they must match.
    assert good.verdict == noise.verdict == "uncertain", (good, noise)


def test_non_ok_result_has_no_quality(tmp_path: Path) -> None:
    bad = tmp_path / "bad.cf32"
    np.array([np.nan + 0j], dtype=np.complex64).tofile(bad)
    workdir = tmp_path / "rx_bad"
    workdir.mkdir()
    res = run_rx(
        _soft_rx(_SYM),
        stage_registry(),
        sample_rate=_SR,
        start=IQ,
        workdir=workdir,
        source=SourceSlice(path=bad),
    )
    assert res.status == "error"
    assert res.quality is None
