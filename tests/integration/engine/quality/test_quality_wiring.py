from __future__ import annotations

from pathlib import Path

import numpy as np
from integration.engine.quality._capture import make_clean_capture

from marconi.engine.modulation.fsk.stages import FskStep, MfskSoftDemapStep
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
        source_io={"path": str(iq)},
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
        source_io={"path": str(iq)},
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
        source_io={"path": str(bad)},
    )
    assert res.status == "error"
    assert res.quality is None
