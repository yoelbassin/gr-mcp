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


def test_hard_slice_of_soft_demod_forwards_soft_confidence(tmp_path: Path) -> None:
    # A bits-final path hard-slices the demod's soft symbols; the run taps that
    # soft wire so the verdict matches the soft-terminal path instead of reading
    # "uncertain" purely because the last stream is hard bits.
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
    assert res.quality.verdict == "decoded"
    assert any(e.metric == "soft_confidence" for e in res.quality.evidence)
    assert res.hints == []  # fsk discriminator has a fixed bit sense, no hint


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
