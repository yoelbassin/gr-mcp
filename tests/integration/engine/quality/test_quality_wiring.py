from __future__ import annotations

from pathlib import Path

import numpy as np

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
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


def _make_clean_capture(tmp_path: Path, n_bits: int = 4096) -> Path:
    rng = np.random.default_rng(7)
    bits_path = tmp_path / "tx.u8"
    rng.integers(0, 2, n_bits).astype(np.uint8).tofile(bits_path)
    iq_path = tmp_path / "clean.cf32"
    tx = Modem(symbol_rate=_SYM, path=[FskStep(deviation=_DEV), SliceStep()])
    gr = compile_modem(
        tx,
        stage_registry(),
        direction="tx",
        sample_rate=_SR,
        start=IQ,
        source_io={"path": str(bits_path)},
        sink_io={"path": str(iq_path)},
    )
    ensure_worker_warm()
    r = GnuRadioBackend().run_pipeline(gr, timeout=120.0)
    assert r.status == "ok", r
    return iq_path


def _soft_rx(symbol_rate: float) -> Modem:
    return Modem(
        symbol_rate=symbol_rate,
        path=[FskStep(deviation=_DEV), MfskSoftDemapStep(levels=[-1.0, 1.0])],
    )


def test_clean_soft_run_is_decoded(tmp_path: Path) -> None:
    iq = _make_clean_capture(tmp_path)
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


def test_hard_path_with_no_evidence_is_uncertain(tmp_path: Path) -> None:
    iq = _make_clean_capture(tmp_path)
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
    assert res.quality is not None
    assert res.quality.verdict == "uncertain"
    assert "no checkable evidence" in res.quality.rationale


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
