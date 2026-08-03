from __future__ import annotations

from pathlib import Path

import numpy as np
from integration.engine.quality._capture import make_clean_capture

from marconi.engine.backends.gnuradio.runner import ensure_worker_warm
from marconi.engine.modulation.css.stages import ChirpSyncStep, DechirpStep
from marconi.engine.modulation.fsk.stages import FskStep, MfskSoftDemapStep
from marconi.engine.run import PipelineResult, run_rx
from marconi.engine.stages.probes import BurstProbeStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

IQ = Descriptor(Level.IQ, ItemType.C)
_SR, _SYM, _DEV = 4.0, 1.0, 1.0
_CSS_SF, _CSS_OSR, _CSS_ZP, _CSS_SYM = 7, 2, 4, 1.0
_CSS_PREAMBLE_LEN, _CSS_SFD_SYMBOLS, _CSS_SYNC_SYMBOLS = 8, 2.25, 2
_CSS_RATE = _CSS_OSR * (1 << _CSS_SF) * _CSS_SYM


def _css_marks_only_modem() -> Modem:
    return Modem(
        symbol_rate=_CSS_SYM,
        path=[
            ChirpSyncStep(
                sf=_CSS_SF,
                oversample=_CSS_OSR,
                zero_pad=_CSS_ZP,
                preamble_len=_CSS_PREAMBLE_LEN,
                sfd_symbols=_CSS_SFD_SYMBOLS,
                sync_symbols=_CSS_SYNC_SYMBOLS,
            ),
            DechirpStep(sf=_CSS_SF, oversample=_CSS_OSR, zero_pad=_CSS_ZP),
            BurstProbeStep(),
        ],
    )


def _soft_rx(symbol_rate: float) -> Modem:
    return Modem(
        symbol_rate=symbol_rate,
        path=[FskStep(deviation=_DEV), MfskSoftDemapStep(levels=[-1.0, 1.0])],
    )


def _run(modem: Modem, iq: Path, workdir: Path) -> PipelineResult:
    workdir.mkdir(exist_ok=True)
    return run_rx(
        modem,
        stage_registry(),
        sample_rate=_SR,
        start=IQ,
        workdir=workdir,
        source_io={"path": str(iq)},
    )


def _not_trusted(res: PipelineResult) -> bool:
    return res.status != "ok" or (
        res.quality is not None and res.quality.verdict != "decoded"
    )


def test_noise_only_capture_is_not_called_decoded(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    noise = (rng.normal(0, 1, 16384) + 1j * rng.normal(0, 1, 16384)) / np.sqrt(2)
    iq = tmp_path / "noise.cf32"
    noise.astype(np.complex64).tofile(iq)
    res = _run(_soft_rx(_SYM), iq, tmp_path / "rx")
    assert _not_trusted(res), res.quality


def test_wrong_symbol_rate_is_not_called_decoded(tmp_path: Path) -> None:
    iq = make_clean_capture(tmp_path)
    res = _run(_soft_rx(1.6), iq, tmp_path / "rx")
    assert _not_trusted(res), res.quality


def test_clean_control_is_decoded(tmp_path: Path) -> None:
    iq = make_clean_capture(tmp_path)
    res = _run(_soft_rx(_SYM), iq, tmp_path / "rx")
    assert res.status == "ok"
    assert res.quality is not None and res.quality.verdict == "decoded"


def test_pure_tone_capture_is_not_called_decoded(tmp_path: Path) -> None:
    n = 16384
    t = np.arange(n)
    tone = np.exp(2j * np.pi * 0.05 * t).astype(np.complex64)
    iq = tmp_path / "tone.cf32"
    tone.tofile(iq)
    res = _run(_soft_rx(_SYM), iq, tmp_path / "rx")
    assert _not_trusted(res), res.quality


def test_pure_noise_css_marks_path_is_not_called_decoded(tmp_path: Path) -> None:
    ensure_worker_warm()
    rng = np.random.default_rng(11)
    n = 256 * 200
    noise = ((rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)) / np.sqrt(2)).astype(
        np.complex64
    )
    iq = tmp_path / "css_noise.iq"
    noise.tofile(iq)
    workdir = tmp_path / "rx"
    workdir.mkdir()
    res = run_rx(
        _css_marks_only_modem(),
        stage_registry(),
        sample_rate=_CSS_RATE,
        start=IQ,
        workdir=workdir,
        source_io={"path": str(iq)},
    )
    assert _not_trusted(res), (res.status, res.error, res.quality, res.marks)
