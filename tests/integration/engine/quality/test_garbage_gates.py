from __future__ import annotations

from pathlib import Path

import numpy as np
from integration.engine.quality._capture import make_clean_capture

from marconi.engine.modulation.fsk.stages import FskStep, MfskSoftDemapStep
from marconi.engine.run import PipelineResult, run_rx
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
