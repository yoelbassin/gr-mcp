from pathlib import Path

import numpy as np
import pytest

from marconi.deadline import RunTimeout
from marconi.engine.io.source import SourceSlice
from marconi.engine.run import run_rx
from marconi.engine.stages.registry import stage_registry, step_models
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem


def _tiny_capture(tmp_path: Path) -> Path:
    # a short clean tone-ish IQ; content irrelevant — the run must time out first
    n = 200_000
    x = np.exp(1j * 2 * np.pi * 0.01 * np.arange(n)).astype(np.complex64)
    p = tmp_path / "cap.cf32"
    x.tofile(p)
    return p


def _fsk_modem() -> Modem:
    return Modem.from_spec(
        {"symbol_rate": 1000.0, "path": [{"conv": "fsk", "deviation": 1.0}]},
        step_models(),
    )


def test_run_rx_times_out_deterministically(tmp_path: Path) -> None:
    cap = _tiny_capture(tmp_path)
    with pytest.raises(RunTimeout):
        run_rx(
            _fsk_modem(),
            stage_registry(),
            sample_rate=8000.0,
            start=Descriptor(Level.IQ, ItemType.C),
            workdir=tmp_path,
            source=SourceSlice(path=cap),
            timeout=0.0,
        )


def test_run_rx_normal_timeout_still_succeeds(tmp_path: Path) -> None:
    cap = _tiny_capture(tmp_path)
    result = run_rx(
        _fsk_modem(),
        stage_registry(),
        sample_rate=8000.0,
        start=Descriptor(Level.IQ, ItemType.C),
        workdir=tmp_path,
        source=SourceSlice(path=cap),
        timeout=180.0,
    )
    assert result.status in ("ok", "empty")  # ran to completion, not timed out
