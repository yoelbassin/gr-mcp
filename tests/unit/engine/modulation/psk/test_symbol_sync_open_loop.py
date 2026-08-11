# tests/unit/engine/modulation/psk/test_symbol_sync_open_loop.py
from __future__ import annotations

import pytest
from pydantic import ValidationError

from marconi.engine.compile.compiler import compile_modem
from marconi.engine.compile.errors import CompileError
from marconi.engine.modulation.psk.stages import SymbolSyncStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Amplitude, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

IQ = Descriptor(Level.IQ, ItemType.C, amplitude=Amplitude.RMS_UNITY)


def _kinds(loop_bw: float, sps: int, rate: float) -> list[str]:
    m = Modem(symbol_rate=1000.0, path=[SymbolSyncStep(sps=sps, loop_bw=loop_bw)])
    gr = compile_modem(
        m,
        stage_registry(),
        direction="rx",
        sample_rate=rate,
        start=IQ,
        source_io={"path": "in"},
        sink_io={"path": "out"},
    )
    return [b.kind for b in gr.blocks]


def test_open_loop_emits_oerder_meyr() -> None:
    assert "oerder_meyr_timing" in _kinds(0.0, 8, 8000.0)
    assert "symbol_sync_cc" not in _kinds(0.0, 8, 8000.0)


def test_closed_loop_still_emits_symbol_sync_cc() -> None:
    assert "symbol_sync_cc" in _kinds(0.045, 8, 8000.0)
    assert "oerder_meyr_timing" not in _kinds(0.045, 8, 8000.0)


def test_open_loop_requires_sps_ge_4() -> None:
    with pytest.raises(CompileError, match="samples per symbol"):
        _kinds(0.0, 2, 2000.0)  # sps 2 < 4


def test_step_accepts_loop_bw_zero() -> None:
    assert SymbolSyncStep(sps=8, loop_bw=0.0).loop_bw == 0.0


def test_open_loop_requires_excess_bandwidth() -> None:
    with pytest.raises(ValidationError, match="excess bandwidth"):
        SymbolSyncStep(sps=8, loop_bw=0.0, alpha=0.05)
    # the closed loop has no such physics: low rolloff stays valid there
    assert SymbolSyncStep(sps=8, loop_bw=0.045, alpha=0.05).alpha == 0.05
