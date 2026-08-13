from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from marconi.engine.compile.compiler import compile_modem
from marconi.engine.compile.errors import CompileError
from marconi.engine.compile.ir import GrPipeline
from marconi.engine.io.source import SourceSlice
from marconi.engine.modulation.fsk.stages import FskStep
from marconi.engine.modulation.psk.stages import PskDemodStep
from marconi.engine.run import run_rx
from marconi.engine.stages.conditioning import (
    AmStep,
    AnalyticStep,
    ChannelizeStep,
    FmDemodStep,
)
from marconi.engine.stages.general import SliceStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Amplitude, Descriptor
from marconi.engine.types.enums import ItemType, PskOrder
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

IQ = Descriptor(Level.IQ, ItemType.C)


def _compile(modem: Modem) -> GrPipeline:
    return compile_modem(
        modem,
        stage_registry(),
        sample_rate=48000.0,
        start=IQ,
        source_io={"path": "/dev/null"},
        sink_io={"path": "/dev/null"},
    )


def _audio_modem() -> Modem:
    return Modem(
        symbol_rate=2400.0,
        path=[
            AmStep(),
            AnalyticStep(),
            ChannelizeStep(decim=1, bandwidth_hz=2400.0, center_hz=1800.0),
            FskStep(deviation=600.0),
            SliceStep(),
        ],
    )


def test_audio_path_compiles_rx() -> None:
    pipe = _compile(_audio_modem())
    kinds = [b.kind for b in pipe.blocks]
    assert "complex_to_mag" in kinds
    assert "dc_blocker_ff" in kinds
    assert "hilbert_fc" in kinds


def _fm_modem(deviation: float = 3000.0) -> Modem:
    return Modem(
        symbol_rate=2400.0,
        path=[
            FmDemodStep(deviation=deviation),
            AnalyticStep(),
            ChannelizeStep(decim=1, bandwidth_hz=2400.0, center_hz=1800.0),
            FskStep(deviation=600.0),
            SliceStep(),
        ],
    )


def test_fm_audio_path_compiles_rx() -> None:
    pipe = _compile(_fm_modem())
    kinds = [b.kind for b in pipe.blocks]
    assert kinds.count("quadrature_demod") == 2  # discriminator + inner fsk
    assert "dc_blocker_ff" in kinds
    assert "hilbert_fc" in kinds


def test_fm_demod_rejects_nonpositive_deviation() -> None:
    # Now rejected at Step CONSTRUCTION (pydantic Field(gt=0)), not at compile.
    with pytest.raises(ValidationError, match="deviation"):
        _fm_modem(deviation=0.0)


def test_run_rx_rejects_audio_final(tmp_path: Path) -> None:
    modem = Modem(
        symbol_rate=1200.0,
        path=[FmDemodStep(deviation=3000.0)],
    )
    with pytest.raises(CompileError, match="ends at AUDIO"):
        run_rx(
            modem,
            stage_registry(),
            sample_rate=48000.0,
            start=IQ,
            workdir=tmp_path,
            source=SourceSlice(path=tmp_path / "in.cf32"),
        )


def test_run_rx_accepts_conditioned_iq_final(tmp_path: Path) -> None:
    # a conditioned-IQ terminal (channelize, no demod) now returns the cleaned
    # sub-band as a complex ("c") stream to survey/inspect further, instead of
    # a dead-end compile reject that pushed the agent back to survey.
    modem = Modem(
        symbol_rate=1200.0,
        path=[ChannelizeStep(decim=2, bandwidth_hz=8000.0, center_hz=10000.0)],
    )
    cap = tmp_path / "in.cf32"
    n = 8192
    np.exp(1j * 2 * np.pi * 0.1 * np.arange(n)).astype(np.complex64).tofile(cap)
    r = run_rx(
        modem,
        stage_registry(),
        sample_rate=48000.0,
        start=IQ,
        workdir=tmp_path,
        source=SourceSlice(path=cap),
    )
    assert r.status == "ok", r
    assert r.symbolstream is not None
    assert r.symbolstream.item_type == "c"
    assert r.symbolstream.path.suffix == ".cf32"


def test_run_rx_accepts_complex_symbol_final(tmp_path: Path) -> None:
    # Now a valid .cf32 terminal (bare demod, no demap), not a compile reject.
    modem = Modem(
        symbol_rate=1.0,
        path=[PskDemodStep(order=PskOrder(2))],
    )
    cap = tmp_path / "in.cf32"
    n = 8000
    np.exp(1j * 2 * np.pi * 0.01 * np.arange(n)).astype(np.complex64).tofile(cap)
    r = run_rx(
        modem,
        stage_registry(),
        sample_rate=4.0,
        start=Descriptor(Level.IQ, ItemType.C, amplitude=Amplitude.RMS_UNITY),
        workdir=tmp_path,
        source=SourceSlice(path=cap),
    )
    assert r.status == "ok", r
    assert r.symbolstream is not None
    assert r.symbolstream.item_type == "c"
    assert r.symbolstream.path.suffix == ".cf32"
    assert r.symbolstream.num_symbols > 0
    assert r.symbolstream.path.stat().st_size // 8 == r.symbolstream.num_symbols
