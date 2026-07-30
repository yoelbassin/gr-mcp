from __future__ import annotations

import pytest
from pydantic import ValidationError

from marconi.engine.compile.compiler import CompileError, compile_modem
from marconi.engine.compile.ir import GrPipeline
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
from marconi.engine.types.enums import PskOrder
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

IQ = Descriptor(Level.IQ, "c")


def _compile(modem: Modem) -> GrPipeline:
    return compile_modem(
        modem,
        stage_registry(),
        direction="rx",
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


def test_run_rx_rejects_audio_final(tmp_path) -> None:
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
            source_io={"path": str(tmp_path / "in.cf32")},
        )


def test_run_rx_rejects_complex_symbol_final(tmp_path) -> None:
    modem = Modem(
        symbol_rate=1.0,
        path=[PskDemodStep(order=PskOrder(2))],
    )
    with pytest.raises(CompileError, match="complex soft symbols"):
        run_rx(
            modem,
            stage_registry(),
            sample_rate=4.0,
            start=Descriptor(Level.IQ, "c", amplitude=Amplitude.RMS_UNITY),
            workdir=tmp_path,
            source_io={"path": str(tmp_path / "in.cf32")},
        )
