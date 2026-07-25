from __future__ import annotations

import pytest

from marconi.engine.compile.compiler import CompileError, compile_modem
from marconi.engine.compile.ir import GrPipeline
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.levels import Level
from marconi.engine.types.models import ModemSpec, ModemStep

IQ = Descriptor(Level.IQ, "c")


def _compile(modem: ModemSpec) -> GrPipeline:
    return compile_modem(
        modem,
        stage_registry(),
        direction="rx",
        sample_rate=48000.0,
        start=IQ,
        source_io={"path": "/dev/null"},
        sink_io={"path": "/dev/null"},
    )


def _audio_modem() -> ModemSpec:
    return ModemSpec(
        symbol_rate=2400.0,
        path=[
            ModemStep(conv="am", params={}),
            ModemStep(conv="analytic", params={}),
            ModemStep(
                conv="channelize",
                params={"decim": 1, "bandwidth_hz": 2400.0, "center_hz": 1800.0},
            ),
            ModemStep(conv="fsk", params={"deviation": 600.0}),
            ModemStep(conv="slice", params={}),
        ],
    )


def test_audio_path_compiles_rx() -> None:
    pipe = _compile(_audio_modem())
    kinds = [b.kind for b in pipe.blocks]
    assert "complex_to_mag" in kinds
    assert "dc_blocker_ff" in kinds
    assert "hilbert_fc" in kinds


def _fm_modem(deviation: float = 3000.0) -> ModemSpec:
    return ModemSpec(
        symbol_rate=2400.0,
        path=[
            ModemStep(conv="fm_demod", params={"deviation": deviation}),
            ModemStep(conv="analytic", params={}),
            ModemStep(
                conv="channelize",
                params={"decim": 1, "bandwidth_hz": 2400.0, "center_hz": 1800.0},
            ),
            ModemStep(conv="fsk", params={"deviation": 600.0}),
            ModemStep(conv="slice", params={}),
        ],
    )


def test_fm_audio_path_compiles_rx() -> None:
    pipe = _compile(_fm_modem())
    kinds = [b.kind for b in pipe.blocks]
    assert kinds.count("quadrature_demod") == 2  # discriminator + inner fsk
    assert "dc_blocker_ff" in kinds
    assert "hilbert_fc" in kinds


def test_fm_demod_rejects_nonpositive_deviation() -> None:
    with pytest.raises(CompileError, match="deviation"):
        _compile(_fm_modem(deviation=0.0))
