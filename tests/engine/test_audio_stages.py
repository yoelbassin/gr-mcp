from __future__ import annotations

import pytest

from marconi.engine.compile.compiler import CompileError, compile_modem
from marconi.engine.compile.ir import GrPipeline
from marconi.engine.run import run_rx
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Amplitude, Descriptor
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


def test_run_rx_rejects_audio_final(tmp_path) -> None:
    modem = ModemSpec(
        symbol_rate=1200.0,
        path=[ModemStep(conv="fm_demod", params={"deviation": 3000.0})],
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
    modem = ModemSpec(
        symbol_rate=1.0,
        path=[ModemStep(conv="psk_demod", params={"order": 2})],
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
