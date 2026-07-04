from __future__ import annotations

from marconi.core.descriptor import Descriptor
from marconi.core.levels import Level
from marconi.phy.compiler import compile_modem
from marconi.phy.models import ModemSpec, ModemStep
from marconi.phy.stages import stage_registry

IQ = Descriptor(Level.IQ, "c")


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
    pipe = compile_modem(
        _audio_modem(),
        stage_registry(),
        direction="rx",
        sample_rate=48000.0,
        start=IQ,
        source_io={"path": "/dev/null"},
        sink_io={"path": "/dev/null"},
    )
    kinds = [b.kind for b in pipe.blocks]
    assert "complex_to_mag" in kinds
    assert "dc_blocker_ff" in kinds
    assert "hilbert_fc" in kinds
