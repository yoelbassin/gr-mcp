from __future__ import annotations

from marconi.engine.compile.compiler import compile_pipeline
from marconi.engine.modulation.fsk.stages import FskStep
from marconi.engine.stages.general import SliceStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

IQ = Descriptor(Level.IQ, ItemType.C)


def test_compiled_pipeline_carries_full_trace() -> None:
    modem = Modem(symbol_rate=1.0, path=[FskStep(deviation=1.0), SliceStep()])
    cp = compile_pipeline(
        modem,
        stage_registry(),
        direction="rx",
        sample_rate=4.0,
        start=IQ,
        source_io={"path": "unused.cf32"},
        sink_io={"path": "unused.u8"},
    )
    assert len(cp.boundaries) == 3
    assert len(cp.rates) == 3
    assert cp.boundaries[0] == IQ
    assert cp.boundaries[-1] == cp.final
    assert cp.rates[0] == 4.0
