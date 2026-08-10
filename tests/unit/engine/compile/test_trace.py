from __future__ import annotations

from pathlib import Path

from helpers._fixtures import FakeDemapStep, FakeDemodStep, fixture_registry

from marconi.engine.compile.compiler import (
    CompiledPipeline,
    compile_pipeline,
    trace_sink_path,
)
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


def _demod_demap(trace_dir: Path | None) -> CompiledPipeline:
    modem = Modem(symbol_rate=2.0, path=[FakeDemodStep(), FakeDemapStep()])
    return compile_pipeline(
        modem,
        fixture_registry(),
        direction="rx",
        sample_rate=8.0,
        start=IQ,
        source_io={"path": "in.iq"},
        sink_io={"path": "out.bits"},
        trace_dir=trace_dir,
    )


def test_trace_dir_taps_each_gr_stage(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    cp = _demod_demap(trace_dir)
    assert cp.gr_steps == 2
    assert cp.gr is not None
    taps = {
        str(b.params.get("path"))
        for b in cp.gr.blocks
        if str(b.params.get("path", "")).startswith(str(trace_dir))
    }
    assert taps == {
        str(trace_sink_path(trace_dir, 0, "fake_demod", cp.boundaries[1].item_type)),
        str(trace_sink_path(trace_dir, 1, "fake_demap", cp.boundaries[2].item_type)),
    }


def test_no_trace_dir_adds_no_taps() -> None:
    base = _demod_demap(None)
    traced = _demod_demap(Path("/tmp/whatever/trace"))
    assert base.gr is not None and traced.gr is not None
    # exactly one extra sink block per GR stage when tracing, nothing otherwise
    assert len(traced.gr.blocks) - len(base.gr.blocks) == base.gr_steps
