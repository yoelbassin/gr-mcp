from __future__ import annotations

from pathlib import Path

from marconi.engine.coding.stages_bits import DescrambleStep
from marconi.engine.compile.compiler import CompiledPipeline, compile_pipeline
from marconi.engine.modulation.fsk.stages import FskStep
from marconi.engine.stages.general import SliceStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem
from marconi.engine.types.step import Step

IQ = Descriptor(Level.IQ, ItemType.C)


def _compile(*steps: Step, quality_tap: bool = True) -> CompiledPipeline:
    return compile_pipeline(
        Modem(symbol_rate=1.0, path=list(steps)),
        stage_registry(),
        sample_rate=4.0,
        start=IQ,
        source_io={"path": "in.cf32"},
        sink_io={"path": "/run/seam.dat"},
        quality_tap=quality_tap,
    )


def test_soft_tap_taps_the_demod_wire_feeding_the_slicer() -> None:
    cp = _compile(FskStep(deviation=1.0), SliceStep())
    assert cp.soft_seam == Path("/run/soft_tap.f32")
    assert cp.gr is not None
    taps = [b for b in cp.gr.blocks if b.kind == "soft_bits_file_sink"]
    assert len(taps) == 1
    assert taps[0].params["path"] == str(cp.soft_seam)
    slicer = next(b for b in cp.gr.blocks if b.kind == "binary_slicer")
    feeds_slicer = next(
        c.src_block for c in cp.gr.connections if c.dst_block == slicer.id
    )
    feeds_tap = next(
        c.src_block for c in cp.gr.connections if c.dst_block == taps[0].id
    )
    assert feeds_tap == feeds_slicer  # same soft-symbol wire


def test_no_tap_when_not_requested() -> None:
    cp = _compile(FskStep(deviation=1.0), SliceStep(), quality_tap=False)
    assert cp.soft_seam is None
    assert cp.gr is not None
    assert not any(b.kind == "soft_bits_file_sink" for b in cp.gr.blocks)


def test_no_tap_for_a_soft_terminal_path() -> None:
    # the seam is already the soft stream (its normal sink IS a soft_bits_file_sink);
    # no separate tap sidecar is added
    cp = _compile(FskStep(deviation=1.0))
    assert cp.soft_seam is None
    assert cp.gr is not None
    assert not any(b.params.get("path") == "/run/soft_tap.f32" for b in cp.gr.blocks)


def test_no_tap_when_a_coding_tail_follows_the_slice() -> None:
    # scoped to all-GR bits-final paths; a coding tail carries its own evidence
    cp = _compile(FskStep(deviation=1.0), SliceStep(), DescrambleStep(sequence="ff"))
    assert cp.coding is not None
    assert cp.soft_seam is None
    assert cp.gr is not None
    assert not any(b.kind == "soft_bits_file_sink" for b in cp.gr.blocks)
