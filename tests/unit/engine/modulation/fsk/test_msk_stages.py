from __future__ import annotations

from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.modulation.fsk.stages import MskStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level

IQ = Descriptor(Level.IQ, ItemType.C)


def test_msk_registered_and_contract() -> None:
    msk = stage_registry()["msk"]
    assert msk.from_level == Level.IQ and msk.to_level == Level.SYMBOLS
    d = msk.out_descriptor(IQ, MskStep())
    assert (d.level, d.item_type, d.carrier) == (Level.SYMBOLS, "f", Carrier.SOFT)


def test_msk_emits_msk_demod_with_ctx_sps() -> None:
    msk = stage_registry()["msk"]
    b = CompileContext(IQ, rate=48000.0, symbol_rate=2400.0)
    msk.emit_rx(b, MskStep())
    pipe = b.build("t", 48000.0)
    (blk,) = [x for x in pipe.blocks if x.kind == "msk_demod"]
    assert blk.params["sps"] == 20.0


def test_msk_exposes_loop_pole_and_mf_oversample_defaults() -> None:
    s = MskStep()
    assert s.loop_pole == 0.52
    assert s.mf_oversample == 12


def test_msk_emit_passes_loop_pole_and_mf_oversample() -> None:
    msk = stage_registry()["msk"]
    b = CompileContext(IQ, rate=48000.0, symbol_rate=2400.0)
    msk.emit_rx(b, MskStep(loop_pole=0.4, mf_oversample=8))
    (blk,) = [x for x in b.build("t", 48000.0).blocks if x.kind == "msk_demod"]
    assert blk.params["loop_pole"] == 0.4
    assert blk.params["mf_oversample"] == 8


def test_msk_defaults_match_block_registration() -> None:
    # loop_bw/loop_pole/mf_oversample are spelled in MskStep, GR_BLOCKS, and
    # make_msk_demod; the stage default is the live value, but a divergence in
    # the block-side fallbacks would be a silent pin drift. Anchor them together.
    import inspect

    from marconi.engine.backends.gnuradio.embedded.msk import make_msk_demod

    sig = inspect.signature(make_msk_demod).parameters
    assert MskStep().loop_bw == sig["loop_bw"].default
    assert MskStep().loop_pole == sig["loop_pole"].default
    assert MskStep().mf_oversample == sig["mf_oversample"].default
