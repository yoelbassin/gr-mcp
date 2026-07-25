from __future__ import annotations

from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.levels import Level

IQ = Descriptor(Level.IQ, "c")


def test_msk_registered_and_contract() -> None:
    msk = stage_registry()["msk"]
    assert msk.from_level == Level.IQ and msk.to_level == Level.SYMBOLS
    d = msk.out_descriptor(IQ, {})
    assert (d.level, d.item_type, d.carrier) == (Level.SYMBOLS, "f", Carrier.SOFT)


def test_msk_emits_msk_demod_with_ctx_sps() -> None:
    msk = stage_registry()["msk"]
    b = CompileContext(IQ, rate=48000.0, symbol_rate=2400.0)
    msk.emit_rx(b, {})
    pipe = b.build("t", 48000.0)
    (blk,) = [x for x in pipe.blocks if x.kind == "msk_demod"]
    assert blk.params["sps"] == 20.0


def test_msk_default_loop_bw_matches_block_registration() -> None:
    # loop_bw is spelled in _MskParams, GR_BLOCKS["msk_demod"], and
    # make_msk_demod; the stage default is the live value, but a divergence in
    # the block-side fallbacks would be a silent pin drift. Anchor them together.
    import inspect

    from marconi.engine.backends.gnuradio.embedded.msk import make_msk_demod
    from marconi.engine.modulation.fsk.stages import _MskParams

    block_default = inspect.signature(make_msk_demod).parameters["loop_bw"].default
    assert _MskParams().loop_bw == block_default
