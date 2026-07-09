from __future__ import annotations

from marconi.core.descriptor import Carrier, Descriptor
from marconi.core.levels import Level
from marconi.phy.compile_context import CompileContext
from marconi.phy.stages import stage_registry

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
