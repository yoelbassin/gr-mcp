import pytest
from pydantic import ValidationError

from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.modulation.psk.stages import (
    PskDemap,
    PskDemapStep,
    PskDemod,
    PskDemodStep,
)
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType, PskOrder
from marconi.engine.types.levels import Level


def test_registered() -> None:
    reg = stage_registry()
    assert reg["psk_demod"].family == "psk" and reg["psk_demap"].family == "psk"


def test_demod_descriptor_is_soft_complex_symbols() -> None:
    out = PskDemod().out_descriptor(
        Descriptor(Level.IQ, ItemType.C), PskDemodStep(order=PskOrder(4))
    )
    assert out == Descriptor(Level.SYMBOLS, ItemType.C, carrier=Carrier.SOFT, order=4)


def test_demap_descriptor_is_hard_bits() -> None:
    out = PskDemap().out_descriptor(
        Descriptor(Level.SYMBOLS, ItemType.C, carrier=Carrier.SOFT),
        PskDemapStep(order=PskOrder(4)),
    )
    assert out == Descriptor(Level.BITS, ItemType.B, carrier=Carrier.HARD)


def test_order_validation_rejects_bad_and_missing() -> None:
    model = PskDemod().step_model
    with pytest.raises(ValidationError):
        model.model_validate({"order": 5})
    with pytest.raises(ValidationError):
        model.model_validate({})  # order is required


def test_demod_rx_uses_iq_rate_and_sps() -> None:
    # rate=8, symbol_rate=2 -> sps=4 != rate; RRC symbol_rate arg must be rate/sps
    b = CompileContext(Descriptor(Level.IQ, ItemType.C), rate=8.0, symbol_rate=2.0)
    PskDemod().emit_rx(b, PskDemodStep(order=PskOrder(4)))
    p = b.build("t", 8.0)
    rrc = next(x for x in p.blocks if x.kind == "rrc_filter_ccf")
    assert rrc.params["rate"] == 8.0 and rrc.params["sps"] == 4.0
    ss = next(x for x in p.blocks if x.kind == "symbol_sync_cc")
    assert ss.params["sps"] == 4.0
    costas = next(x for x in p.blocks if x.kind == "costas_loop_cc")
    assert costas.params["order"] == 4


def test_demod_tx_shaping_interp_is_sps() -> None:
    b = CompileContext(Descriptor(Level.IQ, ItemType.C), rate=8.0, symbol_rate=2.0)
    PskDemod().emit_tx(b, PskDemodStep(order=PskOrder(4)))
    rrc = next(x for x in b.build("t", 8.0).blocks if x.kind == "rrc_filter_ccf")
    assert rrc.params["interpolation"] == 4


def test_demap_tx_packs_then_maps() -> None:
    b = CompileContext(
        Descriptor(Level.SYMBOLS, ItemType.C, carrier=Carrier.SOFT),
        rate=4.0,
        symbol_rate=1.0,
    )
    PskDemap().emit_tx(b, PskDemapStep(order=PskOrder(8)))
    kinds = [x.kind for x in b.build("t", 4.0).blocks]
    assert kinds == ["pack_k_bits_bb", "chunks_to_symbols_bc"]
    pack = next(x for x in b.build("t", 4.0).blocks if x.kind == "pack_k_bits_bb")
    assert pack.params["k"] == 3  # log2(8)
