from marconi.core.descriptor import Carrier, Descriptor
from marconi.core.levels import Level
from marconi.core.stages import validate_params
from marconi.phy.compile_context import CompileContext
from marconi.phy.modulation.qam.stages import QamDemap, QamDemod
from marconi.phy.stages import stage_registry


def test_registered() -> None:
    reg = stage_registry()
    assert reg["qam_demod"].family == "qam" and reg["qam_demap"].family == "qam"


def test_demod_descriptor_is_hard_symbol_index() -> None:
    out = QamDemod().out_descriptor(Descriptor(Level.IQ, "c"), {"order": 16})
    assert out == Descriptor(Level.SYMBOLS, "b", carrier=Carrier.HARD)


def test_demap_descriptor_is_hard_bits() -> None:
    out = QamDemap().out_descriptor(
        Descriptor(Level.SYMBOLS, "b", carrier=Carrier.HARD), {"order": 16}
    )
    assert out == Descriptor(Level.BITS, "b", carrier=Carrier.HARD)


def test_order_validation_rejects_bad_and_missing() -> None:
    bad: list = []
    validate_params("qam_demod[0]", QamDemod().params_model, {"order": 32}, bad)
    assert bad
    missing: list = []
    validate_params("qam_demod[0]", QamDemod().params_model, {}, missing)
    assert missing  # order is required


def test_demod_rx_chain_is_rrc_sync_receiver() -> None:
    b = CompileContext(Descriptor(Level.IQ, "c"), rate=8.0, symbol_rate=2.0)
    QamDemod().emit_rx(b, {"order": 16})
    p = b.build("t", 8.0)
    assert [x.kind for x in p.blocks] == [
        "rrc_filter_ccf",
        "symbol_sync_cc",
        "constellation_receiver_cb",
    ]
    rrc = next(x for x in p.blocks if x.kind == "rrc_filter_ccf")
    assert rrc.params["rate"] == 8.0 and rrc.params["sps"] == 4.0
    cr = next(x for x in p.blocks if x.kind == "constellation_receiver_cb")
    assert cr.params["scheme"] == "qam" and cr.params["order"] == 16


def test_demod_tx_maps_then_shapes_with_interp_sps() -> None:
    b = CompileContext(Descriptor(Level.IQ, "c"), rate=8.0, symbol_rate=2.0)
    QamDemod().emit_tx(b, {"order": 64})
    p = b.build("t", 8.0)
    assert [x.kind for x in p.blocks] == ["chunks_to_symbols_bc", "rrc_filter_ccf"]
    c2s = next(x for x in p.blocks if x.kind == "chunks_to_symbols_bc")
    assert c2s.params["scheme"] == "qam" and c2s.params["order"] == 64
    rrc = next(x for x in p.blocks if x.kind == "rrc_filter_ccf")
    assert rrc.params["interpolation"] == 4  # sps


def test_demap_is_pure_pack_unpack_of_k() -> None:
    tx = CompileContext(
        Descriptor(Level.SYMBOLS, "b", carrier=Carrier.HARD), rate=4.0, symbol_rate=1.0
    )
    QamDemap().emit_tx(tx, {"order": 64})
    pack = next(x for x in tx.build("t", 4.0).blocks if x.kind == "pack_k_bits_bb")
    assert pack.params["k"] == 6  # log2(64)
    rx = CompileContext(
        Descriptor(Level.SYMBOLS, "b", carrier=Carrier.HARD), rate=4.0, symbol_rate=1.0
    )
    QamDemap().emit_rx(rx, {"order": 16})
    unpack = next(x for x in rx.build("t", 4.0).blocks if x.kind == "unpack_k_bits_bb")
    assert unpack.params["k"] == 4  # log2(16)
