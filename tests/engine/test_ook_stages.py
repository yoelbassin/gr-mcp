from marconi.engine.compile_context import CompileContext
from marconi.engine.descriptor import Carrier, Descriptor
from marconi.engine.levels import Level
from marconi.engine.modulation.ook.stages import OokEnvelope
from marconi.engine.stages import stage_registry


def test_registered_with_ook_family() -> None:
    assert stage_registry()["ook_envelope"].family == "ook"


def test_descriptor_is_soft_float_symbols() -> None:
    out = OokEnvelope().out_descriptor(Descriptor(Level.IQ, "c"), {})
    assert out == Descriptor(Level.SYMBOLS, "f", carrier=Carrier.SOFT)


def test_rx_is_mag_centre_then_timing() -> None:
    b = CompileContext(Descriptor(Level.IQ, "c"), rate=4.0, symbol_rate=1.0)
    OokEnvelope().emit_rx(b, {})
    kinds = [x.kind for x in b.build("t", 4.0).blocks]
    assert kinds == [
        "complex_to_mag",
        "multiply_const_ff",
        "add_const_ff",
        "symbol_sync_ff",
    ]


def test_tx_inverts_centre_upsamples_to_complex() -> None:
    b = CompileContext(Descriptor(Level.IQ, "c"), rate=4.0, symbol_rate=1.0)
    OokEnvelope().emit_tx(b, {})
    kinds = [x.kind for x in b.build("t", 4.0).blocks]
    assert kinds == [
        "add_const_ff",
        "multiply_const_ff",
        "repeat_f",
        "float_to_complex",
    ]
    rpt = next(x for x in b.build("t", 4.0).blocks if x.kind == "repeat_f")
    assert rpt.params["interp"] == 4
