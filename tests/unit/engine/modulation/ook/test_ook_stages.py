from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.modulation.ook.stages import OokEnvelope, OokEnvelopeStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level


def test_registered_with_ook_family() -> None:
    assert stage_registry()["ook_envelope"].family == "ook"


def test_description_states_conditional_agc_contract() -> None:
    # accepts_amplitude alone reads [peak_unity, rms_unity] unconditionally; the
    # description is what tells the agent that contract only holds closed-loop,
    # and that open-loop must carry NO agc.
    desc = stage_registry()["ook_envelope"].description
    assert "agc" in desc.lower(), desc
    assert "open-loop" in desc.lower(), desc
    assert len(desc) > 40


def test_descriptor_is_soft_float_symbols() -> None:
    out = OokEnvelope().out_descriptor(
        Descriptor(Level.IQ, ItemType.C), OokEnvelopeStep()
    )
    assert out == Descriptor(Level.SYMBOLS, ItemType.F, carrier=Carrier.SOFT)


def test_rx_is_mag_centre_then_timing() -> None:
    b = CompileContext(Descriptor(Level.IQ, ItemType.C), rate=4.0, symbol_rate=1.0)
    OokEnvelope().emit_rx(b, OokEnvelopeStep())
    kinds = [x.kind for x in b.build("t", 4.0).blocks]
    assert kinds == [
        "complex_to_mag",
        "multiply_const_ff",
        "add_const_ff",
        "symbol_sync_ff",
    ]
