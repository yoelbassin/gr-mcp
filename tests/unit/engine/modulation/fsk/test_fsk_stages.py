import math

import pytest

from marconi.engine.compile.compiler import compile_modem
from marconi.engine.modulation.fsk.stages import FskStep
from marconi.engine.stages.general import SliceStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

IQ = Descriptor(Level.IQ, ItemType.C)


def _modem() -> Modem:
    return Modem(
        symbol_rate=1.0,
        path=[FskStep(deviation=1.0), SliceStep()],
    )


def test_fsk_descriptors_and_levels() -> None:
    reg = stage_registry()
    fsk, sl = reg["fsk"], reg["slice"]
    assert (fsk.from_level, fsk.to_level) == (Level.IQ, Level.SYMBOLS)
    assert (sl.from_level, sl.to_level) == (Level.SYMBOLS, Level.BITS)
    assert fsk.rate_factor({"deviation": 1.0}) == 1.0
    d = fsk.out_descriptor(IQ, {"deviation": 1.0})
    assert (d.level, d.item_type, d.carrier) == (
        Level.SYMBOLS,
        "f",
        Carrier.SOFT,
    )
    d2 = sl.out_descriptor(d, {})
    assert (d2.level, d2.item_type, d2.carrier) == (
        Level.BITS,
        "b",
        Carrier.HARD,
    )


def test_rx_compiles_to_fsk_chain() -> None:
    pipe = compile_modem(
        _modem(),
        stage_registry(),
        sample_rate=4.0,
        start=IQ,
        source_io={"path": "in.iq"},
        sink_io={"path": "out.bits"},
    )
    assert [b.kind for b in pipe.blocks] == [
        "iq_file_source",
        "quadrature_demod",
        "symbol_sync_ff",
        "binary_slicer",
        "bits_file_sink",
    ]
    qd = next(b for b in pipe.blocks if b.kind == "quadrature_demod")
    expected_gain = 4.0 / (2 * math.pi * 1.0)
    assert abs(qd.params["gain"] - expected_gain) < 1e-9  # type: ignore[operator]
    ss = next(b for b in pipe.blocks if b.kind == "symbol_sync_ff")
    assert ss.params["sps"] == 4.0  # IQ-domain sample_rate / symbol_rate


def test_formulas_use_iq_rate_not_sps() -> None:
    # sample_rate=8, symbol_rate=2 -> b.rate=8 but b.sps=4 (distinct), so these
    # assertions fail if any formula swaps b.rate <-> b.sps (rate-model invariant).
    modem = Modem(
        symbol_rate=2.0,
        path=[FskStep(deviation=1.0), SliceStep()],
    )
    rx = compile_modem(
        modem,
        stage_registry(),
        sample_rate=8.0,
        start=IQ,
        source_io={"path": "in.iq"},
        sink_io={"path": "out.bits"},
    )
    qd = next(b for b in rx.blocks if b.kind == "quadrature_demod")
    expected_gain = 8.0 / (2 * math.pi * 1.0)  # uses b.rate=8, not b.sps=4
    assert abs(qd.params["gain"] - expected_gain) < 1e-9  # type: ignore[operator]
    ss = next(b for b in rx.blocks if b.kind == "symbol_sync_ff")
    assert ss.params["sps"] == 4.0  # b.sps, not b.rate=8


def test_fsk_and_ook_accept_open_loop_reject_negative() -> None:
    from pydantic import ValidationError

    from marconi.engine.modulation.ook.stages import OokEnvelopeStep

    assert FskStep(deviation=2400.0, loop_bw=0.0).loop_bw == 0.0
    assert OokEnvelopeStep(loop_bw=0.0).loop_bw == 0.0
    with pytest.raises(ValidationError):
        FskStep(deviation=2400.0, loop_bw=-0.01)
    with pytest.raises(ValidationError):
        OokEnvelopeStep(loop_bw=-0.01)


def test_fsk_deviation_must_be_positive_when_given() -> None:
    # deviation=0 divides the discriminator gain by zero (an [internal_error]
    # at runtime today), and a negative deviation silently complements every
    # output bit — both are spec errors, not signals
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        FskStep(deviation=0.0)
    with pytest.raises(ValidationError):
        FskStep(deviation=-2400.0)


def test_fsk_deviation_defaults_to_half_symbol_rate() -> None:
    # deviation only scales the discriminator output (detection is
    # unaffected), so an agent surveying an unknown FSK signal should not
    # have to invent one; omitted means symbol_rate/2
    modem = Modem(symbol_rate=2.0, path=[FskStep(), SliceStep()])
    rx = compile_modem(
        modem,
        stage_registry(),
        sample_rate=8.0,
        start=IQ,
        source_io={"path": "in.iq"},
        sink_io={"path": "out.bits"},
    )
    qd = next(b for b in rx.blocks if b.kind == "quadrature_demod")
    expected_gain = 8.0 / (2 * math.pi * (2.0 / 2))  # b.rate / (2*pi*sym/2)
    assert abs(qd.params["gain"] - expected_gain) < 1e-9  # type: ignore[operator]
