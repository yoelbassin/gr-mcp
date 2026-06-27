import math

from marconi.core.descriptor import Carrier, Descriptor
from marconi.core.levels import Level
from marconi.phy.compiler import compile_modem
from marconi.phy.models import ModemSpec, ModemStep
from marconi.phy.stages import stage_registry

IQ = Descriptor(Level.IQ, "c")


def _modem() -> ModemSpec:
    return ModemSpec(
        symbol_rate=1.0,
        path=[
            ModemStep(conv="fsk", params={"deviation": 1.0}),
            ModemStep(conv="slice"),
        ],
    )


def test_registry_exposes_fsk_and_slice() -> None:
    assert {"fsk", "slice"} <= set(stage_registry())


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
        direction="rx",
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


def test_tx_compiles_to_reversed_chain() -> None:
    pipe = compile_modem(
        _modem(),
        stage_registry(),
        direction="tx",
        sample_rate=4.0,
        start=IQ,
        source_io={"path": "in.bits"},
        sink_io={"path": "out.iq"},
    )
    assert [b.kind for b in pipe.blocks] == [
        "bits_file_source",
        "chunks_to_symbols",
        "repeat_f",
        "frequency_modulator",
        "iq_file_sink",
    ]
    assert next(b for b in pipe.blocks if b.kind == "repeat_f").params["interp"] == 4
    fm = next(b for b in pipe.blocks if b.kind == "frequency_modulator")
    sens = 2 * math.pi * 1.0 / 4.0
    assert abs(fm.params["sensitivity"] - sens) < 1e-9  # type: ignore[operator]
