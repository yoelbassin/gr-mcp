import math

import pytest

from marconi.core.descriptor import Descriptor
from marconi.core.levels import Level
from marconi.core.stages import StageDirectionError, validate_params
from marconi.phy.compile_context import CompileContext
from marconi.phy.stages import stage_registry
from marconi.phy.stages.conditioning import Channelize, Invert


def test_registered_as_conditioning() -> None:
    reg = stage_registry()
    assert reg["channelize"].family == "conditioning"
    assert reg["invert"].family == "conditioning"


def test_channelize_is_iq_to_iq_and_decimates() -> None:
    c = Channelize()
    out = c.out_descriptor(Descriptor(Level.IQ, "c"), {"decim": 2, "bandwidth_hz": 4.0})
    assert out == Descriptor(Level.IQ, "c")
    assert c.rate_factor({"decim": 2, "bandwidth_hz": 4.0}) == 0.5
    assert c.rate_factor({"decim": 4, "bandwidth_hz": 4.0}) == 0.25


def test_invert_is_iq_to_iq_unity_rate() -> None:
    c = Invert()
    assert c.out_descriptor(Descriptor(Level.IQ, "c"), {}) == Descriptor(Level.IQ, "c")
    assert c.rate_factor({}) == 1.0


def test_rx_only_rejects_tx() -> None:
    b = CompileContext(Descriptor(Level.IQ, "c"), rate=16.0, symbol_rate=1.0)
    with pytest.raises(StageDirectionError):
        Channelize().emit_tx(b, {"decim": 2, "bandwidth_hz": 4.0})
    with pytest.raises(StageDirectionError):
        Invert().emit_tx(b, {})


def test_channelize_emit_uses_input_rate_and_decim() -> None:
    b = CompileContext(Descriptor(Level.IQ, "c"), rate=16.0, symbol_rate=1.0)
    Channelize().emit_rx(b, {"decim": 2, "bandwidth_hz": 4.0, "center_hz": 4.0})
    p = b.build("t", 16.0)
    fx = next(x for x in p.blocks if x.kind == "freq_xlating_fir_filter_ccf")
    assert fx.params["decim"] == 2
    assert fx.params["rate"] == 16.0  # input-rate domain
    assert fx.params["center"] == 4.0
    assert fx.params["cutoff"] == 2.0 and fx.params["transition"] == 2.0  # bandwidth/2


def test_invert_emit_is_conjugate() -> None:
    b = CompileContext(Descriptor(Level.IQ, "c"), rate=16.0, symbol_rate=1.0)
    Invert().emit_rx(b, {})
    assert [x.kind for x in b.build("t", 16.0).blocks] == ["conjugate_cc"]


def test_param_validation() -> None:
    bad: list = []
    validate_params(
        "channelize[0]", Channelize().params_model, {"bandwidth_hz": 4.0}, bad
    )
    assert bad  # decim required
    bad2: list = []
    validate_params(
        "channelize[0]",
        Channelize().params_model,
        {"decim": 0, "bandwidth_hz": 4.0},
        bad2,
    )
    assert bad2  # decim must be >= 1


def test_rate_threads_through_channelize_to_demod() -> None:
    # The novel bit: a rate-changing stage must hand the DECIMATED rate to the demod.
    # Compile [channelize(decim=2), fsk] at input rate 16 -> fsk sees rate 8.
    from marconi.core.descriptor import Descriptor as D
    from marconi.phy.compiler import compile_modem
    from marconi.phy.models import ModemSpec, ModemStep

    modem = ModemSpec(
        symbol_rate=1.0,
        path=[
            ModemStep(conv="channelize", params={"decim": 2, "bandwidth_hz": 4.0}),
            ModemStep(conv="fsk", params={"deviation": 0.75}),
        ],
    )
    pipe = compile_modem(
        modem,
        stage_registry(),
        direction="rx",
        sample_rate=16.0,
        start=D(Level.IQ, "c"),
        source_io={"path": "x"},
        sink_io={"path": "y"},
    )
    fx = next(x for x in pipe.blocks if x.kind == "freq_xlating_fir_filter_ccf")
    assert fx.params["rate"] == 16.0  # channelize at the input rate
    qd = next(x for x in pipe.blocks if x.kind == "quadrature_demod")
    # fsk gain = rate / (2*pi*deviation) at the DECIMATED rate 8, not 16
    assert qd.params["gain"] == pytest.approx(8.0 / (2.0 * math.pi * 0.75))
