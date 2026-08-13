import math

import pytest
from pydantic import ValidationError

from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.stages.conditioning import (
    Channelize,
    ChannelizeStep,
    ClockCorrectStep,
    Invert,
    InvertStep,
)
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level


def test_registered_as_conditioning() -> None:
    reg = stage_registry()
    assert reg["channelize"].family == "conditioning"
    assert reg["invert"].family == "conditioning"


def test_channelize_is_iq_to_iq_and_decimates() -> None:
    c = Channelize()
    step = ChannelizeStep(decim=2, bandwidth_hz=4.0)
    out = c.out_descriptor(Descriptor(Level.IQ, ItemType.C), step)
    assert out == Descriptor(Level.IQ, ItemType.C)
    assert c.rate_factor(step) == 0.5
    assert c.rate_factor(ChannelizeStep(decim=4, bandwidth_hz=4.0)) == 0.25


def test_invert_is_iq_to_iq_unity_rate() -> None:
    c = Invert()
    step = InvertStep()
    out = c.out_descriptor(Descriptor(Level.IQ, ItemType.C), step)
    assert out == Descriptor(Level.IQ, ItemType.C)
    assert c.rate_factor(step) == 1.0


def test_channelize_bounds_center_and_bandwidth_against_rate() -> None:
    c = Channelize()
    ok = ChannelizeStep(decim=4, bandwidth_hz=25_000.0, center_hz=100_000.0)
    assert c.validate_input_rate(ok, 250_000.0) is None
    wrapped = ChannelizeStep(decim=4, bandwidth_hz=25_000.0, center_hz=490_000.0)
    assert "Nyquist" in (c.validate_input_rate(wrapped, 250_000.0) or "")
    folded = ChannelizeStep(decim=4, bandwidth_hz=80_000.0)
    assert "folds" in (c.validate_input_rate(folded, 250_000.0) or "")


def test_translate_bounds_center_against_rate() -> None:
    from marconi.engine.stages.conditioning import Translate, TranslateStep

    t = Translate()
    assert t.validate_input_rate(TranslateStep(center_hz=100_000.0), 250_000.0) is None
    msg = t.validate_input_rate(TranslateStep(center_hz=490_000.0), 250_000.0)
    assert msg is not None and "Nyquist" in msg


def test_compile_rejects_translate_beyond_nyquist() -> None:
    from marconi.engine.compile.compiler import compile_pipeline
    from marconi.engine.compile.errors import CompileError
    from marconi.engine.stages.conditioning import TranslateStep
    from marconi.engine.types.models import Modem

    modem = Modem(symbol_rate=1_000.0, path=[TranslateStep(center_hz=490_000.0)])
    with pytest.raises(CompileError, match="Nyquist"):
        compile_pipeline(
            modem,
            stage_registry(),
            sample_rate=250_000.0,
            start=Descriptor(Level.IQ, ItemType.C),
            source_io={"path": "in.cf32"},
            sink_io={"path": "out.cf32"},
        )


def test_channelize_emit_uses_input_rate_and_decim() -> None:
    b = CompileContext(Descriptor(Level.IQ, ItemType.C), rate=16.0, symbol_rate=1.0)
    Channelize().emit_rx(b, ChannelizeStep(decim=2, bandwidth_hz=4.0, center_hz=4.0))
    p = b.build("t", 16.0)
    fx = next(x for x in p.blocks if x.kind == "freq_xlating_fir_filter_ccf")
    assert fx.params["decim"] == 2
    assert fx.params["rate"] == 16.0  # input-rate domain
    assert fx.params["center"] == 4.0
    assert fx.params["cutoff"] == 2.0 and fx.params["transition"] == 2.0  # bandwidth/2


def test_invert_emit_is_conjugate() -> None:
    b = CompileContext(Descriptor(Level.IQ, ItemType.C), rate=16.0, symbol_rate=1.0)
    Invert().emit_rx(b, InvertStep())
    assert [x.kind for x in b.build("t", 16.0).blocks] == ["conjugate_cc"]


def test_param_validation() -> None:
    with pytest.raises(ValidationError):
        ChannelizeStep(bandwidth_hz=4.0)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        ChannelizeStep(decim=0, bandwidth_hz=4.0)


def test_clock_correct_rejects_pole_ppm() -> None:
    with pytest.raises(ValidationError):
        ClockCorrectStep(ppm=-1e6)


def test_clock_correct_accepts_normal_ppm() -> None:
    ClockCorrectStep(ppm=6.0)


def test_rate_threads_through_channelize_to_demod() -> None:
    # The novel bit: a rate-changing stage must hand the DECIMATED rate to the demod.
    # Compile [channelize(decim=2), fsk] at input rate 16 -> fsk sees rate 8.
    from marconi.engine.compile.compiler import compile_modem
    from marconi.engine.modulation.fsk.stages import FskStep
    from marconi.engine.types.descriptor import Descriptor as D
    from marconi.engine.types.models import Modem

    modem = Modem(
        symbol_rate=1.0,
        path=[
            ChannelizeStep(decim=2, bandwidth_hz=4.0),
            FskStep(deviation=0.75),
        ],
    )
    pipe = compile_modem(
        modem,
        stage_registry(),
        sample_rate=16.0,
        start=D(Level.IQ, ItemType.C),
        source_io={"path": "x"},
        sink_io={"path": "y"},
    )
    fx = next(x for x in pipe.blocks if x.kind == "freq_xlating_fir_filter_ccf")
    assert fx.params["rate"] == 16.0  # channelize at the input rate
    qd = next(x for x in pipe.blocks if x.kind == "quadrature_demod")
    # fsk gain = rate / (2*pi*deviation) at the DECIMATED rate 8, not 16
    assert qd.params["gain"] == pytest.approx(8.0 / (2.0 * math.pi * 0.75))


def test_a_passband_far_below_the_input_rate_is_refused() -> None:
    # firdes sizes the FIR as ~rate/transition taps. Tied 1:1 to the
    # passband, 20 Hz on a 2.048 Msps capture asked for ~493k taps and
    # compiled clean, then spent the whole run budget in the filter and
    # returned a timeout naming nothing. A plausible follow-up after survey
    # reports a very narrow occupied bandwidth.
    from marconi.engine.stages.conditioning import Channelize, ChannelizeStep

    step = ChannelizeStep(decim=1, bandwidth_hz=20.0)
    problem = Channelize().validate_input_rate(step, 2_048_000.0)
    assert problem is not None and "Decimate first" in problem
