from typing import Any, Literal

import pytest

from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.compile.errors import CompileError
from marconi.engine.compile.ir import GrPipeline
from marconi.engine.modulation.ook.stages import OokEnvelopeStep
from marconi.engine.stages.base import DuplexStage, RxStage, Stage
from marconi.engine.stages.conditioning import AgcStep, ChannelizeStep
from marconi.engine.stages.general import SliceStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Amplitude, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem
from marconi.engine.types.step import Step

IQ = Descriptor(Level.IQ, ItemType.C)


class _NeedsNormalizedStep(Step):
    conv: Literal["needs_normalized"] = "needs_normalized"


class _EstablishesStep(Step):
    conv: Literal["establishes"] = "establishes"


class _NeedsNormalized(DuplexStage[CompileContext, _NeedsNormalizedStep]):
    name = "needs_normalized"
    from_level = Level.IQ
    to_level = Level.SYMBOLS
    family = "test"
    accepts_amplitude = frozenset({Amplitude.RMS_UNITY})
    step_model = _NeedsNormalizedStep

    def emit_rx(self, b: CompileContext, step: _NeedsNormalizedStep) -> None:
        b.chain("complex_to_mag")

    def emit_tx(self, b: CompileContext, step: _NeedsNormalizedStep) -> None:
        b.chain("float_to_complex")


class _Establishes(RxStage[CompileContext, _EstablishesStep]):
    name = "establishes"
    from_level = Level.IQ
    to_level = Level.IQ
    family = "test"
    step_model = _EstablishesStep

    def emit_rx(self, b: CompileContext, step: _EstablishesStep) -> None:
        b.chain("multiply_const_cc", value=1.0)

    def out_descriptor(self, in_desc: Descriptor, step: _EstablishesStep) -> Descriptor:
        return Descriptor(
            Level.IQ,
            in_desc.item_type,
            in_desc.carrier,
            Amplitude.RMS_UNITY,
        )


class _NeedsPeakStep(Step):
    conv: Literal["needs_peak"] = "needs_peak"


class _NeedsPeak(DuplexStage[CompileContext, _NeedsPeakStep]):
    name = "needs_peak"
    from_level = Level.IQ
    to_level = Level.SYMBOLS
    family = "test"
    accepts_amplitude = frozenset({Amplitude.PEAK_UNITY})
    step_model = _NeedsPeakStep

    def emit_rx(self, b: CompileContext, step: _NeedsPeakStep) -> None:
        b.chain("complex_to_mag")

    def emit_tx(self, b: CompileContext, step: _NeedsPeakStep) -> None:
        b.chain("float_to_complex")


def _registry() -> dict[str, Stage[Any, Any]]:
    reg = dict(stage_registry())
    reg["needs_normalized"] = _NeedsNormalized()
    reg["establishes"] = _Establishes()
    reg["needs_peak"] = _NeedsPeak()
    return reg


def _compile(path: list[Step], direction: Literal["rx", "tx"] = "rx") -> GrPipeline:
    return compile_modem(
        Modem(symbol_rate=1.0, path=path),
        _registry(),
        direction=direction,
        sample_rate=4.0,
        start=IQ,
        source_io={"path": "/dev/null"},
        sink_io={"path": "/dev/null"},
    )


def test_missing_amplitude_establishment_fails_at_compile() -> None:
    with pytest.raises(CompileError) as exc:
        _compile([_NeedsNormalizedStep()])
    msg = str(exc.value)
    assert "needs_normalized" in msg
    assert "normalized" in msg
    assert "unknown" in msg
    assert "mode='power'" in msg
    assert "selects the statistic" not in msg


def test_established_amplitude_satisfies_the_requirement() -> None:
    _compile([_EstablishesStep(), _NeedsNormalizedStep()])


def test_amplitude_check_does_not_apply_to_tx() -> None:
    _compile([_NeedsNormalizedStep()], direction="tx")


def test_wrong_statistic_names_the_corrective_mode() -> None:
    with pytest.raises(CompileError) as exc:
        _compile([_EstablishesStep(), _NeedsPeakStep()])
    msg = str(exc.value)
    assert "peak_unity" in msg
    assert "rms_unity" in msg
    assert "mode='feedforward'" in msg
    assert "selects the statistic" not in msg


def test_amplitude_requiring_stages_declare_measured_statistics() -> None:
    """Each set is the measured gain-invariant subset, not a guess.

    See tests/integration/engine/contracts/test_amplitude_invariance.py::
    test_amplitude_acceptance_matches_the_declared_set, which re-derives
    these by measurement.
    """
    reg = stage_registry()
    assert reg["qam_demod"].accepts_amplitude == frozenset({Amplitude.RMS_UNITY})
    assert reg["ook_envelope"].accepts_amplitude == frozenset(
        {Amplitude.PEAK_UNITY, Amplitude.RMS_UNITY}
    )
    assert reg["psk_demod"].accepts_amplitude == frozenset(
        {Amplitude.PEAK_UNITY, Amplitude.MEAN_MAG_UNITY, Amplitude.RMS_UNITY}
    )


def test_fsk_does_not_require_amplitude() -> None:
    assert stage_registry()["fsk"].accepts_amplitude is None


def test_agc_after_channelize_satisfies_a_requiring_stage() -> None:
    compile_modem(
        Modem(
            symbol_rate=1.0,
            path=[
                ChannelizeStep(decim=1, bandwidth_hz=1.0),
                AgcStep(),
                OokEnvelopeStep(),
                SliceStep(),
            ],
        ),
        stage_registry(),
        direction="rx",
        sample_rate=4.0,
        start=IQ,
        source_io={"path": "/dev/null"},
        sink_io={"path": "/dev/null"},
    )


def test_agc_before_channelize_is_rejected() -> None:
    with pytest.raises(CompileError) as exc:
        compile_modem(
            Modem(
                symbol_rate=1.0,
                path=[
                    AgcStep(),
                    ChannelizeStep(decim=1, bandwidth_hz=1.0),
                    OokEnvelopeStep(),
                    SliceStep(),
                ],
            ),
            stage_registry(),
            direction="rx",
            sample_rate=4.0,
            start=IQ,
            source_io={"path": "/dev/null"},
            sink_io={"path": "/dev/null"},
        )
    assert "ook_envelope" in str(exc.value)
    assert "channelize" in str(exc.value)
