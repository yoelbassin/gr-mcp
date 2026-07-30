from typing import Literal

from marconi.engine.stages.base import RxStage
from marconi.engine.types.descriptor import Amplitude, Carrier, Descriptor
from marconi.engine.types.levels import Level
from marconi.engine.types.step import Step


class _PassThroughStep(Step):
    conv: Literal["passthrough"] = "passthrough"


class _PassThrough(RxStage[object, _PassThroughStep]):
    name = "passthrough"
    from_level = Level.IQ
    to_level = Level.IQ
    family = "test"
    step_model = _PassThroughStep

    def emit_rx(self, b: object, step: _PassThroughStep) -> None:
        return None


class _Alters(_PassThrough):
    name = "alters"
    alters_amplitude = True


class _Demod(_PassThrough):
    name = "demod"
    to_level = Level.SYMBOLS


_NORMALIZED = Descriptor(Level.IQ, "c", Carrier.HARD, Amplitude.RMS_UNITY)


def test_descriptor_defaults_to_unknown_amplitude() -> None:
    assert Descriptor(Level.IQ, "c").amplitude is Amplitude.UNKNOWN


def test_iq_stage_passes_amplitude_through() -> None:
    out = _PassThrough().out_descriptor(_NORMALIZED, _PassThroughStep())
    assert out.amplitude is Amplitude.RMS_UNITY


def test_altering_stage_invalidates_amplitude() -> None:
    out = _Alters().out_descriptor(_NORMALIZED, _PassThroughStep())
    assert out.amplitude is Amplitude.UNKNOWN


def test_non_iq_output_invalidates_amplitude() -> None:
    out = _Demod().out_descriptor(_NORMALIZED, _PassThroughStep())
    assert out.amplitude is Amplitude.UNKNOWN


def test_stage_declares_no_amplitude_requirement_by_default() -> None:
    assert _PassThrough().accepts_amplitude is None
