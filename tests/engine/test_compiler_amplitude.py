from collections.abc import Mapping
from typing import Any

import pytest

from marconi.engine.compile_context import CompileContext
from marconi.engine.compiler import CompileError, compile_modem
from marconi.engine.descriptor import Amplitude, Descriptor
from marconi.engine.levels import Level
from marconi.engine.models import ModemSpec, ModemStep
from marconi.engine.params import StageParams
from marconi.engine.stage import DuplexStage, RxStage
from marconi.engine.stages import stage_registry

IQ = Descriptor(Level.IQ, "c")


class _NoParams(StageParams):
    pass


class _NeedsNormalized(DuplexStage[CompileContext]):
    name = "needs_normalized"
    from_level = Level.IQ
    to_level = Level.SYMBOLS
    family = "test"
    accepts_amplitude = frozenset({Amplitude.RMS_UNITY})
    params_model = _NoParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain("complex_to_mag")

    def emit_tx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain("float_to_complex")


class _Establishes(RxStage[CompileContext]):
    name = "establishes"
    from_level = Level.IQ
    to_level = Level.IQ
    family = "test"
    params_model = _NoParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain("multiply_const_cc", value=1.0)

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(
            Level.IQ,
            in_desc.item_type,
            in_desc.layout,
            in_desc.carrier,
            Amplitude.RMS_UNITY,
        )


def _registry() -> dict:
    reg = dict(stage_registry())
    reg["needs_normalized"] = _NeedsNormalized()
    reg["establishes"] = _Establishes()
    return reg


def _compile(path: list[ModemStep], direction: str = "rx"):
    return compile_modem(
        ModemSpec(symbol_rate=1.0, path=path),
        _registry(),
        direction=direction,
        sample_rate=4.0,
        start=IQ,
        source_io={"path": "/dev/null"},
        sink_io={"path": "/dev/null"},
    )


def test_missing_amplitude_establishment_fails_at_compile() -> None:
    with pytest.raises(CompileError) as exc:
        _compile([ModemStep(conv="needs_normalized")])
    assert "needs_normalized" in str(exc.value)
    assert "normalized" in str(exc.value)
    assert "unknown" in str(exc.value)


def test_established_amplitude_satisfies_the_requirement() -> None:
    _compile([ModemStep(conv="establishes"), ModemStep(conv="needs_normalized")])


def test_amplitude_check_does_not_apply_to_tx() -> None:
    _compile([ModemStep(conv="needs_normalized")], direction="tx")


def test_amplitude_requiring_stages_declare_measured_statistics() -> None:
    """Each set is the measured gain-invariant subset, not a guess.

    See tests/phy/test_amplitude_invariance.py::test_declared_amplitude_
    statistics_are_the_ones_that_work, which re-derives these by measurement.
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
        ModemSpec(
            symbol_rate=1.0,
            path=[
                ModemStep(conv="channelize", params={"decim": 1, "bandwidth_hz": 1.0}),
                ModemStep(conv="agc"),
                ModemStep(conv="ook_envelope"),
                ModemStep(conv="slice"),
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
            ModemSpec(
                symbol_rate=1.0,
                path=[
                    ModemStep(conv="agc"),
                    ModemStep(
                        conv="channelize", params={"decim": 1, "bandwidth_hz": 1.0}
                    ),
                    ModemStep(conv="ook_envelope"),
                    ModemStep(conv="slice"),
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
