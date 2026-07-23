from collections.abc import Mapping
from typing import Any

import pytest

from marconi.core.descriptor import Amplitude, Descriptor
from marconi.core.levels import Level
from marconi.core.params import StageParams
from marconi.core.stages import DuplexStage, RxStage
from marconi.phy.compile_context import CompileContext
from marconi.phy.compiler import CompileError, compile_modem
from marconi.phy.models import ModemSpec, ModemStep
from marconi.phy.stages import stage_registry

IQ = Descriptor(Level.IQ, "c")


class _NoParams(StageParams):
    pass


class _NeedsNormalized(DuplexStage[CompileContext]):
    name = "needs_normalized"
    from_level = Level.IQ
    to_level = Level.SYMBOLS
    family = "test"
    accepts_amplitude = Amplitude.NORMALIZED
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
            Amplitude.NORMALIZED,
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
