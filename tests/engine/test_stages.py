from typing import Any, Literal

import pytest
from pydantic import StrictInt, ValidationError

from marconi.engine.stages.base import (
    DuplexStage,
    RxStage,
    Stage,
    StageDirectionError,
    validate_path,
)
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.levels import Level
from marconi.engine.types.models import ValidationIssue
from marconi.engine.types.step import Step


class _FakeBuilder:
    def __init__(self) -> None:
        self.ops: list[str] = []


class _DemodStep(Step):
    conv: Literal["demod"] = "demod"
    sps: StrictInt


class Demod(DuplexStage["_FakeBuilder", _DemodStep]):
    name = "demod"
    from_level = Level.IQ
    to_level = Level.SYMBOLS
    family = "fsk"
    step_model = _DemodStep

    def emit_rx(self, b: "_FakeBuilder", step: _DemodStep) -> None:
        b.ops.append("rx")

    def emit_tx(self, b: "_FakeBuilder", step: _DemodStep) -> None:
        b.ops.append("tx")

    def out_descriptor(self, d: Descriptor, step: _DemodStep) -> Descriptor:
        return Descriptor(self.to_level, "f", carrier=Carrier.SOFT)


class _ResampleStep(Step):
    conv: Literal["resample"] = "resample"
    interpolation: StrictInt
    decimation: StrictInt


class Resample(DuplexStage["_FakeBuilder", _ResampleStep]):
    name = "resample"
    from_level = Level.IQ
    to_level = Level.IQ
    family = "general"
    step_model = _ResampleStep

    def emit_rx(self, b: "_FakeBuilder", step: _ResampleStep) -> None: ...
    def emit_tx(self, b: "_FakeBuilder", step: _ResampleStep) -> None: ...

    def rate_factor(self, step: _ResampleStep) -> float:
        return step.interpolation / step.decimation


class _SinkStep(Step):
    conv: Literal["css_demap"] = "css_demap"
    sf: StrictInt


class Sink(RxStage["_FakeBuilder", _SinkStep]):
    name = "css_demap"
    from_level = Level.SYMBOLS
    to_level = Level.BITS
    family = "css"
    step_model = _SinkStep

    def emit_rx(self, b: "_FakeBuilder", step: _SinkStep) -> None: ...


class _FecStep(Step):
    conv: Literal["fec"] = "fec"


class Fec(DuplexStage["_FakeBuilder", _FecStep]):
    name = "fec"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"
    step_model = _FecStep

    def emit_rx(self, b: "_FakeBuilder", step: _FecStep) -> None: ...
    def emit_tx(self, b: "_FakeBuilder", step: _FecStep) -> None: ...


class _UnknownStep(Step):
    conv: Literal["nope"] = "nope"


def _registry() -> dict[str, Stage[Any, Any]]:
    return {s.name: s for s in (Demod(), Resample(), Sink())}


def test_stage_declares_descriptor_and_rate_transforms() -> None:
    out = Demod().out_descriptor(Descriptor(Level.IQ, "c"), _DemodStep(sps=4))
    assert out == Descriptor(Level.SYMBOLS, "f", carrier=Carrier.SOFT)
    assert Resample().rate_factor(_ResampleStep(interpolation=3, decimation=2)) == 1.5
    assert Demod().rate_factor(_DemodStep(sps=4)) == 1.0  # default identity


def test_rx_only_stage_rejects_tx() -> None:
    with pytest.raises(StageDirectionError):
        Sink().emit_tx(_FakeBuilder(), _SinkStep(sf=7))


def test_valid_path_passes() -> None:
    issues: list[ValidationIssue] = []
    validate_path(
        [_DemodStep(sps=4), _SinkStep(sf=7)],
        _registry(),
        Level.IQ,
        "modem",
        issues,
    )
    assert issues == []


def test_nonadjacent_path_is_flagged() -> None:
    issues: list[ValidationIssue] = []
    validate_path([_SinkStep(sf=7)], _registry(), Level.IQ, "modem", issues)
    assert any(
        "not adjacent" in i.message or "must start at" in i.message for i in issues
    )


def test_skipped_conversion_between_stages_is_flagged() -> None:
    """demod ends at SYMBOLS, fec starts at BITS: the missing SYMBOLS->BITS
    stage must fail validation. The old rung-successor allowance let this
    compile — soft symbols punning as soft bits decoded garbage."""
    issues: list[ValidationIssue] = []
    validate_path(
        [_DemodStep(sps=4), _FecStep()],
        {**_registry(), "fec": Fec()},
        Level.IQ,
        "modem",
        issues,
    )
    assert any("does not match" in i.message for i in issues), issues


def test_unknown_converter_is_flagged() -> None:
    issues: list[ValidationIssue] = []
    validate_path([_UnknownStep()], _registry(), Level.IQ, "modem", issues)
    assert any("unknown converter" in i.message for i in issues)


def test_bad_params_rejected_at_construction() -> None:
    with pytest.raises(ValidationError):
        _DemodStep(sps="x")  # type: ignore[arg-type]
